from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
from urllib import error, request


def page_document(source: Mapping[str, Any], *, max_chars: int = 2600) -> str:
    title = " ".join(str(source.get("title_original") or "").split())
    aliases = [
        " ".join(str(value).split())
        for value in source.get("alias_original", ())
        if str(value).strip()
    ][:16]
    headings = [
        " ".join(str(value).split())
        for value in source.get("heading_original", ())
        if str(value).strip()
    ][:16]
    body = " ".join(str(source.get("body_original") or "").split())
    parts = [f"Title: {title}"]
    if aliases:
        parts.append(f"Aliases: {'; '.join(aliases)}")
    if headings:
        parts.append(f"Sections: {' > '.join(headings)}")
    if body:
        parts.append(f"Passage: {body}")
    return "\n".join(parts)[: max(256, int(max_chars))]


@dataclass(frozen=True)
class BuildCheckpoint:
    source_index: str
    target_index: str
    last_sort: tuple[str, ...]
    indexed_pages: int
    complete: bool
    page_id_gte: str = ""
    page_id_lt: str = ""

    @classmethod
    def load(cls, path: Path) -> "BuildCheckpoint":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            source_index=str(value["source_index"]),
            target_index=str(value["target_index"]),
            last_sort=tuple(str(item) for item in value.get("last_sort", ())),
            indexed_pages=int(value.get("indexed_pages") or 0),
            complete=bool(value.get("complete")),
            page_id_gte=str(value.get("page_id_gte") or ""),
            page_id_lt=str(value.get("page_id_lt") or ""),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "finewiki-page-embedding-checkpoint.v1",
                    "source_index": self.source_index,
                    "target_index": self.target_index,
                    "last_sort": list(self.last_sort),
                    "indexed_pages": self.indexed_pages,
                    "complete": self.complete,
                    "page_id_gte": self.page_id_gte,
                    "page_id_lt": self.page_id_lt,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


class ElasticsearchClient:
    def __init__(self, endpoint: str, *, timeout: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)

    def call(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        body: bytes | None = None,
        content_type: str = "application/json",
        timeout: float | None = None,
    ) -> Any:
        if payload is not None and body is not None:
            raise ValueError("payload and body are mutually exclusive")
        data = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if payload is not None
            else body
        )
        req = request.Request(
            self.endpoint + path,
            method=method,
            data=data,
            headers={"Content-Type": content_type},
        )
        try:
            with request.urlopen(req, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"{method} {path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        return json.loads(raw) if raw else None

    def create_vector_index(
        self,
        index: str,
        *,
        dims: int,
        shards: int,
        recreate: bool,
    ) -> None:
        if recreate:
            try:
                self.call("DELETE", f"/{index}")
            except RuntimeError as exc:
                if "index_not_found_exception" not in str(exc):
                    raise
        mapping = {
            "settings": {
                "number_of_shards": max(1, int(shards)),
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            },
            "mappings": {
                "dynamic": "strict",
                "_source": {"excludes": ["embedding"]},
                "properties": {
                    "page_id": {"type": "keyword"},
                    "title": {"type": "keyword", "index": False},
                    "text": {"type": "text", "index": False},
                    "headings": {"type": "keyword", "index": False},
                    "url": {"type": "keyword", "index": False},
                    "language": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": int(dims),
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "int8_hnsw",
                            "m": 16,
                            "ef_construction": 100,
                        },
                    },
                },
            },
        }
        self.call("PUT", f"/{index}", mapping)

    def fetch_lead_pages(
        self,
        index: str,
        *,
        batch_size: int,
        search_after: Sequence[str] = (),
        page_id_gte: str = "",
        page_id_lt: str = "",
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [{"term": {"chunk_id": 0}}]
        page_range = {
            name: value
            for name, value in (("gte", page_id_gte), ("lt", page_id_lt))
            if value
        }
        if page_range:
            filters.append({"range": {"page_id": page_range}})
        payload: dict[str, Any] = {
            "size": max(1, min(5000, int(batch_size))),
            "track_total_hits": False,
            "_source": [
                "page_id",
                "title_original",
                "alias_original",
                "heading_original",
                "body_original",
                "url",
                "language",
            ],
            "query": {"bool": {"filter": filters}},
            "sort": [{"page_id": "asc"}],
        }
        if search_after:
            payload["search_after"] = list(search_after)
        result = self.call("POST", f"/{index}/_search", payload)
        return list(result.get("hits", {}).get("hits", ()))

    def bulk_pages(
        self,
        index: str,
        documents: Sequence[Mapping[str, Any]],
    ) -> None:
        lines = []
        for document in documents:
            lines.append(
                json.dumps(
                    {"index": {"_index": index, "_id": document["page_id"]}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            lines.append(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        result = self.call(
            "POST",
            "/_bulk",
            body=("\n".join(lines) + "\n").encode("utf-8"),
            content_type="application/x-ndjson",
            timeout=max(self.timeout, 300.0),
        )
        if result.get("errors"):
            failures = [
                item.get("index", {}).get("error")
                for item in result.get("items", ())
                if item.get("index", {}).get("error")
            ]
            raise RuntimeError(f"vector bulk failed: {failures[:3]}")

    def finish_index(self, index: str) -> None:
        self.call(
            "PUT",
            f"/{index}/_settings",
            {"index": {"refresh_interval": "30s"}},
        )
        self.call("POST", f"/{index}/_refresh", timeout=300.0)
        self.call("POST", f"/{index}/_flush?wait_if_ongoing=true", timeout=300.0)


class E5Encoder:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
        fp16: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("embedding build requires torch and transformers") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        if fp16:
            self.model = self.model.half()
        self.model.eval()
        self.dims = int(self.model.config.hidden_size)

    def _encode(self, documents: Sequence[str], *, prefix: str) -> list[list[float]]:
        output: list[list[float]] = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(documents), self.batch_size):
                texts = [
                    f"{prefix}: {value}"
                    for value in documents[start : start + self.batch_size]
                ]
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    name: tensor.to(self.device)
                    for name, tensor in inputs.items()
                }
                hidden = self.model(**inputs, return_dict=True).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                output.extend(pooled.cpu().tolist())
        return output

    def encode_passages(self, documents: Sequence[str]) -> list[list[float]]:
        return self._encode(documents, prefix="passage")

    def encode_queries(self, queries: Sequence[str]) -> list[list[float]]:
        return self._encode(queries, prefix="query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated page-level multilingual E5 index from FineWiki lead chunks."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--target-index", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fetch-size", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Write a new bounded partition into an already-created target index.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--page-id-gte", default="")
    parser.add_argument("--page-id-lt", default="")
    parser.add_argument(
        "--defer-finish",
        action="store_true",
        help="Leave refresh disabled so another partition can continue building.",
    )
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if args.resume:
        checkpoint = BuildCheckpoint.load(checkpoint_path)
        if (
            checkpoint.source_index != args.source_index
            or checkpoint.target_index != args.target_index
        ):
            raise SystemExit("checkpoint source/target index does not match")
        if (checkpoint.page_id_gte or checkpoint.page_id_lt) and (
            checkpoint.page_id_gte != args.page_id_gte
            or checkpoint.page_id_lt != args.page_id_lt
        ):
            raise SystemExit("checkpoint page range does not match")
        if checkpoint.complete:
            print(json.dumps({"event": "already_complete", "indexed": checkpoint.indexed_pages}))
            return 0
    else:
        checkpoint = BuildCheckpoint(
            source_index=args.source_index,
            target_index=args.target_index,
            last_sort=(),
            indexed_pages=0,
            complete=False,
            page_id_gte=args.page_id_gte,
            page_id_lt=args.page_id_lt,
        )
    encoder = E5Encoder(
        args.model,
        device=args.device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        fp16=args.fp16,
    )
    client = ElasticsearchClient(args.endpoint)
    if args.recreate and args.append:
        raise SystemExit("--recreate and --append are mutually exclusive")
    if not args.resume and not args.append:
        client.create_vector_index(
            args.target_index,
            dims=encoder.dims,
            shards=args.shards,
            recreate=args.recreate,
        )
    started = time.perf_counter()
    indexed = checkpoint.indexed_pages
    start_indexed = indexed
    last_sort = checkpoint.last_sort
    while True:
        hits = client.fetch_lead_pages(
            args.source_index,
            batch_size=args.fetch_size,
            search_after=last_sort,
            page_id_gte=args.page_id_gte,
            page_id_lt=args.page_id_lt,
        )
        if not hits:
            break
        if args.limit:
            hits = hits[: max(0, args.limit - indexed)]
        if not hits:
            break
        sources = [dict(hit.get("_source") or {}) for hit in hits]
        documents = [page_document(source) for source in sources]
        embeddings = encoder.encode_passages(documents)
        if len(embeddings) != len(sources):
            raise RuntimeError("embedding count does not match source page count")
        payloads = []
        for source, embedding in zip(sources, embeddings):
            payloads.append(
                {
                    "page_id": str(source.get("page_id") or ""),
                    "title": str(source.get("title_original") or ""),
                    "text": str(source.get("body_original") or ""),
                    "headings": list(source.get("heading_original", ()) or ()),
                    "url": str(source.get("url") or ""),
                    "language": str(source.get("language") or ""),
                    "embedding": embedding,
                }
            )
        if any(not item["page_id"] for item in payloads):
            raise RuntimeError("source page is missing page_id")
        client.bulk_pages(args.target_index, payloads)
        indexed += len(payloads)
        last_sort = tuple(str(value) for value in hits[-1].get("sort", ()))
        checkpoint = BuildCheckpoint(
            source_index=args.source_index,
            target_index=args.target_index,
            last_sort=last_sort,
            indexed_pages=indexed,
            complete=False,
            page_id_gte=args.page_id_gte,
            page_id_lt=args.page_id_lt,
        )
        checkpoint.write(checkpoint_path)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "event": "progress",
                    "indexed_pages": indexed,
                    "batch_pages": len(payloads),
                    "pages_per_second": round(
                        (indexed - start_indexed) / max(elapsed, 1e-6),
                        3,
                    ),
                    "last_sort": list(last_sort),
                }
            ),
            flush=True,
        )
        if args.limit and indexed >= args.limit:
            break
    if not args.defer_finish:
        client.finish_index(args.target_index)
    BuildCheckpoint(
        source_index=args.source_index,
        target_index=args.target_index,
        last_sort=last_sort,
        indexed_pages=indexed,
        complete=not bool(args.limit),
        page_id_gte=args.page_id_gte,
        page_id_lt=args.page_id_lt,
    ).write(checkpoint_path)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "event": "complete" if not args.limit else "limit_reached",
                "indexed_pages": indexed,
                "indexed_pages_this_run": indexed - start_indexed,
                "elapsed_seconds": round(elapsed, 3),
                "pages_per_second": round(
                    (indexed - start_indexed) / max(elapsed, 1e-6),
                    3,
                ),
                "embedding_dims": encoder.dims,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
