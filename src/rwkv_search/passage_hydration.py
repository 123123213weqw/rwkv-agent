from __future__ import annotations

from dataclasses import dataclass, replace
import json
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

from .analysis import QueryAnalyzer
from .candidate_index import CandidateHit, CandidateIndexClient


class PassageScorer(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


def candidate_document(passage: Mapping[str, Any], *, max_chars: int = 2400) -> str:
    title = " ".join(str(passage.get("title") or "").split())
    headings = " > ".join(
        " ".join(str(value).split())
        for value in passage.get("headings", ())
        if str(value).strip()
    )
    body = " ".join(str(passage.get("text") or "").split())
    parts = [f"Title: {title}"]
    if headings:
        parts.append(f"Sections: {headings}")
    if body:
        parts.append(f"Passage: {body}")
    return "\n".join(parts)[: max(128, int(max_chars))]


def _query_terms(query: str) -> tuple[list[str], list[str]]:
    analysis = QueryAnalyzer().analyze(query)
    exact = list(dict.fromkeys(str(value) for value in analysis.exact_terms if value))
    words = list(
        dict.fromkeys(
            token.normalized
            for token in analysis.tokens
            if token.normalized
            and token.kind in {"word", "number", "symbol"}
            and token.weight > 0.15
        )
    )
    return exact, words


def build_page_passage_query(
    query: str,
    page_ids: Sequence[str],
    *,
    chunks_per_page: int,
) -> dict[str, Any]:
    """Build a generic passage query restricted to already-ranked pages."""

    pages = list(dict.fromkeys(str(value) for value in page_ids if str(value)))
    if not pages:
        raise ValueError("page_ids must not be empty")
    exact, words = _query_terms(query)
    should: list[dict[str, Any]] = []
    for term in exact:
        should.extend(
            [
                {"term": {"heading_exact": {"value": term, "boost": 4.0}}},
                {"term": {"body_exact": {"value": term, "boost": 1.5}}},
            ]
        )
    if words:
        should.append(
            {
                "multi_match": {
                    "query": " ".join(words),
                    "fields": [
                        "heading_words^4",
                        "body_words^2",
                        "title_words^1.5",
                        "heading_bigrams^1.2",
                        "body_bigrams^0.6",
                    ],
                    "type": "best_fields",
                }
            }
        )
    source_fields = [
        "doc_id",
        "page_id",
        "chunk_id",
        "title_original",
        "heading_original",
        "body_original",
        "url",
        "source",
        "language",
        "modified_at",
        "char_start",
    ]
    scored_query: dict[str, Any] = {
        "bool": {"filter": [{"terms": {"page_id": pages}}]}
    }
    if should:
        scored_query["bool"]["should"] = should
        # Page ranking already admitted these pages. No lexical overlap should
        # still return the page lead as a deterministic fallback.
        scored_query["bool"]["minimum_should_match"] = 0
    return {
        "size": len(pages),
        "_source": source_fields,
        "query": scored_query,
        "sort": [{"_score": "desc"}, {"page_id": "asc"}],
        "collapse": {
            "field": "page_id",
            "inner_hits": [
                {
                    "name": "passage_candidates",
                    "size": max(1, int(chunks_per_page)),
                    "sort": [{"_score": "desc"}, {"char_start": "asc"}],
                    "_source": source_fields,
                },
                {
                    "name": "page_lead",
                    "size": 1,
                    "sort": [{"char_start": "asc"}],
                    "_source": source_fields,
                },
            ],
        },
    }


def _passage_from_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(hit.get("_source") or {})
    return {
        "doc_id": str(hit.get("_id") or source.get("doc_id") or ""),
        "page_id": str(source.get("page_id") or ""),
        "chunk_id": int(source.get("chunk_id", -1) or 0),
        "title": str(source.get("title_original") or ""),
        "headings": list(source.get("heading_original") or ()),
        "text": str(source.get("body_original") or ""),
        "url": str(source.get("url") or ""),
        "source": str(source.get("source") or "finewiki"),
        "language": str(source.get("language") or ""),
        "modified_at": str(source.get("modified_at") or ""),
        "char_start": int(source.get("char_start", 0) or 0),
        "lexical_score": float(hit.get("_score") or 0.0),
    }


class PagePassageClient(CandidateIndexClient):
    """Read-only page hydrator over an existing FineWiki chunk index."""

    def search_pages(
        self,
        query: str,
        *,
        index: str,
        page_ids: Sequence[str],
        chunks_per_page: int = 12,
    ) -> dict[str, list[dict[str, Any]]]:
        body = json.dumps(
            build_page_passage_query(
                query,
                page_ids,
                chunks_per_page=chunks_per_page,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        response = self._request("POST", f"/{index}/_search", body)
        pages: dict[str, list[dict[str, Any]]] = {}
        for outer in response.get("hits", {}).get("hits", ()):
            page_id = str(outer.get("_source", {}).get("page_id") or "")
            inner = outer.get("inner_hits", {})
            candidates = list(
                inner.get("passage_candidates", {})
                .get("hits", {})
                .get("hits", ())
            )
            leads = list(
                inner.get("page_lead", {}).get("hits", {}).get("hits", ())
            )
            if not candidates:
                candidates = [outer]
            ordered: list[dict[str, Any]] = []
            seen: set[str] = set()
            for hit in (*candidates, *leads):
                passage = _passage_from_hit(hit)
                if not passage["doc_id"] or passage["doc_id"] in seen:
                    continue
                seen.add(passage["doc_id"])
                ordered.append(passage)
            if page_id and ordered:
                pages[page_id] = ordered
        return pages


def combine_passages(
    lead: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    max_chars: int = 3200,
) -> dict[str, Any]:
    """Keep the page lead and query-specific passage within one Evidence."""

    lead_value = dict(lead)
    selected_value = dict(selected)
    if str(lead_value.get("doc_id") or "") == str(
        selected_value.get("doc_id") or ""
    ):
        output = lead_value
        output["text"] = str(output.get("text") or "").strip()[
            : max(512, int(max_chars))
        ]
        output["component_doc_ids"] = [str(lead_value.get("doc_id") or "")]
        return output
    separator = "\n\n"
    text_budget = max(512, int(max_chars)) - len(separator)
    lead_budget = text_budget // 2
    selected_budget = text_budget - lead_budget
    lead_text = str(lead_value.get("text") or "").strip()[:lead_budget]
    selected_text = str(selected_value.get("text") or "").strip()[:selected_budget]
    output = dict(selected_value)
    output.update(
        {
            "doc_id": (
                f"{lead_value.get('doc_id', '')}+{selected_value.get('doc_id', '')}"
            ),
            "chunk_id": -1,
            "char_start": min(
                int(lead_value.get("char_start", 0) or 0),
                int(selected_value.get("char_start", 0) or 0),
            ),
            "headings": list(
                dict.fromkeys(
                    [
                        *(
                            str(value)
                            for value in lead_value.get("headings", ())
                            if value
                        ),
                        *(
                            str(value)
                            for value in selected_value.get("headings", ())
                            if value
                        ),
                    ]
                )
            ),
            "text": f"{lead_text}{separator}{selected_text}".strip(),
            "component_doc_ids": [
                str(lead_value.get("doc_id") or ""),
                str(selected_value.get("doc_id") or ""),
            ],
        }
    )
    return output


@dataclass(frozen=True)
class PassageHydrationResult:
    hits: tuple[CandidateHit, ...]
    stats: dict[str, Any]


class PassageHydrator:
    """Hydrate already-ranked pages without changing their order or identity."""

    def __init__(
        self,
        client: PagePassageClient,
        scorer: PassageScorer,
        *,
        max_pages: int = 8,
        chunks_per_page: int = 12,
        max_chars: int = 3200,
    ) -> None:
        self.client = client
        self.scorer = scorer
        self.max_pages = max(1, int(max_pages))
        self.chunks_per_page = max(1, int(chunks_per_page))
        self.max_chars = max(512, int(max_chars))

    def hydrate(
        self,
        query: str,
        *,
        index: str,
        hits: Sequence[CandidateHit],
    ) -> PassageHydrationResult:
        started = time.perf_counter()
        head = list(hits[: self.max_pages])
        tail = list(hits[self.max_pages :])
        page_ids = list(dict.fromkeys(hit.page_id for hit in head if hit.page_id))
        if not page_ids:
            return PassageHydrationResult(
                hits=tuple(hits),
                stats=self._stats(
                    requested=0,
                    returned=0,
                    changed=0,
                    query_ms=0.0,
                    rerank_ms=0.0,
                    total_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )

        query_started = time.perf_counter()
        pages = self.client.search_pages(
            query,
            index=index,
            page_ids=page_ids,
            chunks_per_page=self.chunks_per_page,
        )
        query_ms = (time.perf_counter() - query_started) * 1000.0

        flattened: list[tuple[str, dict[str, Any]]] = []
        for page_id in page_ids:
            for passage in pages.get(page_id, ()):
                if str(passage.get("text") or "").strip():
                    flattened.append((page_id, dict(passage)))
        rerank_started = time.perf_counter()
        scores = list(
            self.scorer.score(
                query,
                [candidate_document(passage) for _, passage in flattened],
            )
        )
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        if len(scores) != len(flattened):
            raise ValueError("passage scorer returned an unexpected score count")

        by_page: dict[str, list[tuple[dict[str, Any], float]]] = {}
        for (page_id, passage), score in zip(flattened, scores):
            by_page.setdefault(page_id, []).append((passage, float(score)))

        output: list[CandidateHit] = []
        changed = 0
        for hit in head:
            candidates = by_page.get(hit.page_id, ())
            if not candidates:
                output.append(hit)
                continue
            passages = [passage for passage, _ in candidates]
            lead = min(
                passages,
                key=lambda passage: (
                    int(passage.get("char_start", 0) or 0),
                    int(passage.get("chunk_id", 0) or 0),
                ),
            )
            selected, selected_score = max(
                candidates,
                key=lambda item: (
                    item[1],
                    -int(item[0].get("char_start", 0) or 0),
                    str(item[0].get("doc_id") or ""),
                ),
            )
            combined = combine_passages(
                lead,
                selected,
                max_chars=self.max_chars,
            )
            text = str(combined.get("text") or "").strip()
            if not text:
                output.append(hit)
                continue
            if text != hit.text:
                changed += 1
            output.append(
                replace(
                    hit,
                    doc_id=str(combined.get("doc_id") or hit.doc_id),
                    text=text,
                    chunk_id=int(combined.get("chunk_id", -1) or 0),
                    char_start=int(combined.get("char_start", 0) or 0),
                    headings=tuple(
                        str(value)
                        for value in combined.get("headings", ())
                        if value
                    ),
                    passage_score=float(selected_score),
                    candidate_chunk_count=len(passages),
                    hydration_strategy="lead_plus_cross",
                    component_doc_ids=tuple(
                        str(value)
                        for value in combined.get("component_doc_ids", ())
                        if value
                    ),
                )
            )
        output.extend(tail)
        return PassageHydrationResult(
            hits=tuple(output),
            stats=self._stats(
                requested=len(page_ids),
                returned=len(pages),
                changed=changed,
                query_ms=query_ms,
                rerank_ms=rerank_ms,
                total_ms=(time.perf_counter() - started) * 1000.0,
            ),
        )

    def _stats(
        self,
        *,
        requested: int,
        returned: int,
        changed: int,
        query_ms: float,
        rerank_ms: float,
        total_ms: float,
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "status": "ok",
            "strategy": "lead_plus_cross",
            "model": str(getattr(self.scorer, "model_name", "")),
            "max_pages": self.max_pages,
            "chunks_per_page": self.chunks_per_page,
            "max_chars": self.max_chars,
            "pages_requested": requested,
            "pages_returned": returned,
            "changed_evidence_count": changed,
            "query_latency_ms": round(float(query_ms), 3),
            "rerank_latency_ms": round(float(rerank_ms), 3),
            "latency_ms": round(float(total_ms), 3),
        }


class TransformersPassageScorer:
    """Lazy, serialized Cross-Encoder scorer for the optional Shadow path."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        fp16: bool = True,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.fp16 = bool(fp16)
        self.local_files_only = bool(local_files_only)
        self._lock = threading.Lock()
        self._torch: Any = None
        self._device: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "passage hydration requires torch and transformers"
            ) from exc
        device_name = self.device_name
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        ).to(device)
        if self.fp16 and device.type == "cuda":
            model = model.half()
        model.eval()
        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        with self._lock:
            self._load()
            pairs = [(query, document) for document in documents]
            values: list[float] = []
            with self._torch.inference_mode():
                for start in range(0, len(pairs), self.batch_size):
                    batch = pairs[start : start + self.batch_size]
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    inputs = {
                        name: tensor.to(self._device)
                        for name, tensor in inputs.items()
                    }
                    logits = self._model(**inputs, return_dict=True).logits
                    values.extend(
                        float(value)
                        for value in logits.view(-1).float().cpu()
                    )
            return values
