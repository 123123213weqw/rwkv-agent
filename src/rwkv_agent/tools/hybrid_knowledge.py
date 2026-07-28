from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib import error, request

from rwkv_search.analysis import QueryAnalyzer
from rwkv_search.candidate_index import CandidateIndexClient


LEXICAL_INDEX_BY_LANGUAGE = {
    "zh": "rwkv-finewiki-zh-full-v1",
    "en": "rwkv-finewiki-en-full-v1",
}
DENSE_INDEX_BY_LANGUAGE = {
    "zh": "rwkv-finewiki-page-e5-small-zh-v1",
    "en": "rwkv-finewiki-page-e5-small-en-v1",
}


class QueryEncoder(Protocol):
    model_name: str

    def encode_queries(self, queries: Sequence[str]) -> Sequence[Sequence[float]]: ...


class PairScorer(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class DenseSearchClient:
    def __init__(self, endpoint: str, *, timeout: float = 45.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)

    def search(
        self,
        index: str,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
    ) -> tuple[list[dict[str, Any]], float]:
        payload = {
            "size": max(1, int(limit)),
            "_source": [
                "page_id",
                "title",
                "text",
                "headings",
                "url",
                "language",
            ],
            "knn": {
                "field": "embedding",
                "query_vector": [float(value) for value in query_vector],
                "k": max(1, int(limit)),
                "num_candidates": max(int(limit), int(num_candidates)),
            },
        }
        started = time.perf_counter()
        req = request.Request(
            f"{self.endpoint}/{index}/_search",
            method="POST",
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"dense search failed with HTTP {exc.code}: {detail}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        hits: list[dict[str, Any]] = []
        for rank, hit in enumerate(
            value.get("hits", {}).get("hits", ()),
            start=1,
        ):
            source = dict(hit.get("_source") or {})
            page_id = str(source.get("page_id") or "")
            if not page_id:
                continue
            hits.append(
                {
                    "doc_id": page_id,
                    "page_id": page_id,
                    "title": str(source.get("title") or ""),
                    "text": str(source.get("text") or ""),
                    "headings": list(source.get("headings") or ()),
                    "url": str(source.get("url") or ""),
                    "source": "finewiki",
                    "language": str(source.get("language") or ""),
                    "score": float(hit.get("_score") or 0.0),
                    "channels": ["dense"],
                    "ranks": {"dense": rank},
                    "chunk_id": 0,
                    "char_start": 0,
                }
            )
        return hits, elapsed_ms


class E5QueryEncoder:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        batch_size: int = 16,
        max_length: int = 256,
        fp16: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "hybrid knowledge search requires torch and transformers"
            ) from exc
        self.model_name = model_path
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        if fp16:
            self.model = self.model.half()
        self.model.eval()
        self._lock = threading.Lock()

    def encode_queries(
        self,
        queries: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        output: list[list[float]] = []
        with self._lock, self.torch.inference_mode():
            for start in range(0, len(queries), self.batch_size):
                texts = [
                    f"query: {value}"
                    for value in queries[start : start + self.batch_size]
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
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                    min=1
                )
                pooled = self.torch.nn.functional.normalize(
                    pooled.float(),
                    p=2,
                    dim=1,
                )
                output.extend(pooled.cpu().tolist())
        return output


class CrossEncoderScorer:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        batch_size: int = 16,
        max_length: int = 512,
        fp16: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hybrid knowledge search requires torch and transformers"
            ) from exc
        self.model_name = model_path
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to(self.device)
        if fp16:
            self.model = self.model.half()
        self.model.eval()
        self._lock = threading.Lock()

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> Sequence[float]:
        if not documents:
            return []
        values: list[float] = []
        pairs = [(query, document) for document in documents]
        with self._lock, self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                inputs = self.tokenizer(
                    pairs[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    name: tensor.to(self.device)
                    for name, tensor in inputs.items()
                }
                logits = self.model(**inputs, return_dict=True).logits
                values.extend(
                    float(value)
                    for value in logits.view(-1).float().cpu()
                )
        return values


def candidate_document(
    hit: Mapping[str, Any],
    *,
    max_chars: int = 2400,
) -> str:
    title = " ".join(str(hit.get("title") or "").split())
    headings = " > ".join(
        " ".join(str(value).split())
        for value in hit.get("headings", ())
        if str(value).strip()
    )
    body = " ".join(str(hit.get("text") or "").split())
    parts = [f"Title: {title}"]
    if headings:
        parts.append(f"Sections: {headings}")
    if body:
        parts.append(f"Passage: {body}")
    return "\n".join(parts)[: max(128, int(max_chars))]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Mapping[str, Any]]],
    *,
    rrf_k: float = 60.0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for channel, ranking in enumerate(rankings):
        for rank, source in enumerate(ranking, start=1):
            page_id = str(source.get("page_id") or "")
            if not page_id:
                continue
            scores[page_id] += 1.0 / (rrf_k + rank)
            documents.setdefault(page_id, dict(source))
            first_seen.setdefault(page_id, (channel, rank))
    ordered = sorted(
        documents,
        key=lambda page_id: (
            -scores[page_id],
            first_seen[page_id],
            page_id,
        ),
    )
    output: list[dict[str, Any]] = []
    for page_id in ordered[: max(1, int(limit))]:
        item = documents[page_id]
        item["fusion_score"] = scores[page_id]
        output.append(item)
    return output


def rerank_candidates(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    scorer: PairScorer,
    *,
    depth: int = 50,
) -> tuple[list[dict[str, Any]], float]:
    raw = [dict(item) for item in candidates]
    selected_depth = min(len(raw), max(1, int(depth)))
    if not selected_depth:
        return raw, 0.0
    started = time.perf_counter()
    scores = list(
        scorer.score(
            query,
            [candidate_document(item) for item in raw[:selected_depth]],
        )
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(scores) != selected_depth:
        raise RuntimeError("cross encoder returned an unexpected score count")
    scored = []
    for position, score in enumerate(scores):
        item = dict(raw[position])
        item["rerank_score"] = float(score)
        scored.append((-float(score), position, item))
    head = [
        item
        for _, _, item in sorted(
            scored,
            key=lambda value: (value[0], value[1]),
        )
    ]
    return head + raw[selected_depth:], elapsed_ms


def _query_terms(query: str) -> tuple[list[str], list[str]]:
    analysis = QueryAnalyzer().analyze(query)
    exact = list(
        dict.fromkeys(
            str(value) for value in analysis.exact_terms if str(value)
        )
    )
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
    chunks_per_page: int = 12,
) -> dict[str, Any]:
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
    bool_query: dict[str, Any] = {
        "filter": [{"terms": {"page_id": pages}}],
    }
    if should:
        bool_query["should"] = should
        bool_query["minimum_should_match"] = 0
    return {
        "size": len(pages),
        "_source": source_fields,
        "query": {"bool": bool_query},
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


def _passage(hit: Mapping[str, Any]) -> dict[str, Any]:
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
    }


class PagePassageClient(CandidateIndexClient):
    def search_pages(
        self,
        query: str,
        *,
        index: str,
        page_ids: Sequence[str],
        chunks_per_page: int = 12,
    ) -> dict[str, list[dict[str, Any]]]:
        payload = json.dumps(
            build_page_passage_query(
                query,
                page_ids,
                chunks_per_page=chunks_per_page,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        response = self._request("POST", f"/{index}/_search", payload)
        output: dict[str, list[dict[str, Any]]] = {}
        for outer in response.get("hits", {}).get("hits", ()):
            page_id = str(outer.get("_source", {}).get("page_id") or "")
            inner = outer.get("inner_hits", {})
            candidates = list(
                inner.get("passage_candidates", {})
                .get("hits", {})
                .get("hits", ())
            )
            leads = list(
                inner.get("page_lead", {})
                .get("hits", {})
                .get("hits", ())
            )
            if not candidates:
                candidates = [outer]
            values: list[dict[str, Any]] = []
            seen: set[str] = set()
            for source in (*candidates, *leads):
                item = _passage(source)
                if not item["doc_id"] or item["doc_id"] in seen:
                    continue
                seen.add(item["doc_id"])
                values.append(item)
            if page_id and values:
                output[page_id] = values
        return output


def _combine_passages(
    lead: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    max_chars: int,
) -> dict[str, Any]:
    first = dict(lead)
    second = dict(selected)
    if str(first.get("doc_id") or "") == str(second.get("doc_id") or ""):
        first["text"] = str(first.get("text") or "")[:max_chars]
        first["component_doc_ids"] = [str(first.get("doc_id") or "")]
        return first
    separator = "\n\n"
    budget = max(512, int(max_chars)) - len(separator)
    first_budget = budget // 2
    second_budget = budget - first_budget
    output = dict(second)
    output.update(
        {
            "doc_id": (
                f"{first.get('doc_id', '')}+{second.get('doc_id', '')}"
            ),
            "chunk_id": -1,
            "char_start": min(
                int(first.get("char_start", 0) or 0),
                int(second.get("char_start", 0) or 0),
            ),
            "text": (
                str(first.get("text") or "")[:first_budget]
                + separator
                + str(second.get("text") or "")[:second_budget]
            ).strip(),
            "component_doc_ids": [
                str(first.get("doc_id") or ""),
                str(second.get("doc_id") or ""),
            ],
        }
    )
    return output


def hydrate_pages(
    query: str,
    pages: Sequence[Mapping[str, Any]],
    passages: Mapping[str, Sequence[Mapping[str, Any]]],
    scorer: PairScorer,
    *,
    max_chars: int = 3200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flattened: list[tuple[str, dict[str, Any]]] = []
    for page in pages:
        page_id = str(page.get("page_id") or "")
        for passage in passages.get(page_id, ()):
            if str(passage.get("text") or "").strip():
                flattened.append((page_id, dict(passage)))
    started = time.perf_counter()
    scores = list(
        scorer.score(
            query,
            [candidate_document(item) for _, item in flattened],
        )
    )
    rerank_ms = (time.perf_counter() - started) * 1000.0
    if len(scores) != len(flattened):
        raise RuntimeError("passage scorer returned an unexpected score count")
    by_page: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for (page_id, passage), score in zip(flattened, scores):
        by_page[page_id].append((passage, float(score)))
    output: list[dict[str, Any]] = []
    changed = 0
    for source in pages:
        page = dict(source)
        candidates = by_page.get(str(page.get("page_id") or ""), ())
        if not candidates:
            output.append(page)
            continue
        lead = min(
            (item for item, _ in candidates),
            key=lambda item: (
                int(item.get("char_start", 0) or 0),
                int(item.get("chunk_id", 0) or 0),
            ),
        )
        selected, score = max(
            candidates,
            key=lambda value: (
                value[1],
                -int(value[0].get("char_start", 0) or 0),
                str(value[0].get("doc_id") or ""),
            ),
        )
        combined = _combine_passages(
            lead,
            selected,
            max_chars=max_chars,
        )
        if str(combined.get("text") or "") != str(page.get("text") or ""):
            changed += 1
        page.update(combined)
        page["passage_score"] = score
        page["hydration_strategy"] = "lead_plus_cross"
        output.append(page)
    return output, {
        "status": "ok",
        "strategy": "lead_plus_cross",
        "requested_pages": len(pages),
        "returned_pages": len(passages),
        "changed_pages": changed,
        "rerank_ms": round(rerank_ms, 3),
    }


@dataclass(frozen=True)
class HybridSearchResult:
    hits: tuple[dict[str, Any], ...]
    stats: dict[str, Any]


class HybridKnowledgeRetriever:
    def __init__(
        self,
        endpoint: str,
        *,
        encoder: QueryEncoder,
        scorer: PairScorer,
        lexical_client: Any | None = None,
        dense_client: Any | None = None,
        passage_client: Any | None = None,
        candidate_limit: int = 100,
        dense_num_candidates: int = 1000,
        rerank_depth: int = 50,
        result_limit: int = 5,
        chunks_per_page: int = 12,
        passage_max_chars: int = 3200,
    ) -> None:
        self.encoder = encoder
        self.scorer = scorer
        self.lexical = lexical_client or CandidateIndexClient(
            endpoint,
            timeout=45.0,
        )
        self.dense = dense_client or DenseSearchClient(
            endpoint,
            timeout=45.0,
        )
        self.passages = passage_client or PagePassageClient(
            endpoint,
            timeout=45.0,
        )
        self.candidate_limit = max(5, int(candidate_limit))
        self.dense_num_candidates = max(
            self.candidate_limit,
            int(dense_num_candidates),
        )
        self.rerank_depth = max(1, int(rerank_depth))
        self.result_limit = max(1, int(result_limit))
        self.chunks_per_page = max(1, int(chunks_per_page))
        self.passage_max_chars = max(512, int(passage_max_chars))

    def search(self, query: str, *, language: str) -> HybridSearchResult:
        if language not in LEXICAL_INDEX_BY_LANGUAGE:
            raise ValueError(f"unsupported language: {language}")
        total_started = time.perf_counter()
        lexical_index = LEXICAL_INDEX_BY_LANGUAGE[language]
        dense_index = DENSE_INDEX_BY_LANGUAGE[language]
        _, lexical_hits, lexical_ms = self.lexical.search(
            query,
            index=lexical_index,
            channel_size=self.candidate_limit,
            limit=self.candidate_limit,
            max_chunks_per_page=2,
        )
        lexical = [
            hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
            for hit in lexical_hits
        ]
        dense_started = time.perf_counter()
        vectors = self.encoder.encode_queries([query])
        if len(vectors) != 1:
            raise RuntimeError("query encoder returned an unexpected vector count")
        dense, dense_search_ms = self.dense.search(
            dense_index,
            vectors[0],
            limit=self.candidate_limit,
            num_candidates=self.dense_num_candidates,
        )
        dense_ms = (time.perf_counter() - dense_started) * 1000.0
        fused = reciprocal_rank_fusion(
            (lexical, dense),
            limit=self.candidate_limit,
        )
        reranked, rerank_ms = rerank_candidates(
            query,
            fused,
            self.scorer,
            depth=self.rerank_depth,
        )
        selected = reranked[: self.result_limit]
        if selected:
            passage_started = time.perf_counter()
            passage_candidates = self.passages.search_pages(
                query,
                index=lexical_index,
                page_ids=[
                    str(item.get("page_id") or "") for item in selected
                ],
                chunks_per_page=self.chunks_per_page,
            )
            passage_query_ms = (
                time.perf_counter() - passage_started
            ) * 1000.0
            hydrated, hydration = hydrate_pages(
                query,
                selected,
                passage_candidates,
                self.scorer,
                max_chars=self.passage_max_chars,
            )
        else:
            passage_query_ms = 0.0
            hydrated = []
            hydration = {
                "status": "empty",
                "strategy": "lead_plus_cross",
                "requested_pages": 0,
                "returned_pages": 0,
                "changed_pages": 0,
                "rerank_ms": 0.0,
            }
        return HybridSearchResult(
            hits=tuple(hydrated),
            stats={
                "status": "ok",
                "strategy": "lexical_dense_rrf_cross_lead_plus_cross",
                "lexical_index": lexical_index,
                "dense_index": dense_index,
                "candidate_limit": self.candidate_limit,
                "rerank_depth": self.rerank_depth,
                "result_limit": self.result_limit,
                "latency_ms": {
                    "lexical": round(float(lexical_ms), 3),
                    "embedding_and_dense": round(dense_ms, 3),
                    "dense_search": round(float(dense_search_ms), 3),
                    "page_rerank": round(rerank_ms, 3),
                    "passage_query": round(passage_query_ms, 3),
                    "passage_rerank": hydration["rerank_ms"],
                    "total": round(
                        (time.perf_counter() - total_started) * 1000.0,
                        3,
                    ),
                },
                "hydration": hydration,
            },
        )


class LazyHybridKnowledgeRetriever:
    def __init__(
        self,
        endpoint: str,
        *,
        embedding_model: str,
        reranker_model: str,
        device: str,
    ) -> None:
        self.endpoint = endpoint
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.device = device
        self._retriever: HybridKnowledgeRetriever | None = None
        self._lock = threading.Lock()

    def _get(self) -> HybridKnowledgeRetriever:
        with self._lock:
            if self._retriever is None:
                encoder = E5QueryEncoder(
                    self.embedding_model,
                    device=self.device,
                )
                scorer = CrossEncoderScorer(
                    self.reranker_model,
                    device=self.device,
                )
                self._retriever = HybridKnowledgeRetriever(
                    self.endpoint,
                    encoder=encoder,
                    scorer=scorer,
                )
            return self._retriever

    def search(self, query: str, *, language: str) -> HybridSearchResult:
        return self._get().search(query, language=language)


class HybridKnowledgeShadow:
    def __init__(
        self,
        retriever: Any,
        *,
        log_path: str = "",
        max_pending: int = 8,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.retriever = retriever
        self.log_path = Path(log_path) if log_path else None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hybrid-knowledge-shadow",
        )
        self._slots = threading.BoundedSemaphore(max(1, int(max_pending)))
        self._write_lock = threading.Lock()

    def compare(
        self,
        query: str,
        *,
        language: str,
        legacy_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = self.retriever.search(query, language=language)
            row = {
                "schema_version": "agent-knowledge-shadow.v1",
                "status": "ok",
                "query": query,
                "language": language,
                "legacy_page_ids": [
                    str(item.get("page_id") or "")
                    for item in legacy_evidence
                ],
                "hybrid_page_ids": [
                    str(item.get("page_id") or "")
                    for item in result.hits
                ],
                "hybrid_hits": list(result.hits),
                "hybrid_stats": result.stats,
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
            }
        except Exception as exc:
            row = {
                "schema_version": "agent-knowledge-shadow.v1",
                "status": "fallback_legacy",
                "query": query,
                "language": language,
                "legacy_page_ids": [
                    str(item.get("page_id") or "")
                    for item in legacy_evidence
                ],
                "hybrid_page_ids": [],
                "hybrid_hits": [],
                "hybrid_stats": {},
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
            }
        self._write(row)
        return row

    def submit(
        self,
        query: str,
        *,
        language: str,
        legacy_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not self._slots.acquire(blocking=False):
            return {
                "enabled": True,
                "submitted": False,
                "reason": "queue_full",
                "visible_strategy": "legacy",
            }
        try:
            future = self._executor.submit(
                self.compare,
                query,
                language=language,
                legacy_evidence=[dict(item) for item in legacy_evidence],
            )
        except RuntimeError:
            self._slots.release()
            return {
                "enabled": True,
                "submitted": False,
                "reason": "shadow_closed",
                "visible_strategy": "legacy",
            }
        future.add_done_callback(lambda _future: self._slots.release())
        return {
            "enabled": True,
            "submitted": True,
            "visible_strategy": "legacy",
        }

    def _write(self, row: Mapping[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        value = json.dumps(
            dict(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._write_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(value + "\n")

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def build_shadow_from_env(endpoint: str) -> HybridKnowledgeShadow | None:
    enabled = os.getenv("RWKV_AGENT_KNOWLEDGE_SHADOW", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    embedding_model = os.getenv(
        "RWKV_AGENT_EMBEDDING_MODEL",
        "models/multilingual-e5-small",
    )
    reranker_model = os.getenv(
        "RWKV_AGENT_RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3",
    )
    device = os.getenv("RWKV_AGENT_RETRIEVAL_DEVICE", "cuda:0")
    log_path = os.getenv(
        "RWKV_AGENT_KNOWLEDGE_SHADOW_LOG",
        "var/knowledge-shadow.jsonl",
    )
    return HybridKnowledgeShadow(
        LazyHybridKnowledgeRetriever(
            endpoint,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            device=device,
        ),
        log_path=log_path,
    )
