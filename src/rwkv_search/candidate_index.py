from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib import error, request

from .analysis import DocumentAnalyzer, QueryAnalyzer
from .analysis.models import QueryAnalysis
from .analysis.normalization import normalize_text
from .wikipedia import WikipediaChunk


DEFAULT_INDEX = "rwkv-wikipedia-candidate-v1"
CHANNEL_WEIGHTS = {"identity": 5.0, "exact": 3.0, "alias": 3.0, "word": 1.2, "bigram": 0.7}
_RETRIEVAL_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "有", "为", "由", "对",
    "中", "上", "下", "一个", "一种", "这个", "那个", "谁", "什么", "怎么",
    "怎样", "如何", "为什么", "请问", "一下", "what", "which", "how", "please",
}
_OPENCC_T2S: Any = None
_PASSAGE_INNER_HITS = "top_passages"
_LEAD_INNER_HITS = "lead_passage"
_DEFINITION_QUERY = re.compile(
    r"(?:什么[是事]|是什[么麼]|是啥|啥是|何为|什么意思|定义|"
    r"\bwhat\s+(?:is|are)\b)",
    re.IGNORECASE,
)
_DEFINITION_PASSAGE = re.compile(
    r"(?:是(?:一[种個个]|指|指的|用于)|指的是|是由|属于|"
    r"refers?\s+to|is\s+(?:an?|the)\b)",
    re.IGNORECASE,
)


def _script_equivalent(left: str, right: str) -> bool:
    global _OPENCC_T2S
    left_normalized = normalize_text(left).text
    right_normalized = normalize_text(right).text
    if left_normalized == right_normalized:
        return True
    if _OPENCC_T2S is None:
        try:
            from opencc import OpenCC
            _OPENCC_T2S = OpenCC("t2s")
        except ImportError:
            _OPENCC_T2S = False
    return bool(
        _OPENCC_T2S
        and normalize_text(_OPENCC_T2S.convert(left)).text
        == normalize_text(_OPENCC_T2S.convert(right)).text
    )


def candidate_index_mapping(*, shards: int = 2) -> Dict[str, Any]:
    whitespace = {"type": "text", "analyzer": "whitespace", "search_analyzer": "whitespace"}
    return {
        "settings": {
            "number_of_shards": max(1, int(shards)),
            "number_of_replicas": 0,
            "refresh_interval": "-1",
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "doc_id": {"type": "keyword"},
                "page_id": {"type": "keyword"},
                "chunk_id": {"type": "integer"},
                "title_original": {"type": "keyword", "index": False, "ignore_above": 32766},
                "title_normalized": {"type": "keyword", "ignore_above": 2048},
                "title_exact": {"type": "keyword", "ignore_above": 2048},
                "title_words": whitespace,
                "title_bigrams": whitespace,
                "heading_original": {"type": "keyword", "index": False},
                "heading_exact": {"type": "keyword", "ignore_above": 2048},
                "heading_words": whitespace,
                "heading_bigrams": whitespace,
                "body_original": {"type": "text", "index": False},
                "body_exact": {"type": "keyword", "ignore_above": 2048},
                "body_words": whitespace,
                "body_bigrams": whitespace,
                "metadata_original": {"type": "text", "index": False},
                "metadata_exact": {"type": "keyword", "ignore_above": 2048},
                "metadata_words": whitespace,
                "metadata_bigrams": whitespace,
                "alias_original": {"type": "keyword", "index": False, "ignore_above": 32766},
                "alias_normalized": {"type": "keyword", "ignore_above": 2048},
                "alias_exact": {"type": "keyword", "ignore_above": 2048},
                "alias_words": whitespace,
                "alias_bigrams": whitespace,
                "url": {"type": "keyword", "index": False},
                "url_exact": {"type": "keyword", "ignore_above": 2048},
                "source": {"type": "keyword"},
                "wikiname": {"type": "keyword"},
                "wikidata_id": {"type": "keyword"},
                "modified_at": {"type": "date", "format": "strict_date_optional_time"},
                "source_version": {"type": "long"},
                "has_math": {"type": "boolean"},
                "language": {"type": "keyword"},
                "snapshot_date": {"type": "date", "format": "yyyyMMdd||yyyy-MM-dd"},
                "page_type": {"type": "keyword"},
                "char_start": {"type": "integer"},
                "char_end": {"type": "integer"},
                "analysis_ms": {"type": "half_float"},
            },
        },
    }


def aliases_to_index_fields(aliases: Sequence[str], analyzer: DocumentAnalyzer) -> Dict[str, Any]:
    aliases = tuple(alias for alias in aliases if alias)
    fields = [
        analyzer.core.analyze(alias, keep_duplicates=True, include_bigrams=True)
        for alias in aliases
    ]
    return {
        "alias_original": list(aliases),
        "alias_normalized": list(dict.fromkeys(field.normalized for field in fields if field.normalized)),
        "alias_exact": list(dict.fromkeys(term for field in fields for term in field.exact_terms)),
        "alias_words": " ".join(term for field in fields for term in field.word_terms),
        "alias_bigrams": " ".join(term for field in fields for term in field.bigram_terms),
    }


def chunk_to_index_document(chunk: WikipediaChunk, analyzer: DocumentAnalyzer) -> Dict[str, Any]:
    result = analyzer.analyze(
        title=chunk.title,
        body=chunk.text,
        headings=chunk.headings,
        url=chunk.url,
    )
    payload = result.to_index_payload()
    metadata_text = getattr(chunk, "metadata_text", "") or ""
    metadata = analyzer.core.analyze(
        metadata_text,
        keep_duplicates=True,
        include_bigrams=True,
    ) if metadata_text else None
    aliases = tuple(getattr(chunk, "aliases", ()) or ())
    payload.update(
        {
            "doc_id": chunk.doc_id,
            "page_id": chunk.page_id,
            "chunk_id": chunk.chunk_id,
            "title_normalized": result.title.normalized,
            "heading_original": list(chunk.headings),
            "body_exact": result.body.exact_terms,
            "url": chunk.url,
            "metadata_original": metadata_text,
            "metadata_exact": metadata.exact_terms if metadata else [],
            "metadata_words": " ".join(metadata.word_terms) if metadata else "",
            "metadata_bigrams": " ".join(metadata.bigram_terms) if metadata else "",
            "source": getattr(chunk, "source", "wikipedia"),
            "language": getattr(chunk, "language", "zh") or "und",
            "snapshot_date": chunk.snapshot_date,
            "page_type": chunk.page_type,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "analysis_ms": round(result.elapsed_ms, 4),
            "wikiname": getattr(chunk, "wikiname", "") or "",
            "wikidata_id": getattr(chunk, "wikidata_id", "") or "",
            "source_version": int(getattr(chunk, "source_version", 0) or 0),
            "has_math": bool(getattr(chunk, "has_math", False)),
        }
    )
    payload.update(aliases_to_index_fields(aliases, analyzer))
    modified_at = getattr(chunk, "modified_at", "") or ""
    if modified_at:
        payload["modified_at"] = modified_at
    return payload


@dataclass(frozen=True)
class CandidateHit:
    doc_id: str
    page_id: str
    title: str
    text: str
    url: str
    page_type: str
    score: float
    channels: Tuple[str, ...]
    ranks: Mapping[str, int]
    source: str = "wikipedia"
    language: str = ""
    wikidata_id: str = ""
    modified_at: str = ""
    chunk_id: int = -1
    char_start: int = 0
    headings: Tuple[str, ...] = ()
    passage_score: float = 0.0
    candidate_chunk_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["channels"] = list(self.channels)
        data["ranks"] = dict(self.ranks)
        data["headings"] = list(self.headings)
        return data


class CandidateIndexClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:19220", *, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        *,
        content_type: str = "application/json",
        timeout: Optional[float] = None,
    ) -> Any:
        req = request.Request(
            self.endpoint + path,
            data=body,
            method=method,
            headers={"Content-Type": content_type},
        )
        try:
            with request.urlopen(req, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
        return json.loads(raw) if raw else None

    def health(self) -> Mapping[str, Any]:
        return self._request("GET", "/_cluster/health")

    def create_index(self, index: str = DEFAULT_INDEX, *, recreate: bool = False, shards: int = 2) -> None:
        if recreate:
            try:
                self._request("DELETE", f"/{index}")
            except RuntimeError as exc:
                if "index_not_found_exception" not in str(exc):
                    raise
        body = json.dumps(candidate_index_mapping(shards=shards)).encode()
        self._request("PUT", f"/{index}", body)

    def put_mapping(self, index: str, properties: Mapping[str, Any]) -> None:
        body = json.dumps({"properties": dict(properties)}).encode()
        self._request("PUT", f"/{index}/_mapping", body)

    def set_refresh_interval(self, index: str, value: str) -> None:
        body = json.dumps({"index": {"refresh_interval": value}}).encode()
        self._request("PUT", f"/{index}/_settings", body)

    def refresh(self, index: str) -> None:
        self._request("POST", f"/{index}/_refresh")

    def flush(self, index: str) -> None:
        self._request("POST", f"/{index}/_flush?wait_if_ongoing=true", timeout=max(self.timeout, 120.0))

    def count(self, index: str) -> int:
        return int(self._request("GET", f"/{index}/_count")["count"])

    def existing_page_ids(
        self,
        index: str,
        page_ids: Sequence[str],
        *,
        batch_size: int = 5000,
    ) -> set[str]:
        """Return qrel page IDs present in an index without fetching their chunks."""

        unique = list(dict.fromkeys(str(item) for item in page_ids if str(item)))
        existing: set[str] = set()
        batch_size = max(1, min(10_000, int(batch_size)))
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            body = json.dumps(
                {
                    "size": len(batch),
                    "_source": ["page_id"],
                    "query": {"terms": {"page_id": batch}},
                    "collapse": {"field": "page_id"},
                },
                separators=(",", ":"),
            ).encode()
            response = self._request("POST", f"/{index}/_search", body)
            existing.update(
                str(hit.get("_source", {}).get("page_id") or "")
                for hit in response.get("hits", {}).get("hits", [])
            )
        existing.discard("")
        return existing

    def stats(self, index: str) -> Mapping[str, Any]:
        return self._request("GET", f"/{index}/_stats/store,docs")

    def bulk(self, index: str, documents: Sequence[Mapping[str, Any]]) -> int:
        lines: List[bytes] = []
        for document in documents:
            lines.append(
                json.dumps(
                    {"index": {"_index": index, "_id": document["doc_id"]}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            lines.append(
                json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
        body = b"\n".join(lines) + b"\n"
        result = self._request(
            "POST",
            "/_bulk",
            body,
            content_type="application/x-ndjson",
            timeout=max(self.timeout, 120.0),
        )
        if result.get("errors"):
            failures = [
                item["index"].get("error")
                for item in result.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise RuntimeError(f"Bulk index failed: {failures[:3]}")
        return len(documents)

    def bulk_update(self, index: str, documents: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
        """Partially update documents and return ``(updated, missing)``."""

        lines: List[bytes] = []
        for document in documents:
            doc_id = str(document["doc_id"])
            lines.append(
                json.dumps(
                    {"update": {"_index": index, "_id": doc_id, "retry_on_conflict": 2}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            lines.append(
                json.dumps(
                    {"doc": {key: value for key, value in document.items() if key != "doc_id"}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        body = b"\n".join(lines) + b"\n"
        result = self._request(
            "POST", "/_bulk", body, content_type="application/x-ndjson",
            timeout=max(self.timeout, 120.0),
        )
        updated = missing = 0
        failures = []
        for item in result.get("items", []):
            update = item.get("update", {})
            if update.get("status") == 404:
                missing += 1
            elif update.get("error"):
                failures.append(update.get("error"))
            else:
                updated += 1
        if failures:
            raise RuntimeError(f"Bulk update failed: {failures[:3]}")
        return updated, missing

    def search(
        self,
        query_text: str,
        *,
        index: str = DEFAULT_INDEX,
        query_analyzer: Optional[QueryAnalyzer] = None,
        channel_size: int = 50,
        limit: int = 10,
        max_chunks_per_page: int = 2,
    ) -> Tuple[QueryAnalysis, List[CandidateHit], float]:
        started = time.perf_counter()
        analysis = (query_analyzer or QueryAnalyzer()).analyze(query_text)
        max_chunks_per_page = max(1, int(max_chunks_per_page))
        channel_queries = build_channel_queries(
            analysis,
            size=channel_size,
            max_chunks_per_page=max_chunks_per_page,
        )
        header = json.dumps({"index": index}, separators=(",", ":"))
        body_lines: List[str] = []
        channels: List[str] = []
        for channel, query_body in channel_queries:
            channels.append(channel)
            body_lines.extend([header, json.dumps(query_body, ensure_ascii=False, separators=(",", ":"))])
        if not channels:
            return analysis, [], (time.perf_counter() - started) * 1000.0
        raw = ("\n".join(body_lines) + "\n").encode("utf-8")
        result = self._request(
            "POST", "/_msearch", raw, content_type="application/x-ndjson", timeout=self.timeout
        )
        ranked: Dict[str, Dict[str, Any]] = {}
        for channel, response in zip(channels, result.get("responses", [])):
            for rank, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
                page_id = str(hit["_source"]["page_id"])
                entry = ranked.setdefault(
                    page_id,
                    {
                        "score": 0.0,
                        "ranks": {},
                        "channels": [],
                        "chunks": {},
                    },
                )
                entry["score"] += CHANNEL_WEIGHTS[channel] / (60.0 + rank)
                entry["ranks"][channel] = rank
                entry["channels"].append(channel)
                if channel == "exact" and "identity" in hit.get("matched_queries", ()):
                    entry["score"] += CHANNEL_WEIGHTS["identity"] / 61.0
                    entry["ranks"]["identity"] = 1
                    entry["channels"].append("identity")
                if (
                    channel == "alias"
                    and analysis.normalized
                    and _script_equivalent(hit["_source"]["title_original"], analysis.normalized)
                ):
                    entry["score"] += CHANNEL_WEIGHTS["identity"] / 61.0
                    entry["ranks"]["script_identity"] = 1
                    entry["channels"].append("script_identity")

                # Page scoring and passage scoring are deliberately separate.
                # The outer collapsed hit is the channel's strongest passage;
                # inner_hits preserves additional relevant passages plus the
                # earliest matching passage. This prevents the first channel
                # that sees a page from accidentally fixing its evidence text.
                passage_hits = list(
                    hit.get("inner_hits", {})
                    .get(_PASSAGE_INNER_HITS, {})
                    .get("hits", {})
                    .get("hits", [])
                )
                if not passage_hits:
                    passage_hits = [hit]
                seen_in_channel = set()
                for passage_rank, passage_hit in enumerate(passage_hits, start=1):
                    passage_doc_id = str(passage_hit.get("_id") or "")
                    if not passage_doc_id or passage_doc_id in seen_in_channel:
                        continue
                    seen_in_channel.add(passage_doc_id)
                    _add_passage_candidate(
                        entry,
                        passage_hit,
                        channel=channel,
                        passage_rank=passage_rank,
                    )

                lead_hits = list(
                    hit.get("inner_hits", {})
                    .get(_LEAD_INNER_HITS, {})
                    .get("hits", {})
                    .get("hits", [])
                )
                for lead_hit in lead_hits:
                    _add_passage_candidate(
                        entry,
                        lead_hit,
                        channel=channel,
                        passage_rank=None,
                        is_lead=True,
                    )
        ordered = sorted(ranked.items(), key=lambda item: (-item[1]["score"], item[0]))
        hits: List[CandidateHit] = []
        for _page_id, item in ordered:
            candidates = list(item["chunks"].values())
            if not candidates:
                continue
            candidates = _limit_passage_candidates(candidates, max_chunks_per_page)
            for candidate in candidates:
                candidate["selection_score"] = _passage_selection_score(
                    candidate,
                    analysis,
                )
            selected = max(
                candidates,
                key=lambda candidate: (
                    candidate["selection_score"],
                    candidate["support_score"],
                    -int(candidate["source"].get("char_start", 0) or 0),
                    str(candidate["doc_id"]),
                ),
            )
            source = selected["source"]
            page_id = str(source["page_id"])
            hits.append(
                CandidateHit(
                    doc_id=selected["doc_id"],
                    page_id=page_id,
                    title=source["title_original"],
                    text=source["body_original"],
                    url=source["url"],
                    page_type=source["page_type"],
                    score=round(float(item["score"]), 8),
                    channels=tuple(item["channels"]),
                    ranks=dict(item["ranks"]),
                    source=source.get("source", "wikipedia"),
                    language=source.get("language", ""),
                    wikidata_id=source.get("wikidata_id", ""),
                    modified_at=source.get("modified_at", ""),
                    chunk_id=int(source.get("chunk_id", -1) or 0),
                    char_start=int(source.get("char_start", 0) or 0),
                    headings=tuple(source.get("heading_original", ()) or ()),
                    passage_score=round(float(selected["selection_score"]), 8),
                    candidate_chunk_count=len(candidates),
                )
            )
            if len(hits) >= limit:
                break
        return analysis, hits, (time.perf_counter() - started) * 1000.0


def _weighted_terms(analysis: QueryAnalysis, kinds: Iterable[str]) -> List[str]:
    accepted = set(kinds)
    return [
        token.normalized
        for token in analysis.tokens
        if token.kind in accepted
        and token.weight > 0.15
        and token.normalized not in _RETRIEVAL_STOPWORDS
    ]


def _add_passage_candidate(
    page: Dict[str, Any],
    hit: Mapping[str, Any],
    *,
    channel: str,
    passage_rank: Optional[int],
    is_lead: bool = False,
) -> None:
    source = dict(hit.get("_source") or {})
    doc_id = str(hit.get("_id") or source.get("doc_id") or "")
    if not doc_id or not source:
        return
    candidate = page["chunks"].setdefault(
        doc_id,
        {
            "doc_id": doc_id,
            "source": source,
            "support_score": 0.0,
            "channel_ranks": {},
            "is_lead": False,
        },
    )
    candidate["is_lead"] = bool(candidate["is_lead"] or is_lead)
    if passage_rank is not None:
        previous = candidate["channel_ranks"].get(channel)
        if previous is None or passage_rank < previous:
            if previous is not None:
                candidate["support_score"] -= CHANNEL_WEIGHTS[channel] / (10.0 + previous)
            candidate["channel_ranks"][channel] = passage_rank
            candidate["support_score"] += CHANNEL_WEIGHTS[channel] / (10.0 + passage_rank)


def _limit_passage_candidates(
    candidates: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Keep the strongest channel candidates without dropping the page lead."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["support_score"]),
            int(candidate["source"].get("char_start", 0) or 0),
            str(candidate["doc_id"]),
        ),
    )
    selected = list(ordered[: max(1, limit)])
    lead = min(
        (candidate for candidate in candidates if candidate["is_lead"]),
        key=lambda candidate: int(candidate["source"].get("char_start", 0) or 0),
        default=None,
    )
    if lead is not None and all(item["doc_id"] != lead["doc_id"] for item in selected):
        if len(selected) >= limit:
            selected[-1] = lead
        else:
            selected.append(lead)
    return selected


def _passage_selection_score(candidate: Mapping[str, Any], analysis: QueryAnalysis) -> float:
    source = candidate["source"]
    body_terms = set(str(source.get("body_words") or "").split())
    heading_terms = set(str(source.get("heading_words") or "").split())
    terms = list(dict.fromkeys(_weighted_terms(analysis, {"word", "number", "symbol"})))
    matched_body = sum(1 for term in terms if term in body_terms)
    matched_heading = sum(1 for term in terms if term in heading_terms)
    denominator = max(1, len(terms))
    coverage = matched_body / denominator
    heading_coverage = matched_heading / denominator

    body = str(source.get("body_original") or "")
    title = str(source.get("title_original") or "")
    body_normalized = normalize_text(body).text
    title_normalized = normalize_text(title).text
    query_normalized = normalize_text(analysis.normalized).text
    original_query = str(analysis.original or "")
    definition_query = bool(_DEFINITION_QUERY.search(original_query))
    subject_matches_page = bool(
        title_normalized
        and (
            query_normalized == title_normalized
            or title_normalized in query_normalized
        )
    )
    prefer_lead = subject_matches_page and (
        definition_query or query_normalized == title_normalized
    )
    lead_bonus = 1.25 if prefer_lead and candidate.get("is_lead") else 0.0
    definition_bonus = (
        0.65
        if definition_query and _DEFINITION_PASSAGE.search(body_normalized[:600])
        else 0.0
    )
    char_start = max(0, int(source.get("char_start", 0) or 0))
    early_bonus = 0.12 / (1.0 + char_start / 800.0)

    return (
        float(candidate["support_score"])
        + coverage * 0.9
        + heading_coverage * 0.3
        + lead_bonus
        + definition_bonus
        + early_bonus
    )


def build_channel_queries(
    analysis: QueryAnalysis,
    *,
    size: int = 50,
    max_chunks_per_page: int = 2,
) -> List[Tuple[str, Dict[str, Any]]]:
    source_fields = [
        "doc_id",
        "page_id",
        "title_original",
        "body_original",
        "url",
        "page_type",
        "source",
        "language",
        "wikidata_id",
        "modified_at",
        "chunk_id",
        "char_start",
        "heading_original",
        "body_words",
        "heading_words",
    ]
    collapse = {
        "field": "page_id",
        "inner_hits": [
            {
                "name": _PASSAGE_INNER_HITS,
                "size": max(1, int(max_chunks_per_page)),
                "sort": [{"_score": "desc"}, {"char_start": "asc"}],
                "_source": source_fields,
            },
            {
                "name": _LEAD_INNER_HITS,
                "size": 1,
                "sort": [{"char_start": "asc"}],
                "_source": source_fields,
            },
        ],
    }
    output: List[Tuple[str, Dict[str, Any]]] = []
    exact_should: List[Dict[str, Any]] = []
    if analysis.normalized:
        exact_should.append({
            "term": {
                "title_normalized": {
                    "value": analysis.normalized, "boost": 10.0, "_name": "identity",
                }
            }
        })
    for term in analysis.exact_terms:
        exact_should.extend(
            [
                {"term": {"title_exact": {"value": term, "boost": 5.0}}},
                {"term": {"heading_exact": {"value": term, "boost": 3.0}}},
                {"term": {"body_exact": {"value": term, "boost": 1.0}}},
                {"term": {"metadata_exact": {"value": term, "boost": 0.8}}},
            ]
        )
    if exact_should:
        output.append(
            (
                "exact",
                {
                    "size": size,
                    "_source": source_fields,
                    "collapse": collapse,
                    "query": {"bool": {"should": exact_should, "minimum_should_match": 1}},
                },
            )
        )
    words = _weighted_terms(analysis, {"word", "number", "symbol"})
    alias_should: List[Dict[str, Any]] = []
    if analysis.normalized:
        alias_should.append({"term": {"alias_normalized": {"value": analysis.normalized, "boost": 12.0}}})
    for term in analysis.exact_terms:
        alias_should.append({"term": {"alias_exact": {"value": term, "boost": 6.0}}})
    if len(words) == 1:
        alias_should.append({"term": {"alias_words": {"value": words[0], "boost": 2.0}}})
    if alias_should:
        alias_query: Dict[str, Any] = {
            "bool": {
                "must": [{"bool": {"should": alias_should, "minimum_should_match": 1}}]
            }
        }
        answer_type = str(analysis.constraints.get("answer_type") or "")
        if answer_type:
            alias_query["bool"]["should"] = [
                {"term": {"title_words": {"value": answer_type, "boost": 12.0}}},
                {"term": {"heading_words": {"value": answer_type, "boost": 6.0}}},
                {"term": {"body_words": {"value": answer_type, "boost": 3.0}}},
                {"term": {"metadata_words": {"value": answer_type, "boost": 3.0}}},
            ]
        output.append(
            (
                "alias",
                {
                    "size": size,
                    "_source": source_fields,
                    "collapse": collapse,
                    "query": alias_query,
                },
            )
        )
    if words:
        output.append(
            (
                "word",
                {
                    "size": size,
                    "_source": source_fields,
                    "collapse": collapse,
                    "query": {
                        "multi_match": {
                            "query": " ".join(words),
                            "fields": [
                                "title_words^4", "heading_words^3", "body_words^1.5",
                                "metadata_words^0.8",
                            ],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    },
                },
            )
        )
    latin_terms = {
        token.normalized
        for token in analysis.tokens
        if token.script == "latin" and token.kind == "word" and token.weight > 0.15
    }
    bigrams = _weighted_terms(analysis, {"bigram"}) if len(latin_terms) < 2 else []
    if bigrams:
        output.append(
            (
                "bigram",
                {
                    "size": size,
                    "_source": source_fields,
                    "collapse": collapse,
                    "query": {
                        "multi_match": {
                            "query": " ".join(bigrams),
                            "fields": [
                                "title_bigrams^1.2", "heading_bigrams^0.8", "body_bigrams^0.3",
                                "metadata_bigrams^0.15",
                            ],
                            "type": "best_fields",
                            "operator": "or",
                            "minimum_should_match": "60%",
                        }
                    },
                },
            )
        )
    return output
