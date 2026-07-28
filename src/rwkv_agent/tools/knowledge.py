from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Protocol, Sequence

from rwkv_search.candidate_index import CandidateHit, CandidateIndexClient


INDEX_BY_LANGUAGE = {
    "zh": "rwkv-finewiki-zh-full-v1",
    "en": "rwkv-finewiki-en-full-v1",
}
_SPACE_RE = re.compile(r"\s+")
_UNSAFE_MARKUP = {
    "<tool_call>": "[tool_call]",
    "</tool_call>": "[/tool_call]",
    "<tool_result>": "[tool_result]",
    "</tool_result>": "[/tool_result]",
}


class SearchClient(Protocol):
    def search(
        self,
        query_text: str,
        *,
        index: str,
        channel_size: int,
        limit: int,
        max_chunks_per_page: int,
    ) -> tuple[Any, Sequence[CandidateHit], float]: ...


@dataclass(frozen=True)
class KnowledgeEvidence:
    id: str
    title: str
    content: str
    source: str
    uri: str
    page_id: str
    doc_id: str
    chunk_id: int
    language: str
    score: float


def detect_language(text: str) -> str:
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    latin = sum(("a" <= char.lower() <= "z") for char in text)
    return "zh" if cjk >= max(1, latin // 3) else "en"


def _clean_content(text: str, max_chars: int) -> str:
    value = _SPACE_RE.sub(" ", str(text or "")).strip()
    for source, replacement in _UNSAFE_MARKUP.items():
        value = value.replace(source, replacement)
    if len(value) <= max_chars:
        return value
    return value[: max(1, max_chars - 1)].rstrip() + "…"


class KnowledgeSearchAdapter:
    """Read-only adapter over the existing FineWiki candidate indexes."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:19220",
        *,
        client: SearchClient | None = None,
        top_k: int = 5,
        channel_size: int = 100,
        max_evidence_chars: int = 900,
        shadow: Any | None = None,
    ) -> None:
        self.client = client or CandidateIndexClient(endpoint, timeout=30.0)
        self.top_k = max(1, min(10, int(top_k)))
        self.channel_size = max(10, int(channel_size))
        self.max_evidence_chars = max(120, int(max_evidence_chars))
        if shadow is None:
            from .hybrid_knowledge import build_shadow_from_env

            shadow = build_shadow_from_env(endpoint)
        self.shadow = shadow

    def execute(self, query: str, *, language: str | None = None) -> dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {
                "status": "invalid",
                "evidence": [],
                "message": "knowledge_search requires a non-empty query.",
            }
        selected_language = language or detect_language(normalized_query)
        if selected_language not in INDEX_BY_LANGUAGE:
            return {
                "status": "invalid",
                "evidence": [],
                "message": f"unsupported knowledge language: {selected_language}",
            }
        index = INDEX_BY_LANGUAGE[selected_language]
        _analysis, hits, latency_ms = self.client.search(
            normalized_query,
            index=index,
            channel_size=self.channel_size,
            limit=self.top_k,
            max_chunks_per_page=2,
        )
        evidence = [
            KnowledgeEvidence(
                id=f"K{position}",
                title=hit.title,
                content=_clean_content(hit.text, self.max_evidence_chars),
                source=hit.source,
                uri=hit.url,
                page_id=hit.page_id,
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                language=hit.language or selected_language,
                score=hit.score,
            )
            for position, hit in enumerate(hits, start=1)
        ]
        result = {
            "status": "ok" if evidence else "empty",
            "query": normalized_query,
            "language": selected_language,
            "index": index,
            "evidence": [asdict(item) for item in evidence],
            "retrieval": {
                "latency_ms": round(float(latency_ms), 3),
                "returned": len(evidence),
                "top_k": self.top_k,
                "channel_size": self.channel_size,
            },
            "message": (
                "Use only the supplied evidence and cite supporting IDs."
                if evidence
                else "No local knowledge evidence was found."
            ),
        }
        if self.shadow not in (None, False):
            result["retrieval"]["shadow"] = self.shadow.submit(
                normalized_query,
                language=selected_language,
                legacy_evidence=result["evidence"],
            )
        return result

    def close(self) -> None:
        close = getattr(self.shadow, "close", None)
        if callable(close):
            close()


def cited_evidence(
    result: Mapping[str, Any], citation_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    wanted = set(citation_ids)
    return [
        item
        for item in result.get("evidence", [])
        if isinstance(item, Mapping) and str(item.get("id") or "") in wanted
    ]
