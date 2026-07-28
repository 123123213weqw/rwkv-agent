from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import SearchConfig
from .passage_selection import PassageScorer, select_page_passages
from .search import SearchResult
from .text import best_snippet


ROLE_MARKERS = re.compile(r"(?im)^(system|assistant|developer|user)\s*:")


@dataclass
class Evidence:
    """Canonical evidence.v1 record shared by every retrieval backend.

    ``id``, ``authority`` and ``score`` remain available during the migration
    because the current answerer and frontend use them.  ``to_dict`` publishes
    both the canonical names and those legacy aliases.
    """

    id: str
    title: str
    url: str
    source_type: str
    published_at: Optional[str]
    fetched_at: float
    authority: float
    text: str
    score: float
    updated_at: Optional[str] = None
    freshness_score: float = 0.0
    matched_channels: Tuple[str, ...] = ()
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION = "evidence.v1"

    @property
    def evidence_id(self) -> str:
        return self.id

    @property
    def retrieval_score(self) -> float:
        return self.score

    @property
    def authority_score(self) -> float:
        return self.authority

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "evidence_id": self.id,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "fetched_at": self.fetched_at,
            "authority_score": self.authority,
            "freshness_score": self.freshness_score,
            "retrieval_score": self.score,
            "matched_channels": list(self.matched_channels),
            "text": self.text,
            "metadata": dict(self.metadata),
            # Compatibility aliases.  Remove only after the web protocol and
            # answerer have both migrated to evidence.v1.
            "id": self.id,
            "authority": self.authority,
            "score": self.score,
        }

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        *,
        evidence_id: str,
        text: str,
    ) -> "Evidence":
        components = dict(result.score_components or {})
        return cls(
            id=evidence_id,
            title=result.title,
            url=result.url,
            source_type=result.source_type,
            published_at=result.published_at,
            fetched_at=result.fetched_at,
            authority=result.authority,
            text=text,
            score=result.score,
            updated_at=result.updated_at,
            freshness_score=float(components.get("freshness") or 0.0),
            matched_channels=tuple(result.matched_channels) or tuple(
                key for key, value in components.items() if float(value or 0.0) != 0.0
            ),
            source_id=result.source_id or str(result.document_id),
            metadata={"score_components": components},
        )

    @classmethod
    def from_candidate_hit(
        cls,
        hit: Any,
        *,
        evidence_id: str,
        authority: float = 0.85,
        fetched_at: float = 0.0,
    ) -> "Evidence":
        ranks: Mapping[str, int] = getattr(hit, "ranks", {}) or {}
        return cls(
            id=evidence_id,
            title=str(getattr(hit, "title", "")),
            url=str(getattr(hit, "url", "")),
            source_type=str(getattr(hit, "source", "finewiki") or "finewiki"),
            published_at=None,
            fetched_at=fetched_at,
            authority=authority,
            text=cls._sanitize_text(str(getattr(hit, "text", ""))),
            score=float(getattr(hit, "score", 0.0) or 0.0),
            updated_at=str(getattr(hit, "modified_at", "") or "") or None,
            freshness_score=0.0,
            matched_channels=tuple(
                dict.fromkeys(getattr(hit, "channels", ()) or ())
            ),
            source_id=str(getattr(hit, "doc_id", "") or "") or None,
            metadata={
                "page_id": str(getattr(hit, "page_id", "") or ""),
                "chunk_id": int(getattr(hit, "chunk_id", -1) or 0),
                "char_start": int(getattr(hit, "char_start", 0) or 0),
                "headings": list(getattr(hit, "headings", ()) or ()),
                "passage_score": float(getattr(hit, "passage_score", 0.0) or 0.0),
                "candidate_chunk_count": int(
                    getattr(hit, "candidate_chunk_count", 1) or 1
                ),
                "hydration_strategy": str(
                    getattr(hit, "hydration_strategy", "") or ""
                ),
                "component_doc_ids": list(
                    getattr(hit, "component_doc_ids", ()) or ()
                ),
                "page_type": str(getattr(hit, "page_type", "") or ""),
                "wikidata_id": str(getattr(hit, "wikidata_id", "") or ""),
                "ranks": dict(ranks),
            },
        )

    @staticmethod
    def _sanitize_text(text: str) -> str:
        value = text.replace("<|", "＜|").replace("|>", "|＞")
        return ROLE_MARKERS.sub(lambda match: match.group(1) + "﹕", value).strip()


class EvidenceBuilder:
    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        *,
        passage_scorer: PassageScorer | None = None,
    ) -> None:
        self.config = config or SearchConfig()
        self.passage_scorer = passage_scorer

    def build(self, query: str, results: Sequence[SearchResult]) -> List[Evidence]:
        selected: List[Evidence] = []
        remaining = self.config.evidence_character_budget
        # Strongest evidence is placed last in the model prompt to account for recurrent recency.
        chosen = list(results[: self.config.evidence_limit])
        for index, result in enumerate(chosen, start=1):
            budget = min(3200, max(600, remaining // max(1, len(chosen) - index + 1)))
            selection = None
            if (
                self.config.passage_selection_enabled
                and len(result.content) >= self.config.passage_min_document_chars
                and result.source_type not in {"finewiki"}
            ):
                selection = select_page_passages(
                    query,
                    result.title,
                    result.content,
                    max_passages=self.config.passage_max_per_document,
                    max_chars=min(
                        budget,
                        self.config.passage_max_chars_per_evidence,
                    ),
                    target_chars=self.config.passage_target_chars,
                    hard_max_chars=self.config.passage_hard_max_chars,
                    scorer=self.passage_scorer,
                )
            text = (
                selection.text
                if selection is not None and selection.text
                else best_snippet(result.content, query, limit=budget)
            )
            text = self._sanitize(text)
            if not text:
                continue
            evidence = Evidence.from_search_result(
                result,
                evidence_id=f"S{index}",
                text=text,
            )
            if selection is not None:
                evidence.metadata["passage_selection"] = selection.metadata()
            selected.append(evidence)
            remaining -= len(text)
            if remaining <= 0:
                break
        return selected

    @staticmethod
    def _sanitize(text: str) -> str:
        return Evidence._sanitize_text(text)


def validate_citations(answer: Dict[str, Any], evidence: Sequence[Evidence]) -> Dict[str, Any]:
    allowed = {item.id for item in evidence}
    citations = answer.get("citations") if isinstance(answer, dict) else []
    if not isinstance(citations, list):
        citations = []
    valid = [str(item) for item in citations if str(item) in allowed]
    invalid = [str(item) for item in citations if str(item) not in allowed]
    return {
        "valid": not invalid,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "has_citation": bool(valid),
    }
