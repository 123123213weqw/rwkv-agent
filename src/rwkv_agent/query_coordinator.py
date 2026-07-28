from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from rwkv_search.search_reasoning import query_anchors, validate_generated_query


_PRIMARY_SOURCE_LABELS = frozenset(
    {
        "company_filing",
        "github",
        "government",
        "official",
        "official_docs",
        "paper",
        "regulator",
        "repository",
    }
)
_PIVOT_NOISE = frozenset(
    {
        "creator",
        "founder",
        "github",
        "latest",
        "official",
        "project",
        "projects",
        "source",
        "update",
        "updates",
        "what",
        "which",
        "who",
        "什么",
        "创始人",
        "原文",
        "哪些",
        "官网",
        "官方",
        "更新",
        "最新",
        "项目",
    }
)


@dataclass(frozen=True)
class QueryView:
    query: str
    strategy: str
    branch_index: int
    round_index: int
    anchors: tuple[str, ...]
    retained_anchors: tuple[str, ...]
    evidence_based: bool
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    def to_trace(self) -> dict[str, Any]:
        return asdict(self)


class QueryCoordinator:
    """Coordinate complementary query views without classifying a topic domain."""

    def coordinate(
        self,
        question: str,
        generated_query: str,
        *,
        branch_index: int,
        round_index: int,
        observation: Mapping[str, Any] | None,
        used_queries: set[str],
    ) -> QueryView:
        raw = self._clean(question)
        generated = self._clean(generated_query)
        anchors = query_anchors(raw, limit=6)
        exact = self._exact_anchor_view(anchors)
        is_zh = any("\u3400" <= character <= "\u9fff" for character in raw)
        primary = "官网 官方" if is_zh else "official primary source"
        original = "原文 来源" if is_zh else "original source"
        pivot = self._supported_pivot(raw, observation, anchors)

        if int(round_index) <= 1:
            candidates = (
                (generated, "model", False),
                (exact, "exact_anchors", False),
                (self._join(generated, primary), "primary_source", False),
                (raw, "raw_question", False),
            )
        else:
            candidates = (
                (generated, "model_gap", False),
                (self._join(generated, primary), "gap_primary", False),
                (self._join(generated, original), "gap_original", False),
                (
                    self._join(exact, pivot) if pivot else raw,
                    "evidence_pivot" if pivot else "raw_fallback",
                    bool(pivot),
                ),
                (raw, "raw_fallback", False),
            )

        preferred_index = min(max(0, int(branch_index)), len(candidates) - 1)
        preferred = candidates[preferred_index]
        ordered = (preferred,) + tuple(
            candidate
            for index, candidate in enumerate(candidates)
            if index != preferred_index
        )
        rejection_reasons: list[str] = []
        for value, strategy, evidence_based in ordered:
            query = self._clean(value)[:240].strip()
            if not query:
                rejection_reasons.append("empty_query")
                continue
            validation = validate_generated_query(
                raw,
                query,
                observation=self._observation_text(observation),
                allow_observation_grounding=evidence_based,
                max_chars=240,
            )
            key = self._key(query)
            reasons = list(validation.reasons)
            if key in used_queries and "duplicate_query" not in reasons:
                reasons.append("duplicate_query")
            if reasons:
                rejection_reasons.extend(reasons)
                continue
            used_queries.add(key)
            return QueryView(
                query=query,
                strategy=strategy,
                branch_index=int(branch_index),
                round_index=int(round_index),
                anchors=tuple(anchors),
                retained_anchors=tuple(validation.retained_anchors),
                evidence_based=evidence_based,
                accepted=True,
            )
        return QueryView(
            query="",
            strategy="skipped_duplicate",
            branch_index=int(branch_index),
            round_index=int(round_index),
            anchors=tuple(anchors),
            retained_anchors=(),
            evidence_based=False,
            accepted=False,
            rejection_reasons=tuple(
                dict.fromkeys(rejection_reasons or ["no_unique_query"])
            ),
        )

    @staticmethod
    def _supported_pivot(
        question: str,
        observation: Mapping[str, Any] | None,
        anchors: Sequence[str],
    ) -> str:
        core = [
            anchor
            for anchor in anchors
            if anchor.casefold() not in _PIVOT_NOISE and len(anchor.strip()) >= 2
        ]
        if not core:
            return ""
        for item in list((observation or {}).get("evidence") or ())[:5]:
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "").casefold().strip()
            if source not in _PRIMARY_SOURCE_LABELS:
                continue
            title = QueryCoordinator._clean(str(item.get("title") or ""))
            content = QueryCoordinator._clean(str(item.get("content") or ""))
            searchable = f"{title} {content}".casefold()
            if not title or not any(anchor.casefold() in searchable for anchor in core):
                continue
            validation = validate_generated_query(
                question,
                f"{title} {content[:160]}",
                observation=searchable,
                allow_observation_grounding=True,
            )
            if validation.accepted:
                return title[:120]
        return ""

    @staticmethod
    def _observation_text(observation: Mapping[str, Any] | None) -> str:
        return " ".join(
            f"{item.get('title') or ''} {item.get('content') or ''}"
            for item in list((observation or {}).get("evidence") or ())[:5]
            if isinstance(item, Mapping)
        )

    @staticmethod
    def _exact_anchor_view(anchors: Sequence[str]) -> str:
        return " ".join(
            f'"{anchor}"' if len(anchor) >= 4 else anchor for anchor in anchors[:4]
        )

    @staticmethod
    def _join(*values: str) -> str:
        return " ".join(value for value in values if value).strip()

    @staticmethod
    def _clean(value: str) -> str:
        return " ".join(str(value or "").split()).strip()

    @staticmethod
    def _key(value: str) -> str:
        return QueryCoordinator._clean(value).casefold()
