from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from rwkv_search.search_reasoning import query_anchors, validate_generated_query
from rwkv_search.text import search_tokens

from .page_quality import classify_page_quality


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
    safe_observation_count: int = 0
    gap_validated: bool = False
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
        safe_observation = self._safe_observation(raw, observation)
        observation_text = self._observation_text(safe_observation)

        if int(round_index) <= 1:
            candidates = (
                (generated, "model", False),
                (exact, "exact_anchors", False),
                (self._join(generated, primary), "primary_source", False),
                (raw, "raw_question", False),
            )
        else:
            candidates = (
                (generated, "model_gap", True),
                (self._join(generated, primary), "gap_primary", True),
                (self._join(generated, original), "gap_original", True),
                (self._join(exact, generated), "gap_anchor_merge", True),
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
                observation=observation_text,
                allow_observation_grounding=evidence_based,
                max_chars=240,
            )
            key = self._key(query)
            reasons = list(validation.reasons)
            if (
                key in used_queries
                or (
                    int(round_index) > 1
                    and self._near_duplicate(query, used_queries)
                )
            ) and "duplicate_query" not in reasons:
                reasons.append("duplicate_query")
            if int(round_index) > 1 and not generated:
                reasons.append("missing_gap_query")
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
                safe_observation_count=len(safe_observation),
                gap_validated=int(round_index) > 1,
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
            safe_observation_count=len(safe_observation),
            gap_validated=False,
            rejection_reasons=tuple(
                dict.fromkeys(rejection_reasons or ["no_unique_query"])
            ),
        )

    @staticmethod
    def _safe_observation(
        question: str,
        observation: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in list((observation or {}).get("evidence") or ())[:5]:
            if not isinstance(item, Mapping):
                continue
            quality = classify_page_quality(question, item)
            if not quality.pivot_allowed:
                continue
            output.append(dict(item))
        return output

    @staticmethod
    def _observation_text(
        observation: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    ) -> str:
        items = (
            list((observation or {}).get("evidence") or ())
            if isinstance(observation, Mapping)
            else list(observation or ())
        )
        return " ".join(
            f"{item.get('title') or ''} {item.get('content') or ''}"
            for item in items[:5]
            if isinstance(item, Mapping)
        )

    @staticmethod
    def _near_duplicate(value: str, used_queries: set[str]) -> bool:
        wanted = set(search_tokens(value))
        if not wanted:
            return False
        for previous in used_queries:
            existing = set(search_tokens(previous))
            if not existing:
                continue
            union = wanted | existing
            if union and len(wanted & existing) / len(union) >= 0.88:
                return True
        return False

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
