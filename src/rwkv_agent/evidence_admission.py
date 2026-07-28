from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence

from rwkv_search.search_reasoning import query_anchors
from rwkv_search.text import search_tokens


_RELATION_NOISE = frozenset(
    {
        "address",
        "author",
        "company",
        "create",
        "created",
        "creator",
        "founder",
        "github",
        "latest",
        "maintain",
        "maintained",
        "maintains",
        "official",
        "owner",
        "project",
        "projects",
        "repository",
        "route",
        "source",
        "update",
        "updates",
        "what",
        "where",
        "which",
        "who",
        "怎么样",
        "公司",
        "创始人",
        "原文",
        "地址",
        "如何",
        "官网",
        "官方",
        "更新",
        "最新",
        "路线",
        "项目",
    }
)


@dataclass(frozen=True)
class EvidenceAdmissionTrace:
    anchors: tuple[str, ...]
    admitted: int
    rejected: int
    rejected_ids: tuple[str, ...]
    rejection_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntityEvidenceAdmission:
    """Reject evidence that never mentions the question's named subject.

    This is deliberately domain-neutral.  It extracts stable entity-like
    anchors from the user's question and checks only title, content, and URI.
    When no reliable anchor exists, it preserves recall by admitting all input.
    """

    def admit(
        self,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], EvidenceAdmissionTrace]:
        anchors = self.entity_anchors(question)
        admitted: list[dict[str, Any]] = []
        rejected_ids: list[str] = []
        for item in evidence:
            value = dict(item)
            if not anchors or self._matches(value, anchors):
                admitted.append(value)
                continue
            rejected_ids.append(str(value.get("id") or value.get("uri") or ""))
        rejected = len(evidence) - len(admitted)
        trace = EvidenceAdmissionTrace(
            anchors=anchors,
            admitted=len(admitted),
            rejected=rejected,
            rejected_ids=tuple(rejected_ids),
            rejection_counts={"entity_mismatch": rejected} if rejected else {},
        )
        return admitted, trace

    @staticmethod
    def entity_anchors(question: str) -> tuple[str, ...]:
        anchors = query_anchors(question, limit=8)
        identifiers = [
            value
            for value in anchors
            if EntityEvidenceAdmission._is_identifier(value)
        ]
        if identifiers:
            return tuple(identifiers[:3])
        named = [
            value
            for value in anchors
            if value.casefold() not in _RELATION_NOISE and len(value.strip()) >= 2
        ]
        return tuple(named[:3])

    @staticmethod
    def _is_identifier(value: str) -> bool:
        folded = value.casefold().strip()
        return bool(
            folded not in _RELATION_NOISE
            and len(folded) >= 3
            and folded.isascii()
            and any(character.isalpha() for character in folded)
        )

    @staticmethod
    def _matches(item: Mapping[str, Any], anchors: Sequence[str]) -> bool:
        document = unicodedata.normalize(
            "NFKC",
            " ".join(
                str(item.get(field) or "")
                for field in ("title", "content", "uri")
            ),
        ).casefold()
        tokens = set(search_tokens(document))
        compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", document)
        for anchor in anchors:
            folded = unicodedata.normalize("NFKC", anchor).casefold()
            aliases = set(search_tokens(folded))
            if aliases and aliases.intersection(tokens):
                return True
            normalized = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", folded)
            if len(normalized) >= 2 and normalized in compact:
                return True
        return False
