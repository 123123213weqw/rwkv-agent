from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .semantic_selection import PairScorer, select_diverse_items
from .text import search_tokens


_CLAUSE_SPLIT_RE = re.compile(
    r"[\n，,；;。！？!?]+|(?:以及|并且|同时|还有|另外|然后)"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.])\s+|\n+")
_REFERENCE_PREFIX_RE = re.compile(
    r"^(?:它|其|该|此|这些|这项|其中|此外|因此|同时|the\s+(?:project|release|model)|it\b)",
    re.IGNORECASE,
)
PassageScorer = PairScorer


@dataclass(frozen=True)
class WebPassage:
    passage_id: str
    index: int
    text: str
    char_start: int
    char_end: int
    token_count: int

    def to_dict(self, *, score: float | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "passage_id": self.passage_id,
            "index": self.index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "text": self.text,
        }
        if score is not None:
            value["score"] = round(float(score), 6)
        return value


@dataclass(frozen=True)
class PassageSelection:
    text: str
    passages: tuple[WebPassage, ...]
    scores: tuple[float, ...]
    total_passages: int
    query_aspects: tuple[str, ...]
    strategy: str
    reranker_model: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query_aspects": list(self.query_aspects),
            "total_passages": self.total_passages,
            "selected_passages": [
                passage.to_dict(score=score)
                for passage, score in zip(self.passages, self.scores)
            ],
            "reranker_model": self.reranker_model,
        }


def query_aspects(query: str) -> tuple[str, ...]:
    """Split a compound question while preserving a stable entity anchor."""

    clean = " ".join(str(query or "").split())
    clauses = [item.strip(" 的地得请问一下") for item in _CLAUSE_SPLIT_RE.split(clean)]
    clauses = [item for item in clauses if len(item) >= 2]
    if not clauses:
        return (clean,) if clean else ()

    anchors = [
        token
        for token in search_tokens(clauses[0])
        if token.isascii() and len(token) >= 2
    ][:2]
    output: list[str] = []
    for index, clause in enumerate(clauses[:4]):
        value = clause
        if index and anchors and not any(anchor in value.casefold() for anchor in anchors):
            value = f"{' '.join(anchors)} {value}"
        if value not in output:
            output.append(value)
    return tuple(output)


def split_web_passages(
    text: str,
    *,
    target_chars: int = 900,
    max_chars: int = 1400,
    min_chars: int = 120,
) -> list[WebPassage]:
    """Paragraph-first, sentence-safe splitting for already extracted text."""

    source = str(text or "").strip()
    if not source:
        return []
    target = max(256, int(target_chars))
    hard_max = max(target, int(max_chars))
    minimum = max(40, min(int(min_chars), target))

    units: list[tuple[str, int, int]] = []
    cursor = 0
    for raw in re.split(r"\n{1,}", source):
        value = " ".join(raw.split()).strip()
        if not value:
            continue
        start = source.find(raw, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(raw)
        if len(value) <= hard_max:
            units.append((value, start, start + len(raw)))
            continue
        sentence_cursor = start
        for sentence in _SENTENCE_SPLIT_RE.split(value):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_start = source.find(sentence, sentence_cursor)
            if sentence_start < 0:
                sentence_start = sentence_cursor
            sentence_cursor = sentence_start + len(sentence)
            if len(sentence) <= hard_max:
                units.append((sentence, sentence_start, sentence_cursor))
                continue
            for offset in range(0, len(sentence), hard_max):
                piece = sentence[offset : offset + hard_max].strip()
                if piece:
                    piece_start = sentence_start + offset
                    units.append((piece, piece_start, piece_start + len(piece)))

    groups: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_chars = 0
    for unit in units:
        separator = 2 if current else 0
        projected = current_chars + separator + len(unit[0])
        if current and projected > hard_max and current_chars >= minimum:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += (2 if current_chars else 0) + len(unit[0])
        if current_chars >= target:
            groups.append(current)
            current = []
            current_chars = 0
    if current:
        if groups and current_chars < minimum:
            merged_chars = sum(len(value) for value, _, _ in groups[-1]) + current_chars
            if merged_chars <= hard_max:
                groups[-1].extend(current)
            else:
                groups.append(current)
        else:
            groups.append(current)

    passages: list[WebPassage] = []
    for index, group in enumerate(groups):
        value = "\n\n".join(item[0] for item in group).strip()
        if not value:
            continue
        passages.append(
            WebPassage(
                passage_id=f"p{index + 1}",
                index=index,
                text=value,
                char_start=min(item[1] for item in group),
                char_end=max(item[2] for item in group),
                token_count=len(search_tokens(value)),
            )
        )
    return passages


def select_page_passages(
    query: str,
    title: str,
    text: str,
    *,
    max_passages: int = 3,
    max_chars: int = 3200,
    target_chars: int = 900,
    hard_max_chars: int = 1400,
    scorer: PassageScorer | None = None,
) -> PassageSelection:
    """Select complementary passages from one page for a compound question."""

    passage_limit = max(1, int(max_passages))
    per_passage_budget = max(
        256,
        (max(512, int(max_chars)) - 2 * (passage_limit - 1)) // passage_limit,
    )
    passages = split_web_passages(
        text,
        target_chars=min(int(target_chars), per_passage_budget),
        max_chars=min(int(hard_max_chars), per_passage_budget),
    )
    aspects = query_aspects(query)
    if not passages:
        return PassageSelection("", (), (), 0, aspects, "no_passages")

    selection = select_diverse_items(
        query,
        aspects,
        [
            {
                "passage_id": passage.passage_id,
                "title": title,
                "content": passage.text,
                "uri": f"https://passage.invalid/{passage.passage_id}",
                "_best_position": passage.index + 1,
            }
            for passage in passages
        ],
        limit=passage_limit,
        scorer=scorer,
    )
    index_by_id = {passage.passage_id: passage.index for passage in passages}
    selected_indices = [
        index_by_id[str(item.get("passage_id") or "")]
        for item in selection.items
        if str(item.get("passage_id") or "") in index_by_id
    ]
    score_by_index = {
        index: score
        for index, score in zip(selected_indices, selection.selected_scores)
    }

    neighbor_indices: list[int] = []
    for index in list(selected_indices):
        passage = passages[index]
        if (
            len(passage.text) < 260
            or _REFERENCE_PREFIX_RE.search(passage.text)
        ):
            neighbor = index - 1 if index > 0 else index + 1
            if (
                0 <= neighbor < len(passages)
                and neighbor not in selected_indices
                and neighbor not in neighbor_indices
            ):
                neighbor_indices.append(neighbor)

    # Reserve the budget for the relevant passages before adding contextual
    # neighbors. Otherwise an early neighbor can crowd out a later passage that
    # answers another explicit sub-question. Preserve source order only after
    # the final set has been chosen.
    kept: list[int] = []
    used_chars = 0
    for index in selected_indices + neighbor_indices:
        value = passages[index].text
        separator = 2 if kept else 0
        if kept and used_chars + separator + len(value) > max(512, int(max_chars)):
            continue
        kept.append(index)
        used_chars += separator + len(value)
    if not kept:
        kept = [0]
    kept.sort()

    selected_passages = tuple(passages[index] for index in kept)
    text_value = "\n\n".join(item.text for item in selected_passages)
    return PassageSelection(
        text=text_value[: max(512, int(max_chars))],
        passages=selected_passages,
        scores=tuple(score_by_index.get(index, 0.0) for index in kept),
        total_passages=len(passages),
        query_aspects=aspects,
        strategy="paragraph_" + selection.strategy,
        reranker_model=selection.scorer_model,
    )
