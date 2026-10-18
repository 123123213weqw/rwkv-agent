from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "long-knowledge-case.v1"
EXPECTATIONS = frozenset({"relevant", "missing"})


@dataclass(frozen=True)
class RelevantPage:
    page_id: str
    relevance: int

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RelevantPage":
        page_id = str(value.get("page_id") or "").strip()
        relevance = int(value.get("relevance") or 0)
        if not page_id:
            raise ValueError("relevant page_id must not be empty")
        if relevance <= 0:
            raise ValueError("relevant page relevance must be positive")
        return cls(page_id=page_id, relevance=relevance)

    def to_dict(self) -> dict:
        return {"page_id": self.page_id, "relevance": self.relevance}


@dataclass(frozen=True)
class LongKnowledgeCase:
    id: str
    query: str
    language: str
    source_dataset: str
    source_split: str
    source_qid: str
    relevant_pages: tuple[RelevantPage, ...]
    expectation: str = "relevant"
    query_type: str = "unspecified"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LongKnowledgeCase":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {value.get('schema_version')!r}")
        fields = {
            name: str(value.get(name) or "").strip()
            for name in ("id", "query", "language", "source_dataset", "source_split", "source_qid")
        }
        missing = [name for name, item in fields.items() if not item]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        raw_pages = value.get("relevant_pages")
        if not isinstance(raw_pages, Sequence) or isinstance(raw_pages, (str, bytes)):
            raise ValueError("relevant_pages must be a list")
        if any(not isinstance(item, Mapping) for item in raw_pages):
            raise ValueError("each relevant_pages item must be an object")
        pages = tuple(RelevantPage.from_dict(item) for item in raw_pages)
        expectation = str(value.get("expectation") or "relevant").strip()
        query_type = str(value.get("query_type") or "unspecified").strip()
        if expectation not in EXPECTATIONS:
            raise ValueError(f"unsupported expectation: {expectation!r}")
        if expectation == "relevant" and not pages:
            raise ValueError("at least one positive relevant page is required")
        if expectation == "missing" and pages:
            raise ValueError("missing cases must not contain relevant pages")
        page_ids = [item.page_id for item in pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("relevant page_ids must be unique")
        return cls(
            **fields,
            relevant_pages=pages,
            expectation=expectation,
            query_type=query_type,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "query": self.query,
            "language": self.language,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "source_qid": self.source_qid,
            "expectation": self.expectation,
            "query_type": self.query_type,
            "relevant_pages": [item.to_dict() for item in self.relevant_pages],
        }


def load_cases(path: str | Path) -> list[LongKnowledgeCase]:
    cases: list[LongKnowledgeCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = LongKnowledgeCase.from_dict(json.loads(line))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid case at line {line_number}: {exc}") from exc
        if case.id in seen_ids:
            raise ValueError(f"duplicate case id at line {line_number}: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("test set is empty")
    return cases


def write_cases(path: str | Path, cases: Iterable[LongKnowledgeCase]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
