"""Bounded long-text QA using parallel RWKV chunk workers.

The design borrows the useful, general part of the Three Body reproduction:
each selected text chunk becomes an independent recurrent-state job, candidate
answers are produced in parallel, and a deterministic reducer turns the
successful candidates into evidence for the Agent's final answer stage.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import math
import re
import time
from typing import Any, Callable


Completion = Callable[..., dict[str, Any]]
_CJK = re.compile(r"[\u3400-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_.+\-]{2,}")
_SPACE = re.compile(r"\s+")
_NUMBER_OR_CODE = re.compile(
    r"(?:[A-Za-z]+\s*)?\d+(?:\.\d+)?"
    r"(?:\s*[A-Za-z]+|兆赫|小时|米|次|号|人|个|名)?"
)
_STRUCTURED_CODE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?=[A-Z0-9_-]*[A-Z])(?=[A-Z0-9_-]*\d)"
    r"[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+"
    r"(?![A-Za-z0-9])"
)
_QUERY_STOP = (
    "请根据",
    "根据",
    "这份文档",
    "这个文档",
    "文档中",
    "材料中",
    "回答",
    "什么",
    "多少",
    "哪些",
    "如何",
    "为什么",
    "where",
    "what",
    "which",
    "who",
    "when",
    "how",
    "the",
    "from",
    "document",
)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ChunkCandidate:
    chunk: TextChunk
    answer: str
    quote: str
    retrieval_score: float
    raw: str
    model_elapsed_ms: float


def _normalize_space(value: str) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _query_features(question: str) -> set[str]:
    text = str(question or "").lower()
    for token in _QUERY_STOP:
        text = text.replace(token, " ")
    features = {match.group(0) for match in _WORD.finditer(text)}
    compact_cjk = "".join(_CJK.findall(text))
    for size in (1, 2, 3):
        for start in range(max(0, len(compact_cjk) - size + 1)):
            features.add(compact_cjk[start : start + size])
    return {feature for feature in features if feature.strip()}


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 160,
) -> list[TextChunk]:
    """Split on paragraph/line boundaries and retain bounded character overlap."""

    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []

    chunks: list[TextChunk] = []
    start = 0
    length = len(normalized)
    while start < length:
        hard_end = min(length, start + max_chars)
        end = hard_end
        if hard_end < length:
            candidates = [
                normalized.rfind("\n\n", start + max_chars // 2, hard_end),
                normalized.rfind("\n", start + max_chars // 2, hard_end),
                normalized.rfind("。", start + max_chars // 2, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (1 if normalized[boundary] != "\n" else 0)
        content = normalized[start:end].strip()
        if content:
            chunks.append(
                TextChunk(
                    chunk_id=len(chunks),
                    text=content,
                    char_start=start,
                    char_end=end,
                )
            )
        if end >= length:
            break
        next_start = max(start + 1, end - overlap_chars)
        while next_start < end and normalized[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def rank_chunks(
    question: str,
    chunks: list[TextChunk],
    *,
    top_k: int = 16,
) -> list[tuple[float, TextChunk]]:
    """Rank chunks with query-feature IDF without gold labels or task rules."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    features = _query_features(question)
    if not features:
        return [(0.0, chunk) for chunk in chunks[:top_k]]
    lowered = [chunk.text.lower() for chunk in chunks]
    document_frequency = Counter(
        feature
        for feature in features
        for text in lowered
        if feature in text
    )
    total = max(1, len(chunks))
    ranked: list[tuple[float, TextChunk]] = []
    for chunk, text in zip(chunks, lowered):
        score = sum(
            math.log((total + 1) / (document_frequency[feature] + 1)) + 1.0
            for feature in features
            if feature in text
        )
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: (item[0], -item[1].chunk_id), reverse=True)
    positive = [item for item in ranked if item[0] > 0]
    selected = positive[:top_k]
    if len(selected) < min(top_k, len(ranked)):
        selected_ids = {item[1].chunk_id for item in selected}
        selected.extend(
            item
            for item in ranked
            if item[1].chunk_id not in selected_ids
        )
    return selected[: min(top_k, len(ranked))]


def render_chunk_worker_prompt(question: str, chunk: TextChunk) -> str:
    """Use a JSON answer prefix, which is materially stronger on current G1I."""

    system = {
        "role": "long_text_chunk_worker",
        "rules": [
            "Use only the supplied chunk.",
            "Return null when the chunk does not answer the question.",
            "When non-null, quote must be an exact substring of the chunk.",
            "Keep the answer short.",
            "Do not call tools.",
        ],
    }
    payload = {
        "question": str(question).strip(),
        "chunk_id": chunk.chunk_id,
        "chunk": chunk.text,
    }
    return (
        "System: "
        + json.dumps(system, ensure_ascii=False, separators=(",", ":"))
        + "\n\nUser: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + '\n\nAssistant: {"answer":'
    )


def parse_chunk_candidate(
    raw: str,
    *,
    chunk: TextChunk,
    retrieval_score: float,
    model_elapsed_ms: float = 0.0,
    question: str = "",
) -> ChunkCandidate | None:
    """Parse the completion after the fixed ``{"answer":`` prefix."""

    suffix = str(raw or "").strip()
    prefixed = '{"answer":' + suffix
    candidates = [prefixed]
    if suffix.startswith("{"):
        candidates.append(suffix)
    value: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            parsed, _end = json.JSONDecoder().raw_decode(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            value = parsed
            break
    if value is None:
        answer_value = _decode_json_field(prefixed, "answer")
        if answer_value is _MISSING:
            return None
        quote_value = _decode_json_field(prefixed, "quote")
        value = {
            "answer": answer_value,
            "quote": "" if quote_value is _MISSING else quote_value,
        }

    answer_value = value.get("answer")
    if answer_value is None:
        return None
    if isinstance(answer_value, (dict, list)):
        answer = json.dumps(
            answer_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        answer = _normalize_space(str(answer_value))
    if not answer or answer.lower() == "null":
        return None

    quote = _normalize_space(str(value.get("quote") or ""))
    normalized_chunk = _normalize_space(chunk.text)
    if quote and quote not in normalized_chunk:
        quote = ""
    grounded_answer = _grounded_answer_fragment(
        answer,
        normalized_chunk,
        question,
    )
    if grounded_answer:
        answer = grounded_answer
    if not quote and grounded_answer:
        position = normalized_chunk.find(grounded_answer)
        quote = normalized_chunk[
            max(0, position - 80) : min(
                len(normalized_chunk),
                position + len(grounded_answer) + 80,
            )
        ]
    if not quote:
        return None
    return ChunkCandidate(
        chunk=chunk,
        answer=answer,
        quote=quote,
        retrieval_score=float(retrieval_score),
        raw=suffix,
        model_elapsed_ms=float(model_elapsed_ms),
    )


_MISSING = object()


def _decode_json_field(text: str, field: str) -> Any:
    """Decode one JSON field even when later generation was truncated."""

    match = re.search(rf'"{re.escape(field)}"\s*:\s*', text)
    if not match:
        return _MISSING
    try:
        value, _end = json.JSONDecoder().raw_decode(text, match.end())
    except (json.JSONDecodeError, TypeError):
        return _MISSING
    return value


def _grounded_answer_fragment(
    answer: str,
    chunk_text: str,
    question: str,
) -> str:
    """Find a short answer fragment that is literally present in the chunk."""

    clean_answer = _normalize_space(answer).strip("`*_ ，,。；;：:！？!?")
    clean_question = _normalize_space(question)
    if clean_answer in chunk_text:
        return clean_answer

    proposed: list[tuple[int, str]] = []

    def add(value: str, priority: int) -> None:
        clean = _normalize_space(value).strip(
            "`*_ ，,。；;：:！？!?（）()[]{}\"'“”‘’"
        )
        if not clean or clean not in chunk_text:
            return
        if len(clean) < 2 and not clean.isdigit():
            return
        if clean in clean_question and not any(char.isdigit() for char in clean):
            priority -= 40
        proposed.append((priority + min(len(clean), 30), clean))

    for quoted in re.findall(r"[“”\"'‘’]([^“”\"'‘’]{1,80})[“”\"'‘’]", answer):
        add(quoted, 100)
    for bold in re.findall(r"\*\*([^*]{1,80})\*\*", answer):
        for unit in re.split(r"[！!？?。；;，,]", bold):
            add(unit, 95)
    for token in _NUMBER_OR_CODE.findall(answer):
        add(token.replace(" ", ""), 110)
    for unit in re.split(r"[：:。；;，,！!？?\n]", answer):
        add(unit, 30)

    matcher = SequenceMatcher(None, clean_answer, chunk_text, autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size >= 2:
            add(
                chunk_text[block.b : block.b + block.size],
                60,
            )
    if not proposed:
        return ""
    proposed.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return proposed[0][1]


def _structured_code_candidate(
    question: str,
    ranked: list[tuple[float, TextChunk]],
) -> ChunkCandidate | None:
    """Return an exact, high-confidence code from the best grounded sentence.

    Identifiers such as release IDs, approval codes and ticket numbers are
    already self-delimiting.  Extracting them deterministically avoids an
    expensive model pass while retaining an exact source quote.  Free-form
    answers still use the parallel recurrent-state workers below.
    """

    query_features = {
        feature for feature in _query_features(question) if len(feature) >= 2
    }
    question_upper = str(question or "").upper()
    best: tuple[float, ChunkCandidate] | None = None
    for retrieval_score, chunk in ranked:
        sentences = re.split(r"(?<=[。！？!?；;])|\n+", chunk.text)
        for sentence in sentences:
            quote = _normalize_space(sentence)
            if not quote:
                continue
            overlap = sum(
                len(feature)
                for feature in query_features
                if feature in quote.lower()
            )
            if query_features and overlap == 0:
                continue
            for match in _STRUCTURED_CODE.finditer(quote):
                answer = match.group(0)
                if answer.upper() in question_upper:
                    continue
                score = float(retrieval_score) + 0.25 * overlap
                candidate = ChunkCandidate(
                    chunk=chunk,
                    answer=answer,
                    quote=quote,
                    retrieval_score=float(retrieval_score),
                    raw="lexical_structured_code",
                    model_elapsed_ms=0.0,
                )
                if best is None or score > best[0]:
                    best = (score, candidate)
    return best[1] if best else None


class LongTextQAAdapter:
    """Analyze bounded pasted text with parallel chunk extraction."""

    def __init__(
        self,
        complete: Completion,
        *,
        top_k: int = 16,
        concurrency: int = 8,
        chunk_chars: int = 1200,
        overlap_chars: int = 160,
        max_document_chars: int = 1_000_000,
        max_evidence: int = 8,
        worker_max_tokens: int = 64,
    ) -> None:
        self.complete = complete
        self.top_k = max(1, int(top_k))
        self.concurrency = max(1, int(concurrency))
        self.chunk_chars = int(chunk_chars)
        self.overlap_chars = int(overlap_chars)
        self.max_document_chars = max(1, int(max_document_chars))
        self.max_evidence = max(1, int(max_evidence))
        self.worker_max_tokens = max(16, min(int(worker_max_tokens), 192))

    def _worker(
        self,
        question: str,
        score: float,
        chunk: TextChunk,
    ) -> tuple[ChunkCandidate | None, dict[str, Any]]:
        started = time.perf_counter()
        try:
            completion = self.complete(
                render_chunk_worker_prompt(question, chunk),
                max_tokens=self.worker_max_tokens,
                stops=["}", "\nUser:", "\nSystem:", "</s>"],
            )
            candidate = parse_chunk_candidate(
                completion.get("raw", ""),
                chunk=chunk,
                retrieval_score=score,
                model_elapsed_ms=float(
                    completion.get("model_elapsed_ms") or 0.0
                ),
                question=question,
            )
            return candidate, {
                "chunk_id": chunk.chunk_id,
                "status": "candidate" if candidate else "null_or_invalid",
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
                "output_tokens": int(completion.get("output_tokens") or 0),
            }
        except Exception as exc:
            return None, {
                "chunk_id": chunk.chunk_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
            }

    @staticmethod
    def _reduce(
        question: str,
        candidates: list[ChunkCandidate],
        max_evidence: int,
    ) -> list[ChunkCandidate]:
        question_features = _query_features(question)
        support = Counter(
            _normalize_space(candidate.answer).lower()
            for candidate in candidates
        )

        def key(candidate: ChunkCandidate) -> tuple[float, float, int]:
            quote_lower = candidate.quote.lower()
            overlap = sum(
                len(feature)
                for feature in question_features
                if feature in quote_lower
            )
            answer_key = _normalize_space(candidate.answer).lower()
            return (
                candidate.retrieval_score
                + 0.25 * overlap
                + 2.0 * math.log1p(support[answer_key]),
                candidate.retrieval_score,
                -candidate.chunk.chunk_id,
            )

        best_by_answer: dict[str, ChunkCandidate] = {}
        for candidate in sorted(candidates, key=key, reverse=True):
            answer_key = _normalize_space(candidate.answer).lower()
            best_by_answer.setdefault(answer_key, candidate)
        return list(best_by_answer.values())[:max_evidence]

    def execute(
        self,
        text: str,
        question: str,
        *,
        document_name: str = "pasted-text",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        clean_question = str(question or "").strip()
        if not clean_question:
            return {
                "status": "invalid",
                "evidence": [],
                "message": "long_text_qa requires a non-empty question.",
            }
        document_text = str(text or "")
        if not document_text.strip():
            return {
                "status": "empty",
                "evidence": [],
                "message": "No pasted long text is active in this session.",
            }
        if len(document_text) > self.max_document_chars:
            return {
                "status": "invalid",
                "evidence": [],
                "message": (
                    f"pasted text exceeds {self.max_document_chars} "
                    "character limit"
                ),
            }
        clean_name = _normalize_space(document_name)[:80] or "pasted-text"

        chunks = chunk_text(
            document_text,
            max_chars=self.chunk_chars,
            overlap_chars=self.overlap_chars,
        )
        if not chunks:
            return {
                "status": "empty",
                "evidence": [],
                "message": "document contains no usable text",
            }
        ranked = rank_chunks(clean_question, chunks, top_k=self.top_k)
        structured_candidate = _structured_code_candidate(
            clean_question,
            ranked,
        )
        if structured_candidate is not None:
            return {
                "status": "ok",
                "tool": "long_text_qa",
                "answer_hint": structured_candidate.answer,
                "answer_hint_evidence_id": "L1",
                "document": {
                    "source": "session_pasted_text",
                    "name": clean_name,
                    "chars": len(document_text),
                    "chunks": len(chunks),
                },
                "retrieval": {
                    "method": "query_feature_idf+structured_code",
                    "selected_chunks": len(ranked),
                    "top_k": self.top_k,
                },
                "workers": {
                    "submitted": 0,
                    "completed": 0,
                    "concurrency": 0,
                    "candidates": 1,
                    "errors": 0,
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000.0,
                        3,
                    ),
                },
                "evidence": [
                    {
                        "id": "L1",
                        "title": (
                            f"{clean_name} · chunk "
                            f"{structured_candidate.chunk.chunk_id}"
                        ),
                        "content": structured_candidate.quote,
                        "uri": (
                            "session-text://current#chunk="
                            f"{structured_candidate.chunk.chunk_id}"
                        ),
                        "chunk_id": structured_candidate.chunk.chunk_id,
                        "answer_candidate": structured_candidate.answer,
                        "retrieval_score": round(
                            structured_candidate.retrieval_score,
                            6,
                        ),
                    }
                ],
                "trace": [
                    {
                        "chunk_id": structured_candidate.chunk.chunk_id,
                        "status": "lexical_structured_code",
                        "elapsed_ms": 0.0,
                        "output_tokens": 0,
                    }
                ],
                "message": "",
            }
        traces: list[dict[str, Any]] = []
        candidates: list[ChunkCandidate] = []
        workers = min(self.concurrency, len(ranked))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="rwkv-long-text",
        ) as executor:
            futures = {
                executor.submit(self._worker, clean_question, score, chunk): chunk
                for score, chunk in ranked
            }
            for future in as_completed(futures):
                candidate, trace = future.result()
                traces.append(trace)
                if candidate is not None:
                    candidates.append(candidate)

        reduced = self._reduce(
            clean_question,
            candidates,
            self.max_evidence,
        )
        evidence = [
            {
                "id": f"L{index}",
                "title": f"{clean_name} · chunk {candidate.chunk.chunk_id}",
                "content": candidate.quote,
                "uri": f"session-text://current#chunk={candidate.chunk.chunk_id}",
                "chunk_id": candidate.chunk.chunk_id,
                "answer_candidate": candidate.answer,
                "retrieval_score": round(candidate.retrieval_score, 6),
            }
            for index, candidate in enumerate(reduced, 1)
        ]
        errors = sum(trace["status"] == "error" for trace in traces)
        return {
            "status": "ok" if evidence else "empty",
            "tool": "long_text_qa",
            "answer_hint": reduced[0].answer if reduced else None,
            "answer_hint_evidence_id": "L1" if reduced else None,
            "document": {
                "source": "session_pasted_text",
                "name": clean_name,
                "chars": len(document_text),
                "chunks": len(chunks),
            },
            "retrieval": {
                "method": "query_feature_idf",
                "selected_chunks": len(ranked),
                "top_k": self.top_k,
            },
            "workers": {
                "submitted": len(ranked),
                "completed": len(traces),
                "concurrency": workers,
                "candidates": len(candidates),
                "errors": errors,
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
            },
            "evidence": evidence,
            "trace": sorted(traces, key=lambda item: item["chunk_id"]),
            "message": (
                ""
                if evidence
                else "No chunk worker returned a grounded candidate."
            ),
        }
