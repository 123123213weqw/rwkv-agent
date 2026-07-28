"""Bounded recurrent-state branch reader for LongBench-style choices."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping, Sequence
import uuid

from .tools.long_text import TextChunk, rank_chunks


CHOICE = re.compile(r"(?:\"choice\"\s*:\s*\")?\s*([ABCD])\b", re.I)


def split_choice_question(question: str) -> tuple[str, dict[str, str]]:
    """Separate the question stem from A-D options without dataset metadata."""

    value = str(question or "").strip()
    options = {
        label.upper(): text.strip()
        for label, text in re.findall(
            r"(?m)^\s*([A-D])[.)]\s*(.+?)\s*$",
            value,
        )
        if text.strip()
    }
    stem = re.split(r"(?m)^\s*A[.)]\s+", value, maxsplit=1)[0].strip()
    return stem or value, options


def _root_prompt(
    question: str,
    primary: Sequence[tuple[float, TextChunk]],
) -> str:
    excerpts = "\n\n".join(
        f"[Primary chunk {chunk.chunk_id}; retrieval {score:.4f}]\n{chunk.text}"
        for score, chunk in primary
    )
    return (
        "System: You are the retained root of a recurrent-state long-document "
        "reader. Retain the primary Top-6 excerpts and question below. Four "
        "branches will inherit this state; one answers from the primary view and "
        "three inspect small complementary packets. In every branch choose "
        "exactly one of A, B, C, or D. Do not call tools or expose reasoning."
        "\n\nPrimary excerpts:\n"
        + excerpts
        + "\n\nQuestion:\n"
        + str(question).strip()
    )


def _branch_input(
    question: str,
    selected: Sequence[tuple[float, TextChunk]],
) -> str:
    excerpts = "\n\n".join(
        f"[Chunk {chunk.chunk_id}; retrieval {score:.4f}]\n{chunk.text}"
        for score, chunk in selected
    )
    return (
        "\n\nUser: Combine the primary Top-6 evidence already retained in state "
        "with the following small complementary packet. Choose the best-supported "
        "option. If evidence "
        "is weak, still choose the most likely option. Output exactly one letter: "
        "A, B, C, or D.\n\nEvidence view:\n"
        + (excerpts or "[No additional excerpts; judge the retained Top-6 view.]")
        + "\n\nQuestion:\n"
        + str(question).strip()
        + '\n\nAssistant: {"choice":"'
    )


def _final_input(question: str, reports: Sequence[Mapping[str, Any]]) -> str:
    compact = json.dumps(list(reports), ensure_ascii=False, separators=(",", ":"))
    return (
        "\n\nTool: <branch_reports>"
        + compact
        + "</branch_reports>\n\nUser: Final choice stage. Reconcile the branch "
        "reports against the original question and output exactly one letter: A, "
        "B, C, or D. Do not explain. Original question:\n"
        + str(question).strip()
        + '\n\nAssistant: {"choice":"'
    )


def _parse_choice(text: str) -> str:
    match = CHOICE.search(str(text or "").strip())
    return match.group(1).upper() if match else ""


def _permutation_root_prompt(
    question: str,
    selected: Sequence[tuple[float, TextChunk]],
) -> tuple[str, dict[str, str]]:
    stem, options = split_choice_question(question)
    if set(options) != set("ABCD"):
        raise ValueError("question must contain exactly A-D options")
    excerpts = "\n\n".join(
        f"[Evidence {chunk.chunk_id}; retrieval {score:.4f}]\n{chunk.text}"
        for score, chunk in selected[:6]
    )
    prompt = (
        "System: You are the retained evidence state for a multiple-choice "
        "reader. Read the evidence and question stem carefully. Four branches "
        "will receive the same answer options under different neutral labels. "
        "Judge from evidence, not label position. Do not call tools or expose "
        "reasoning.\n\nEvidence:\n"
        + excerpts
        + "\n\nQuestion stem:\n"
        + stem
    )
    return prompt, options


def _permutation_branch_input(
    options: Mapping[str, str],
    mapping: Mapping[str, str],
) -> str:
    rendered = "\n".join(
        f"{display}. {options[original]}" for display, original in mapping.items()
    )
    return (
        "\n\nUser: Select the single best-supported option below. Reply with "
        "exactly its displayed label A, B, C, or D.\n"
        + rendered
        + '\n\nAssistant: {"choice":"'
    )


def run_state_longbench_permutation_reader(
    state_model: Any,
    *,
    question: str,
    selected: Sequence[tuple[float, TextChunk]],
    session_id: str,
) -> dict[str, Any]:
    """Debias choice labels by scoring four cyclic option permutations in state."""

    if not selected:
        raise ValueError("selected excerpts are required")
    started = time.perf_counter()
    owner_id = f"long-permute-{session_id[:32]}-{uuid.uuid4().hex}"
    home_url = ""
    state_ids: list[str] = []
    release: dict[str, Any] = {}
    response: dict[str, Any] | None = None
    try:
        root_prompt, options = _permutation_root_prompt(question, selected)
        root = state_model.state_prefill(owner_id=owner_id, prompt=root_prompt)
        root_id = str(root["state_id"])
        home_url = str(root["home_url"])
        state_ids.append(root_id)
        labels = tuple("ABCD")
        mappings = [
            {
                display: labels[(index + shift) % len(labels)]
                for index, display in enumerate(labels)
            }
            for shift in range(len(labels))
        ]
        branches = state_model.state_fork(
            home_url=home_url,
            owner_id=owner_id,
            parent_state_id=root_id,
            branches=[f"permutation-{index}" for index in range(len(mappings))],
        )
        state_ids.extend(str(item["state_id"]) for item in branches)
        classifications = state_model.state_batch_classify(
            home_url=home_url,
            owner_id=owner_id,
            items=[
                {
                    "state_id": str(branch["state_id"]),
                    "input": _permutation_branch_input(options, mapping),
                }
                for branch, mapping in zip(branches, mappings, strict=True)
            ],
            labels={label: label for label in labels},
        )
        aggregate = {label: 0.0 for label in labels}
        reports: list[dict[str, Any]] = []
        identity_scores: dict[str, float] = {}
        for index, (mapping, item) in enumerate(
            zip(mappings, classifications, strict=True)
        ):
            displayed_scores = {
                label: float(score)
                for label, score in dict(item.get("scores") or {}).items()
                if label in labels
            }
            mapped_scores = {
                mapping[display]: score
                for display, score in displayed_scores.items()
            }
            for original, score in mapped_scores.items():
                aggregate[original] += score
            if index == 0:
                identity_scores = mapped_scores
            reports.append(
                {
                    "branch": str(item.get("branch") or ""),
                    "mapping": dict(mapping),
                    "displayed_scores": displayed_scores,
                    "mapped_scores": mapped_scores,
                }
            )
        best_score = max(aggregate.values())
        tied = [label for label in labels if aggregate[label] == best_score]
        choice = (
            max(tied, key=lambda label: identity_scores.get(label, float("-inf")))
            if len(tied) > 1
            else tied[0]
        )
        response = {
            "choice": choice,
            "reports": reports,
            "choice_scores": aggregate,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "states_created": len(state_ids),
            "states_released": 0,
            "state_leak_count": len(state_ids),
            "root_output_tokens": 0,
            "branch_output_tokens": 0,
            "release": release,
        }
        return response
    finally:
        if home_url and state_ids:
            try:
                release = state_model.state_release(
                    home_url=home_url,
                    owner_id=owner_id,
                    state_ids=state_ids,
                )
            except Exception as exc:
                release = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "released": 0,
                }
        if response is not None:
            released = int(release.get("released") or 0)
            response["release"] = release
            response["states_released"] = released
            response["state_leak_count"] = max(0, len(state_ids) - released)


def run_state_longbench_chunk_ensemble(
    state_model: Any,
    *,
    question: str,
    selected: Sequence[tuple[float, TextChunk]],
    session_id: str,
    branch_width: int = 8,
) -> dict[str, Any]:
    """Read Top-K chunks concurrently and cancel A-D positional logit priors.

    The root retains the question stem once.  Each fork receives one chunk and
    one cyclic relabeling of the options.  With eight branches every original
    option occupies every displayed label exactly twice, so summing mapped
    next-token logits cancels a fixed label-position bias while keeping all
    evidence reads recurrent-state native.
    """

    if branch_width not in {4, 8}:
        raise ValueError("branch_width must be 4 or 8")
    packet = list(selected[:branch_width])
    if len(packet) < branch_width:
        raise ValueError("selected excerpts do not fill branch_width")
    stem, options = split_choice_question(question)
    if set(options) != set("ABCD"):
        raise ValueError("question must contain exactly A-D options")
    started = time.perf_counter()
    owner_id = f"long-chunks-{session_id[:32]}-{uuid.uuid4().hex}"
    home_url = ""
    state_ids: list[str] = []
    release: dict[str, Any] = {}
    response: dict[str, Any] | None = None
    try:
        root = state_model.state_prefill(
            owner_id=owner_id,
            prompt=(
                "System: You are the retained root of a parallel long-document "
                "reader. Each fork will inspect one evidence chunk. Score the "
                "answer options only from the supplied chunk and the retained "
                "question. Do not call tools or expose reasoning.\n\nQuestion "
                "stem:\n"
                + stem
            ),
        )
        root_id = str(root["state_id"])
        home_url = str(root["home_url"])
        state_ids.append(root_id)
        labels = tuple("ABCD")
        mappings = [
            {
                display: labels[(index + branch_index) % len(labels)]
                for index, display in enumerate(labels)
            }
            for branch_index in range(branch_width)
        ]
        branches = state_model.state_fork(
            home_url=home_url,
            owner_id=owner_id,
            parent_state_id=root_id,
            branches=[f"chunk-{chunk.chunk_id}" for _score, chunk in packet],
        )
        state_ids.extend(str(item["state_id"]) for item in branches)
        items = []
        for branch, mapping, (retrieval_score, chunk) in zip(
            branches, mappings, packet, strict=True
        ):
            options_text = "\n".join(
                f"{display}. {options[original]}"
                for display, original in mapping.items()
            )
            items.append(
                {
                    "state_id": str(branch["state_id"]),
                    "input": (
                        f"\n\nUser: Evidence chunk {chunk.chunk_id} "
                        f"(retrieval {retrieval_score:.4f}):\n{chunk.text}\n\n"
                        "Choose the option best supported by this chunk. If this "
                        "chunk is incomplete, choose the most plausible option. "
                        "Reply exactly with its displayed label.\n"
                        + options_text
                        + '\n\nAssistant: {"choice":"'
                    ),
                }
            )
        classifications = state_model.state_batch_classify(
            home_url=home_url,
            owner_id=owner_id,
            items=items,
            labels={label: label for label in labels},
        )
        aggregate = {label: 0.0 for label in labels}
        reports: list[dict[str, Any]] = []
        identity_scores: dict[str, float] = {}
        for index, (mapping, classification, selected_item) in enumerate(
            zip(mappings, classifications, packet, strict=True)
        ):
            retrieval_score, chunk = selected_item
            displayed = {
                label: float(score)
                for label, score in dict(classification.get("scores") or {}).items()
                if label in labels
            }
            mapped = {
                mapping[display]: score for display, score in displayed.items()
            }
            for original, score in mapped.items():
                aggregate[original] += score
            if index == 0:
                identity_scores = mapped
            reports.append(
                {
                    "branch": str(classification.get("branch") or ""),
                    "chunk_id": chunk.chunk_id,
                    "retrieval_score": float(retrieval_score),
                    "mapping": dict(mapping),
                    "displayed_scores": displayed,
                    "mapped_scores": mapped,
                }
            )
        best_score = max(aggregate.values())
        tied = [label for label in labels if aggregate[label] == best_score]
        choice = (
            max(tied, key=lambda label: identity_scores.get(label, float("-inf")))
            if len(tied) > 1
            else tied[0]
        )
        response = {
            "choice": choice,
            "reports": reports,
            "choice_scores": aggregate,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "states_created": len(state_ids),
            "states_released": 0,
            "state_leak_count": len(state_ids),
            "root_output_tokens": 0,
            "branch_output_tokens": 0,
            "release": release,
        }
        return response
    finally:
        if home_url and state_ids:
            try:
                release = state_model.state_release(
                    home_url=home_url,
                    owner_id=owner_id,
                    state_ids=state_ids,
                )
            except Exception as exc:
                release = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "released": 0,
                }
        if response is not None:
            released = int(release.get("released") or 0)
            response["release"] = release
            response["states_released"] = released
            response["state_leak_count"] = max(0, len(state_ids) - released)


def build_state_evidence_views(
    question: str,
    chunks: Sequence[TextChunk],
    *,
    branch_width: int = 4,
    chunks_per_view: int = 6,
) -> list[list[tuple[float, TextChunk]]]:
    """Build generic complementary views without labels or task-specific cases."""

    if not chunks:
        return []
    if not 1 <= int(branch_width) <= 4:
        raise ValueError("branch_width must be between 1 and 4")
    if chunks_per_view <= 0:
        raise ValueError("chunks_per_view must be positive")
    values = list(chunks)
    core_question, parsed_options = split_choice_question(question)
    choice_lines = list(parsed_options.values())
    lexical = rank_chunks(question, values, top_k=chunks_per_view)
    core = rank_chunks(core_question or question, values, top_k=chunks_per_view)

    contrast: list[tuple[float, TextChunk]] = []
    seen: set[int] = set()
    for option in choice_lines[:4]:
        for score, chunk in rank_chunks(
            f"{core_question} {option}",
            values,
            top_k=2,
        ):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            contrast.append((score, chunk))
            break
    for score, chunk in lexical:
        if len(contrast) >= chunks_per_view:
            break
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            contrast.append((score, chunk))

    by_id = {chunk.chunk_id: chunk for chunk in values}
    neighbor: list[tuple[float, TextChunk]] = []
    seen.clear()
    for score, anchor in lexical[:3]:
        for chunk_id in (anchor.chunk_id - 1, anchor.chunk_id, anchor.chunk_id + 1):
            chunk = by_id.get(chunk_id)
            if chunk is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            neighbor.append((score if chunk_id == anchor.chunk_id else score * 0.9, chunk))
            if len(neighbor) >= chunks_per_view:
                break
        if len(neighbor) >= chunks_per_view:
            break
    for score, chunk in core:
        if len(neighbor) >= chunks_per_view:
            break
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            neighbor.append((score, chunk))

    candidates = [lexical, core, contrast, neighbor]
    return [list(view[:chunks_per_view]) for view in candidates[:branch_width]]


def run_state_longbench_reader(
    state_model: Any,
    *,
    question: str,
    selected: Sequence[tuple[float, TextChunk]],
    session_id: str,
    branch_width: int = 4,
    views: Sequence[Sequence[tuple[float, TextChunk]]] | None = None,
) -> dict[str, Any]:
    """Read disjoint excerpt packets in forked states and resume the root."""

    if not 1 <= int(branch_width) <= 4:
        raise ValueError("branch_width must be between 1 and 4")
    if not selected:
        raise ValueError("selected excerpts are required")
    started = time.perf_counter()
    owner_id = f"long-{session_id[:40]}-{uuid.uuid4().hex}"
    home_url = ""
    state_ids: list[str] = []
    release: dict[str, Any] = {}
    response: dict[str, Any] | None = None
    try:
        primary = list(views[0]) if views is not None else list(selected[:6])
        root = state_model.state_prefill(
            owner_id=owner_id,
            prompt=_root_prompt(question, primary),
        )
        root_id = str(root["state_id"])
        home_url = str(root["home_url"])
        state_ids.append(root_id)
        branches = state_model.state_fork(
            home_url=home_url,
            owner_id=owner_id,
            parent_state_id=root_id,
            branches=[f"reader-{index + 1}" for index in range(branch_width)],
        )
        state_ids.extend(str(item["state_id"]) for item in branches)
        complementary = list(selected[6:])
        primary_ids = {chunk.chunk_id for _score, chunk in primary}
        packets = (
            [[]]
            + [
                [
                    (score, chunk)
                    for score, chunk in view
                    if chunk.chunk_id not in primary_ids
                ][:2]
                for view in views[1:branch_width]
            ]
            if views is not None
            else [
                list(complementary[index::branch_width])
                for index in range(branch_width)
            ]
        )
        if len(packets) != branch_width:
            raise ValueError("views must contain one packet per branch")
        if views is None and not any(packets):
            packets = [list(selected[index::branch_width]) for index in range(branch_width)]
        completions = state_model.state_batch_continue(
            home_url=home_url,
            owner_id=owner_id,
            items=[
                {
                    "state_id": str(branch["state_id"]),
                    "input": _branch_input(question, packet),
                }
                for branch, packet in zip(branches, packets, strict=True)
            ],
            stops=['"}', "\n", "</answer>", "\n\nUser:", "\nSystem:"],
            max_tokens=4,
        )
        reports = []
        for packet, completion in zip(packets, completions, strict=True):
            text = str(completion.get("text") or "")
            reports.append(
                {
                    "branch": str(completion.get("branch") or ""),
                    "choice": _parse_choice(text),
                    "report": text[:240],
                    "chunk_ids": [chunk.chunk_id for _score, chunk in packet],
                }
            )
        weights = (1.5, 1.0, 0.9, 0.8)
        vote_scores: dict[str, float] = {}
        for index, item in enumerate(reports):
            choice = str(item["choice"])
            if choice:
                vote_scores[choice] = vote_scores.get(choice, 0.0) + weights[index]
        choice = (
            max(sorted(vote_scores), key=lambda value: vote_scores[value])
            if vote_scores
            else ""
        )
        response = {
            "choice": choice,
            "reports": reports,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "states_created": len(state_ids),
            "states_released": 0,
            "state_leak_count": len(state_ids),
            "root_output_tokens": 0,
            "branch_output_tokens": sum(
                int(item.get("output_tokens") or 0) for item in completions
            ),
            "release": release,
            "vote_scores": vote_scores,
        }
        return response
    finally:
        if home_url and state_ids:
            try:
                release = state_model.state_release(
                    home_url=home_url,
                    owner_id=owner_id,
                    state_ids=state_ids,
                )
            except Exception as exc:
                release = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "released": 0,
                }
        if response is not None:
            released = int(release.get("released") or 0)
            response["release"] = release
            response["states_released"] = released
            response["state_leak_count"] = max(0, len(state_ids) - released)
