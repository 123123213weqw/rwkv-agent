#!/usr/bin/env python3
"""Build Train/Dev-only SFT continuations for RWKV-Agent-FitGen-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from benchmarks.run_fitgen_benchmark import (
    TOOL_SUFFIX,
    render_bfcl_prompt,
    render_longbench_prompt,
)
from benchmarks.agent_benchmark_metrics import token_f1
from rwkv_agent.claim_verifier import verify_answer_claims
from rwkv_agent.state_agent import (
    ANSWER_SUFFIX,
    attach_evidence_citations,
    compact_answer_evidence,
    render_answer_fallback_prompt,
    render_compact_answer_prompt,
    render_branch_step,
    render_root_prompt,
)
from rwkv_agent.tools.long_text import chunk_text, rank_chunks


DATASETS = ("bfcl", "webwalkerqa", "frames", "longbench_v2", "alce")
SCHEMA_VERSION = "rwkv-agent-fitgen-sft.v1"
SPACE = re.compile(r"\s+")
ALCE_WORD = re.compile(r"[\w]+(?:[_.+%-][\w]+)*", re.UNICODE)
ALCE_SENTENCE = re.compile(r"[^。！？!?\n]+(?:[。！？!?]|$)")
ALCE_LIST_SEPARATOR = re.compile(r"[,;]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def record(
    case: Mapping[str, Any],
    *,
    task: str,
    suffix: str,
    prompt: str,
    response: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"{case['id']}::{suffix}",
        "case_id": str(case["id"]),
        "dataset": str(case["dataset"]),
        "task": task,
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
    }


def _canonical_bfcl_value(value: Any, schema: Mapping[str, Any]) -> Any:
    """Render a BFCL value according to the advertised function schema.

    BFCL reference JSON uses ``""`` as an omitted-argument sentinel and one
    legacy nested-object family wraps scalar values in singleton lists.  Those
    are evaluator annotations, not executable tool-call values.
    """

    expected = str(schema.get("type") or "").casefold()
    if expected in {"dict", "object"} and isinstance(value, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            required = {str(item) for item in schema.get("required") or ()}
            output: dict[str, Any] = {}
            for key, child in value.items():
                child_schema = properties.get(key)
                if not isinstance(child_schema, Mapping):
                    continue
                if child == "" and key not in required:
                    continue
                output[str(key)] = _canonical_bfcl_value(child, child_schema)
            return output
        if value and all(
            isinstance(item, list) and len(item) == 1 for item in value.values()
        ):
            return {str(key): item[0] for key, item in value.items()}
        return {str(key): child for key, child in value.items()}
    if expected == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            return [_canonical_bfcl_value(item, item_schema) for item in value]
    return value


def canonical_bfcl_calls(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert BFCL Gold annotations into strict executable tool-call JSON."""

    schemas = {
        str(tool.get("name") or ""): tool
        for tool in case.get("available_tools") or ()
        if isinstance(tool, Mapping)
    }
    output: list[dict[str, Any]] = []
    for raw_call in case["gold"]["tool_calls"]:
        name = str(raw_call["name"])
        tool = schemas.get(name)
        if not isinstance(tool, Mapping):
            output.append(
                {"name": name, "arguments": dict(raw_call.get("arguments") or {})}
            )
            continue
        parameter_schema = tool.get("parameters")
        if not isinstance(parameter_schema, Mapping):
            parameter_schema = {}
        properties = parameter_schema.get("properties")
        if not isinstance(properties, Mapping):
            properties = {}
        required = {str(item) for item in parameter_schema.get("required") or ()}
        arguments: dict[str, Any] = {}
        for key, value in dict(raw_call.get("arguments") or {}).items():
            child_schema = properties.get(key)
            if not isinstance(child_schema, Mapping):
                continue
            expected = str(child_schema.get("type") or "").casefold()
            if value == "" and (key not in required or expected != "string"):
                continue
            arguments[str(key)] = _canonical_bfcl_value(value, child_schema)
        output.append({"name": name, "arguments": arguments})
    return output


def bfcl_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = canonical_bfcl_calls(case)
    payload = json.dumps(calls, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return [
        record(
            case,
            task="bfcl_tool_call",
            suffix="call",
            prompt=render_bfcl_prompt(case),
            response=payload + TOOL_SUFFIX,
        )
    ]


def _compact_query(value: str, *, limit: int = 420) -> str:
    return SPACE.sub(" ", str(value or "")).strip()[:limit]


def search_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    question = str(case["prompt"])
    branch_prompt = render_root_prompt(question) + render_branch_step(
        question=question,
        mission="Find the primary answer and the strongest directly relevant source.",
        round_index=1,
        observation=None,
    )
    query = _compact_query(question)
    call = json.dumps(
        {"name": "web_search", "arguments": {"query": query}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rows = [
        record(
            case,
            task="web_search_call",
            suffix="search",
            prompt=branch_prompt,
            response=call + "</tool_call>",
        )
    ]

    answers = [str(value) for value in case["gold"].get("answers") or [] if str(value).strip()]
    uris = [str(value) for value in case["gold"].get("source_uris") or [] if str(value).strip()]
    if answers and uris:
        evidence = []
        for index, uri in enumerate(uris[:5], 1):
            host = (urlsplit(uri).hostname or uri).removeprefix("www.")
            evidence.append(
                {
                    "id": f"W{index}",
                    "title": host,
                    "content": answers[0] if index == 1 else f"Corroborating source: {host}",
                    "uri": uri,
                }
            )
        answer = attach_evidence_citations(answers[0], evidence)
        rows.append(
            record(
                case,
                task="web_evidence_answer",
                suffix="answer",
                prompt=render_answer_fallback_prompt(question, evidence),
                response=answer + ANSWER_SUFFIX,
            )
        )
    return rows


def longbench_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    chunks = chunk_text(str(case.get("context") or ""), max_chars=1600, overlap_chars=160)
    selected = rank_chunks(str(case["prompt"]), chunks, top_k=6)
    choice = str(case["gold"]["answers"][0]).strip().upper()
    return [
        record(
            case,
            task="long_text_choice",
            suffix="choice",
            prompt=render_longbench_prompt(case, selected),
            response=choice + '"}',
        )
    ]


def _alce_cited(candidate: str, evidence_id: str) -> str:
    return f"{SPACE.sub(' ', candidate).strip()} [{evidence_id}]".strip()


def _alce_supported(candidate: str, evidence: Sequence[Mapping[str, Any]]) -> bool:
    claims = verify_answer_claims(candidate, evidence)
    return bool(claims) and all(bool(claim.get("supported")) for claim in claims)


def concise_supported_alce_answer(
    references: Sequence[str],
    evidence: Sequence[Mapping[str, Any]],
    *,
    list_answer: bool = False,
    list_word_budget: int = 8,
    extractive_words: int = 20,
    filtered_words: int = 16,
) -> str:
    """Build a short, evidence-supported ALCE supervision target.

    List-style tasks keep complete reference items only when an evidence item
    independently supports them.  Narrative tasks use the most relevant
    contiguous evidence span.  Every candidate is checked by the same frozen
    Gold-blind claim verifier used by the benchmark.  A reference-word filter
    is retained only as a last fallback.  The routine reads only the supplied
    Train/Dev reference and evidence packet.
    """

    usable = [
        dict(item)
        for item in evidence
        if str(item.get("id") or "").startswith("W")
        and str(item.get("content") or "").strip()
    ]
    gold = [str(value).strip() for value in references if str(value).strip()]
    if not usable or not gold:
        return ""

    if list_answer:
        selected: list[str] = []
        used_words = 0
        for raw_item in ALCE_LIST_SEPARATOR.split(gold[0]):
            item = raw_item.strip().strip(".[]")
            if not item:
                continue
            item_words = len(ALCE_WORD.findall(item))
            if selected and used_words + item_words > max(1, int(list_word_budget)):
                continue
            supported_item = ""
            for source in usable:
                candidate = _alce_cited(item, str(source["id"]))
                if _alce_supported(candidate, usable):
                    supported_item = candidate + "."
                    break
            if supported_item:
                selected.append(supported_item)
                used_words += item_words
        candidate = " ".join(selected)
        if candidate and _alce_supported(candidate, usable):
            return candidate

    ranked: list[tuple[float, int, int, str]] = []
    # Prefer a contiguous evidence span: it teaches readable grounded prose,
    # not a metric-shaped bag of reference terms.
    for evidence_index, item in enumerate(usable):
        evidence_id = str(item["id"])
        content = str(item.get("content") or "")
        for sentence in ALCE_SENTENCE.findall(content):
            words = [match.group(0) for match in ALCE_WORD.finditer(sentence)]
            if not words:
                continue
            width = min(len(words), max(1, int(extractive_words)))
            for start in range(0, len(words) - width + 1):
                candidate = _alce_cited(
                    " ".join(words[start : start + width]), evidence_id
                )
                if not _alce_supported(candidate, usable):
                    continue
                ranked.append(
                    (
                        max(token_f1(candidate, value) for value in gold),
                        -evidence_index,
                        -start,
                        candidate,
                    )
                )
    if ranked:
        return max(ranked)[3]

    # Last fallback: retain only reference words attested by one source.  This
    # keeps a usable target for very short or unusually formatted evidence.
    for evidence_index, item in enumerate(usable):
        evidence_id = str(item["id"])
        source_words = {
            match.group(0).casefold()
            for match in ALCE_WORD.finditer(
                f"{item.get('title') or ''} {item.get('content') or ''}"
            )
        }
        for reference_index, reference in enumerate(gold):
            kept: list[str] = []
            for match in ALCE_WORD.finditer(reference):
                word = match.group(0)
                if word.casefold() in source_words:
                    kept.append(word)
                if len(kept) >= max(1, int(filtered_words)):
                    break
            if not kept:
                continue
            candidate = _alce_cited(" ".join(kept), evidence_id)
            if _alce_supported(candidate, usable):
                ranked.append(
                    (
                        max(token_f1(candidate, value) for value in gold),
                        -evidence_index,
                        -reference_index,
                        candidate,
                    )
                )

    if ranked:
        return max(ranked)[3]
    return ""


def alce_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = compact_answer_evidence(
        str(case["prompt"]),
        [dict(item) for item in list(case.get("evidence_context") or [])[:5]],
    )
    answers = list(case["gold"].get("answers") or [])
    if not evidence or not answers:
        return []
    answer = concise_supported_alce_answer(
        answers,
        evidence,
        list_answer=str(case.get("metadata", {}).get("subset") or "").casefold()
        == "qampari",
    )
    if not answer:
        return []
    return [
        record(
            case,
            task="cited_evidence_answer",
            suffix="answer",
            prompt=render_compact_answer_prompt(str(case["prompt"]), evidence),
            response=answer + ANSWER_SUFFIX,
        )
    ]


def case_records(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = str(case["dataset"])
    if dataset == "bfcl":
        return bfcl_records(case)
    if dataset in {"webwalkerqa", "frames"}:
        return search_records(case)
    if dataset == "longbench_v2":
        return longbench_records(case)
    if dataset == "alce":
        return alce_records(case)
    raise ValueError(f"unsupported dataset: {dataset}")


def build(split_root: Path, output_dir: Path) -> dict[str, Any]:
    split_root = split_root.expanduser().resolve()
    expected_parent = split_root / "training"
    if not (expected_parent / "train").is_dir() or not (expected_parent / "dev").is_dir():
        raise ValueError("split root must contain training/train and training/dev")
    if (split_root / "locked").resolve() in expected_parent.parents:
        raise ValueError("training input cannot resolve through locked data")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split_root": str(split_root),
        "inputs": {},
        "outputs": {},
        "locked_files_read": 0,
    }
    for split in ("train", "dev"):
        rows: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for dataset in DATASETS:
            source = expected_parent / split / f"{dataset}.jsonl"
            cases = jsonl_load(source)
            generated = [item for case in cases for item in case_records(case)]
            rows.extend(generated)
            counts[dataset] = len(generated)
            manifest["inputs"][f"{split}/{dataset}.jsonl"] = {
                "cases": len(cases),
                "sha256": sha256(source),
            }
        rows.sort(key=lambda item: str(item["id"]))
        destination = output_dir / f"{split}.jsonl"
        total = jsonl_dump(destination, rows)
        manifest["outputs"][destination.name] = {
            "records": total,
            "by_dataset": counts,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_sha256"] = sha256(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.split_root, args.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
