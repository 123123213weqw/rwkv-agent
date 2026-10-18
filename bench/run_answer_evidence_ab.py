from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Mapping, Sequence
import unicodedata
from urllib.request import Request, urlopen

from rwkv_search.evidence import Evidence
from rwkv_search.router import RouteDecision
from rwkv_search.rwkv_answerer import (
    build_grounded_repair_prompt,
    build_rwkv_grounded_prompt,
    grounded_text_envelope,
)

_CITATION_GROUP = re.compile(r"\[((?:S\d+\s*,?\s*)+)\]", re.I)
_INSUFFICIENT = re.compile(
    r"(证据不足|资料不足|无法确定|无法回答|没有足够|未找到|未检索到|"
    r"insufficient evidence|not enough evidence|cannot determine|"
    r"unable to answer|does not support)",
    re.I,
)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("id") or "") for row in rows]
    if not rows or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("answer A/B cases must have unique non-empty IDs")
    for row in rows:
        if row.get("language") not in {"zh", "en"}:
            raise ValueError(f"unsupported language in {row.get('id')}")
        if not row.get("source_case_id") or not row.get("user_query"):
            raise ValueError(f"incomplete answer case {row.get('id')}")
        if not row.get("relevant_page_ids"):
            raise ValueError(f"positive case lacks relevant pages: {row.get('id')}")
        if not isinstance(row.get("required_any", []), list):
            raise ValueError(f"invalid required_any in {row.get('id')}")
    return rows


def load_shadow_rows(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            if not case_id or case_id in output:
                raise ValueError(f"duplicate or missing shadow case ID: {case_id}")
            if row.get("legacy_page_ids") != row.get("hydrated_page_ids"):
                raise ValueError(f"page order changed in shadow row {case_id}")
            output[case_id] = row
    return output


def compact_text(evidence: Mapping[str, Any], *, max_chars: int = 900) -> str:
    """Keep both lead and selected passage when compacting hydrated Evidence."""

    text = " ".join(str(evidence.get("text") or "").split())
    metadata = evidence.get("metadata") or {}
    components = list(metadata.get("component_doc_ids") or ())
    if len(components) < 2 or "\n\n" not in str(evidence.get("text") or ""):
        return text[: max(128, int(max_chars))]
    raw_parts = [
        " ".join(part.split())
        for part in str(evidence.get("text") or "").split("\n\n", 1)
    ]
    separator = " … "
    budget = max(128, int(max_chars)) - len(separator)
    left = budget // 2
    right = budget - left
    return (raw_parts[0][:left] + separator + raw_parts[1][:right]).strip()


def build_compact_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    max_evidence: int = 5,
    max_chars: int = 900,
) -> tuple[list[Evidence], list[dict[str, Any]]]:
    items: list[Evidence] = []
    ledger: list[dict[str, Any]] = []
    for position, source in enumerate(evidence[: max(1, max_evidence)], start=1):
        evidence_id = f"S{position}"
        page_id = str((source.get("metadata") or {}).get("page_id") or "")
        content = compact_text(source, max_chars=max_chars)
        item = Evidence(
            id=evidence_id,
            title=str(source.get("title") or ""),
            url=str(source.get("url") or ""),
            source_type=str(source.get("source_type") or "finewiki"),
            published_at=source.get("published_at"),
            fetched_at=float(source.get("fetched_at") or 0.0),
            authority=float(
                source.get("authority_score", source.get("authority", 0.85))
                or 0.85
            ),
            text=content,
            score=float(
                source.get("retrieval_score", source.get("score", 0.0)) or 0.0
            ),
            updated_at=source.get("updated_at"),
            freshness_score=float(source.get("freshness_score") or 0.0),
            matched_channels=tuple(source.get("matched_channels") or ()),
            source_id=str(source.get("source_id") or "") or None,
            metadata=dict(source.get("metadata") or {}),
        )
        items.append(item)
        ledger.append(
            {
                "id": evidence_id,
                "title": item.title,
                "content": content,
                "page_id": page_id,
                "source_id": item.source_id or "",
            }
        )
    return items, ledger


def complete(
    endpoint: str,
    prompt: str,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "stop": ["\n\nUser:", "</s>", "</tool_call>", "</tool_calls>"],
            "max_tokens": max(1, int(max_tokens)),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    started = time.perf_counter()
    request = Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        data = json.load(response)
    result = dict(data.get("g1i") or {})
    return {
        "raw": str(result.get("text") or "").strip(),
        "stop_reason": str(result.get("stop_reason") or ""),
        "output_tokens": len(result.get("token_ids") or ()),
        "model_elapsed_ms": float(result.get("elapsed_ms") or 0.0),
        "request_elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "model": str(data.get("model") or ""),
        "endpoint": endpoint,
    }


def citation_ids(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            citation.upper()
            for group in _CITATION_GROUP.findall(text)
            for citation in re.findall(r"S\d+", group, re.I)
        )
    )


def generate_grounded_answer(
    endpoint: str,
    query: str,
    evidence: Sequence[Evidence],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    route = RouteDecision(
        "static_knowledge",
        ["local_search"],
        "stable",
        "single",
        False,
        [query],
        [],
        "offline evidence A/B",
    )
    prompt = build_rwkv_grounded_prompt(
        query,
        route,
        evidence,
        as_of="2026-07-25T00:00:00+00:00",
        timezone="Asia/Shanghai",
    )
    first = complete(endpoint, prompt, max_tokens=max_tokens)
    parsed = grounded_text_envelope(
        first["raw"],
        evidence,
        as_of="2026-07-25T00:00:00+00:00",
    )
    repair: dict[str, Any] | None = None
    if parsed is None:
        repair = complete(
            endpoint,
            build_grounded_repair_prompt(query, first["raw"], evidence),
            max_tokens=max_tokens,
        )
        parsed = grounded_text_envelope(
            repair["raw"],
            evidence,
            as_of="2026-07-25T00:00:00+00:00",
        )
    return {
        **first,
        "raw_first": first["raw"],
        "raw_repair": repair["raw"] if repair else "",
        "raw": (
            first["raw"] + ("\n<REPAIR>\n" + repair["raw"] if repair else "")
        ),
        "visible_answer": str((parsed or {}).get("answer") or ""),
        "answer_accepted": parsed is not None,
        "repaired": repair is not None,
        "insufficient_evidence_envelope": bool(
            (parsed or {}).get("insufficient_evidence")
        ),
        "model_elapsed_ms": first["model_elapsed_ms"]
        + (repair["model_elapsed_ms"] if repair else 0.0),
        "request_elapsed_ms": first["request_elapsed_ms"]
        + (repair["request_elapsed_ms"] if repair else 0.0),
        "output_tokens": first["output_tokens"]
        + (repair["output_tokens"] if repair else 0),
    }


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def _contains_any(text: str, values: Sequence[str]) -> bool:
    normalized = _normalized(text)
    return any(_normalized(value) in normalized for value in values if value)


def score_answer(
    case: Mapping[str, Any],
    answer: str,
    ledger: Sequence[Mapping[str, Any]],
    *,
    retrieval_hit: bool,
) -> dict[str, Any]:
    answer = str(answer or "").strip()
    required_all = [str(value) for value in case.get("required_all", ())]
    required_any = [
        [str(value) for value in group]
        for group in case.get("required_any", ())
    ]
    forbidden = [str(value) for value in case.get("forbidden", ())]
    all_ok = all(_contains_any(answer, [value]) for value in required_all)
    any_ok = all(_contains_any(answer, group) for group in required_any)
    forbidden_ok = not _contains_any(answer, forbidden) if forbidden else True
    content_success = bool(answer and all_ok and any_ok and forbidden_ok)

    citations = citation_ids(answer)
    by_id = {str(item.get("id") or ""): item for item in ledger}
    citation_valid = bool(citations) and all(value in by_id for value in citations)
    relevant_pages = {str(value) for value in case.get("relevant_page_ids", ())}
    cited = [by_id[value] for value in citations if value in by_id]
    relevant_citation = any(
        str(item.get("page_id") or "") in relevant_pages for item in cited
    )
    cited_text = "\n".join(str(item.get("content") or "") for item in cited)
    required_support = all(
        _contains_any(cited_text, [value]) for value in required_all
    ) and all(_contains_any(cited_text, group) for group in required_any)
    insufficient = bool(_INSUFFICIENT.search(answer))
    strict_grounded = bool(
        retrieval_hit
        and content_success
        and citation_valid
        and relevant_citation
        and required_support
        and not insufficient
    )
    return {
        "answer_nonempty": bool(answer),
        "content_success": content_success,
        "citations": citations,
        "citation_present": bool(citations),
        "citation_valid": citation_valid,
        "relevant_citation": relevant_citation,
        "required_support": required_support,
        "insufficient_evidence": insufficient,
        "strict_grounded": strict_grounded,
        "safe_abstention": bool(not retrieval_hit and insufficient),
        "unsupported_answer": bool(
            not retrieval_hit and content_success and not insufficient
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * fraction) - 1),
    )
    return ordered[position]


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return (
        statistics.fmean(float(row.get(key, False)) for row in rows)
        if rows
        else 0.0
    )


def summarize_strategy(rows: Sequence[Mapping[str, Any]], strategy: str) -> dict[str, Any]:
    values = [row["strategies"][strategy] for row in rows]
    hit = [value for value in values if value["retrieval_hit"]]
    miss = [value for value in values if not value["retrieval_hit"]]
    latencies = [float(value["model_elapsed_ms"]) for value in values]
    output = {
        "cases": len(values),
        "answer_accepted_rate": _rate(values, "answer_accepted"),
        "repair_attempt_rate": _rate(values, "repaired"),
        "answer_nonempty_rate": _rate(values, "answer_nonempty"),
        "content_success_rate": _rate(values, "content_success"),
        "citation_present_rate": _rate(values, "citation_present"),
        "citation_valid_rate": _rate(values, "citation_valid"),
        "relevant_citation_rate": _rate(values, "relevant_citation"),
        "required_support_rate": _rate(values, "required_support"),
        "strict_grounded_rate": _rate(values, "strict_grounded"),
        "model_latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "output_tokens": {
            "mean": statistics.fmean(
                float(value.get("output_tokens") or 0.0) for value in values
            )
            if values
            else 0.0,
            "p95": _percentile(
                [float(value.get("output_tokens") or 0.0) for value in values],
                0.95,
            ),
        },
        "correct_page_bucket": {
            "cases": len(hit),
            "content_success_rate": _rate(hit, "content_success"),
            "citation_valid_rate": _rate(hit, "citation_valid"),
            "relevant_citation_rate": _rate(hit, "relevant_citation"),
            "required_support_rate": _rate(hit, "required_support"),
            "strict_grounded_rate": _rate(hit, "strict_grounded"),
            "insufficient_evidence_rate": _rate(hit, "insufficient_evidence"),
        },
        "wrong_page_bucket": {
            "cases": len(miss),
            "safe_abstention_rate": _rate(miss, "safe_abstention"),
            "unsupported_answer_rate": _rate(miss, "unsupported_answer"),
            "content_success_rate": _rate(miss, "content_success"),
            "citation_present_rate": _rate(miss, "citation_present"),
        },
    }
    return output


def _paired_change(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    bucket: str,
    higher_is_better: bool = True,
) -> dict[str, int]:
    selected = [
        row
        for row in rows
        if (
            bucket == "all"
            or bool(row["retrieval_hit"]) == (bucket == "correct_page")
        )
    ]
    improved = regressed = unchanged = 0
    for row in selected:
        legacy = bool(row["strategies"]["legacy"].get(key))
        hydrated = bool(row["strategies"]["hydrated"].get(key))
        if hydrated == higher_is_better and legacy != higher_is_better:
            improved += 1
        elif legacy == higher_is_better and hydrated != higher_is_better:
            regressed += 1
        else:
            unchanged += 1
    return {
        "cases": len(selected),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "net": improved - regressed,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty answer A/B results")
    return {
        "schema_version": "answer-evidence-ab-summary.v1",
        "status": "offline_ab_not_production",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(rows),
        "correct_page_cases": sum(bool(row["retrieval_hit"]) for row in rows),
        "wrong_page_cases": sum(not bool(row["retrieval_hit"]) for row in rows),
        "page_order_identity_rate": _rate(rows, "page_order_identical"),
        "evidence_text_changed_case_rate": _rate(rows, "evidence_text_changed"),
        "strategies": {
            strategy: summarize_strategy(rows, strategy)
            for strategy in ("legacy", "hydrated")
        },
        "paired": {
            "correct_page_strict_grounded": _paired_change(
                rows,
                "strict_grounded",
                bucket="correct_page",
            ),
            "correct_page_content": _paired_change(
                rows,
                "content_success",
                bucket="correct_page",
            ),
            "correct_page_relevant_citation": _paired_change(
                rows,
                "relevant_citation",
                bucket="correct_page",
            ),
            "wrong_page_safe_abstention": _paired_change(
                rows,
                "safe_abstention",
                bucket="wrong_page",
            ),
            "wrong_page_unsupported_answer": _paired_change(
                rows,
                "unsupported_answer",
                bucket="wrong_page",
                higher_is_better=False,
            ),
        },
        "visible_output_changed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline paired answer A/B over frozen legacy and hydrated Evidence."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--shadow-trace", action="append", required=True)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--max-evidence", type=int, default=5)
    parser.add_argument("--evidence-chars", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    shadow = load_shadow_rows(args.shadow_trace)
    endpoints = [
        value.strip().rstrip("/")
        for value in args.endpoints.split(",")
        if value.strip()
    ]
    if len(endpoints) != 2:
        raise SystemExit("paired benchmark requires exactly two sidecar endpoints")
    rows: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            source_case = str(case["source_case_id"])
            if source_case not in shadow:
                raise KeyError(f"missing shadow row for {source_case}")
            source = shadow[source_case]
            legacy_evidence, legacy_ledger = build_compact_evidence(
                source.get("legacy_evidence") or (),
                max_evidence=args.max_evidence,
                max_chars=args.evidence_chars,
            )
            hydrated_evidence, hydrated_ledger = build_compact_evidence(
                source.get("hydrated_evidence") or (),
                max_evidence=args.max_evidence,
                max_chars=args.evidence_chars,
            )
            legacy_pages = [
                str(item.get("page_id") or "") for item in legacy_ledger
            ]
            hydrated_pages = [
                str(item.get("page_id") or "") for item in hydrated_ledger
            ]
            if legacy_pages != hydrated_pages:
                raise RuntimeError(f"page order changed for {case['id']}")
            relevant = {str(value) for value in case["relevant_page_ids"]}
            retrieval_hit = bool(relevant.intersection(legacy_pages))
            # Swap GPU assignment deterministically to avoid making one strategy
            # synonymous with one sidecar/GPU.
            swap = int(
                hashlib.sha256(str(case["id"]).encode()).hexdigest()[:2],
                16,
            ) % 2
            assignment = (
                {"legacy": endpoints[1], "hydrated": endpoints[0]}
                if swap
                else {"legacy": endpoints[0], "hydrated": endpoints[1]}
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    strategy: executor.submit(
                        generate_grounded_answer,
                        assignment[strategy],
                        case["user_query"],
                        (
                            legacy_evidence
                            if strategy == "legacy"
                            else hydrated_evidence
                        ),
                        max_tokens=args.max_tokens,
                    )
                    for strategy in ("legacy", "hydrated")
                }
                completions = {
                    strategy: future.result()
                    for strategy, future in futures.items()
                }
            strategies: dict[str, Any] = {}
            for strategy, ledger in (
                ("legacy", legacy_ledger),
                ("hydrated", hydrated_ledger),
            ):
                completion = completions[strategy]
                score = score_answer(
                    case,
                    completion["visible_answer"],
                    ledger,
                    retrieval_hit=retrieval_hit,
                )
                strategies[strategy] = {
                    **score,
                    **completion,
                    "retrieval_hit": retrieval_hit,
                    "evidence_count": len(ledger),
                    "evidence_page_ids": [
                        str(item.get("page_id") or "") for item in ledger
                    ],
                }
            row = {
                "schema_version": "answer-evidence-ab-row.v1",
                "case_id": case["id"],
                "source_case_id": source_case,
                "language": case["language"],
                "user_query": case["user_query"],
                "relevant_page_ids": sorted(relevant),
                "retrieval_hit": retrieval_hit,
                "page_order_identical": legacy_pages == hydrated_pages,
                "evidence_text_changed": (
                    [item["content"] for item in legacy_ledger]
                    != [item["content"] for item in hydrated_ledger]
                ),
                "assignment_swapped": bool(swap),
                "strategies": strategies,
            }
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            rows.append(row)
            print(
                f"[{position}/{len(cases)}] {case['id']} "
                f"bucket={'hit' if retrieval_hit else 'miss'} "
                f"strict={int(strategies['legacy']['strict_grounded'])}->"
                f"{int(strategies['hydrated']['strict_grounded'])}",
                flush=True,
            )
    summary = summarize(rows)
    summary.update(
        {
            "prompt_mode": "production_grounded_with_one_repair",
            "model": next(
                (
                    row["strategies"]["legacy"]["model"]
                    for row in rows
                    if row["strategies"]["legacy"]["model"]
                ),
                "",
            ),
            "max_evidence": max(1, args.max_evidence),
            "evidence_chars": max(128, args.evidence_chars),
            "max_tokens": max(1, args.max_tokens),
            "dataset_sha256": hashlib.sha256(
                Path(args.cases).read_bytes()
            ).hexdigest(),
        }
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
