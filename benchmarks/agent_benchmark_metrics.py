from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from benchmarks.agent_benchmark_schema import validate_case, validate_result
from rwkv_agent.citations import extract_citation_ids, strip_citations


PROTOCOL_TAG = re.compile(
    r"</?(?:tool_call|tool_result|answer|think|analysis)(?:\s[^>]*)?>",
    re.I,
)
ROLE_HEADER = re.compile(
    r"(?:^|\n)\s*(?:System|User|Assistant|Tool)\s*:",
    re.I,
)
EVIDENCE_JSON = re.compile(
    r'\{\s*"(?:status|evidence)"\s*:',
    re.I,
)


HIGHER_IS_BETTER = {
    "answer_exact_match",
    "answer_token_f1",
    "answerability_accuracy",
    "citation_presence",
    "citation_validity_precision",
    "citation_source_precision",
    "citation_source_recall",
    "citation_source_domain_precision",
    "citation_source_domain_recall",
    "citation_exact_page_precision",
    "citation_exact_page_recall",
    "claim_citation_coverage",
    "evidence_id_precision",
    "evidence_id_recall",
    "evidence_nonempty",
    "result_status_ok",
    "source_precision",
    "source_recall",
    "exact_page_precision",
    "exact_page_recall",
    "source_domain_precision",
    "source_domain_recall",
    "state_cleanup_success",
    "state_release_rate",
    "state_reuse_success",
    "supported_claim_rate",
    "tool_arguments_exact_match",
    "tool_call_exact_match",
    "tool_group_exact_match",
    "tool_name_exact_match",
    "tool_needed_accuracy",
    "tool_protocol_valid",
    "tool_sequence_exact_match",
    "within_latency_budget",
    "within_request_budget",
    "within_round_budget",
}
LOWER_IS_BETTER = {
    "citation_invalid_rate",
    "cpu_state_peak_mib",
    "gpu_peak_mib",
    "input_tokens",
    "latency_ms",
    "output_tokens",
    "protocol_leak",
    "request_count",
    "round_count",
    "state_leak_count",
    "tool_false_negative",
    "tool_false_positive",
    "ttft_ms",
    "unsupported_claim_rate",
}
RATE_METRICS = {
    "answer_token_f1",
    "citation_invalid_rate",
    "citation_source_precision",
    "citation_source_recall",
    "citation_source_domain_precision",
    "citation_source_domain_recall",
    "citation_exact_page_precision",
    "citation_exact_page_recall",
    "citation_validity_precision",
    "claim_citation_coverage",
    "evidence_id_precision",
    "evidence_id_recall",
    "source_precision",
    "source_recall",
    "exact_page_precision",
    "exact_page_recall",
    "source_domain_precision",
    "source_domain_recall",
    "state_release_rate",
    "supported_claim_rate",
    "unsupported_claim_rate",
}


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def answer_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    output: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            output.append("".join(buffer))
            buffer.clear()

    for character in normalized:
        if _is_cjk(character):
            flush()
            output.append(character)
        elif character.isalnum():
            buffer.append(character)
        else:
            flush()
    flush()
    return output


def normalized_answer(value: str) -> str:
    return " ".join(answer_tokens(value))


def token_f1(prediction: str, reference: str) -> float:
    predicted = answer_tokens(prediction)
    expected = answer_tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _best_answer_score(answer: str, references: Sequence[str]) -> tuple[bool, float]:
    if not references:
        return False, 0.0
    answer = strip_citations(answer)
    normalized = normalized_answer(answer)
    exact = any(normalized == normalized_answer(reference) for reference in references)
    return exact, max(token_f1(answer, reference) for reference in references)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_call(call: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            "name": str(call.get("name") or ""),
            "arguments": dict(call.get("arguments") or {}),
        }
    )


def _canonical_arguments(call: Mapping[str, Any]) -> str:
    return _canonical_json(dict(call.get("arguments") or {}))


def _canonical_bfcl_gold_value(value: Any, schema: Mapping[str, Any]) -> Any:
    """Convert BFCL evaluator annotations to executable JSON values."""

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
                output[str(key)] = _canonical_bfcl_gold_value(child, child_schema)
            return output
        if value and all(
            isinstance(item, list) and len(item) == 1 for item in value.values()
        ):
            return {str(key): item[0] for key, item in value.items()}
        return {str(key): child for key, child in value.items()}
    if expected in {"array", "list"} and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            return [_canonical_bfcl_gold_value(item, item_schema) for item in value]
    return value


def _canonical_bfcl_gold_calls(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Drop BFCL scoring sentinels and schema-external annotation fields.

    BFCL uses empty strings in Gold as evaluator wildcards for omitted optional
    parameters.  Comparing those annotations directly with executable JSON
    undercounts strict calls, so this mirrors the frozen SFT target conversion.
    """

    schemas = {
        str(tool.get("name") or ""): tool
        for tool in case.get("available_tools") or ()
        if isinstance(tool, Mapping)
    }
    output: list[dict[str, Any]] = []
    for raw_call in case.get("gold", {}).get("tool_calls") or ():
        name = str(raw_call.get("name") or "")
        tool = schemas.get(name)
        arguments = dict(raw_call.get("arguments") or {})
        if isinstance(tool, Mapping):
            parameters = tool.get("parameters")
            if not isinstance(parameters, Mapping):
                parameters = {}
            properties = parameters.get("properties")
            if not isinstance(properties, Mapping):
                properties = {}
            required = {str(item) for item in parameters.get("required") or ()}
            normalized: dict[str, Any] = {}
            for key, value in arguments.items():
                child_schema = properties.get(key)
                if not isinstance(child_schema, Mapping):
                    continue
                expected = str(child_schema.get("type") or "").casefold()
                if value == "" and (key not in required or expected != "string"):
                    continue
                normalized[str(key)] = _canonical_bfcl_gold_value(
                    value, child_schema
                )
            arguments = normalized
        output.append({"name": name, "arguments": arguments})
    return output


def _tool_groups(calls: Sequence[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for index, call in enumerate(calls):
        group = call.get("parallel_group")
        key = str(group) if group is not None else f"sequential:{index}"
        groups[key][_canonical_call(call)] += 1
    return dict(groups)


def canonical_uri(value: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value or "").strip().casefold()
    hostname = (parsed.hostname or "").casefold()
    port = parsed.port
    if port and not (
        (parsed.scheme.casefold() == "http" and port == 80)
        or (parsed.scheme.casefold() == "https" and port == 443)
    ):
        hostname += f":{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.casefold(), hostname, path, parsed.query, "")
    )


def _uri_domain(value: str) -> str:
    return (urlsplit(str(value or "")).hostname or "").casefold().strip(".")


def _domain_matches(actual: str, expected: str) -> bool:
    return bool(
        actual
        and expected
        and (
            actual == expected
            or actual.endswith("." + expected)
            or expected.endswith("." + actual)
        )
    )


def _domain_scores(actual: set[str], expected: set[str]) -> tuple[float, float]:
    if not expected:
        return (1.0 if not actual else 0.0), 1.0
    matched_actual = {
        domain
        for domain in actual
        if any(_domain_matches(domain, wanted) for wanted in expected)
    }
    matched_expected = {
        domain
        for domain in expected
        if any(_domain_matches(found, domain) for found in actual)
    }
    precision = len(matched_actual) / len(actual) if actual else 0.0
    recall = len(matched_expected) / len(expected)
    return precision, recall


def _set_scores(actual: set[str], expected: set[str]) -> tuple[float, float]:
    if not expected:
        return (1.0 if not actual else 0.0), 1.0
    overlap = len(actual & expected)
    recall = overlap / len(expected)
    precision = overlap / len(actual) if actual else 0.0
    return precision, recall


def _numeric(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return value


def _protocol_leak(answer: str) -> tuple[bool, list[str]]:
    kinds: list[str] = []
    if PROTOCOL_TAG.search(answer):
        kinds.append("protocol_tag")
    if ROLE_HEADER.search(answer):
        kinds.append("role_header")
    if EVIDENCE_JSON.search(answer):
        kinds.append("evidence_json")
    try:
        decoded = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        kinds.append("json_payload")
    return bool(kinds), kinds


def evaluate_agent_case(
    case: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    case = validate_case(case)
    result = validate_result(result)
    if result["case_id"] != case["id"]:
        raise ValueError(
            f"result {result['case_id']} does not match case {case['id']}"
        )

    gold = dict(case["gold"])
    answer = str(result.get("answer") or "")
    metrics: dict[str, bool | int | float] = {
        "result_status_ok": result["status"] == "ok",
    }
    diagnostics: dict[str, Any] = {}

    references = [str(value) for value in gold.get("answers", [])]
    if references:
        exact, f1 = _best_answer_score(answer, references)
        metrics["answer_exact_match"] = exact
        metrics["answer_token_f1"] = round(f1, 6)

    if "answerable" in gold:
        abstained = bool(
            result.get("abstained", result["status"] == "insufficient")
        )
        metrics["answerability_accuracy"] = abstained != bool(gold["answerable"])

    expected_calls = list(gold.get("tool_calls") or [])
    if str(case.get("dataset") or "").casefold() == "bfcl":
        expected_calls = _canonical_bfcl_gold_calls(case)
    predicted_calls = list(result.get("tool_calls") or [])
    if "should_call_tools" in gold:
        should_call = bool(gold["should_call_tools"])
        did_call = bool(predicted_calls)
        metrics["tool_needed_accuracy"] = should_call == did_call
        metrics["tool_false_positive"] = did_call and not should_call
        metrics["tool_false_negative"] = should_call and not did_call
        expected_names = [str(call["name"]) for call in expected_calls]
        predicted_names = [str(call["name"]) for call in predicted_calls]
        metrics["tool_name_exact_match"] = Counter(expected_names) == Counter(
            predicted_names
        )
        expected_canonical = [_canonical_call(call) for call in expected_calls]
        predicted_canonical = [_canonical_call(call) for call in predicted_calls]
        expected_arguments = [
            _canonical_arguments(call) for call in expected_calls
        ]
        predicted_arguments = [
            _canonical_arguments(call) for call in predicted_calls
        ]
        metrics["tool_arguments_exact_match"] = Counter(
            expected_arguments
        ) == Counter(predicted_arguments)
        metrics["tool_call_exact_match"] = Counter(expected_canonical) == Counter(
            predicted_canonical
        )
        metrics["tool_sequence_exact_match"] = (
            expected_canonical == predicted_canonical
        )
        if any("parallel_group" in call for call in expected_calls):
            metrics["tool_group_exact_match"] = _tool_groups(
                expected_calls
            ) == _tool_groups(predicted_calls)
        protocol = dict(result.get("protocol") or {})
        if "tool_call_valid" in protocol:
            metrics["tool_protocol_valid"] = bool(protocol["tool_call_valid"])

    evidence = list(result.get("evidence") or [])
    evidence_by_id = {str(item["id"]): item for item in evidence}
    evidence_ids = set(evidence_by_id)
    metrics["evidence_nonempty"] = bool(evidence)
    expected_evidence_ids = {str(value) for value in gold.get("evidence_ids", [])}
    actual_gold_ids = {
        str(item.get("gold_id") or item.get("id")) for item in evidence
    }
    if expected_evidence_ids:
        precision, recall = _set_scores(actual_gold_ids, expected_evidence_ids)
        metrics["evidence_id_precision"] = round(precision, 6)
        metrics["evidence_id_recall"] = round(recall, 6)

    expected_sources = {
        canonical_uri(str(value)) for value in gold.get("source_uris", [])
    }
    actual_sources = {
        canonical_uri(str(item.get("uri") or ""))
        for item in evidence
        if item.get("uri")
    }
    if expected_sources:
        precision, recall = _set_scores(actual_sources, expected_sources)
        metrics["source_precision"] = round(precision, 6)
        metrics["source_recall"] = round(recall, 6)
        # Exact canonical page recall is the primary retrieval-quality signal.
        # Keep the legacy source_* names as compatible aliases.
        metrics["exact_page_precision"] = round(precision, 6)
        metrics["exact_page_recall"] = round(recall, 6)
        expected_domains = {_uri_domain(value) for value in expected_sources}
        actual_domains = {_uri_domain(value) for value in actual_sources}
        precision, recall = _domain_scores(actual_domains, expected_domains)
        metrics["source_domain_precision"] = round(precision, 6)
        metrics["source_domain_recall"] = round(recall, 6)

    citations = set(extract_citation_ids(answer))
    valid_citations = citations & evidence_ids
    invalid_citations = citations - evidence_ids
    if citations:
        metrics["citation_validity_precision"] = round(
            len(valid_citations) / len(citations),
            6,
        )
        metrics["citation_invalid_rate"] = round(
            len(invalid_citations) / len(citations),
            6,
        )
    elif gold.get("requires_citations"):
        metrics["citation_validity_precision"] = 0.0
        metrics["citation_invalid_rate"] = 0.0
    if "requires_citations" in gold:
        metrics["citation_presence"] = bool(citations) == bool(
            gold["requires_citations"]
        )
    if expected_sources:
        cited_sources = {
            canonical_uri(str(evidence_by_id[citation].get("uri") or ""))
            for citation in valid_citations
            if evidence_by_id[citation].get("uri")
        }
        precision, recall = _set_scores(cited_sources, expected_sources)
        metrics["citation_source_precision"] = round(precision, 6)
        metrics["citation_source_recall"] = round(recall, 6)
        metrics["citation_exact_page_precision"] = round(precision, 6)
        metrics["citation_exact_page_recall"] = round(recall, 6)
        cited_domains = {_uri_domain(value) for value in cited_sources}
        expected_domains = {_uri_domain(value) for value in expected_sources}
        precision, recall = _domain_scores(cited_domains, expected_domains)
        metrics["citation_source_domain_precision"] = round(precision, 6)
        metrics["citation_source_domain_recall"] = round(recall, 6)
    diagnostics["citations"] = sorted(citations)
    diagnostics["invalid_citations"] = sorted(invalid_citations)

    claims = list(result.get("claims") or [])
    required_claims = [
        claim for claim in claims if claim.get("requires_citation", True)
    ]
    if required_claims:
        cited_claims = sum(bool(claim.get("citations")) for claim in required_claims)
        metrics["claim_citation_coverage"] = round(
            cited_claims / len(required_claims),
            6,
        )
    judged_claims = [claim for claim in claims if "supported" in claim]
    if judged_claims:
        supported = sum(bool(claim["supported"]) for claim in judged_claims)
        metrics["supported_claim_rate"] = round(
            supported / len(judged_claims),
            6,
        )
        metrics["unsupported_claim_rate"] = round(
            (len(judged_claims) - supported) / len(judged_claims),
            6,
        )

    leaked, leak_kinds = _protocol_leak(answer)
    metrics["protocol_leak"] = leaked
    diagnostics["leak_kinds"] = leak_kinds

    trace = dict(result.get("trace") or {})
    resources = dict(result.get("resources") or {})
    trace_numeric = {
        "request_count": trace.get("requests"),
        "round_count": trace.get("rounds"),
        "state_leak_count": trace.get("states_leaked"),
    }
    resource_numeric = {
        "latency_ms": resources.get("latency_ms"),
        "ttft_ms": resources.get("ttft_ms"),
        "gpu_peak_mib": resources.get("gpu_peak_mib"),
        "cpu_state_peak_mib": resources.get("cpu_state_peak_mib"),
        "input_tokens": resources.get("input_tokens"),
        "output_tokens": resources.get("output_tokens"),
    }
    for name, value in {**trace_numeric, **resource_numeric}.items():
        number = _numeric(value)
        if number is not None:
            metrics[name] = number

    created = _numeric(trace.get("states_created"))
    released = _numeric(trace.get("states_released"))
    if created is not None and released is not None:
        metrics["state_release_rate"] = round(
            min(float(released), float(created)) / max(1.0, float(created)),
            6,
        )
        metrics["state_cleanup_success"] = created == released and not bool(
            trace.get("states_leaked", 0)
        )
    reused = _numeric(trace.get("states_reused"))
    if reused is not None:
        metrics["state_reuse_success"] = reused > 0

    limits = dict(case.get("limits") or {})
    if "max_requests" in limits and "request_count" in metrics:
        metrics["within_request_budget"] = (
            metrics["request_count"] <= limits["max_requests"]
        )
    if "max_rounds" in limits and "round_count" in metrics:
        metrics["within_round_budget"] = (
            metrics["round_count"] <= limits["max_rounds"]
        )
    if "max_latency_ms" in limits and "latency_ms" in metrics:
        metrics["within_latency_budget"] = (
            metrics["latency_ms"] <= limits["max_latency_ms"]
        )

    return {
        "schema_version": "rwkv-agent-benchmark-evaluation.v1",
        "case_id": str(case["id"]),
        "dataset": str(case["dataset"]),
        "split": str(case["split"]),
        "track": str(case["track"]),
        "language": str(case["language"]),
        "context_bucket": str(case.get("metadata", {}).get("context_bucket", "")),
        "metadata": dict(case.get("metadata") or {}),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values_by_metric: dict[str, list[bool | float | int]] = defaultdict(list)
    for row in rows:
        for name, value in dict(row.get("metrics") or {}).items():
            if isinstance(value, (bool, int, float)):
                values_by_metric[name].append(value)
    summaries: dict[str, Any] = {}
    for name, raw_values in sorted(values_by_metric.items()):
        if all(isinstance(value, bool) for value in raw_values):
            passed = sum(bool(value) for value in raw_values)
            summaries[name] = {
                "kind": "rate",
                "applicable": len(raw_values),
                "passed": passed,
                "rate": round(passed / len(raw_values), 6),
            }
            continue
        values = [float(value) for value in raw_values]
        summary = {
            "kind": "macro_rate" if name in RATE_METRICS else "distribution",
            "applicable": len(values),
            "mean": round(sum(values) / len(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
        if name not in RATE_METRICS:
            summary.update(
                {
                    "p50": round(_percentile(values, 0.50), 6),
                    "p95": round(_percentile(values, 0.95), 6),
                    "p99": round(_percentile(values, 0.99), 6),
                    "sum": round(sum(values), 6),
                }
            )
        summaries[name] = summary
    return {"cases": len(rows), "metrics": summaries}


def aggregate_evaluations(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    evaluations = list(rows)

    def grouped(key: str, *, skip_empty: bool = False) -> dict[str, Any]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in evaluations:
            value = str(row.get(key) or "")
            if skip_empty and not value:
                continue
            groups[value or "unknown"].append(row)
        return {
            name: _summarize(group) for name, group in sorted(groups.items())
        }

    return {
        "schema_version": "rwkv-agent-benchmark-summary.v1",
        "overall": _summarize(evaluations),
        "by_track": grouped("track"),
        "by_dataset": grouped("dataset"),
        "by_language": grouped("language"),
        "by_context_bucket": grouped("context_bucket", skip_empty=True),
        "grand_score": None,
        "grand_score_reason": "track metrics are intentionally not collapsed",
    }


def compare_evaluations(
    baseline: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(row["case_id"]): row for row in baseline}
    candidate_by_id = {str(row["case_id"]): row for row in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        missing = sorted(set(baseline_by_id) - set(candidate_by_id))
        extra = sorted(set(candidate_by_id) - set(baseline_by_id))
        raise ValueError(
            f"paired comparison requires identical case IDs; missing={missing}, "
            f"extra={extra}"
        )

    names = sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)
    metrics: dict[str, Any] = {}
    for name in names:
        pairs: list[tuple[float, float]] = []
        for case_id in sorted(baseline_by_id):
            before = baseline_by_id[case_id].get("metrics", {}).get(name)
            after = candidate_by_id[case_id].get("metrics", {}).get(name)
            if isinstance(before, (bool, int, float)) and isinstance(
                after, (bool, int, float)
            ):
                pairs.append((float(before), float(after)))
        if not pairs:
            continue
        direction = "higher" if name in HIGHER_IS_BETTER else "lower"
        wins = losses = ties = 0
        for before, after in pairs:
            delta = after - before
            if abs(delta) <= 1e-12:
                ties += 1
            elif (delta > 0) == (direction == "higher"):
                wins += 1
            else:
                losses += 1
        before_mean = sum(value[0] for value in pairs) / len(pairs)
        after_mean = sum(value[1] for value in pairs) / len(pairs)
        metrics[name] = {
            "direction": direction,
            "applicable": len(pairs),
            "baseline_mean": round(before_mean, 6),
            "candidate_mean": round(after_mean, 6),
            "delta": round(after_mean - before_mean, 6),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
    return {
        "schema_version": "rwkv-agent-benchmark-comparison.v1",
        "paired_cases": len(baseline_by_id),
        "metrics": metrics,
        "grand_score": None,
    }
