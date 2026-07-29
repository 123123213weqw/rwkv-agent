#!/usr/bin/env python3
"""Evaluate every RWKV-Agent-FitGen-v1 release gate without a grand score.

The script consumes only already-scored summaries and frozen manifests.  It
does not open case JSONL files, so running the audit cannot expose locked Gold
to an optimization process.  A missing metric is a failed gate, never an
implicit pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "rwkv-agent-fitgen-release-gates.v1"
DATASETS = ("bfcl", "webwalkerqa", "frames", "longbench_v2", "alce")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def metric(summary: Mapping[str, Any], name: str, statistic: str | None = None) -> float | None:
    value = nested(summary, f"unified_metrics.{name}")
    if not isinstance(value, Mapping):
        return None
    if statistic is not None:
        return number(value.get(statistic))
    kind = str(value.get("kind") or "")
    key = "rate" if kind == "rate" else "mean"
    return number(value.get(key))


@dataclass(frozen=True)
class Gate:
    gate_id: str
    track: str
    actual: Any
    operator: str
    target: Any
    passed: bool
    source: str


class GateBook:
    def __init__(self) -> None:
        self.items: list[Gate] = []

    def add(
        self,
        gate_id: str,
        track: str,
        actual: Any,
        operator: str,
        target: Any,
        source: str,
    ) -> None:
        measured = number(actual)
        wanted = number(target)
        passed = False
        if operator == ">=" and measured is not None and wanted is not None:
            passed = measured >= wanted
        elif operator == "<=" and measured is not None and wanted is not None:
            passed = measured <= wanted
        elif operator == "==":
            passed = actual == target
        else:
            if operator not in {">=", "<=", "=="}:
                raise ValueError(f"unsupported operator: {operator}")
        self.items.append(Gate(gate_id, track, actual, operator, target, passed, source))


class RunBundle:
    def __init__(self, path: Path, datasets: Iterable[str]) -> None:
        self.path = path.expanduser().resolve()
        self.manifest_path = self.path / "run-manifest.json"
        self.manifest = load_json(self.manifest_path)
        self.summaries = {
            dataset: load_json(self.path / f"{dataset}.score-summary.json")
            for dataset in datasets
        }

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id") or "")

    def summary(self, dataset: str) -> dict[str, Any]:
        return self.summaries[dataset]


def group_mean(summary: Mapping[str, Any], group_path: str, group: str) -> float | None:
    groups = nested(summary, group_path)
    if not isinstance(groups, Mapping):
        return None
    item = next(
        (
            value
            for name, value in groups.items()
            if str(name).casefold() == str(group).casefold()
        ),
        None,
    )
    if not isinstance(item, Mapping):
        return None
    return number(item.get("mean", item.get("rate")))


def minimum_group(summary: Mapping[str, Any], path: str) -> tuple[float | None, int]:
    groups = nested(summary, path)
    if not isinstance(groups, Mapping) or not groups:
        return None, 0
    values = [
        measured
        for item in groups.values()
        if isinstance(item, Mapping)
        for measured in [number(item.get("mean", item.get("rate")))]
        if measured is not None
    ]
    return (min(values), len(values)) if values else (None, 0)


def group_spread(summary: Mapping[str, Any], path: str) -> tuple[float | None, int]:
    groups = nested(summary, path)
    if not isinstance(groups, Mapping) or not groups:
        return None, 0
    values = [
        measured
        for item in groups.values()
        if isinstance(item, Mapping)
        for measured in [number(item.get("mean", item.get("rate")))]
        if measured is not None
    ]
    return (max(values) - min(values), len(values)) if values else (None, 0)


def weighted_rate(summaries: Sequence[Mapping[str, Any]], path: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for summary in summaries:
        value = number(nested(summary, path))
        cases = int(number(summary.get("cases")) or 0)
        if value is None or cases <= 0:
            return None
        numerator += value * cases
        denominator += cases
    return numerator / denominator if denominator else None


def verify_run_binding(
    book: GateBook,
    bundle: RunBundle,
    *,
    label: str,
    checkpoint_record_sha: str,
) -> None:
    binding = bundle.manifest.get("checkpoint_manifest")
    bound_sha = binding.get("sha256") if isinstance(binding, Mapping) else None
    book.add(f"binding.{label}.full", "binding", bundle.manifest.get("mode"), "==", "full", str(bundle.manifest_path))
    book.add(
        f"binding.{label}.checkpoint",
        "binding",
        bound_sha,
        "==",
        checkpoint_record_sha,
        str(bundle.manifest_path),
    )
    completed = sorted(set(map(str, bundle.manifest.get("completed_datasets") or [])))
    book.add(
        f"binding.{label}.completed",
        "binding",
        completed,
        "==",
        sorted(bundle.summaries),
        str(bundle.manifest_path),
    )


def validate_split_inputs(
    book: GateBook,
    bundle: RunBundle,
    split_manifest: Mapping[str, Any],
    split: str,
    label: str,
) -> None:
    artifacts = dict(split_manifest.get("artifacts") or {})
    prefix = f"locked/{split}/"
    for dataset, summary in bundle.summaries.items():
        expected = artifacts.get(prefix + dataset + ".jsonl")
        expected = expected if isinstance(expected, Mapping) else {}
        book.add(
            f"binding.{label}.{dataset}.cases",
            "binding",
            summary.get("cases"),
            "==",
            expected.get("cases"),
            str(bundle.path / f"{dataset}.score-summary.json"),
        )
        book.add(
            f"binding.{label}.{dataset}.source_sha",
            "binding",
            nested(summary, "inputs.normalized_source_sha256"),
            "==",
            expected.get("sha256"),
            str(bundle.path / f"{dataset}.score-summary.json"),
        )


def add_metric_gate(
    book: GateBook,
    gate_id: str,
    track: str,
    summary: Mapping[str, Any],
    metric_name: str,
    operator: str,
    target: float,
    source: str,
    statistic: str | None = None,
) -> None:
    book.add(gate_id, track, metric(summary, metric_name, statistic), operator, target, source)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_record_path = args.checkpoint_manifest.expanduser().resolve()
    checkpoint_record = load_json(checkpoint_record_path)
    checkpoint_record_sha = sha256(checkpoint_record_path)
    split_manifest_path = args.split_manifest.expanduser().resolve()
    split_manifest = load_json(split_manifest_path)

    fit = RunBundle(args.fit_id_run, DATASETS)
    ood = RunBundle(args.ood_run, DATASETS)
    fresh = RunBundle(args.fresh_run, ("webwalkerqa",))
    eli5 = RunBundle(args.eli5_run, ("alce",))
    state_fit = RunBundle(args.longbench_state_fit_id_run, ("longbench_v2",))
    state_ood = RunBundle(args.longbench_state_ood_run, ("longbench_v2",))
    bundles = {
        "fit_id": fit,
        "structural_ood": ood,
        "fresh": fresh,
        "eli5": eli5,
        "state_fit_id": state_fit,
        "state_ood": state_ood,
    }
    book = GateBook()
    for label, bundle in bundles.items():
        verify_run_binding(book, bundle, label=label, checkpoint_record_sha=checkpoint_record_sha)
    validate_split_inputs(book, fit, split_manifest, "fit_id", "fit_id")
    validate_split_inputs(book, ood, split_manifest, "structural_ood", "structural_ood")

    fresh_manifest_path = args.fresh_manifest.expanduser().resolve()
    fresh_claim_path = args.fresh_blind_claim.expanduser().resolve()
    fresh_scoring_manifest_path = args.fresh_scoring_manifest.expanduser().resolve()
    load_json(fresh_manifest_path)
    fresh_claim = load_json(fresh_claim_path)
    fresh_scoring_manifest = load_json(fresh_scoring_manifest_path)
    book.add("fresh.case_count", "fresh", fresh.summary("webwalkerqa").get("cases"), "==", 200, str(fresh.path))
    book.add("fresh.claim.run_id", "fresh", fresh_claim.get("run_id"), "==", fresh.run_id, str(fresh_claim_path))
    book.add("fresh.claim.collection", "fresh", fresh_claim.get("manifest_sha256"), "==", sha256(fresh_manifest_path), str(fresh_claim_path))
    book.add(
        "fresh.claim.checkpoint",
        "fresh",
        fresh_claim.get("checkpoint_manifest_sha256"),
        "==",
        checkpoint_record.get("checkpoint_manifest_sha256"),
        str(fresh_claim_path),
    )
    book.add("fresh.scoring.event", "fresh", fresh_scoring_manifest.get("event"), "==", "fresh_blind_scoring_cases_materialized", str(fresh_scoring_manifest_path))
    book.add("fresh.scoring.run_id", "fresh", fresh_scoring_manifest.get("run_id"), "==", fresh.run_id, str(fresh_scoring_manifest_path))
    book.add("fresh.scoring.collection", "fresh", fresh_scoring_manifest.get("collection_manifest_sha256"), "==", sha256(fresh_manifest_path), str(fresh_scoring_manifest_path))
    book.add("fresh.scoring.claim", "fresh", fresh_scoring_manifest.get("blind_claim_sha256"), "==", sha256(fresh_claim_path), str(fresh_scoring_manifest_path))
    scoring_artifact = dict(fresh_scoring_manifest.get("artifacts") or {}).get(
        "webwalkerqa.jsonl"
    )
    scoring_artifact = scoring_artifact if isinstance(scoring_artifact, Mapping) else {}
    book.add("fresh.scoring.cases", "fresh", scoring_artifact.get("cases"), "==", 200, str(fresh_scoring_manifest_path))
    book.add("fresh.scoring.source_sha", "fresh", nested(fresh.summary("webwalkerqa"), "inputs.normalized_source_sha256"), "==", scoring_artifact.get("sha256"), str(fresh_scoring_manifest_path))
    book.add("eli5.case_count", "alce", eli5.summary("alce").get("cases"), "==", 300, str(eli5.path))

    # BFCL.
    bfcl_id = fit.summary("bfcl")
    bfcl_ood = ood.summary("bfcl")
    book.add("bfcl.id.ast", "bfcl", nested(bfcl_id, "bfcl_ast_exact_match.rate"), ">=", 0.90, str(fit.path))
    book.add("bfcl.ood.ast", "bfcl", nested(bfcl_ood, "bfcl_ast_exact_match.rate"), ">=", 0.80, str(ood.path))
    book.add("bfcl.ood.parallel", "bfcl", nested(bfcl_ood, "by_category.parallel.rate"), ">=", 0.75, str(ood.path))
    book.add("bfcl.ood.parallel_multiple", "bfcl", nested(bfcl_ood, "by_category.parallel_multiple.rate"), ">=", 0.65, str(ood.path))
    for label, summary, source in (("id", bfcl_id, fit.path), ("ood", bfcl_ood, ood.path)):
        add_metric_gate(book, f"bfcl.{label}.protocol", "bfcl", summary, "tool_protocol_valid", ">=", 0.99, str(source))
        add_metric_gate(book, f"bfcl.{label}.strict", "bfcl", summary, "tool_call_exact_match", ">=", 0.75, str(source))
    book.add(
        "bfcl.id_ood_drop",
        "bfcl",
        (number(nested(bfcl_id, "bfcl_ast_exact_match.rate")) or 0.0) - (number(nested(bfcl_ood, "bfcl_ast_exact_match.rate")) or 0.0),
        "<=",
        0.10,
        "fit-id vs structural-ood",
    )

    # WebWalkerQA and Fresh-Web.
    web_id = fit.summary("webwalkerqa")
    web_ood = ood.summary("webwalkerqa")
    web_fresh = fresh.summary("webwalkerqa")
    for label, summary, f1, domain, exact, source in (
        ("id", web_id, 0.30, 0.35, 0.10, fit.path),
        ("ood", web_ood, 0.25, 0.25, 0.06, ood.path),
        ("fresh", web_fresh, 0.25, 0.25, 0.06, fresh.path),
    ):
        add_metric_gate(book, f"web.{label}.f1", "web", summary, "answer_token_f1", ">=", f1, str(source))
        add_metric_gate(book, f"web.{label}.domain_recall", "web", summary, "source_domain_recall", ">=", domain, str(source))
        add_metric_gate(book, f"web.{label}.exact_url_recall", "web", summary, "source_recall", ">=", exact, str(source))
        add_metric_gate(book, f"web.{label}.citation_presence", "web", summary, "citation_presence", ">=", 0.80, str(source))
        add_metric_gate(book, f"web.{label}.cited_domain_recall", "web", summary, "citation_source_domain_recall", ">=", 0.15, str(source))
    book.add("web.id_ood_drop", "web", (metric(web_id, "answer_token_f1") or 0.0) - (metric(web_ood, "answer_token_f1") or 0.0), "<=", 0.06, "fit-id vs structural-ood")
    for name in ("answer_token_f1", "source_domain_recall", "source_recall"):
        id_value = metric(web_id, name)
        fresh_value = metric(web_fresh, name)
        ratio = fresh_value / id_value if id_value and fresh_value is not None else None
        book.add(f"web.fresh_ratio.{name}", "fresh", ratio, ">=", 0.80, "fresh vs fit-id")

    # FRAMES.
    frames_id = fit.summary("frames")
    frames_ood = ood.summary("frames")
    for label, summary, f1, domain, exact, source in (
        ("id", frames_id, 0.20, 0.15, 0.05, fit.path),
        ("ood", frames_ood, 0.15, 0.10, 0.03, ood.path),
    ):
        add_metric_gate(book, f"frames.{label}.f1", "frames", summary, "answer_token_f1", ">=", f1, str(source))
        add_metric_gate(book, f"frames.{label}.domain_recall", "frames", summary, "source_domain_recall", ">=", domain, str(source))
        add_metric_gate(book, f"frames.{label}.exact_url_recall", "frames", summary, "source_recall", ">=", exact, str(source))
        add_metric_gate(book, f"frames.{label}.citation_presence", "frames", summary, "citation_presence", ">=", 0.60, str(source))
    reasoning_min, reasoning_groups = minimum_group(frames_ood, "answer_f1_by_reasoning_type")
    book.add("frames.ood.reasoning_group_count", "frames", reasoning_groups, ">=", 5, str(ood.path))
    book.add("frames.ood.reasoning_min_f1", "frames", reasoning_min, ">=", 0.10, str(ood.path))
    book.add("frames.id_ood_drop", "frames", (metric(frames_id, "answer_token_f1") or 0.0) - (metric(frames_ood, "answer_token_f1") or 0.0), "<=", 0.06, "fit-id vs structural-ood")

    # LongBench v2, including the true State-reader comparison.
    long_id = fit.summary("longbench_v2")
    long_ood = ood.summary("longbench_v2")
    long_state_id = state_fit.summary("longbench_v2")
    long_state_ood = state_ood.summary("longbench_v2")
    book.add("long.id.accuracy", "longbench", nested(long_id, "choice_accuracy.rate"), ">=", 0.45, str(fit.path))
    book.add("long.ood.accuracy", "longbench", nested(long_ood, "choice_accuracy.rate"), ">=", 0.40, str(ood.path))
    book.add("long.ood.hard", "longbench", group_mean(long_ood, "accuracy_by_difficulty", "hard"), ">=", 0.37, str(ood.path))
    domain_min, domain_count = minimum_group(long_ood, "accuracy_by_domain")
    book.add("long.ood.domain_count", "longbench", domain_count, ">=", 6, str(ood.path))
    book.add("long.ood.domain_min", "longbench", domain_min, ">=", 0.30, str(ood.path))
    context_spread, context_count = group_spread(long_ood, "accuracy_by_context_bucket")
    book.add("long.ood.context_bucket_count", "longbench", context_count, ">=", 3, str(ood.path))
    book.add("long.ood.context_spread", "longbench", context_spread, "<=", 0.07, str(ood.path))
    book.add("long.id_ood_drop", "longbench", (number(nested(long_id, "choice_accuracy.rate")) or 0.0) - (number(nested(long_ood, "choice_accuracy.rate")) or 0.0), "<=", 0.07, "fit-id vs structural-ood")
    for label, lexical_summary, state_summary, source in (
        ("fit_id", long_id, long_state_id, state_fit.path),
        ("ood", long_ood, long_state_ood, state_ood.path),
    ):
        book.add(f"long.state.{label}.same_cases", "longbench", nested(state_summary, "inputs.cases_sha256"), "==", nested(lexical_summary, "inputs.cases_sha256"), str(source))
    lexical_rate = weighted_rate((long_id, long_ood), "choice_accuracy.rate")
    state_rate = weighted_rate((long_state_id, long_state_ood), "choice_accuracy.rate")
    state_gain = state_rate - lexical_rate if state_rate is not None and lexical_rate is not None else None
    book.add("long.state.gain", "longbench", state_gain, ">=", 0.05, "state vs lexical Top-6 on locked cases")

    # ALCE.
    alce_id = fit.summary("alce")
    alce_ood = ood.summary("alce")
    alce_eli5 = eli5.summary("alce")
    add_metric_gate(book, "alce.id.f1", "alce", alce_id, "answer_token_f1", ">=", 0.32, str(fit.path))
    add_metric_gate(book, "alce.ood.f1", "alce", alce_ood, "answer_token_f1", ">=", 0.27, str(ood.path))
    add_metric_gate(book, "alce.eli5.f1", "alce", alce_eli5, "answer_token_f1", ">=", 0.22, str(eli5.path))
    book.add("alce.id.asqa", "alce", group_mean(alce_id, "answer_f1_by_subset", "asqa"), ">=", 0.35, str(fit.path))
    book.add("alce.id.qampari", "alce", group_mean(alce_id, "answer_f1_by_subset", "qampari"), ">=", 0.22, str(fit.path))
    for label, summary, source in (("id", alce_id, fit.path), ("ood", alce_ood, ood.path), ("eli5", alce_eli5, eli5.path)):
        add_metric_gate(book, f"alce.{label}.citation_presence", "alce", summary, "citation_presence", ">=", 0.95, str(source))
        add_metric_gate(book, f"alce.{label}.citation_validity", "alce", summary, "citation_validity_precision", ">=", 0.98, str(source))
        add_metric_gate(book, f"alce.{label}.gold_doc_recall", "alce", summary, "citation_source_recall", ">=", 0.35, str(source))
        add_metric_gate(book, f"alce.{label}.supported_claim", "alce", summary, "supported_claim_rate", ">=", 0.85, str(source))
        add_metric_gate(book, f"alce.{label}.unsupported_claim", "alce", summary, "unsupported_claim_rate", "<=", 0.10, str(source))
    book.add("alce.id_ood_drop", "alce", (metric(alce_id, "answer_token_f1") or 0.0) - (metric(alce_ood, "answer_token_f1") or 0.0), "<=", 0.07, "fit-id vs structural-ood")

    # Reliability is aggregated over every one-shot/full run.  Per-track P95
    # uses the worst locked split so a fast split cannot mask a slow one.
    all_summaries = [summary for bundle in bundles.values() for summary in bundle.summaries.values()]
    total_cases = sum(int(number(summary.get("cases")) or 0) for summary in all_summaries)
    total_ok = sum(int(number(summary.get("status_ok")) or 0) for summary in all_summaries)
    book.add("reliability.status_ok", "reliability", total_ok / total_cases if total_cases else None, ">=", 0.995, "all locked runs")
    for name in ("http_409_count", "state_leak_count", "protocol_leak_count", "budget_overrun_count"):
        total = sum(int(number(nested(summary, f"reliability.{name}")) or 0) for summary in all_summaries)
        book.add(f"reliability.{name}", "reliability", total, "==", 0, "all locked runs")
    latency_limits = {"bfcl": 2600.0, "webwalkerqa": 20000.0, "frames": 21000.0, "longbench_v2": 4000.0, "alce": 4000.0}
    for dataset, limit in latency_limits.items():
        candidates = [summary for bundle in bundles.values() for name, summary in bundle.summaries.items() if name == dataset]
        p95_values = [metric(summary, "latency_ms", "p95") for summary in candidates]
        worst = max(value for value in p95_values if value is not None) if any(value is not None for value in p95_values) else None
        book.add(f"reliability.{dataset}.p95_ms", "reliability", worst, "<=", limit, "worst locked split")
    for dataset in ("webwalkerqa", "frames"):
        candidates = [summary for bundle in bundles.values() for name, summary in bundle.summaries.items() if name == dataset]
        means = [metric(summary, "request_count", "mean") for summary in candidates]
        worst = max(value for value in means if value is not None) if any(value is not None for value in means) else None
        book.add(f"reliability.{dataset}.mean_requests", "reliability", worst, "<=", 8.0, "worst locked split")

    failed = [asdict(item) for item in book.items if not item.passed]
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "RWKV-Agent-FitGen-v1",
        "created_at": utc_now(),
        "release_passed": not failed,
        "grand_score": None,
        "gate_counts": {
            "total": len(book.items),
            "passed": len(book.items) - len(failed),
            "failed": len(failed),
        },
        "checkpoint": {
            "record": str(checkpoint_record_path),
            "record_sha256": checkpoint_record_sha,
            "artifact_manifest_sha256": checkpoint_record.get("checkpoint_manifest_sha256"),
        },
        "split_manifest": {
            "path": str(split_manifest_path),
            "sha256": sha256(split_manifest_path),
        },
        "runs": {label: {"path": str(bundle.path), "run_id": bundle.run_id} for label, bundle in bundles.items()},
        "gates": [asdict(item) for item in book.items],
        "failed_gates": failed,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fit-id-run", type=Path, required=True)
    parser.add_argument("--ood-run", type=Path, required=True)
    parser.add_argument("--fresh-run", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--fresh-blind-claim", type=Path, required=True)
    parser.add_argument("--fresh-scoring-manifest", type=Path, required=True)
    parser.add_argument("--eli5-run", type=Path, required=True)
    parser.add_argument("--longbench-state-fit-id-run", type=Path, required=True)
    parser.add_argument("--longbench-state-ood-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"release_passed": report["release_passed"], "gate_counts": report["gate_counts"], "output": str(output)}, sort_keys=True))
    return 0 if report["release_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
