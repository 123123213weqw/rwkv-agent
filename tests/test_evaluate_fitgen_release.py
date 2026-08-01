from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.evaluate_fitgen_release import DATASETS, evaluate


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def rate(value: float) -> dict[str, object]:
    return {"kind": "rate", "rate": value, "passed": int(value > 0), "applicable": 1}


def macro(value: float) -> dict[str, object]:
    return {"kind": "macro_rate", "mean": value, "min": value, "max": value, "applicable": 1}


def distribution(*, mean: float, p95: float) -> dict[str, object]:
    return {"kind": "distribution", "mean": mean, "p95": p95, "min": mean, "max": p95, "applicable": 1}


def summary(dataset: str, cases: int, source_sha: str, *, long_rate: float = 1.0) -> dict[str, object]:
    metrics = {
        name: rate(1.0)
        for name in (
            "tool_protocol_valid",
            "tool_call_exact_match",
            "citation_presence",
            "result_status_ok",
        )
    }
    for name in (
        "answer_token_f1",
        "source_domain_recall",
        "source_recall",
        "citation_source_domain_recall",
        "citation_source_recall",
        "citation_validity_precision",
        "supported_claim_rate",
    ):
        metrics[name] = macro(1.0)
    metrics["unsupported_claim_rate"] = macro(0.0)
    metrics["latency_ms"] = distribution(mean=1.0, p95=1.0)
    metrics["request_count"] = distribution(mean=1.0, p95=1.0)
    value: dict[str, object] = {
        "dataset": dataset,
        "cases": cases,
        "status_ok": cases,
        "status_ok_rate": 1.0,
        "inputs": {"normalized_source_sha256": source_sha, "cases_sha256": source_sha},
        "unified_metrics": metrics,
        "reliability": {
            "http_409_count": 0,
            "state_leak_count": 0,
            "protocol_leak_count": 0,
            "budget_overrun_count": 0,
        },
    }
    if dataset == "bfcl":
        value.update(
            {
                "bfcl_ast_exact_match": {"rate": 1.0},
                "by_category": {
                    "parallel": {"rate": 1.0},
                    "parallel_multiple": {"rate": 1.0},
                },
            }
        )
    elif dataset == "frames":
        value["answer_f1_by_reasoning_type"] = {
            name: {"cases": 10, "mean": 1.0}
            for name in ("Multiple constraints", "Numerical reasoning", "Temporal reasoning", "Tabular reasoning", "Post processing")
        }
    elif dataset == "longbench_v2":
        value.update(
            {
                "choice_accuracy": {"rate": long_rate},
                "accuracy_by_difficulty": {"hard": {"cases": 1, "mean": long_rate}},
                "accuracy_by_domain": {f"d{index}": {"cases": 1, "mean": long_rate} for index in range(6)},
                "accuracy_by_context_bucket": {name: {"cases": 1, "mean": long_rate} for name in ("short", "medium", "long")},
            }
        )
    elif dataset == "alce":
        value["answer_f1_by_subset"] = {
            "ASQA": {"cases": 1, "mean": 1.0},
            "QAMPARI": {"cases": 1, "mean": 1.0},
        }
    return value


class FitGenReleaseGateTests(unittest.TestCase):
    def test_complete_synthetic_release_passes_every_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.json"
            dump(checkpoint, {"checkpoint_manifest_sha256": "artifact-sha"})
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            counts = {
                "fit_id": {"bfcl": 100, "webwalkerqa": 80, "frames": 100, "longbench_v2": 70, "alce": 100},
                "structural_ood": {"bfcl": 149, "webwalkerqa": 120, "frames": 140, "longbench_v2": 98, "alce": 145},
            }
            split_artifacts = {}
            for split, values in counts.items():
                for dataset, cases in values.items():
                    split_artifacts[f"locked/{split}/{dataset}.jsonl"] = {
                        "cases": cases,
                        "sha256": f"{split}-{dataset}",
                    }
            split_manifest = root / "split-manifest.json"
            dump(split_manifest, {"artifacts": split_artifacts})

            def make_run(name: str, datasets: tuple[str, ...], run_counts: dict[str, int], *, state: bool = False) -> Path:
                path = root / name
                manifest = {
                    "run_id": name,
                    "mode": "full",
                    "completed_datasets": list(datasets),
                    "checkpoint_manifest": {"sha256": checkpoint_sha},
                }
                dump(path / "run-manifest.json", manifest)
                for dataset in datasets:
                    source_sha = (
                        f"fit_id-{dataset}"
                        if name in {"fit", "state-fit"}
                        else f"structural_ood-{dataset}"
                        if name in {"ood", "state-ood"}
                        else f"{name}-{dataset}"
                    )
                    dump(
                        path / f"{dataset}.score-summary.json",
                        summary(
                            dataset,
                            run_counts[dataset],
                            source_sha,
                            long_rate=0.60 if state else 0.50,
                        ),
                    )
                return path

            fit = make_run("fit", DATASETS, counts["fit_id"])
            ood = make_run("ood", DATASETS, counts["structural_ood"])
            state_fit = make_run("state-fit", ("longbench_v2",), {"longbench_v2": 70}, state=True)
            state_ood = make_run("state-ood", ("longbench_v2",), {"longbench_v2": 98}, state=True)
            fresh = make_run("fresh", ("webwalkerqa",), {"webwalkerqa": 200})
            fresh_run_manifest = json.loads(
                (fresh / "run-manifest.json").read_text(encoding="utf-8")
            )
            fresh_run_manifest.update(
                {
                    "defer_scoring": True,
                    "web_api_providers": ["github", "crossref", "mediawiki"],
                }
            )
            dump(fresh / "run-manifest.json", fresh_run_manifest)
            eli5 = make_run("eli5", ("alce",), {"alce": 300})

            fresh_manifest = root / "fresh-manifest.json"
            dump(fresh_manifest, {"checkpoint_manifest_sha256": "artifact-sha"})
            fresh_claim = root / "blind-run-claim.json"
            dump(
                fresh_claim,
                {
                    "run_id": "fresh",
                    "manifest_sha256": hashlib.sha256(fresh_manifest.read_bytes()).hexdigest(),
                    "checkpoint_manifest_sha256": "artifact-sha",
                },
            )
            fresh_prediction_seal = root / "fresh-prediction-seal.json"
            dump(
                fresh_prediction_seal,
                {
                    "event": "fresh_blind_predictions_sealed",
                    "run_id": "fresh",
                    "collection_manifest_sha256": hashlib.sha256(
                        fresh_manifest.read_bytes()
                    ).hexdigest(),
                    "blind_claim_sha256": hashlib.sha256(
                        fresh_claim.read_bytes()
                    ).hexdigest(),
                    "require_no_tavily": True,
                    "web_api_providers": ["github", "crossref", "mediawiki"],
                },
            )
            fresh_summary_path = fresh / "webwalkerqa.score-summary.json"
            fresh_summary = json.loads(
                fresh_summary_path.read_text(encoding="utf-8")
            )
            fresh_summary["blind_scoring"] = {
                "prediction_seal_sha256": hashlib.sha256(
                    fresh_prediction_seal.read_bytes()
                ).hexdigest(),
                "gold_revealed_after_predictions": True,
            }
            dump(fresh_summary_path, fresh_summary)
            fresh_scoring_manifest = root / "fresh-scoring-manifest.json"
            dump(
                fresh_scoring_manifest,
                {
                    "event": "fresh_blind_scoring_cases_materialized",
                    "run_id": "fresh",
                    "collection_manifest_sha256": hashlib.sha256(
                        fresh_manifest.read_bytes()
                    ).hexdigest(),
                    "blind_claim_sha256": hashlib.sha256(
                        fresh_claim.read_bytes()
                    ).hexdigest(),
                    "prediction_seal_sha256": hashlib.sha256(
                        fresh_prediction_seal.read_bytes()
                    ).hexdigest(),
                    "artifacts": {
                        "webwalkerqa.jsonl": {
                            "cases": 200,
                            "sha256": "fresh-webwalkerqa",
                        }
                    },
                },
            )
            args = argparse.Namespace(
                checkpoint_manifest=checkpoint,
                split_manifest=split_manifest,
                fit_id_run=fit,
                ood_run=ood,
                fresh_run=fresh,
                fresh_manifest=fresh_manifest,
                fresh_blind_claim=fresh_claim,
                fresh_prediction_seal=fresh_prediction_seal,
                fresh_scoring_manifest=fresh_scoring_manifest,
                eli5_run=eli5,
                longbench_state_fit_id_run=state_fit,
                longbench_state_ood_run=state_ood,
            )
            report = evaluate(args)
            self.assertTrue(report["release_passed"], report["failed_gates"])
            json.dumps(report)


if __name__ == "__main__":
    unittest.main()
