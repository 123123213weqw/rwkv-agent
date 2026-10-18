#!/usr/bin/env python3
"""Score sealed Fresh-Web predictions only after private Gold is materialized."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.run_agent_benchmark_metrics import build_report
from benchmarks.run_fitgen_benchmark import json_dump, jsonl_dump, summarize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def score_sealed_run(
    *,
    collection_manifest_path: Path,
    claim_path: Path,
    prediction_seal_path: Path,
    scoring_manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    collection_manifest_path = collection_manifest_path.expanduser().resolve()
    claim_path = claim_path.expanduser().resolve()
    prediction_seal_path = prediction_seal_path.expanduser().resolve()
    scoring_manifest_path = scoring_manifest_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    collection = load_json(collection_manifest_path)
    claim = load_json(claim_path)
    seal = load_json(prediction_seal_path)
    scoring = load_json(scoring_manifest_path)
    if collection.get("event") != "fresh_collection_frozen":
        raise ValueError("invalid Fresh-Web collection manifest")
    if claim.get("event") != "fresh_blind_run_claimed":
        raise ValueError("invalid Fresh-Web blind-run claim")
    if seal.get("event") != "fresh_blind_predictions_sealed":
        raise ValueError("invalid Fresh-Web prediction seal")
    if scoring.get("event") != "fresh_blind_scoring_cases_materialized":
        raise ValueError("invalid Fresh-Web scoring manifest")
    collection_sha = sha256(collection_manifest_path)
    claim_sha = sha256(claim_path)
    seal_sha = sha256(prediction_seal_path)
    if claim.get("manifest_sha256") != collection_sha:
        raise ValueError("blind-run claim does not bind the collection")
    if seal.get("collection_manifest_sha256") != collection_sha:
        raise ValueError("prediction seal does not bind the collection")
    if seal.get("blind_claim_sha256") != claim_sha:
        raise ValueError("prediction seal does not bind the claim")
    if scoring.get("collection_manifest_sha256") != collection_sha:
        raise ValueError("scoring manifest does not bind the collection")
    if scoring.get("blind_claim_sha256") != claim_sha:
        raise ValueError("scoring manifest does not bind the claim")
    if scoring.get("prediction_seal_sha256") != seal_sha:
        raise ValueError("scoring manifest does not bind the prediction seal")
    run_id = str(claim.get("run_id") or "")
    if not run_id or seal.get("run_id") != run_id or scoring.get("run_id") != run_id:
        raise ValueError("Fresh-Web run IDs differ")

    results_artifact = dict(seal.get("artifacts") or {}).get("results")
    if not isinstance(results_artifact, Mapping):
        raise ValueError("prediction seal has no results artifact")
    results_path = Path(str(results_artifact.get("path") or "")).resolve()
    results_sha = str(results_artifact.get("sha256") or "")
    if (
        not results_path.is_file()
        or sha256(results_path) != results_sha
        or scoring.get("sealed_results_sha256") != results_sha
    ):
        raise ValueError("sealed predictions changed before scoring")
    expected_results = run_dir / "webwalkerqa.results.jsonl"
    if results_path != expected_results:
        raise ValueError("prediction seal does not point to the selected run directory")

    scoring_artifact = dict(scoring.get("artifacts") or {}).get("webwalkerqa.jsonl")
    if not isinstance(scoring_artifact, Mapping):
        raise ValueError("scoring manifest has no webwalkerqa artifact")
    cases_path = scoring_manifest_path.parent / "webwalkerqa.jsonl"
    cases_sha = str(scoring_artifact.get("sha256") or "")
    if not cases_path.is_file() or sha256(cases_path) != cases_sha:
        raise ValueError("materialized scoring cases changed")
    cases = load_jsonl(cases_path, kind="case")
    results = load_jsonl(results_path, kind="result")
    if len(cases) != 200 or int(scoring_artifact.get("cases") or 0) != 200:
        raise ValueError("Fresh-Web scoring requires exactly 200 cases")
    if {str(row["id"]) for row in cases} != {
        str(row["case_id"]) for row in results
    }:
        raise ValueError("Fresh-Web cases and sealed results differ")

    report, evaluations = build_report(
        cases_path=cases_path,
        results_path=results_path,
    )
    report_path = run_dir / "webwalkerqa.report.json"
    evaluations_path = run_dir / "webwalkerqa.evaluations.jsonl"
    summary_path = run_dir / "webwalkerqa.score-summary.json"
    json_dump(report_path, report)
    jsonl_dump(evaluations_path, evaluations)
    summary = summarize("webwalkerqa", results, report, evaluations)
    summary["inputs"] = {
        "cases_sha256": cases_sha,
        "results_sha256": results_sha,
        "normalized_source_sha256": cases_sha,
    }
    summary["blind_scoring"] = {
        "run_id": run_id,
        "collection_manifest_sha256": collection_sha,
        "blind_claim_sha256": claim_sha,
        "prediction_seal_sha256": seal_sha,
        "scoring_manifest_sha256": sha256(scoring_manifest_path),
        "gold_revealed_after_predictions": True,
    }
    json_dump(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--prediction-seal", type=Path, required=True)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = score_sealed_run(
        collection_manifest_path=args.collection_manifest,
        claim_path=args.claim,
        prediction_seal_path=args.prediction_seal,
        scoring_manifest_path=args.scoring_manifest,
        run_dir=args.run_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
