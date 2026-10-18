from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from benchmarks.summarize_web_shadow_metrics import load_rows, summarize


def _row(*, fallback: bool, elapsed_ms: float) -> dict[str, object]:
    return {
        "schema_version": "agent-web-shadow-metrics.v1",
        "status": "fallback_legacy_evidence" if fallback else "ok",
        "legacy_evidence_count": 2,
        "enhanced_evidence_count": 0 if fallback else 3,
        "effective_evidence_count": 2 if fallback else 3,
        "evidence_url_overlap_count": 1,
        "fallback_used": fallback,
        "elapsed_ms": elapsed_ms,
        "legacy": {
            "candidate_count": 10,
            "result_count": 2,
            "fetch_count": 2,
            "fetch_success_count": 1,
            "latency_ms": 5.0,
        },
        "enhanced": {
            "candidate_count": 20,
            "result_count": 3,
            "fetch_count": 4,
            "fetch_success_count": 3,
            "latency_ms": 7.0,
        },
    }


def test_summary_reports_operational_metrics_without_payload_fields() -> None:
    summary = summarize(
        [_row(fallback=False, elapsed_ms=10.0), _row(fallback=True, elapsed_ms=20.0)]
    )

    assert summary["records"] == 2
    assert summary["fallback_rate"] == 0.5
    assert summary["enhanced"]["mean_candidate_count"] == 20
    assert summary["enhanced"]["fetch_success_rate"] == 0.75
    assert summary["shadow_elapsed_ms"] == {
        "mean": 15.0,
        "p95": 20.0,
        "max": 20.0,
    }
    assert summary["privacy"] == {
        "contains_queries": False,
        "contains_urls": False,
        "contains_page_content": False,
    }


def test_loader_rejects_full_trace_or_unknown_schema() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shadow.jsonl"
        path.write_text(
            json.dumps({"schema_version": "agent-web-shadow.v1"}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="unexpected schema"):
            load_rows(path)
