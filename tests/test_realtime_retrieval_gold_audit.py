from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bench.retrieval_schema import load_cases


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "bench/realtime_web_retrieval.jsonl"
AUDITED = ROOT / "bench/realtime_web_retrieval_audited_v2.jsonl"
MANIFEST = ROOT / "bench/realtime_web_retrieval_audited_v2_manifest.json"
ALLOWED_CHANGED_FIELDS = {
    "expected_domains_any",
    "target_url_patterns_any",
    "gold_revision",
    "gold_annotation_status",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_audited_realtime_gold_keeps_base_queries_immutable() -> None:
    base = _rows(BASE)
    audited = _rows(AUDITED)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert len(base) == len(audited) == 50
    assert [row["id"] for row in base] == [row["id"] for row in audited]
    assert manifest["base"]["sha256"] == _sha256(BASE)
    assert manifest["audited_dataset"]["sha256"] == _sha256(AUDITED)

    changed = []
    for before, after in zip(base, audited, strict=True):
        before_common = {
            key: value
            for key, value in before.items()
            if key not in ALLOWED_CHANGED_FIELDS
        }
        after_common = {
            key: value
            for key, value in after.items()
            if key not in ALLOWED_CHANGED_FIELDS
        }
        assert before_common == after_common
        if before != after:
            changed.append(before["id"])
            assert after["gold_revision"] == "audited-v2-2026-07-30"
            assert after["gold_annotation_status"] == "primary_source_reverified"

    assert sorted(changed) == manifest["changed_case_ids"]
    assert len(changed) == manifest["changed_case_count"] == 9


def test_audited_realtime_gold_passes_runtime_schema_validation() -> None:
    assert len(load_cases(AUDITED)) == 50


def test_audit_manifest_records_old_new_labels_and_public_sources() -> None:
    base = {row["id"]: row for row in _rows(BASE)}
    audited = {row["id"]: row for row in _rows(AUDITED)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for change in manifest["changes"]:
        case_id = change["id"]
        assert change["old"] == {
            "expected_domains_any": base[case_id]["expected_domains_any"],
            "target_url_patterns_any": base[case_id]["target_url_patterns_any"],
        }
        assert change["new"] == {
            "expected_domains_any": audited[case_id]["expected_domains_any"],
            "target_url_patterns_any": audited[case_id]["target_url_patterns_any"],
        }
        assert change["reason"]
        assert change["verification_urls"]
        assert all(
            url.startswith(("https://", "http://"))
            for url in change["verification_urls"]
        )
