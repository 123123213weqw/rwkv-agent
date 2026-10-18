from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.fresh_web_workflow import (
    CATEGORIES,
    claim_blind_run,
    freeze_checkpoint,
    freeze_collection,
    materialize_blind_scoring_cases,
    seal_blind_predictions,
)
from benchmarks.score_sealed_fresh_web import score_sealed_run


class FreshWebWorkflowTests(unittest.TestCase):
    def test_freeze_and_one_shot_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "adapter.bin").write_bytes(b"adapter")
            checkpoint_record = root / "checkpoint-freeze.json"
            freeze_checkpoint(checkpoint, checkpoint_record)
            snapshot_root = root / "snapshots"
            snapshot_root.mkdir()
            draft = root / "draft.jsonl"
            rows = []
            for index in range(200):
                language = "zh" if index < 100 else "en"
                category = CATEGORIES[index % len(CATEGORIES)]
                relative = Path(f"page-{index}.html")
                body = f"answer {index}".encode()
                (snapshot_root / relative).write_bytes(body)
                rows.append(
                    {
                        "id": f"fresh-{index:03d}",
                        "language": language,
                        "category": category,
                        "prompt": f"question {index}",
                        "answers": [f"answer {index}"],
                        "sources": [
                            {
                                "uri": f"https://fresh-{index}.test/page",
                                "authoritative": True,
                                "snapshot_path": str(relative),
                                "snapshot_sha256": hashlib.sha256(body).hexdigest(),
                                "evidence_spans": [f"answer {index}"],
                            }
                        ],
                    }
                )
            draft.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "frozen"
            manifest = freeze_collection(
                draft,
                snapshot_root,
                checkpoint_record,
                output,
                training_domains={"trained.test"},
            )
            self.assertEqual(manifest["summary"]["cases"], 200)
            self.assertEqual(
                manifest["summary"]["outside_training_domain_cases"], 200
            )
            claim = claim_blind_run(output / "manifest.json", "run-1")
            self.assertEqual(claim["run_id"], "run-1")
            run_dir = root / "run"
            run_dir.mkdir()
            run_manifest = run_dir / "run-manifest.json"
            run_manifest.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "datasets": ["webwalkerqa"],
                        "mode": "full",
                        "defer_scoring": True,
                        "web_api_providers": ["github", "crossref", "mediawiki"],
                        "checkpoint_manifest": {
                            "sha256": hashlib.sha256(
                                checkpoint_record.read_bytes()
                            ).hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )
            results = run_dir / "webwalkerqa.results.jsonl"
            results.write_text(
                "".join(
                    json.dumps(
                        {
                            "schema_version": "rwkv-agent-benchmark-result.v1",
                            "case_id": f"fresh-{index:03d}",
                            "status": "ok",
                            "answer": "",
                            "abstained": True,
                            "tool_calls": [],
                            "evidence": [],
                            "claims": [],
                            "trace": {},
                            "resources": {},
                            "protocol": {},
                        }
                    )
                    + "\n"
                    for index in range(200)
                ),
                encoding="utf-8",
            )
            prediction_seal = output / "prediction-seal.json"
            sealed = seal_blind_predictions(
                output / "manifest.json",
                output / "blind-run-claim.json",
                run_manifest,
                output / "cases.public.jsonl",
                results,
                prediction_seal,
            )
            self.assertFalse(sealed["gold_revealed_before_predictions"])
            self.assertNotIn("tavily", sealed["web_api_providers"])
            scoring = root / "scoring"
            scoring_manifest = materialize_blind_scoring_cases(
                output / "manifest.json",
                output / "blind-run-claim.json",
                prediction_seal,
                scoring,
            )
            self.assertEqual(
                scoring_manifest["artifacts"]["webwalkerqa.jsonl"]["cases"],
                200,
            )
            first = json.loads(
                (scoring / "webwalkerqa.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(first["dataset"], "webwalkerqa")
            self.assertEqual(first["gold"]["answers"], ["answer 0"])
            self.assertEqual(
                first["gold"]["source_uris"],
                ["https://fresh-0.test/page"],
            )
            summary = score_sealed_run(
                collection_manifest_path=output / "manifest.json",
                claim_path=output / "blind-run-claim.json",
                prediction_seal_path=prediction_seal,
                scoring_manifest_path=scoring / "manifest.json",
                run_dir=run_dir,
            )
            self.assertEqual(summary["cases"], 200)
            self.assertTrue(
                summary["blind_scoring"]["gold_revealed_after_predictions"]
            )
            with self.assertRaises(FileExistsError):
                materialize_blind_scoring_cases(
                    output / "manifest.json",
                    output / "blind-run-claim.json",
                    prediction_seal,
                    scoring,
                )
            with self.assertRaises(FileExistsError):
                claim_blind_run(output / "manifest.json", "run-2")

    def test_prediction_seal_rejects_tavily(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_record = root / "checkpoint-freeze.json"
            freeze_checkpoint(checkpoint, checkpoint_record)
            snapshot = root / "page.html"
            snapshot.write_text("answer", encoding="utf-8")
            draft = root / "draft.jsonl"
            rows = []
            for index in range(200):
                rows.append(
                    {
                        "id": f"fresh-{index:03d}",
                        "language": "zh" if index < 100 else "en",
                        "category": CATEGORIES[index % len(CATEGORIES)],
                        "prompt": f"question {index}",
                        "answers": ["answer"],
                        "sources": [
                            {
                                "uri": f"https://fresh-{index}.test/page",
                                "authoritative": True,
                                "snapshot_path": "page.html",
                                "snapshot_sha256": hashlib.sha256(
                                    snapshot.read_bytes()
                                ).hexdigest(),
                                "evidence_spans": ["answer"],
                            }
                        ],
                    }
                )
            draft.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            frozen = root / "frozen"
            freeze_collection(
                draft,
                root,
                checkpoint_record,
                frozen,
                training_domains=set(),
            )
            claim_blind_run(frozen / "manifest.json", "run-tavily")
            run_manifest = root / "run.json"
            run_manifest.write_text(
                json.dumps(
                    {
                        "run_id": "run-tavily",
                        "datasets": ["webwalkerqa"],
                        "mode": "full",
                        "defer_scoring": True,
                        "web_api_providers": ["tavily"],
                        "checkpoint_manifest": {
                            "sha256": hashlib.sha256(
                                checkpoint_record.read_bytes()
                            ).hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )
            results = root / "results.jsonl"
            results.write_text(
                "".join(
                    json.dumps({"case_id": f"fresh-{index:03d}"}) + "\n"
                    for index in range(200)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot enable Tavily"):
                seal_blind_predictions(
                    frozen / "manifest.json",
                    frozen / "blind-run-claim.json",
                    run_manifest,
                    frozen / "cases.public.jsonl",
                    results,
                    frozen / "prediction-seal.json",
                )


if __name__ == "__main__":
    unittest.main()
