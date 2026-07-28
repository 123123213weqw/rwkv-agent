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
)


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
            scoring = root / "scoring"
            scoring_manifest = materialize_blind_scoring_cases(
                output / "manifest.json",
                output / "blind-run-claim.json",
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
            with self.assertRaises(FileExistsError):
                materialize_blind_scoring_cases(
                    output / "manifest.json",
                    output / "blind-run-claim.json",
                    scoring,
                )
            with self.assertRaises(FileExistsError):
                claim_blind_run(output / "manifest.json", "run-2")


if __name__ == "__main__":
    unittest.main()
