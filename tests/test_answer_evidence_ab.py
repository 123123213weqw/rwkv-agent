from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from bench.run_answer_evidence_ab import compact_text, score_answer, summarize


class AnswerEvidenceABTests(unittest.TestCase):
    def test_hydrated_compaction_keeps_lead_and_selected_passage(self) -> None:
        evidence = {
            "text": ("lead " * 200) + "\n\n" + ("selected " * 200),
            "metadata": {
                "hydration_strategy": "lead_plus_cross",
                "component_doc_ids": ["42#0", "42#7"],
            },
        }
        value = compact_text(evidence, max_chars=300)
        self.assertLessEqual(len(value), 300)
        self.assertIn("lead", value)
        self.assertIn("selected", value)
        self.assertIn("…", value)

    def test_correct_page_answer_requires_valid_relevant_supported_citation(
        self,
    ) -> None:
        case = {
            "required_all": [],
            "required_any": [["编程语言", "programming language"]],
            "forbidden": [],
            "relevant_page_ids": ["42"],
        }
        ledger = [
            {
                "id": "S1",
                "page_id": "42",
                "content": "Python 是一种编程语言。",
            }
        ]
        score = score_answer(
            case,
            "Python 是一种编程语言 [S1]。",
            ledger,
            retrieval_hit=True,
        )
        self.assertTrue(score["content_success"])
        self.assertTrue(score["citation_valid"])
        self.assertTrue(score["relevant_citation"])
        self.assertTrue(score["required_support"])
        self.assertTrue(score["strict_grounded"])

    def test_wrong_page_explicit_insufficiency_is_safe_abstention(self) -> None:
        case = {
            "required_all": [],
            "required_any": [["编程语言"]],
            "forbidden": [],
            "relevant_page_ids": ["42"],
        }
        score = score_answer(
            case,
            "当前证据不足，无法回答。",
            [{"id": "S1", "page_id": "99", "content": "不相关内容"}],
            retrieval_hit=False,
        )
        self.assertTrue(score["safe_abstention"])
        self.assertFalse(score["unsupported_answer"])
        self.assertFalse(score["strict_grounded"])

    def test_wrong_page_memory_answer_is_marked_unsupported(self) -> None:
        case = {
            "required_all": [],
            "required_any": [["编程语言"]],
            "forbidden": [],
            "relevant_page_ids": ["42"],
        }
        score = score_answer(
            case,
            "Python 是一种编程语言 [S1]。",
            [{"id": "S1", "page_id": "99", "content": "不相关内容"}],
            retrieval_hit=False,
        )
        self.assertTrue(score["content_success"])
        self.assertTrue(score["unsupported_answer"])
        self.assertFalse(score["safe_abstention"])

    def test_summary_keeps_correct_and_wrong_page_buckets_separate(self) -> None:
        def arm(**values: object) -> dict[str, object]:
            defaults: dict[str, object] = {
                "retrieval_hit": True,
                "answer_nonempty": True,
                "content_success": False,
                "citation_present": False,
                "citation_valid": False,
                "relevant_citation": False,
                "required_support": False,
                "strict_grounded": False,
                "insufficient_evidence": False,
                "safe_abstention": False,
                "unsupported_answer": False,
                "model_elapsed_ms": 10.0,
            }
            defaults.update(values)
            return defaults

        rows = [
            {
                "retrieval_hit": True,
                "page_order_identical": True,
                "evidence_text_changed": True,
                "strategies": {
                    "legacy": arm(),
                    "hydrated": arm(
                        content_success=True,
                        citation_present=True,
                        citation_valid=True,
                        relevant_citation=True,
                        required_support=True,
                        strict_grounded=True,
                    ),
                },
            },
            {
                "retrieval_hit": False,
                "page_order_identical": True,
                "evidence_text_changed": False,
                "strategies": {
                    "legacy": arm(
                        retrieval_hit=False,
                        unsupported_answer=True,
                    ),
                    "hydrated": arm(
                        retrieval_hit=False,
                        insufficient_evidence=True,
                        safe_abstention=True,
                    ),
                },
            },
        ]
        value = summarize(rows)
        self.assertEqual(value["correct_page_cases"], 1)
        self.assertEqual(value["wrong_page_cases"], 1)
        self.assertEqual(
            value["paired"]["correct_page_strict_grounded"]["net"],
            1,
        )
        self.assertEqual(
            value["paired"]["wrong_page_safe_abstention"]["net"],
            1,
        )
        self.assertEqual(
            value["paired"]["wrong_page_unsupported_answer"]["net"],
            1,
        )

    def test_frozen_case_set_is_schema_valid_and_hash_locked(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "bench"
            / "answer_ab_v1"
            / "cases.jsonl"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "5c6156d06491a105f1b1b5f0f4b4efba8abda4d80a5a8559b1210fe2041fcb06",
        )
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            {row["language"] for row in rows},
            {"en", "zh"},
        )
        self.assertEqual(
            len({row["id"] for row in rows}),
            len(rows),
        )

    def test_public_frozen_summary_is_hash_locked_and_safe(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "bench"
            / "baselines"
            / "long_knowledge"
            / "answer-evidence-ab-v1"
        )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "answer-evidence-ab-manifest.v1",
        )
        for item in manifest["files"]:
            path = root / item["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                item["sha256"],
            )
            value = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/wzu/", value)
            self.assertNotIn("127.0.0.1", value)
        comparison = json.loads(
            (root / "comparison.json").read_text(encoding="utf-8")
        )
        self.assertFalse(
            comparison["decision"]["answer_model_admission_gate_passed"]
        )
        self.assertEqual(
            comparison["decision"]["hydration_quality_conclusion"],
            "inconclusive",
        )
        self.assertFalse(comparison["decision"]["production_enabled"])


if __name__ == "__main__":
    unittest.main()
