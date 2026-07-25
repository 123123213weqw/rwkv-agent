from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

from rwkv_search.analysis import AnalyzerCore, DocumentAnalyzer, QueryAnalyzer
from rwkv_search.analysis.entities import EntityProtector
from rwkv_search.analysis.normalization import normalize_text


FIXTURES = Path(__file__).parent / "fixtures" / "analyzer_cases.jsonl"


def load_cases():
    with FIXTURES.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class QueryAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = QueryAnalyzer()

    def test_fixture_invariants(self) -> None:
        for case in load_cases():
            with self.subTest(case=case["id"]):
                result = self.analyzer.analyze(case["query"])
                all_terms = set(result.exact_terms + result.word_terms + result.bigram_terms)
                self.assertTrue(all_terms)
                for term in case.get("must_preserve", []):
                    self.assertIn(term, all_terms)
                for script in case.get("scripts", []):
                    self.assertIn(script, result.scripts)
                if case.get("freshness"):
                    self.assertEqual(result.constraints.get("freshness"), case["freshness"])
                if case.get("multi_query"):
                    self.assertTrue(result.needs_multi_query)
                    self.assertGreaterEqual(len(result.search_queries), 2)

    def test_structured_entities_are_preserved_and_decomposed(self) -> None:
        result = self.analyzer.analyze("RWKV-7与C++、C#、.NET和Node.js")
        self.assertTrue({"rwkv-7", "c++", "c#", ".net", "node.js"} <= set(result.exact_terms))
        self.assertTrue({"rwkv", "7", "c", "net", "node", "js"} <= set(result.word_terms))

    def test_acronym_does_not_swallow_following_chinese(self) -> None:
        result = self.analyzer.analyze("像RNN一样推理的语言模型")
        self.assertNotIn("rnn一样", result.exact_terms)
        self.assertIn("rnn", result.word_terms)

    def test_query_tokens_are_deduplicated_within_each_channel(self) -> None:
        result = self.analyzer.analyze("RWKV RWKV 搜索搜索")
        self.assertEqual(len(result.word_terms), len(set(result.word_terms)))
        self.assertEqual(len(result.bigram_terms), len(set(result.bigram_terms)))

    def test_offsets_point_back_to_original_surface(self) -> None:
        query = "  Ｃ＋＋ 与 Thé  "
        result = self.analyzer.analyze(query)
        for token in result.tokens:
            self.assertGreaterEqual(token.start, 0)
            self.assertGreater(token.end, token.start)
            self.assertLessEqual(token.end, len(result.resolved_query))
            self.assertEqual(token.surface, result.resolved_query[token.start:token.end])

    def test_trace_exposes_every_stage(self) -> None:
        result = self.analyzer.analyze("RWKV-7最新进展")
        stages = [event.stage for event in result.trace]
        self.assertEqual(stages, [
            "clean_query", "normalize", "protect_entities", "segment", "query_constraints", "query_plan"
        ])

    def test_resolved_query_is_analyzed_without_losing_original(self) -> None:
        result = self.analyzer.analyze("为什么火", resolved_query="奶龙为什么火")
        self.assertEqual(result.original, "为什么火")
        self.assertEqual(result.resolved_query, "奶龙为什么火")
        self.assertIn("奶龙", set(result.word_terms + result.bigram_terms))

    def test_surface_cleaning_collapses_initialism_and_definition_shell(self) -> None:
        result = self.analyzer.analyze("什么是r w k v")
        self.assertEqual(result.resolved_query, "RWKV")
        self.assertEqual(result.search_queries, ("RWKV",))
        self.assertIn("rwkv", result.word_terms)

        suffix = self.analyzer.analyze("python是什么东西")
        self.assertEqual(suffix.resolved_query, "python")
        self.assertEqual(suffix.word_terms, ("python",))

        country = self.analyzer.analyze("中国是什么国家")
        self.assertEqual(country.resolved_query, "中国")
        self.assertEqual(country.constraints["answer_type"], "国家")

        overview = self.analyzer.analyze("中华人民共和国基本情况")
        self.assertEqual(overview.resolved_query, "中华人民共和国")

    def test_surface_cleaning_strips_search_shell_but_keeps_meaning(self) -> None:
        result = self.analyzer.analyze("帮我搜索 RWKV 最新进展，请给出来源")
        self.assertEqual(result.resolved_query, "RWKV 最新进展")
        self.assertEqual(result.constraints.get("freshness"), "latest")

        topic = self.analyzer.analyze("搜索系统是什么")
        self.assertEqual(topic.resolved_query, "搜索系统")

    def test_fallback_segmenter_still_has_cjk_recall(self) -> None:
        analyzer = QueryAnalyzer(AnalyzerCore(enable_jieba=False))
        result = analyzer.analyze("本地关键词检索")
        self.assertTrue({"本地", "关键", "检索"} <= set(result.bigram_terms))
        self.assertEqual(result.tokens, analyzer.analyze("本地关键词检索").tokens)

    def test_dynamic_protected_dictionary(self) -> None:
        analyzer = QueryAnalyzer(AnalyzerCore(protected_terms=["奶龙大电影"]))
        self.assertIn("奶龙大电影", analyzer.analyze("奶龙大电影什么时候上映").exact_terms)


class DocumentAnalyzerTests(unittest.TestCase):
    def test_document_retains_frequency(self) -> None:
        result = DocumentAnalyzer().analyze(
            title="RWKV RWKV",
            body="搜索系统搜索系统，RWKV RWKV",
            headings=["搜索系统"],
            url="https://example.com/rwkv-7",
        )
        self.assertEqual(result.title.word_terms.count("rwkv"), 2)
        self.assertEqual(result.body.word_terms.count("rwkv"), 2)
        self.assertEqual(result.body.bigram_terms.count("搜索"), 2)
        payload = result.to_index_payload()
        self.assertEqual(payload["title_words"].split().count("rwkv"), 2)
        self.assertIn("rwkv-7", payload["url_exact"])

    def test_email_and_domain_candidates_keep_valid_entities(self) -> None:
        spans = EntityProtector().find(
            "Contact Foo.Bar+tag@example.co.uk or read docs.python.org today."
        )
        entities = {(span.entity_type, span.normalized) for span in spans}
        self.assertIn(("email", "Foo.Bar+tag@example.co.uk"), entities)
        self.assertIn("docs.python.org", {span.normalized for span in spans})

    def test_fastq_quality_chart_does_not_stall_entity_analysis(self) -> None:
        body = "\n".join(
            [
                "S" * 45,
                "." * 79,
                "X" * 46,
                "." * 54,
                "I" * 43,
                "." * 57,
                "J" * 40,
                "." * 53,
                "N" * 53 + "." * 43,
                "".join(chr(value) for value in range(33, 127)),
                "0" + "." * 26 + "31" + "." * 40,
                "0" + "." * 20 + "30" + "." * 80 + "93",
                "S - Sanger Phred+33, raw reads typically (0, 40)",
                "X - Solexa Solexa+64, raw reads typically (-5, 40)",
                "I - Illumina 1.3+ Phred+64, raw reads typically (0, 40)",
            ]
        )
        started = time.perf_counter()
        result = DocumentAnalyzer().analyze(
            title="FASTQ format",
            body=body,
            headings=("Encoding",),
        )
        elapsed = time.perf_counter() - started
        self.assertTrue(result.body.word_terms)
        self.assertLess(elapsed, 1.0)


class NormalizationPropertyTests(unittest.TestCase):
    def test_normalization_is_idempotent(self) -> None:
        for case in load_cases():
            first = normalize_text(case["query"]).text
            second = normalize_text(first).text
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
