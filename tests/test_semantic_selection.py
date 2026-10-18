from __future__ import annotations

import unittest

from rwkv_search.semantic_selection import (
    rank_capabilities,
    select_diverse_items,
    unique_query_views,
)


class OpaqueScorer:
    model_name = "fake-semantic"

    def score(self, query, documents):
        query = query.casefold()
        if "ownership" in query:
            target = "established"
        elif "offerings" in query:
            target = "products"
        elif "newest" in query:
            target = "announcement"
        else:
            target = "examplecloud"
        return [float(target in document.casefold()) for document in documents]


class SemanticSelectionTests(unittest.TestCase):
    def test_query_views_are_deduplicated_without_a_fixed_schema(self) -> None:
        self.assertEqual(
            unique_query_views("ExampleCloud", [" examplecloud ", "ownership"]),
            ("ExampleCloud", "ownership"),
        )

    def test_mmr_covers_model_generated_views_and_deduplicates_urls(self) -> None:
        selection = select_diverse_items(
            "Tell me about ExampleCloud",
            [
                "ExampleCloud ownership",
                "ExampleCloud offerings",
                "ExampleCloud newest development",
            ],
            [
                {
                    "title": "Company",
                    "content": "ExampleCloud was established by Mira Chen.",
                    "uri": "https://example.com/company",
                },
                {
                    "title": "Products",
                    "content": "ExampleCloud products include Atlas and Beacon.",
                    "uri": "https://example.com/products",
                },
                {
                    "title": "News",
                    "content": "The newest announcement launched Beacon 2.",
                    "uri": "https://news.example.net/beacon-2",
                },
                {
                    "title": "Duplicate company page",
                    "content": "ExampleCloud was established by Mira Chen.",
                    "uri": "https://example.com/company?utm_source=test",
                },
                {
                    "title": "Noise",
                    "content": "Unrelated cooking instructions.",
                    "uri": "https://noise.invalid/a",
                },
            ],
            limit=3,
            scorer=OpaqueScorer(),
        )
        uris = {item["uri"] for item in selection.items}
        self.assertEqual(len(uris), 3)
        self.assertIn("https://example.com/company", uris)
        self.assertIn("https://example.com/products", uris)
        self.assertIn("https://news.example.net/beacon-2", uris)
        self.assertIn("cross_encoder", selection.strategy)

    def test_bm25_fallback_generalizes_to_policy_facets(self) -> None:
        selection = select_diverse_items(
            "新能源补贴政策怎么执行？",
            ["新能源补贴发布机关", "新能源补贴适用范围", "新能源补贴生效日期"],
            [
                {
                    "title": "发布机关",
                    "content": "政策由示例部门联合发布。",
                    "uri": "https://gov.example/a",
                },
                {
                    "title": "适用范围",
                    "content": "补贴适用于符合目录要求的新能源车辆。",
                    "uri": "https://gov.example/b",
                },
                {
                    "title": "生效日期",
                    "content": "该办法自2027年1月1日起生效。",
                    "uri": "https://law.example/c",
                },
                {
                    "title": "无关页面",
                    "content": "旅游攻略和餐厅推荐。",
                    "uri": "https://blog.example/d",
                },
            ],
            limit=3,
        )
        self.assertEqual(
            {item["uri"] for item in selection.items},
            {
                "https://gov.example/a",
                "https://gov.example/b",
                "https://law.example/c",
            },
        )

    def test_capabilities_are_ranked_from_descriptions(self) -> None:
        selected = rank_capabilities(
            "Find the DOI for a research paper",
            {
                "code": "source code repositories commits releases",
                "papers": "research papers journal publications DOI citations",
                "encyclopedia": "people concepts definitions history",
            },
            limit=1,
        )
        self.assertEqual(selected, ("papers",))


if __name__ == "__main__":
    unittest.main()
