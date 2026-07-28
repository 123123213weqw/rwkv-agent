from __future__ import annotations

import unittest

from rwkv_search.realtime.precision_discovery import (
    build_pivot_queries,
    discover_one_hop_links,
    merge_candidate_groups,
    organization_domain,
    select_pivot_domains,
    source_channel_query,
    select_source_channels,
)
from rwkv_search.realtime.types import DiscoveredURL, RealtimeDocument


class PrecisionDiscoveryTest(unittest.TestCase):
    def test_source_channels_only_add_an_explicit_source_shape(self) -> None:
        self.assertEqual(
            select_source_channels("Python latest stable release"), ("general",)
        )
        self.assertEqual(
            select_source_channels("Find the official GitHub repository"),
            ("general", "repos"),
        )
        self.assertEqual(
            select_source_channels("查找 RWKV 原始论文"),
            ("general", "science"),
        )
        self.assertLessEqual(
            len(select_source_channels("GitHub repository and original paper")), 2
        )
        self.assertEqual(
            source_channel_query("llama.cpp latest release GitHub", "repos"),
            "llama.cpp latest release GitHub",
        )
        self.assertEqual(
            source_channel_query(
                "RWKV architecture original paper official", "science"
            ),
            "RWKV architecture original paper official",
        )

    def test_organization_domain_is_conservative_for_government_and_subdomains(
        self,
    ) -> None:
        self.assertEqual(
            organization_domain("https://investor.nvidia.com/a"), "nvidia.com"
        )
        self.assertEqual(
            organization_domain("https://earthquake.usgs.gov/a"), "usgs.gov"
        )
        self.assertEqual(
            organization_domain("https://www.stats.gov.cn/a"), "stats.gov.cn"
        )
        self.assertEqual(organization_domain("www.gov.cn"), "gov.cn")
        self.assertEqual(organization_domain("https://sousuo.www.gov.cn/a"), "gov.cn")

    def test_pivot_domains_require_generic_first_party_signals(self) -> None:
        candidates = [
            DiscoveredURL(
                url="https://www.apple.com/",
                title="Apple Official Site",
                snippet="Official Apple information",
                rank=1,
                candidate_score=0.8,
                score_components={"official_alignment": 1.0},
            ),
            DiscoveredURL(
                url="https://medium.com/a-review",
                title="A review",
                snippet="Independent commentary",
                rank=2,
                candidate_score=0.7,
            ),
            DiscoveredURL(
                url="https://blog.example/python-official-guide",
                title="Python 官网最新版本指南",
                snippet="文章整理官方发布信息",
                rank=3,
                candidate_score=0.9,
            ),
            DiscoveredURL(
                url="https://www.whitehouse.gov/employment",
                title="The Employment Situation in May",
                snippet="Official government report",
                rank=4,
                candidate_score=0.95,
            ),
        ]
        self.assertEqual(
            select_pivot_domains(
                "Find Apple's latest official newsroom announcement",
                ["Apple latest hardware announcement"],
                candidates,
                max_domains=2,
            ),
            ["apple.com"],
        )
        self.assertEqual(
            select_pivot_domains("Apple phone colors", [], candidates, max_domains=2),
            ["apple.com"],
        )

    def test_government_scope_needs_entity_or_title_alignment(self) -> None:
        candidates = [
            DiscoveredURL(
                url="https://www.gsxt.gov.cn/",
                title="国家企业信用信息公示系统",
                snippet="政府网站",
                rank=1,
            ),
            DiscoveredURL(
                url="https://www.stats.gov.cn/",
                title="国家统计局",
                snippet="统计数据",
                rank=2,
            ),
        ]
        self.assertEqual(
            select_pivot_domains(
                "国家统计局最新公布的GDP数据",
                ["国家统计局 最新 GDP 数据"],
                candidates,
                max_domains=2,
            ),
            ["stats.gov.cn"],
        )

    def test_multiword_entity_can_align_with_compact_domain(self) -> None:
        candidates = [
            DiscoveredURL(
                url="https://www.federalreserve.gov/",
                title="Federal Reserve Board",
                rank=1,
            )
        ]
        self.assertEqual(
            select_pivot_domains(
                "Find the Federal Reserve official statement",
                ["Federal Reserve latest FOMC statement"],
                candidates,
                max_domains=2,
            ),
            ["federalreserve.gov"],
        )

    def test_institutional_domain_suppresses_same_label_commercial_lookalike(
        self,
    ) -> None:
        candidates = [
            DiscoveredURL(url="https://earthquake.usgs.gov/", title="USGS", rank=1),
            DiscoveredURL(
                url="https://usgs.com/", title="USGS commercial site", rank=2
            ),
        ]
        self.assertEqual(
            select_pivot_domains(
                "Find the latest official USGS earthquake advisory",
                ["USGS earthquake latest"],
                candidates,
                max_domains=2,
            ),
            ["usgs.gov"],
        )

    def test_build_pivot_queries_removes_existing_site_constraint(self) -> None:
        self.assertEqual(
            build_pivot_queries(
                "site:old.example Python latest release", ["python.org"]
            ),
            ["site:python.org Python latest release"],
        )

    def test_merge_preserves_stage_and_engine_provenance(self) -> None:
        initial = [
            DiscoveredURL(
                url="https://python.org/downloads/",
                title="Downloads",
                engine="mwmbl",
                engines=["mwmbl"],
                matched_queries=["Python release"],
                query_positions={"Python release": 4},
                rrf_score=0.02,
            )
        ]
        pivot = [
            DiscoveredURL(
                url="https://python.org/downloads",
                title="Download Python",
                engine="bing",
                engines=["bing"],
                matched_queries=["site:python.org Python release"],
                query_positions={"site:python.org Python release": 1},
                rrf_score=0.03,
            )
        ]
        merged = merge_candidate_groups(initial, pivot, max_candidates=10)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].discovery_stages, ["initial", "domain_pivot"])
        self.assertEqual(merged[0].engines, ["mwmbl", "bing"])
        self.assertEqual(len(merged[0].matched_queries), 2)

    def test_one_hop_links_are_same_organization_scored_and_bounded(self) -> None:
        document = RealtimeDocument(
            url="https://www.python.org/",
            title="Python",
            text="Official Python information " * 10,
            published_at=None,
            fetched_at=0.0,
            source_type="official_docs",
            authority=1.0,
            extraction_quality=1.0,
            links=[
                "https://www.python.org/downloads/",
                "https://docs.python.org/3/whatsnew/",
                "https://www.python.org/accounts/login/",
                "https://www.python.org/static/logo.png",
                "https://example.com/releases/",
            ],
        )
        links = discover_one_hop_links(
            "What is the latest Python release? Use the official site.",
            ["Python latest stable release"],
            [document],
            allowed_domains=["python.org"],
            max_links=2,
        )
        self.assertLessEqual(len(links), 2)
        self.assertTrue(links)
        self.assertTrue(
            all(organization_domain(item.url) == "python.org" for item in links)
        )
        self.assertTrue(all(item.discovery_stage == "one_hop_link" for item in links))
        self.assertTrue(all(item.parent_url == document.url for item in links))
        self.assertNotIn("login", " ".join(item.url for item in links))
        self.assertNotIn("example.com", " ".join(item.url for item in links))


if __name__ == "__main__":
    unittest.main()
