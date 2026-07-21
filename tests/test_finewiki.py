from __future__ import annotations

import json
import unittest

from rwkv_search.finewiki import (
    FineWikiArticle,
    FineWikiChunker,
    clean_finewiki_markdown,
    extract_wikitext_aliases,
    flatten_infoboxes,
)


class FineWikiCleaningTests(unittest.TestCase):
    def test_removes_duplicate_title_and_localized_reference_tail(self) -> None:
        value = """# Python
Python 是一种编程语言。

## 历史
由吉多·范罗苏姆开发。

## 参考资料
不应进入索引。

### 书籍
也不应进入索引。
"""
        cleaned = clean_finewiki_markdown("Python", value)
        self.assertNotIn("# Python", cleaned)
        self.assertIn("Python 是一种编程语言", cleaned)
        self.assertIn("历史", cleaned)
        self.assertNotIn("不应进入索引", cleaned)
        self.assertNotIn("书籍", cleaned)

    def test_flattens_and_bounds_infobox_json(self) -> None:
        value = json.dumps({"name": "Python", "designer": "Guido van Rossum", "url": "https://x"})
        flattened = flatten_infoboxes(value, max_chars=80)
        self.assertIn("name: Python", flattened)
        self.assertIn("designer: Guido van Rossum", flattened)
        self.assertNotIn("https://", flattened)

    def test_extracts_regional_variant_alias_from_noteta_without_rules(self) -> None:
        wikitext = """{{noteTA
|1=zh-cn:胰腺; zh-tw:胰臟; zh-hk:胰臟;
|2=zh-cn:胰腺腺癌; zh-tw:胰臟腺癌;
}}
'''胰臟癌'''是疾病。
"""
        aliases = extract_wikitext_aliases("胰臟癌", wikitext)
        self.assertIn("胰腺癌", aliases)
        self.assertNotIn("胰腺腺癌", aliases)

    def test_extracts_generic_short_name_from_rendered_naming_sentence(self) -> None:
        aliases = extract_wikitext_aliases(
            "中华人民共和国",
            "",
            rendered_text=(
                "# 中华人民共和国\n中华人民共和国是位于东亚的国家。\n\n"
                "## 国名\n「中国」也逐渐成为国际社会对中华人民共和国的常见称呼。"
            ),
        )
        self.assertIn("中国", aliases)

    def test_does_not_treat_another_subjects_common_name_as_article_alias(self) -> None:
        aliases = extract_wikitext_aliases(
            "胰臟癌",
            "",
            rendered_text=(
                "胰臟癌是恶性肿瘤；该部位也可能发生其他几种"
                "通称为非腺癌（non-adenocarcinomas）的癌症。"
            ),
        )
        self.assertNotIn("非腺癌（non-adenocarcinomas）的癌症", aliases)

    def test_does_not_treat_bold_year_event_labels_as_article_aliases(self) -> None:
        aliases = extract_wikitext_aliases(
            "75年",
            "{{yearTOC|75}}\n\n== 大事记 ==\n*'''[[中国]]'''\n**事件。",
        )
        self.assertNotIn("中国", aliases)

    def test_does_not_treat_bold_title_component_as_alias(self) -> None:
        aliases = extract_wikitext_aliases(
            "中国湿地公园",
            "'''中国湿地公园'''是'''中国'''的湿地公园体系。",
        )
        self.assertNotIn("中国", aliases)


class FineWikiChunkerTests(unittest.TestCase):
    def test_markdown_headings_become_metadata_not_body_noise(self) -> None:
        article = FineWikiArticle(
            page_id="42",
            url="https://zh.wikipedia.org/wiki/Python",
            title="Python",
            text="# Python\n导言。\n\n## 很长：但仍然应该被识别为结构化标题而不是正文\n" + "正文。" * 250,
            date_modified="2025-07-01T00:00:00Z",
            wikidata_id="Q28865",
            infoboxes=json.dumps({"设计者": "Guido van Rossum"}),
            aliases=("Python语言",),
        )
        chunks = FineWikiChunker(target_chars=300, max_chars=400).chunk(article)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("很长：但仍然应该被识别" in heading for c in chunks for heading in c.headings))
        self.assertTrue(all("# Python" not in chunk.text for chunk in chunks))
        self.assertEqual(chunks[0].source, "finewiki")
        self.assertEqual(chunks[0].wikidata_id, "Q28865")
        self.assertIn("Guido van Rossum", chunks[0].metadata_text)
        self.assertIn("Python语言", chunks[0].metadata_text)
        self.assertEqual(chunks[0].aliases, ("Python语言",))
        self.assertTrue(all(not chunk.metadata_text for chunk in chunks[1:]))
        self.assertTrue(all(not chunk.aliases for chunk in chunks[1:]))


if __name__ == "__main__":
    unittest.main()
