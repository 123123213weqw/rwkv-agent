from __future__ import annotations

import unittest

from rwkv_search.wikipedia import (
    WikipediaArticle,
    WikipediaChunker,
    classify_wikipedia_page,
    clean_wikipedia_text,
)


class WikipediaCleaningTests(unittest.TestCase):
    def test_removes_common_mediawiki_residue_and_selects_simplified_variant(self) -> None:
        value = "{{Infobox person\n|name=X\n}}\n-{zh-cn:宾·;zh-tw:賓·;}-人物__NOTOC__ [[ 編輯]]"
        self.assertEqual(clean_wikipedia_text(value), "宾·人物")

    def test_page_types_are_generic_metadata_not_domain_routing(self) -> None:
        self.assertEqual(classify_wikipedia_page("苏丹 (消歧义)", "正文"), "disambiguation")
        self.assertEqual(classify_wikipedia_page("小行星列表/1-100", "正文"), "list")
        self.assertEqual(classify_wikipedia_page("Python", "编程语言"), "article")


class WikipediaChunkerTests(unittest.TestCase):
    def test_short_article_is_preserved(self) -> None:
        chunks = WikipediaChunker().chunk(
            WikipediaArticle("1", "https://example.test/1", "NGC 326", "NGC 326 是双鱼座的一个星系。"),
            snapshot_date="20260301",
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "NGC 326 是双鱼座的一个星系。")

    def test_paragraph_first_chunking_preserves_heading_and_overlap(self) -> None:
        paragraphs = ["第一段。" + "甲" * 260, "第二段。" + "乙" * 260, "第三段。" + "丙" * 260]
        body = "简介\n\n" + "\n\n".join(paragraphs)
        chunks = WikipediaChunker(target_chars=400, max_chars=600, overlap_chars=100).chunk(
            WikipediaArticle("2", "https://example.test/2", "测试文章", body),
            snapshot_date="20260301",
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all("简介" in chunk.headings for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 600 for chunk in chunks))
        self.assertIn(paragraphs[1], chunks[0].text + chunks[1].text)

    def test_chunk_ids_and_metadata_are_stable(self) -> None:
        article = WikipediaArticle("42", "https://example.test/42", "标题", "正文。" * 400)
        chunker = WikipediaChunker(target_chars=300, max_chars=400, overlap_chars=50)
        first = chunker.chunk(article, snapshot_date="20260301")
        second = chunker.chunk(article, snapshot_date="20260301")
        self.assertEqual(first, second)
        self.assertEqual([chunk.doc_id for chunk in first], [f"42#{i}" for i in range(len(first))])
        self.assertTrue(all(len(chunk.text) <= 400 for chunk in first))


if __name__ == "__main__":
    unittest.main()
