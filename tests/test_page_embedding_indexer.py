from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.index_finewiki_page_embeddings import (
    BuildCheckpoint,
    ElasticsearchClient,
    page_document,
)


class PageEmbeddingIndexerTests(unittest.TestCase):
    def test_page_document_is_bounded_and_preserves_semantic_fields(self) -> None:
        value = page_document(
            {
                "title_original": "RWKV",
                "alias_original": ["Receptance Weighted Key Value"],
                "heading_original": ["Architecture", "Inference"],
                "body_original": "A recurrent language model. " * 1000,
            },
            max_chars=512,
        )
        self.assertLessEqual(len(value), 512)
        self.assertIn("Title: RWKV", value)
        self.assertIn("Aliases: Receptance Weighted Key Value", value)
        self.assertIn("Sections: Architecture > Inference", value)

    def test_checkpoint_round_trip_is_atomic_and_typed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            checkpoint = BuildCheckpoint(
                source_index="source",
                target_index="target",
                last_sort=("123",),
                indexed_pages=500,
                complete=False,
                page_id_gte="5",
                page_id_lt="8",
            )
            checkpoint.write(path)
            self.assertEqual(BuildCheckpoint.load(path), checkpoint)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"],
                "finewiki-page-embedding-checkpoint.v1",
            )
            self.assertEqual(payload["page_id_gte"], "5")
            self.assertEqual(payload["page_id_lt"], "8")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_partition_range_is_added_to_source_query(self) -> None:
        client = ElasticsearchClient("http://example.invalid")
        captured = {}

        def fake_call(method, path, payload=None, **kwargs):
            captured.update(payload)
            return {"hits": {"hits": []}}

        client.call = fake_call  # type: ignore[method-assign]
        client.fetch_lead_pages(
            "source",
            batch_size=10,
            page_id_gte="5",
            page_id_lt="8",
        )
        filters = captured["query"]["bool"]["filter"]
        self.assertIn({"term": {"chunk_id": 0}}, filters)
        self.assertIn({"range": {"page_id": {"gte": "5", "lt": "8"}}}, filters)


if __name__ == "__main__":
    unittest.main()
