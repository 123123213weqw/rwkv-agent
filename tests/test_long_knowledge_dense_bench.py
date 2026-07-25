from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from bench.run_long_knowledge_dense_bench import _load_lexical_cache, _semantic_order


class LongKnowledgeDenseBenchTests(unittest.TestCase):
    def test_semantic_order_only_reorders_configured_head(self) -> None:
        candidates = [
            {"page_id": "a"},
            {"page_id": "b"},
            {"page_id": "c"},
        ]
        result = _semantic_order(candidates, [0.1, 0.9, 10.0], depth=2)
        self.assertEqual([item["page_id"] for item in result], ["b", "a", "c"])

    def test_frozen_lexical_cache_preserves_hits_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.jsonl"
            path.write_text(
                '{"id":"a","hits":[{"page_id":"1"}],"search_elapsed_ms":12.5}\n',
                encoding="utf-8",
            )
            cache = _load_lexical_cache(str(path))
        self.assertEqual(cache["a"]["hits"][0]["page_id"], "1")
        self.assertEqual(cache["a"]["search_elapsed_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
