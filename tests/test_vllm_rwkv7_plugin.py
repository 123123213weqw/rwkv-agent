from __future__ import annotations

from pathlib import Path
import sys

import pytest


PLUGIN_SRC = (
    Path(__file__).parents[1]
    / "integrations"
    / "vllm_rwkv7"
    / "src"
)


def test_vllm_rwkv7_query_bounds() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    sys.path.insert(0, str(PLUGIN_SRC))
    try:
        from vllm_rwkv7.model import _query_bounds

        metadata = type(
            "Metadata",
            (),
            {
                "query_start_loc": torch.tensor(
                    [0, 3, 5], dtype=torch.int32
                )
            },
        )()
        assert _query_bounds(
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([7, 8]),
            metadata,
        ) == [(0, 3), (3, 5)]
    finally:
        sys.path.remove(str(PLUGIN_SRC))

