"""Optional local reranker construction shared by the live agent services."""

from __future__ import annotations

import os
from typing import Any


def build_semantic_scorer_from_env() -> Any | None:
    """Build one lazy local reranker shared by routing and evidence selection."""

    model_name = os.getenv("RWKV_AGENT_RERANKER_MODEL", "").strip()
    if not model_name:
        return None
    from rwkv_search.semantic_selection import TransformersPairScorer

    return TransformersPairScorer(
        model_name,
        device=os.getenv("RWKV_AGENT_RERANKER_DEVICE", "auto").strip() or "auto",
        batch_size=int(os.getenv("RWKV_AGENT_RERANKER_BATCH_SIZE", "16")),
        max_length=int(os.getenv("RWKV_AGENT_RERANKER_MAX_LENGTH", "512")),
        fp16=os.getenv("RWKV_AGENT_RERANKER_FP16", "1").strip().casefold()
        not in {"0", "false", "no"},
        local_files_only=True,
    )
