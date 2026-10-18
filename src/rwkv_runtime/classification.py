"""Shared next-token classification helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def finite_label_scores(
    logits: Any,
    labels: Mapping[str, int],
    *,
    error_message: str = "non-finite classification logits",
) -> dict[str, float]:
    scores = {
        str(name): float(logits[int(token)].item())
        for name, token in labels.items()
    }
    if not all(math.isfinite(score) for score in scores.values()):
        raise RuntimeError(error_message)
    return scores
