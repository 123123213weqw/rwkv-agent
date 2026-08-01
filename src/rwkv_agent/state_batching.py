"""Persistent-State row contract used by the unified inference queue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StateContinuationItem:
    state_id: str
    branch: str
    token_ids: tuple[int, ...]
