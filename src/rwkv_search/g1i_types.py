from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class G1ICompletion:
    """One deterministic G1I decode with the complete token trace."""

    text: str
    stop: str = ""
    token_ids: Tuple[int, ...] = ()
    elapsed_ms: Optional[float] = None
