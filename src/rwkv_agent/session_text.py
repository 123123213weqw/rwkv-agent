"""Bounded in-memory pasted-text buffers scoped by Agent session."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading
import time


@dataclass(frozen=True)
class PastedText:
    name: str
    text: str
    chars: int
    sha256: str
    created_at: float


class SessionTextBuffer:
    """Keep one transient long text per session with bounded LRU eviction."""

    def __init__(
        self,
        *,
        max_sessions: int = 32,
        max_chars: int = 1_000_000,
    ) -> None:
        self.max_sessions = max(1, int(max_sessions))
        self.max_chars = max(1, int(max_chars))
        self._items: OrderedDict[str, PastedText] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _session_id(value: str) -> str:
        session_id = str(value or "").strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        return session_id

    def put(self, session_id: str, text: str) -> PastedText:
        normalized_session = self._session_id(session_id)
        content = str(text or "")
        if not content.strip():
            raise ValueError("pasted text must not be empty")
        if len(content) > self.max_chars:
            raise ValueError(
                f"pasted text exceeds {self.max_chars} character limit"
            )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        item = PastedText(
            name=f"pasted-{digest[:8]}",
            text=content,
            chars=len(content),
            sha256=digest,
            created_at=time.time(),
        )
        with self._lock:
            self._items.pop(normalized_session, None)
            self._items[normalized_session] = item
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)
        return item

    def get(self, session_id: str) -> PastedText | None:
        normalized_session = self._session_id(session_id)
        with self._lock:
            item = self._items.get(normalized_session)
            if item is not None:
                self._items.move_to_end(normalized_session)
            return item

    def clear(self, session_id: str) -> bool:
        normalized_session = self._session_id(session_id)
        with self._lock:
            return self._items.pop(normalized_session, None) is not None

    def health(self) -> dict[str, int | str]:
        with self._lock:
            sessions = len(self._items)
            chars = sum(item.chars for item in self._items.values())
        return {
            "mode": "transient_session_text",
            "sessions": sessions,
            "chars": chars,
            "max_sessions": self.max_sessions,
            "max_chars_per_session": self.max_chars,
        }

    def close(self) -> None:
        with self._lock:
            self._items.clear()
