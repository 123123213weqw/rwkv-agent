"""Bounded Controller-side ownership for recurrent chat session states."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import threading


@dataclass(slots=True)
class ChatSessionState:
    session_id: str
    owner_id: str
    state_id: str
    home_url: str
    last_message_id: int
    stop_reason: str
    seen_tokens: int


class ChatStateCache:
    """Keep a small LRU of GPU-resident chat states.

    The Sidecar owns the tensors.  This cache owns only opaque IDs and ensures
    that two HTTP turns for the same session never mutate one recurrent state
    concurrently.
    """

    def __init__(self, *, capacity: int = 3, lock_stripes: int = 64) -> None:
        if capacity < 1:
            raise ValueError("chat state capacity must be positive")
        if lock_stripes < 1:
            raise ValueError("lock_stripes must be positive")
        self.capacity = int(capacity)
        self._items: OrderedDict[str, ChatSessionState] = OrderedDict()
        self._busy: set[str] = set()
        self._lock = threading.RLock()
        self._turn_locks = [threading.RLock() for _ in range(lock_stripes)]
        self._metrics: Counter[str] = Counter()

    @staticmethod
    def normalize_session_id(value: str) -> str:
        session_id = str(value or "").strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        return session_id

    @staticmethod
    def owner_id(session_id: str) -> str:
        normalized = ChatStateCache.normalize_session_id(session_id)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return "chat-" + digest[:32]

    def turn_lock(self, session_id: str) -> threading.RLock:
        normalized = self.normalize_session_id(session_id)
        digest = hashlib.blake2s(normalized.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, "big") % len(self._turn_locks)
        return self._turn_locks[index]

    def get(
        self,
        session_id: str,
        *,
        last_message_id: int,
    ) -> tuple[ChatSessionState | None, ChatSessionState | None]:
        """Return a matching state and an optional stale record to release."""

        normalized = self.normalize_session_id(session_id)
        with self._lock:
            record = self._items.get(normalized)
            if record is None:
                self._metrics["misses"] += 1
                return None, None
            if record.last_message_id != int(last_message_id):
                self._items.pop(normalized, None)
                self._busy.discard(normalized)
                self._metrics["transcript_mismatches"] += 1
                return None, record
            self._items.move_to_end(normalized)
            self._busy.add(normalized)
            self._metrics["hits"] += 1
            return record, None

    def put(self, record: ChatSessionState) -> list[ChatSessionState]:
        normalized = self.normalize_session_id(record.session_id)
        if normalized != record.session_id:
            raise ValueError("chat state contains a non-normalized session_id")
        evicted: list[ChatSessionState] = []
        with self._lock:
            self._busy.discard(normalized)
            previous = self._items.pop(normalized, None)
            if previous is not None and previous.state_id != record.state_id:
                evicted.append(previous)
            self._items[normalized] = record
            while len(self._items) > self.capacity:
                victim = next(
                    (
                        session_id
                        for session_id in self._items
                        if session_id not in self._busy
                    ),
                    None,
                )
                if victim is None:
                    break
                stale = self._items.pop(victim)
                evicted.append(stale)
                self._metrics["evictions"] += 1
            self._metrics["stores"] += 1
        return evicted

    def pop(self, session_id: str) -> ChatSessionState | None:
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            self._busy.discard(normalized)
            record = self._items.pop(normalized, None)
            if record is not None:
                self._metrics["invalidations"] += 1
            return record

    def clear(self) -> list[ChatSessionState]:
        with self._lock:
            records = list(self._items.values())
            self._items.clear()
            self._busy.clear()
            if records:
                self._metrics["invalidations"] += len(records)
            return records

    def count(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._metrics[str(name)] += int(value)

    def health(self, *, enabled: bool) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": bool(enabled),
                "mode": "gpu_recurrent_lru",
                "capacity": self.capacity,
                "allocated": len(self._items),
                "busy": len(self._busy),
                "free": self.capacity - len(self._items),
                "metrics": dict(self._metrics),
            }
