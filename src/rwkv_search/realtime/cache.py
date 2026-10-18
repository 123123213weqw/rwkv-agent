from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float
    size: int


class TTLByteCache(Generic[T]):
    """Small process-local TTL/LRU cache; never persists page content."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, max_bytes)
        self._bytes = 0
        self._items: "OrderedDict[str, _Entry[T]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[T]:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item.expires_at <= now:
                self._drop(key)
                return None
            self._items.move_to_end(key)
            return item.value

    def put(self, key: str, value: T, ttl_seconds: float, size: Optional[int] = None) -> None:
        if self.max_bytes <= 0 or ttl_seconds <= 0:
            return
        measured = max(1, size if size is not None else sys.getsizeof(value))
        if measured > self.max_bytes:
            return
        with self._lock:
            if key in self._items:
                self._drop(key)
            self._items[key] = _Entry(value, time.monotonic() + ttl_seconds, measured)
            self._bytes += measured
            while self._bytes > self.max_bytes and self._items:
                self._drop(next(iter(self._items)))

    def _drop(self, key: str) -> None:
        item = self._items.pop(key, None)
        if item:
            self._bytes -= item.size
