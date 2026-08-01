"""Ownership and lifecycle registry for opaque persistent scheduler states."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
import time
import uuid

from rwkv_runtime.protocols import SchedulerProtocol


@dataclass(slots=True)
class PersistentState:
    state_id: str
    owner_id: str
    parent_state_id: str | None
    branch: str
    created_at: float
    last_used_at: float


class PersistentStateRegistry:
    """Track ownership, TTL and bounded capacity independently of inference."""

    def __init__(
        self,
        *,
        scheduler: SchedulerProtocol,
        capacity: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scheduler = scheduler
        self.capacity = int(capacity)
        self.ttl_seconds = float(ttl_seconds)
        self.clock = clock
        self._records: dict[str, PersistentState] = {}

    @staticmethod
    def clean_owner(owner_id: str) -> str:
        value = str(owner_id or "").strip()
        if not value:
            raise ValueError("owner_id must not be empty")
        if len(value) > 128:
            raise ValueError("owner_id is too long")
        return value

    @staticmethod
    def new_state_id() -> str:
        return "state-" + uuid.uuid4().hex

    @property
    def allocated(self) -> int:
        return len(self._records)

    @property
    def free(self) -> int:
        return self.capacity - self.allocated

    def require(self, state_id: str, owner_id: str) -> PersistentState:
        clean_id = str(state_id or "").strip()
        try:
            record = self._records[clean_id]
        except KeyError as exc:
            raise KeyError(f"unknown state_id: {clean_id}") from exc
        if record.owner_id != owner_id:
            raise PermissionError("state owner mismatch")
        return record

    def ensure_capacity(self, additional: int) -> None:
        if additional < 1:
            return
        if self.allocated + additional > self.capacity:
            raise RuntimeError(
                f"persistent state capacity exceeded: "
                f"{self.allocated}+{additional}>{self.capacity}"
            )

    def add(self, record: PersistentState) -> None:
        if record.state_id in self._records:
            raise ValueError(f"duplicate state_id: {record.state_id}")
        self._records[record.state_id] = record

    def cleanup_expired(
        self,
        *,
        exclude: Collection[str] = (),
    ) -> list[str]:
        now = self.clock()
        protected = set(exclude)
        expired = [
            state_id
            for state_id, record in self._records.items()
            if state_id not in protected
            and now - record.last_used_at >= self.ttl_seconds
        ]
        for state_id in expired:
            self._records.pop(state_id, None)
            try:
                self.scheduler.release(state_id)
            except KeyError:
                pass
        return expired

    def release_records(self, records: Sequence[PersistentState]) -> None:
        for record in records:
            self.scheduler.release(record.state_id)
            self._records.pop(record.state_id, None)

    def oldest_idle_seconds(self) -> float:
        now = self.clock()
        return max(
            (now - record.last_used_at for record in self._records.values()),
            default=0.0,
        )

    def clear(self) -> None:
        for state_id in list(self._records):
            try:
                self.scheduler.release(state_id)
            except KeyError:
                pass
            self._records.pop(state_id, None)
