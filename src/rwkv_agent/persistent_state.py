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

    def children_of(self, state_id: str) -> list[str]:
        """Direct children of a node."""
        return [
            record.state_id
            for record in self._records.values()
            if record.parent_state_id == state_id
        ]

    def descendants_of(self, state_id: str) -> list[str]:
        """Every node below `state_id`, excluding itself: iterative and cycle-safe."""
        out: list[str] = []
        seen = {state_id}
        stack = [state_id]
        while stack:
            current = stack.pop()
            for child in self.children_of(current):
                if child in seen:
                    continue
                seen.add(child)
                out.append(child)
                stack.append(child)
        return out

    def root_of(self, state_id: str) -> str:
        """Walk parent links up to the root. Returns state_id if it is orphaned."""
        seen = {state_id}
        current = state_id
        while True:
            record = self._records.get(current)
            if record is None or record.parent_state_id is None:
                return current
            parent = record.parent_state_id
            if parent in seen or parent not in self._records:
                return current
            seen.add(parent)
            current = parent

    def subtree_last_active(self, state_id: str) -> tuple[float, int]:
        """(most recent last_used_at anywhere in the subtree, node count).

        The MOST RECENT use decides: a group is only idle when even its busiest
        member has been untouched for the whole TTL.
        """
        node = self._records.get(state_id)
        if node is None:
            return 0.0, 0
        ids = [state_id] + self.descendants_of(state_id)
        newest = node.last_used_at
        count = 0
        for i in ids:
            r = self._records.get(i)
            if r is None:
                continue
            count += 1
            newest = max(newest, r.last_used_at)
        return newest, count

    def cleanup_expired(
        self,
        *,
        exclude: Collection[str] = (),
    ) -> list[str]:
        now = self.clock()
        protected = set(exclude)
        # Tree-aware expiry: a node is reclaimable only when its WHOLE subtree is
        # idle. The most recently used member decides, so a single active
        # descendant keeps its ancestors alive.
        expired: list[str] = []
        handled: set[str] = set()
        for state_id in list(self._records):
            if state_id in handled or state_id in protected:
                continue
            record = self._records.get(state_id)
            if record is None:
                continue
            newest, _count = self.subtree_last_active(state_id)
            if now - newest < self.ttl_seconds:
                continue
            subtree = [state_id] + self.descendants_of(state_id)
            if any(node in protected for node in subtree):
                continue
            for node in subtree:
                handled.add(node)
                self._records.pop(node, None)
                try:
                    self.scheduler.release(node)
                except KeyError:
                    pass
            expired.extend(subtree)
        return expired

    def release_records(self, records: Sequence[PersistentState]) -> None:
        """Release each node together with its whole subtree.

        A forked worker branch is disposable by construction -- fork_from_root
        already assumes this -- so dropping a parent without its children would
        leave rows that can never be addressed or reclaimed deterministically.
        """
        targets: list[str] = []
        seen: set[str] = set()
        for record in records:
            for node in [record.state_id] + self.descendants_of(record.state_id):
                if node in seen:
                    continue
                seen.add(node)
                targets.append(node)
        for state_id in targets:
            self._records.pop(state_id, None)
            try:
                self.scheduler.release(state_id)
            except KeyError:
                pass

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
