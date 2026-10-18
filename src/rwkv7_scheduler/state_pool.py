"""Preallocated recurrent-state slots and reusable contiguous batch workspaces."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


class StatePoolError(RuntimeError):
    """Base state-pool error."""


class StatePoolFull(StatePoolError):
    """Raised when no recurrent-state slot is available."""


class StaleStateHandle(StatePoolError):
    """Raised when a released/reused handle is accessed."""


@dataclass(frozen=True, slots=True)
class StateHandle:
    """Opaque ownership token for one row in the recurrent-state slab."""

    state_id: str
    slot: int
    generation: int
    device: str


def _pipeline_state(state: list[Any]) -> bool:
    return len(state) == 3 and isinstance(state[0], list)


def _batch_size(state: list[Any]) -> int:
    if len(state) != 3:
        raise ValueError("Albatross state must contain shift, WKV and elapsed tensors")
    shift, wkv, elapsed = state
    if _pipeline_state(state):
        if not shift or len(shift) != len(wkv) or not isinstance(elapsed, list) or not elapsed:
            raise ValueError("invalid pipeline-parallel Albatross state")
        batch = int(shift[0].shape[1])
        if any(
            value.ndim != 3
            or int(value.shape[0]) != 2
            or int(value.shape[1]) != batch
            for value in shift
        ):
            raise ValueError("pipeline shift state must use [2,B,C] per layer")
        if any(value.ndim != 4 or int(value.shape[0]) != batch for value in wkv):
            raise ValueError("pipeline WKV state must use [B,H,N,N] per layer")
        if any(value.ndim != 1 or int(value.shape[0]) != batch for value in elapsed):
            raise ValueError("pipeline elapsed state must use [B] per device")
        return batch
    if shift.ndim != 4 or wkv.ndim != 5 or elapsed.ndim != 1:
        raise ValueError(
            "unsupported Albatross state layout; expected "
            "[L,2,B,C], [L,B,H,N,N], [B]"
        )
    batch = int(shift.shape[2])
    if int(wkv.shape[1]) != batch or int(elapsed.shape[0]) != batch:
        raise ValueError("inconsistent Albatross state batch dimensions")
    return batch


def state_bytes(state: list[Any]) -> int:
    def size(value: Any) -> int:
        if isinstance(value, list):
            return sum(size(item) for item in value)
        return int(value.numel()) * int(value.element_size())

    return sum(size(value) for value in state)


def _state_device(state: list[Any]) -> str:
    if _pipeline_state(state):
        devices = []
        for value in list(state[0]) + list(state[1]) + list(state[2]):
            name = str(value.device)
            if name not in devices:
                devices.append(name)
        return "pipeline:" + ",".join(devices)
    return str(state[2].device)


def _clone_state(state: list[Any]) -> list[Any]:
    return [
        [item.clone() for item in value] if isinstance(value, list) else value.clone()
        for value in state
    ]


class AlbatrossStatePool:
    """Fixed-capacity single-GPU state slab.

    Model forward mutates a contiguous state batch. Serving requests live in
    stable slab slots; a reusable workspace gathers selected slots, executes
    one model call, then scatters the updated rows back. This bounds allocation
    count and avoids one Python-owned state object per request.
    """

    def __init__(
        self,
        model: Any,
        *,
        capacity: int,
        max_batch_size: int,
        pool_id: str | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if max_batch_size < 1 or max_batch_size > capacity:
            raise ValueError("max_batch_size must be within pool capacity")
        self.model = model
        self.capacity = int(capacity)
        self.max_batch_size = int(max_batch_size)
        self.pool_id = pool_id or uuid.uuid4().hex
        # Keep pool tensors as ordinary non-grad tensors. Workspaces may be
        # lazily created while the scheduler is inside inference_mode; explicit
        # disable avoids producing inference tensors that cannot later be
        # inspected or offloaded outside that context.
        with torch.inference_mode(False):
            self.state = model.zero_state(self.capacity)
        _batch_size(self.state)
        self.pipeline_parallel = _pipeline_state(self.state)
        self.device = _state_device(self.state)
        self._workspaces: dict[int, list[torch.Tensor]] = {}
        self._free = list(reversed(range(self.capacity)))
        self._generation = [0] * self.capacity
        self._owners: list[str | None] = [None] * self.capacity
        self._lock = threading.RLock()
        # Every scheduler facade sharing this model/pool must serialize model
        # forward calls. Albatross mutates the supplied recurrent state and its
        # reusable embedding buffers are not safe for concurrent forwards.
        self.execution_lock = threading.RLock()

    @property
    def allocated(self) -> int:
        with self._lock:
            return self.capacity - len(self._free)

    @property
    def free(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def bytes_per_slot(self) -> int:
        return state_bytes(self.state) // self.capacity

    @property
    def slab_bytes(self) -> int:
        return state_bytes(self.state)

    @property
    def workspace_bytes(self) -> int:
        return sum(state_bytes(state) for state in self._workspaces.values())

    def allocate(self, owner: str) -> StateHandle:
        clean_owner = str(owner or "").strip()
        if not clean_owner:
            raise ValueError("owner must not be empty")
        with self._lock:
            if not self._free:
                raise StatePoolFull(
                    f"state pool {self.pool_id} is full ({self.capacity} slots)"
                )
            slot = self._free.pop()
            self._generation[slot] += 1
            self._owners[slot] = clean_owner
            self._zero_slots([slot])
            return StateHandle(
                state_id=f"{self.pool_id}:{slot}:{self._generation[slot]}",
                slot=slot,
                generation=self._generation[slot],
                device=self.device,
            )

    def allocate_many(self, owners: Iterable[str]) -> list[StateHandle]:
        values = list(owners)
        with self._lock:
            if len(values) > len(self._free):
                raise StatePoolFull(
                    f"need {len(values)} state slots, only {len(self._free)} free"
                )
            handles = []
            try:
                for owner in values:
                    handles.append(self.allocate(owner))
            except Exception:
                for handle in handles:
                    self.release(handle)
                raise
            return handles

    def clone_many(
        self,
        source: StateHandle,
        owners: Iterable[str],
    ) -> list[StateHandle]:
        """Allocate slots containing exact GPU-side copies of ``source``.

        Agent branches share the already-computed prefix represented by one
        recurrent state.  The copy stays on the slab device: no CPU snapshot
        or model forward is involved.  All destinations are rolled back when
        allocation or copying fails.
        """

        values = list(owners)
        if not values:
            return []
        if len(values) > self.max_batch_size:
            raise ValueError(
                f"fork width {len(values)} exceeds "
                f"max_batch_size={self.max_batch_size}"
            )
        with self._lock:
            self._validate(source)
            handles = self.allocate_many(values)
            try:
                count = len(handles)
                # Clone the B1 source rows before index_copy_.  Reading and
                # writing overlapping views of the slab directly is not a
                # supported PyTorch operation even though destinations never
                # include the source slot.
                slots = [handle.slot for handle in handles]
                if self.pipeline_parallel:
                    for target in self.state[0]:
                        indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                        row = target[:, source.slot : source.slot + 1, :].clone()
                        target.index_copy_(1, indices, row.expand(-1, count, -1))
                    for target in self.state[1]:
                        indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                        row = target[source.slot : source.slot + 1, :, :, :].clone()
                        target.index_copy_(0, indices, row.expand(count, -1, -1, -1))
                    for target in self.state[2]:
                        indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                        row = target[source.slot : source.slot + 1].clone()
                        target.index_copy_(0, indices, row.expand(count))
                else:
                    indices = torch.tensor(slots, dtype=torch.long, device=self.state[2].device)
                    shift = self.state[0][:, :, source.slot : source.slot + 1, :].clone()
                    wkv = self.state[1][:, source.slot : source.slot + 1, :, :, :].clone()
                    elapsed = self.state[2][source.slot : source.slot + 1].clone()
                    self.state[0].index_copy_(2, indices, shift.expand(-1, -1, count, -1))
                    self.state[1].index_copy_(1, indices, wkv.expand(-1, count, -1, -1, -1))
                    self.state[2].index_copy_(0, indices, elapsed.expand(count))
            except Exception:
                self.release_many(handles)
                raise
            return handles

    def owner(self, handle: StateHandle) -> str:
        with self._lock:
            self._validate(handle)
            owner = self._owners[handle.slot]
            assert owner is not None
            return owner

    def release(self, handle: StateHandle) -> None:
        with self._lock:
            self._validate(handle)
            self._owners[handle.slot] = None
            self._free.append(handle.slot)

    def release_many(self, handles: Iterable[StateHandle]) -> None:
        with self._lock:
            for handle in handles:
                self.release(handle)

    def gather(self, handles: list[StateHandle]) -> list[torch.Tensor]:
        if not handles:
            raise ValueError("cannot gather an empty state batch")
        if len(handles) > self.max_batch_size:
            raise ValueError(
                f"batch {len(handles)} exceeds max_batch_size={self.max_batch_size}"
            )
        with self._lock:
            self._validate_unique(handles)
            workspace = self._workspace(len(handles))
            slots = [handle.slot for handle in handles]
            if self.pipeline_parallel:
                for source, target in zip(self.state[0], workspace[0]):
                    indices = torch.tensor(slots, dtype=torch.long, device=source.device)
                    torch.index_select(source, 1, indices, out=target)
                for source, target in zip(self.state[1], workspace[1]):
                    indices = torch.tensor(slots, dtype=torch.long, device=source.device)
                    torch.index_select(source, 0, indices, out=target)
                for source, target in zip(self.state[2], workspace[2]):
                    indices = torch.tensor(slots, dtype=torch.long, device=source.device)
                    torch.index_select(source, 0, indices, out=target)
            else:
                indices = torch.tensor(slots, dtype=torch.long, device=self.state[2].device)
                torch.index_select(self.state[0], 2, indices, out=workspace[0])
                torch.index_select(self.state[1], 1, indices, out=workspace[1])
                torch.index_select(self.state[2], 0, indices, out=workspace[2])
            return workspace

    def checkout(
        self,
        handles: list[StateHandle],
    ) -> tuple[list[torch.Tensor], bool]:
        """Return a mutable batch and whether it directly borrows slab storage.

        Consecutive slots are borrowed only when all three resulting tensors
        are contiguous. Custom Albatross CUDA kernels consume flat state
        layouts; a narrow view through the slab's middle batch dimension has
        larger outer strides and must not be passed as if it were contiguous.
        """

        if not handles:
            raise ValueError("cannot checkout an empty state batch")
        with self._lock:
            self._validate_unique(handles)
            slots = [handle.slot for handle in handles]
            start = slots[0]
            if slots == list(range(start, start + len(slots))):
                end = start + len(slots)
                if self.pipeline_parallel:
                    views = [
                        [value[:, start:end, :] for value in self.state[0]],
                        [value[start:end, :, :, :] for value in self.state[1]],
                        [value[start:end] for value in self.state[2]],
                    ]
                    contiguous = all(
                        value.is_contiguous()
                        for group in views
                        for value in group
                    )
                else:
                    views = [
                        self.state[0][:, :, start:end, :],
                        self.state[1][:, start:end, :, :, :],
                        self.state[2][start:end],
                    ]
                    contiguous = all(value.is_contiguous() for value in views)
                if contiguous:
                    return views, True
            return self.gather(handles), False

    def checkin(
        self,
        handles: list[StateHandle],
        state: list[torch.Tensor],
        *,
        borrowed: bool,
    ) -> None:
        """Commit a checkout; borrowed slab views are already up to date."""

        if borrowed:
            with self._lock:
                self._validate_unique(handles)
                if _batch_size(state) != len(handles):
                    raise ValueError("borrowed state batch no longer matches handles")
            return
        self.commit(handles, state)

    def commit(
        self,
        handles: list[StateHandle],
        workspace: list[torch.Tensor],
    ) -> None:
        if not handles:
            raise ValueError("cannot commit an empty state batch")
        with self._lock:
            self._validate_unique(handles)
            if _batch_size(workspace) != len(handles):
                raise ValueError("workspace batch does not match handle count")
            slots = [handle.slot for handle in handles]
            if self.pipeline_parallel:
                for target, source in zip(self.state[0], workspace[0]):
                    indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                    target.index_copy_(1, indices, source)
                for target, source in zip(self.state[1], workspace[1]):
                    indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                    target.index_copy_(0, indices, source)
                for target, source in zip(self.state[2], workspace[2]):
                    indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                    target.index_copy_(0, indices, source)
            else:
                indices = torch.tensor(slots, dtype=torch.long, device=self.state[2].device)
                self.state[0].index_copy_(2, indices, workspace[0])
                self.state[1].index_copy_(1, indices, workspace[1])
                self.state[2].index_copy_(0, indices, workspace[2])

    def snapshot(self, handle: StateHandle) -> list[torch.Tensor]:
        """Clone one row for correctness tests or explicit cold offload."""

        workspace = self.gather([handle])
        return _clone_state(workspace)

    def prewarm(self, batch_sizes: Iterable[int]) -> None:
        with self._lock:
            for value in batch_sizes:
                batch_size = int(value)
                if batch_size < 1 or batch_size > self.max_batch_size:
                    raise ValueError(
                        f"invalid workspace batch size {batch_size}"
                    )
                self._workspace(batch_size)

    def _workspace(self, batch_size: int) -> list[torch.Tensor]:
        workspace = self._workspaces.get(batch_size)
        if workspace is None:
            with torch.inference_mode(False):
                workspace = self.model.zero_state(batch_size)
            if _batch_size(workspace) != batch_size:
                raise ValueError("model.zero_state returned an invalid workspace")
            self._workspaces[batch_size] = workspace
        return workspace

    def _zero_slots(self, slots: list[int]) -> None:
        if self.pipeline_parallel:
            for target in self.state[0]:
                indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                target.index_fill_(1, indices, 0)
            for target in self.state[1]:
                indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                target.index_fill_(0, indices, 0)
            for target in self.state[2]:
                indices = torch.tensor(slots, dtype=torch.long, device=target.device)
                target.index_fill_(0, indices, 0)
        else:
            indices = torch.tensor(slots, dtype=torch.long, device=self.state[2].device)
            self.state[0].index_fill_(2, indices, 0)
            self.state[1].index_fill_(1, indices, 0)
            self.state[2].index_fill_(0, indices, 0)

    def _validate(self, handle: StateHandle) -> None:
        if not isinstance(handle, StateHandle):
            raise TypeError("expected StateHandle")
        if handle.device != self.device:
            raise StaleStateHandle(
                f"state {handle.state_id} belongs to {handle.device}, not {self.device}"
            )
        if handle.slot < 0 or handle.slot >= self.capacity:
            raise StaleStateHandle(f"invalid state slot {handle.slot}")
        if (
            self._owners[handle.slot] is None
            or self._generation[handle.slot] != handle.generation
        ):
            raise StaleStateHandle(f"stale or released state {handle.state_id}")

    def _validate_unique(self, handles: list[StateHandle]) -> None:
        seen = set()
        for handle in handles:
            self._validate(handle)
            if handle.slot in seen:
                raise ValueError(f"duplicate state slot {handle.slot} in one batch")
            seen.add(handle.slot)
