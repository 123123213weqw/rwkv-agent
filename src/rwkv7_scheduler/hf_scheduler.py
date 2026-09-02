"""ROCm-friendly recurrent-State scheduler for the RWKV-7 HF adapter.

The Albatross scheduler owns one preallocated CUDA state slab.  That runtime
uses NVIDIA-specific MMA kernels, so it cannot be the Radeon fallback.  The HF
adapter exposes the same recurrent cache semantics with ordinary PyTorch
tensors.  This module adapts that cache to ``SchedulerProtocol`` while keeping
the externally visible State lifecycle unchanged.

Independent requests never share a cache object.  Ready rows are gathered into
one temporary batched cache, advanced by one model call, and split back into
per-request caches.  Only model weights and the scheduler are shared.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .scheduler import SchedulerConfig


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def _cache_bytes(cache: Any) -> int:
    if cache is None:
        return 0
    state, xpa, xpf, v_first = tuple(cache)
    return sum(
        _tensor_bytes(value)
        for values in (state, xpa, xpf)
        for value in (values or [])
    ) + (0 if v_first is None else _tensor_bytes(v_first))


def _set_cache_seen(cache: Any, seen_tokens: int) -> Any:
    if hasattr(cache, "seen_tokens"):
        cache.seen_tokens = int(seen_tokens)
    elif hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = int(seen_tokens)
    return cache


def _new_cache_like(
    prototype: Any,
    state: list[torch.Tensor],
    xpa: list[torch.Tensor],
    xpf: list[torch.Tensor],
    v_first: torch.Tensor,
    *,
    seen_tokens: int,
) -> Any:
    cache_type = type(prototype)
    try:
        return cache_type(
            state,
            xpa,
            xpf,
            v_first,
            seen_tokens=int(seen_tokens),
        )
    except TypeError:
        return _set_cache_seen(
            cache_type(state, xpa, xpf, v_first),
            seen_tokens,
        )


def _concat_caches(caches: Sequence[Any]) -> Any:
    if not caches or any(cache is None for cache in caches):
        raise ValueError("batched continuation requires initialized caches")
    values = [tuple(cache) for cache in caches]
    if any(len(parts) != 4 for parts in values):
        raise TypeError("RWKV HF cache must expose four recurrent components")
    layer_counts = {
        (len(parts[0]), len(parts[1]), len(parts[2]))
        for parts in values
    }
    if len(layer_counts) != 1:
        raise ValueError("RWKV HF caches have inconsistent layer counts")
    state = [torch.cat(items, dim=0) for items in zip(*(parts[0] for parts in values), strict=True)]
    xpa = [torch.cat(items, dim=0) for items in zip(*(parts[1] for parts in values), strict=True)]
    xpf = [torch.cat(items, dim=0) for items in zip(*(parts[2] for parts in values), strict=True)]
    v_first = torch.cat([parts[3] for parts in values], dim=0)
    return _new_cache_like(
        caches[0],
        state,
        xpa,
        xpf,
        v_first,
        seen_tokens=0,
    )


def _select_cache_row(cache: Any, row: int, seen_tokens: int) -> Any:
    index = torch.tensor([int(row)], dtype=torch.long)
    if hasattr(cache, "select_batch"):
        selected = cache.select_batch(index, inplace=False)
        return _set_cache_seen(selected, seen_tokens)
    state, xpa, xpf, v_first = tuple(cache)
    selected = _new_cache_like(
        cache,
        [value[int(row) : int(row) + 1].clone() for value in state],
        [value[int(row) : int(row) + 1].clone() for value in xpa],
        [value[int(row) : int(row) + 1].clone() for value in xpf],
        v_first[int(row) : int(row) + 1].clone(),
        seen_tokens=seen_tokens,
    )
    return selected


@dataclass(slots=True)
class HFRequestState:
    """One owner-independent recurrent row managed by the HF scheduler."""

    request_id: str
    token_ids: tuple[int, ...]
    offset: int = 0
    seen_tokens: int = 0
    logits: torch.Tensor | None = None
    cache: Any = None
    output_ids: list[int] = field(default_factory=list)
    admitted_at: float = field(default_factory=time.monotonic)
    first_service_at: float | None = None
    completed_at: float | None = None
    cancelled: bool = False

    @property
    def remaining(self) -> int:
        return len(self.token_ids) - self.offset


class HFStatePoolView:
    """Read-only pool counters expected by the shared Agent runtime."""

    def __init__(self, scheduler: "HFRecurrentScheduler", capacity: int) -> None:
        self.scheduler = scheduler
        self.capacity = int(capacity)
        self.max_batch_size = scheduler.config.max_batch_size
        self.device = str(scheduler.device)

    @property
    def allocated(self) -> int:
        return len(self.scheduler._requests)

    @property
    def free(self) -> int:
        return self.capacity - self.allocated

    @property
    def slab_bytes(self) -> int:
        return sum(_cache_bytes(request.cache) for request in self.scheduler._requests.values())

    @property
    def bytes_per_slot(self) -> int:
        allocated = self.allocated
        return self.slab_bytes // allocated if allocated else 0

    @property
    def workspace_bytes(self) -> int:
        return int(self.scheduler._last_workspace_bytes)


class HFRecurrentScheduler:
    """Batch independent RWKV recurrent caches through the HF native backend."""

    def __init__(
        self,
        model: Any,
        *,
        config: SchedulerConfig,
        device: str | torch.device = "cuda",
        capacity: int | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.capacity = int(capacity or config.max_queue_size)
        if self.capacity < config.max_batch_size:
            raise ValueError("capacity must be at least max_batch_size")
        self.pool = HFStatePoolView(self, self.capacity)
        self._requests: dict[str, HFRequestState] = {}
        self._model_lock = threading.RLock()
        self._metrics = Counter()
        self._shape_counts: Counter[tuple[int, int]] = Counter()
        self._last_workspace_bytes = 0
        self._peak_workspace_bytes = 0

    def request(self, request_id: str) -> HFRequestState:
        try:
            return self._requests[str(request_id)]
        except KeyError as exc:
            raise KeyError(f"unknown request_id: {request_id}") from exc

    def admit(self, request_id: str, token_ids: Sequence[int]) -> HFRequestState:
        clean_id = str(request_id or "").strip()
        values = tuple(int(token) for token in token_ids)
        if not clean_id:
            raise ValueError("request_id must not be empty")
        if not values:
            raise ValueError("token_ids must not be empty")
        if any(token < 0 for token in values):
            raise ValueError("token IDs must be non-negative")
        if len(values) > self.config.max_input_tokens:
            raise ValueError(
                f"input has {len(values)} tokens, limit is {self.config.max_input_tokens}"
            )
        with self._model_lock:
            if clean_id in self._requests:
                raise ValueError(f"duplicate request_id: {clean_id}")
            if len(self._requests) >= self.capacity:
                raise RuntimeError("HF recurrent state pool is full")
            request = HFRequestState(clean_id, values)
            self._requests[clean_id] = request
            self._metrics["admitted"] += 1
            return request

    def release(self, request_id: str) -> None:
        with self._model_lock:
            self._requests.pop(str(request_id))
            self._metrics["released"] += 1

    def export_state(self, request_id: str):
        del request_id
        raise RuntimeError(
            "exact snapshot is not enabled for the HF recurrent cache backend"
        )

    def import_state(self, request_id: str, manifest, tensors):
        del request_id, manifest, tensors
        raise RuntimeError(
            "exact restore is not enabled for the HF recurrent cache backend"
        )

    def cancel(self, request_id: str) -> None:
        with self._model_lock:
            self.request(request_id).cancelled = True
            self._metrics["cancelled"] += 1

    def fork(
        self,
        parent_request_id: str,
        child_request_ids: Sequence[str],
    ) -> list[HFRequestState]:
        child_ids = [str(value or "").strip() for value in child_request_ids]
        if not child_ids or any(not value for value in child_ids):
            raise ValueError("child_request_ids must be non-empty strings")
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("child_request_ids must be unique")
        with self._model_lock:
            parent = self.request(parent_request_id)
            if parent.remaining or parent.cache is None or parent.logits is None:
                raise RuntimeError("parent prefill must finish before fork")
            if parent_request_id in child_ids or any(
                child_id in self._requests for child_id in child_ids
            ):
                raise ValueError("fork child request already exists")
            if len(self._requests) + len(child_ids) > self.capacity:
                raise RuntimeError("HF recurrent state pool is full")
            now = time.monotonic()
            children = []
            try:
                for child_id in child_ids:
                    child = HFRequestState(
                        request_id=child_id,
                        token_ids=(),
                        seen_tokens=parent.seen_tokens,
                        logits=parent.logits.detach().clone(),
                        cache=parent.cache.clone(),
                        admitted_at=now,
                        first_service_at=now,
                        completed_at=now,
                    )
                    self._requests[child_id] = child
                    children.append(child)
            except Exception:
                for child in children:
                    self._requests.pop(child.request_id, None)
                raise
            self._metrics["fork_calls"] += 1
            self._metrics["forked_states"] += len(children)
            return children

    def _selected(self, request_ids: Sequence[str] | None) -> list[HFRequestState]:
        if request_ids is None:
            return list(self._requests.values())
        values = [str(value) for value in request_ids]
        if len(set(values)) != len(values):
            raise ValueError("duplicate request_id")
        return [self.request(value) for value in values]

    @staticmethod
    def _active(requests: Sequence[HFRequestState]) -> list[HFRequestState]:
        return [request for request in requests if request.remaining and not request.cancelled]

    @torch.inference_mode()
    def prefill(self, request_ids: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
        with self._model_lock:
            requests = self._selected(request_ids)
            while self._active(requests):
                self._prefill_round_locked(requests)
            return {
                request.request_id: request.logits
                for request in requests
                if request.logits is not None
            }

    @torch.inference_mode()
    def prefill_round(self, request_ids: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
        with self._model_lock:
            requests = self._selected(request_ids)
            self._prefill_round_locked(requests)
            self._metrics["prefill_rounds"] += 1
            return {
                request.request_id: request.logits
                for request in requests
                if request.logits is not None
            }

    def _prefill_round_locked(self, requests: Sequence[HFRequestState]) -> None:
        active = self._active(requests)
        if not active:
            return
        # A first-prompt batch has no cache; continuation batches do.  They use
        # separate calls so HF never receives a partially initialized cache.
        groups = (
            [request for request in active if request.cache is None],
            [request for request in active if request.cache is not None],
        )
        for group in groups:
            for start in range(0, len(group), self.config.max_batch_size):
                self._run_batch(group[start : start + self.config.max_batch_size])

    def _run_batch(self, requests: Sequence[HFRequestState]) -> None:
        if not requests:
            return
        chunks = [
            request.token_ids[
                request.offset : request.offset + self.config.prefill_chunk_size
            ]
            for request in requests
        ]
        lengths = [len(chunk) for chunk in chunks]
        if any(length < 1 for length in lengths):
            raise RuntimeError("scheduler selected an empty prefill chunk")
        width = max(lengths)
        tokens = torch.zeros((len(requests), width), dtype=torch.long, device=self.device)
        mask = torch.zeros_like(tokens)
        for row, chunk in enumerate(chunks):
            length = len(chunk)
            tokens[row, :length] = torch.tensor(chunk, dtype=torch.long, device=self.device)
            mask[row, :length] = 1
        cache = None
        single_row = len(requests) == 1
        if requests[0].cache is not None:
            if any(request.cache is None for request in requests):
                raise RuntimeError("cannot mix initialized and empty caches")
            # A recurrent B1 decode already owns a correctly shaped cache.
            # Concatenating one row and selecting it again copied the complete
            # ~62 MiB RWKV State twice per generated token.  Preserve the
            # cache object for B1; only multi-row batches need gather/scatter.
            cache = requests[0].cache if single_row else _concat_caches(
                [request.cache for request in requests]
            )
            self._last_workspace_bytes = 0 if single_row else _cache_bytes(cache)
            self._peak_workspace_bytes = max(
                self._peak_workspace_bytes,
                self._last_workspace_bytes,
            )
        else:
            self._last_workspace_bytes = 0
        started = time.perf_counter()
        output = self.model(
            input_ids=tokens,
            attention_mask=mask if len(set(lengths)) > 1 else None,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        elapsed = time.perf_counter() - started
        new_cache = output.past_key_values
        if new_cache is None:
            raise RuntimeError("HF recurrent backend returned no cache")
        logits = output.logits[:, -1, :]
        now = time.monotonic()
        for row, (request, length) in enumerate(zip(requests, lengths, strict=True)):
            request.offset += length
            request.seen_tokens += length
            request.logits = logits[row].detach()
            request.cache = (
                _set_cache_seen(new_cache, request.seen_tokens)
                if single_row
                else _select_cache_row(new_cache, row, request.seen_tokens)
            )
            if request.first_service_at is None:
                request.first_service_at = now
            if not request.remaining:
                request.completed_at = now
        self._shape_counts[(len(requests), width)] += 1
        self._metrics["forward_calls"] += 1
        self._metrics["forward_rows"] += len(requests)
        self._metrics["forward_tokens"] += sum(lengths)
        self._metrics["forward_time_us"] += int(elapsed * 1_000_000)
        self._metrics["max_batch_observed"] = max(
            self._metrics["max_batch_observed"],
            len(requests),
        )

    def install_continuations(
        self,
        rows: Sequence[tuple[str, Sequence[int]]],
    ) -> list[HFRequestState]:
        normalized = [
            (str(request_id or "").strip(), tuple(int(token) for token in tokens))
            for request_id, tokens in rows
        ]
        if not normalized:
            return []
        ids = [request_id for request_id, _tokens in normalized]
        if any(not request_id for request_id in ids):
            raise ValueError("request_id must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate request_id in continuations")
        if any(not values for _request_id, values in normalized):
            raise ValueError("continuation token_ids must not be empty")
        if any(token < 0 for _request_id, values in normalized for token in values):
            raise ValueError("token IDs must be non-negative")
        with self._model_lock:
            requests = self._selected(ids)
            by_id = dict(normalized)
            for request in requests:
                values = by_id[request.request_id]
                if request.cancelled:
                    raise RuntimeError(f"request {request.request_id} is cancelled")
                if request.remaining or request.cache is None:
                    raise RuntimeError(
                        f"request {request.request_id} has unfinished prefill"
                    )
                if request.seen_tokens + len(values) > self.config.max_input_tokens:
                    raise ValueError(
                        f"continuation for {request.request_id} would exceed context limit "
                        f"{self.config.max_input_tokens}"
                    )
            for request in requests:
                request.token_ids = by_id[request.request_id]
                request.offset = 0
                request.completed_at = None
            self._metrics["continuation_batches"] += 1
            self._metrics["continuations"] += len(requests)
            return requests

    @torch.inference_mode()
    def continue_many(
        self,
        rows: Sequence[tuple[str, Sequence[int]]],
    ) -> dict[str, torch.Tensor]:
        requests = self.install_continuations(rows)
        return self.prefill([request.request_id for request in requests]) if requests else {}

    @torch.inference_mode()
    def sample_next(self, request_ids: Sequence[str]) -> dict[str, int]:
        with self._model_lock:
            requests = self._selected(request_ids)
            for request in requests:
                if request.remaining or request.logits is None:
                    raise RuntimeError(
                        f"request {request.request_id} is not ready for decode"
                    )
            if not requests:
                return {}
            sampled = torch.argmax(
                torch.stack([request.logits for request in requests]),
                dim=-1,
            ).tolist()
            self._metrics["sample_calls"] += 1
            return {
                request.request_id: int(token)
                for request, token in zip(requests, sampled, strict=True)
            }

    @torch.inference_mode()
    def advance_tokens(self, tokens_by_request: dict[str, int]) -> None:
        if not tokens_by_request:
            return
        self.continue_many(
            [
                (request_id, (int(token),))
                for request_id, token in tokens_by_request.items()
            ]
        )
        self._metrics["decode_calls"] += 1
        self._metrics["decode_tokens"] += len(tokens_by_request)

    def metrics(self) -> dict[str, Any]:
        with self._model_lock:
            metrics = dict(self._metrics)
            metrics["forward_time_ms"] = round(
                metrics.get("forward_time_us", 0) / 1000.0,
                3,
            )
            metrics["shape_counts"] = {
                f"B{batch}T{tokens}": count
                for (batch, tokens), count in sorted(self._shape_counts.items())
            }
            metrics["pool"] = {
                "backend": "hf_recurrent",
                "capacity": self.pool.capacity,
                "allocated": self.pool.allocated,
                "free": self.pool.free,
                "bytes_per_slot": self.pool.bytes_per_slot,
                "slab_bytes": self.pool.slab_bytes,
                "workspace_bytes": self.pool.workspace_bytes,
                "peak_workspace_bytes": self._peak_workspace_bytes,
                "device": self.pool.device,
            }
            return metrics
