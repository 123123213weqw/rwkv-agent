"""Production-oriented exact-chunk prefill and continuous decode scheduler."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from rwkv_runtime.decode import append_greedy_token
from .state_pool import AlbatrossStatePool, StateHandle


@dataclass(slots=True)
class RequestState:
    request_id: str
    handle: StateHandle
    token_ids: tuple[int, ...]
    offset: int = 0
    seen_tokens: int = 0
    logits: torch.Tensor | None = None
    output_ids: list[int] = field(default_factory=list)
    admitted_at: float = field(default_factory=time.monotonic)
    first_service_at: float | None = None
    completed_at: float | None = None
    cancelled: bool = False

    @property
    def remaining(self) -> int:
        return len(self.token_ids) - self.offset


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    prefill_chunk_size: int = 256
    max_batch_size: int = 8
    max_queue_size: int = 256
    max_input_tokens: int = 12288
    eos_token_id: int = 0

    def __post_init__(self) -> None:
        if self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.max_queue_size < self.max_batch_size:
            raise ValueError("max_queue_size must be at least max_batch_size")
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")


class AlbatrossChunkScheduler:
    """Exact-state scheduler for one loaded Albatross model instance.

    Prefill processes at most one fixed-size chunk per active request per
    scheduling round. Short tails are grouped only by exact length—RWKV state is
    never advanced through padding. Decode uses a T=1 active-row batch.
    """

    def __init__(
        self,
        model: Any,
        *,
        pool: AlbatrossStatePool,
        config: SchedulerConfig,
        token_device: str = "cpu",
    ) -> None:
        if pool.max_batch_size < config.max_batch_size:
            raise ValueError("state pool max batch is smaller than scheduler max batch")
        self.model = model
        self.pool = pool
        self.config = config
        self.token_device = token_device
        self._requests: dict[str, RequestState] = {}
        self._model_lock = pool.execution_lock
        self._metrics = Counter()
        self._shape_counts: Counter[tuple[int, int]] = Counter()

    def admit(
        self,
        request_id: str,
        token_ids: Sequence[int],
    ) -> RequestState:
        clean_id = str(request_id or "").strip()
        if not clean_id:
            raise ValueError("request_id must not be empty")
        values = tuple(int(token) for token in token_ids)
        if not values:
            raise ValueError("token_ids must not be empty")
        if len(values) > self.config.max_input_tokens:
            raise ValueError(
                f"input has {len(values)} tokens, limit is "
                f"{self.config.max_input_tokens}"
            )
        if any(token < 0 for token in values):
            raise ValueError("token IDs must be non-negative")
        with self._model_lock:
            if clean_id in self._requests:
                raise ValueError(f"duplicate request_id: {clean_id}")
            if len(self._requests) >= self.config.max_queue_size:
                raise RuntimeError("scheduler queue is full")
            handle = self.pool.allocate(clean_id)
            request = RequestState(clean_id, handle, values)
            self._requests[clean_id] = request
            self._metrics["admitted"] += 1
            return request

    def admit_many(
        self,
        rows: Iterable[tuple[str, Sequence[int]]],
    ) -> list[RequestState]:
        admitted = []
        try:
            for request_id, token_ids in rows:
                admitted.append(self.admit(request_id, token_ids))
        except Exception:
            for request in admitted:
                self.release(request.request_id)
            raise
        return admitted

    def fork(
        self,
        parent_request_id: str,
        child_request_ids: Sequence[str],
    ) -> list[RequestState]:
        """Fork one completed prefix state into independent child requests."""

        child_ids = [str(value or "").strip() for value in child_request_ids]
        if not child_ids or any(not value for value in child_ids):
            raise ValueError("child_request_ids must be non-empty strings")
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("child_request_ids must be unique")
        with self._model_lock:
            parent = self.request(parent_request_id)
            if parent.cancelled:
                raise RuntimeError(f"request {parent_request_id} is cancelled")
            if parent.remaining:
                raise RuntimeError("parent prefill must finish before fork")
            if parent.logits is None:
                raise RuntimeError("parent has no next-token logits")
            if parent_request_id in child_ids or any(
                child_id in self._requests for child_id in child_ids
            ):
                raise ValueError("fork child request already exists")
            if len(self._requests) + len(child_ids) > self.config.max_queue_size:
                raise RuntimeError("scheduler queue is full")

            handles = self.pool.clone_many(parent.handle, child_ids)
            now = time.monotonic()
            children: list[RequestState] = []
            try:
                for child_id, handle in zip(child_ids, handles, strict=True):
                    child = RequestState(
                        request_id=child_id,
                        handle=handle,
                        token_ids=(),
                        seen_tokens=parent.seen_tokens,
                        logits=parent.logits.detach().clone(),
                        admitted_at=now,
                        first_service_at=now,
                        completed_at=now,
                    )
                    self._requests[child_id] = child
                    children.append(child)
            except Exception:
                for child in children:
                    self._requests.pop(child.request_id, None)
                self.pool.release_many(handles)
                raise
            self._metrics["fork_calls"] += 1
            self._metrics["forked_states"] += len(children)
            return children

    def request(self, request_id: str) -> RequestState:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request_id: {request_id}") from exc

    def cancel(self, request_id: str) -> None:
        with self._model_lock:
            request = self.request(request_id)
            request.cancelled = True
            self._metrics["cancelled"] += 1

    def release(self, request_id: str) -> None:
        with self._model_lock:
            request = self._requests.pop(request_id)
            self.pool.release(request.handle)
            self._metrics["released"] += 1

    def export_state(
        self,
        request_id: str,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Copy one decode-ready recurrent row to a portable CPU tensor map."""

        with self._model_lock:
            request = self.request(request_id)
            if request.cancelled or request.remaining or request.logits is None:
                raise RuntimeError(f"request {request_id} is not snapshot-ready")
            state = self.pool.snapshot(request.handle)
            tensors: dict[str, torch.Tensor] = {
                "logits": request.logits.detach().to(device="cpu").contiguous(),
            }
            if self.pool.pipeline_parallel:
                group_lengths = []
                for group_index, group in enumerate(state):
                    group_lengths.append(len(group))
                    for item_index, value in enumerate(group):
                        tensors[f"state.{group_index}.{item_index}"] = (
                            value.detach().to(device="cpu").contiguous()
                        )
                layout = "pipeline"
            else:
                group_lengths = []
                for index, value in enumerate(state):
                    tensors[f"state.{index}"] = (
                        value.detach().to(device="cpu").contiguous()
                    )
                layout = "stacked"
            self._metrics["state_exports"] += 1
            return (
                {
                    "format": "rwkv-recurrent-state.v1",
                    "backend": "albatross",
                    "layout": layout,
                    "group_lengths": group_lengths,
                    "seen_tokens": request.seen_tokens,
                },
                tensors,
            )

    def import_state(
        self,
        request_id: str,
        manifest: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ) -> RequestState:
        """Install a validated exact snapshot into a newly allocated slab row."""

        clean_id = str(request_id or "").strip()
        if not clean_id:
            raise ValueError("request_id must not be empty")
        if manifest.get("format") != "rwkv-recurrent-state.v1" or manifest.get(
            "backend"
        ) != "albatross":
            raise ValueError("snapshot backend is incompatible with Albatross")
        seen_tokens = int(manifest.get("seen_tokens", -1))
        if seen_tokens < 0 or seen_tokens > self.config.max_input_tokens:
            raise ValueError("snapshot seen_tokens is outside the context limit")
        with self._model_lock:
            if clean_id in self._requests:
                raise ValueError(f"duplicate request_id: {clean_id}")
            if len(self._requests) >= self.config.max_queue_size:
                raise RuntimeError("scheduler queue is full")
            handle = self.pool.allocate(clean_id)
            try:
                expected = self.pool.snapshot(handle)
                incoming_state: list[Any]
                if self.pool.pipeline_parallel:
                    lengths = manifest.get("group_lengths")
                    expected_lengths = [len(group) for group in expected]
                    if manifest.get("layout") != "pipeline" or lengths != expected_lengths:
                        raise ValueError("snapshot pipeline layout mismatch")
                    incoming_state = []
                    expected_names = {"logits"}
                    for group_index, expected_group in enumerate(expected):
                        incoming_group = []
                        for item_index, target in enumerate(expected_group):
                            name = f"state.{group_index}.{item_index}"
                            expected_names.add(name)
                            incoming_group.append(
                                self._validated_snapshot_tensor(name, tensors, target)
                            )
                        incoming_state.append(incoming_group)
                else:
                    if manifest.get("layout") != "stacked" or manifest.get(
                        "group_lengths"
                    ) not in ([], None):
                        raise ValueError("snapshot state layout mismatch")
                    expected_names = {"logits"}
                    incoming_state = []
                    for index, target in enumerate(expected):
                        name = f"state.{index}"
                        expected_names.add(name)
                        incoming_state.append(
                            self._validated_snapshot_tensor(name, tensors, target)
                        )
                if set(tensors) != expected_names:
                    raise ValueError("snapshot contains unexpected or missing tensors")
                logits = tensors["logits"]
                if logits.ndim != 1 or logits.numel() < 2:
                    raise ValueError("snapshot logits must be a non-empty vocabulary vector")
                expected_vocab = getattr(self.model, "vocab_size", None)
                head_weight = getattr(getattr(self.model, "head", None), "weight", None)
                if expected_vocab is None and head_weight is not None:
                    expected_vocab = int(head_weight.shape[0])
                if expected_vocab is not None and logits.numel() != int(expected_vocab):
                    raise ValueError("snapshot logits vocabulary size mismatch")
                if head_weight is not None:
                    logits_device = head_weight.device
                elif self.pool.pipeline_parallel:
                    logits_device = expected[0][-1].device
                else:
                    logits_device = expected[0].device
                logits = logits.to(device=logits_device)
                self.pool.commit([handle], incoming_state)
                now = time.monotonic()
                request = RequestState(
                    request_id=clean_id,
                    handle=handle,
                    token_ids=(),
                    seen_tokens=seen_tokens,
                    logits=logits.contiguous(),
                    admitted_at=now,
                    first_service_at=now,
                    completed_at=now,
                )
                self._requests[clean_id] = request
                self._metrics["state_imports"] += 1
                return request
            except Exception:
                self.pool.release(handle)
                raise

    @staticmethod
    def _validated_snapshot_tensor(
        name: str,
        tensors: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        try:
            value = tensors[name]
        except KeyError as exc:
            raise ValueError(f"snapshot is missing tensor {name}") from exc
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"snapshot tensor {name} shape or dtype mismatch")
        return value.to(device=target.device).contiguous()

    @torch.inference_mode()
    def prefill(
        self,
        request_ids: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        with self._model_lock:
            requests = self._selected(request_ids)
            if not requests:
                return {}
            while active := self._active_prefill(requests):
                self._run_prefill_wave(active)
            self._mark_prefill_complete(requests)
            return self._available_logits(requests)

    @torch.inference_mode()
    def prefill_round(
        self,
        request_ids: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run at most one exact chunk for each selected request.

        This is the serving primitive used by the continuous batcher. It lets
        decode-ready rows make progress between long-prompt prefill quanta,
        instead of allowing one admission wave to monopolize the model until
        every prompt has been consumed.
        """

        with self._model_lock:
            requests = self._selected(request_ids)
            active = self._active_prefill(requests)
            if not active:
                return self._available_logits(requests)

            self._run_prefill_wave(active)
            self._mark_prefill_complete(requests)
            self._metrics["prefill_rounds"] += 1
            return self._available_logits(requests)

    @torch.inference_mode()
    def continue_tokens(
        self,
        request_id: str,
        token_ids: Sequence[int],
    ) -> torch.Tensor:
        """Vectorized continuation prefill into an existing recurrent state."""

        values = tuple(int(token) for token in token_ids)
        if not values:
            raise ValueError("continuation token_ids must not be empty")
        with self._model_lock:
            request = self.request(request_id)
            if request.cancelled:
                raise RuntimeError(f"request {request_id} is cancelled")
            if request.remaining:
                raise RuntimeError("initial prefill must finish before continuation")
            if request.seen_tokens + len(values) > self.config.max_input_tokens:
                raise ValueError(
                    f"continuation would exceed context limit "
                    f"{self.config.max_input_tokens}"
                )
            request.token_ids = values
            request.offset = 0
            request.completed_at = None
            self.prefill([request_id])
            assert request.logits is not None
            self._metrics["continuations"] += 1
            return request.logits

    @torch.inference_mode()
    def continue_many(
        self,
        rows: Sequence[tuple[str, Sequence[int]]],
    ) -> dict[str, torch.Tensor]:
        """Batch variable continuations into existing recurrent states.

        Continuations are installed atomically after validation and then use
        the scheduler's exact-length chunk waves.  This is the state-native
        equivalent of resuming multiple Agent branches with different tool
        observations without rebuilding their common prefix.
        """

        requests = self.install_continuations(rows)
        if not requests:
            return {}
        request_ids = [request.request_id for request in requests]
        return self.prefill(request_ids)

    def install_continuations(
        self,
        rows: Sequence[tuple[str, Sequence[int]]],
    ) -> list[RequestState]:
        """Atomically install continuation tokens without running a forward.

        The unified serving worker uses this primitive to put persistent-State
        rows into the same exact-length prefill rounds as newly admitted
        completion and classification rows. ``continue_many`` remains the
        synchronous convenience API and delegates to this method.
        """

        normalized = [
            (str(request_id or "").strip(), tuple(int(token) for token in tokens))
            for request_id, tokens in rows
        ]
        if not normalized:
            return []
        request_ids = [request_id for request_id, _tokens in normalized]
        if any(not request_id for request_id in request_ids):
            raise ValueError("request_id must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request_id in continuations")
        if any(not tokens for _request_id, tokens in normalized):
            raise ValueError("continuation token_ids must not be empty")
        if any(
            token < 0
            for _request_id, tokens in normalized
            for token in tokens
        ):
            raise ValueError("token IDs must be non-negative")

        with self._model_lock:
            requests = self._selected(request_ids)
            by_id = dict(normalized)
            for request in requests:
                values = by_id[request.request_id]
                if request.cancelled:
                    raise RuntimeError(
                        f"request {request.request_id} is cancelled"
                    )
                if request.remaining:
                    raise RuntimeError(
                        f"request {request.request_id} has unfinished prefill"
                    )
                if request.seen_tokens + len(values) > self.config.max_input_tokens:
                    raise ValueError(
                        f"continuation for {request.request_id} would exceed "
                        f"context limit {self.config.max_input_tokens}"
                    )

            for request in requests:
                request.token_ids = by_id[request.request_id]
                request.offset = 0
                request.completed_at = None
            self._metrics["continuation_batches"] += 1
            self._metrics["continuations"] += len(requests)
            return requests

    @torch.inference_mode()
    def greedy_decode(
        self,
        request_ids: Sequence[str],
        *,
        max_new_tokens: int | dict[str, int],
        commit_last_token: bool = True,
    ) -> dict[str, list[int]]:
        if isinstance(max_new_tokens, int):
            if max_new_tokens < 1:
                raise ValueError("max_new_tokens must be positive")
            budgets = {request_id: max_new_tokens for request_id in request_ids}
        else:
            budgets = {key: int(value) for key, value in max_new_tokens.items()}
            if any(value < 1 for value in budgets.values()):
                raise ValueError("decode budgets must be positive")
        with self._model_lock:
            requests = self._selected(request_ids)
            for request in requests:
                if request.remaining:
                    raise RuntimeError(
                        f"request {request.request_id} has unfinished prefill"
                    )
                if request.logits is None:
                    raise RuntimeError(
                        f"request {request.request_id} has no next-token logits"
                    )
                request.output_ids.clear()

            active = list(requests)
            while active:
                active_ids = [request.request_id for request in active]
                sampled = self.sample_next(active_ids)
                advance: dict[str, int] = {}
                next_active: list[RequestState] = []
                for request in active:
                    status = append_greedy_token(
                        request.output_ids,
                        sampled[request.request_id],
                        eos_token_id=self.config.eos_token_id,
                        max_tokens=budgets[request.request_id],
                    )
                    if not status.eos and (
                        commit_last_token or not status.budget_reached
                    ):
                        advance[request.request_id] = status.token
                    if not status.finished:
                        next_active.append(request)

                if advance:
                    self.advance_tokens(advance)
                active = next_active

            self._metrics["decoded_requests"] += len(requests)
            self._metrics["decoded_tokens"] += sum(
                len(request.output_ids) for request in requests
            )
            return {
                request.request_id: list(request.output_ids)
                for request in requests
            }

    @torch.inference_mode()
    def sample_next(self, request_ids: Sequence[str]) -> dict[str, int]:
        """Greedily sample one token without advancing recurrent state."""

        with self._model_lock:
            requests = self._selected(request_ids)
            if not requests:
                return {}
            for request in requests:
                if request.cancelled:
                    raise RuntimeError(
                        f"request {request.request_id} is cancelled"
                    )
                if request.remaining:
                    raise RuntimeError(
                        f"request {request.request_id} has unfinished prefill"
                    )
                if request.logits is None:
                    raise RuntimeError(
                        f"request {request.request_id} has no next-token logits"
                    )
            logits = torch.stack([request.logits for request in requests])
            sampled = torch.argmax(logits, dim=-1).tolist()
            self._metrics["sample_calls"] += 1
            return {
                request.request_id: int(token)
                for request, token in zip(requests, sampled, strict=True)
            }

    @torch.inference_mode()
    def advance_tokens(self, tokens_by_request: dict[str, int]) -> None:
        """Commit one supplied token for each selected active request."""

        if not tokens_by_request:
            return
        with self._model_lock:
            requests = self._selected(list(tokens_by_request))
            for batch in self._batches(requests):
                tokens = [
                    int(tokens_by_request[request.request_id])
                    for request in batch
                ]
                if any(token < 0 for token in tokens):
                    raise ValueError("token IDs must be non-negative")
                self._run_token_batch(batch, tokens)
            self._metrics["advanced_tokens"] += len(requests)

    def metrics(self) -> dict[str, Any]:
        return {
            **dict(self._metrics),
            "shape_counts": {
                f"B{batch}T{tokens}": count
                for (batch, tokens), count in sorted(self._shape_counts.items())
            },
            "pool": {
                "capacity": self.pool.capacity,
                "allocated": self.pool.allocated,
                "free": self.pool.free,
                "bytes_per_slot": self.pool.bytes_per_slot,
                "slab_bytes": self.pool.slab_bytes,
                "workspace_bytes": self.pool.workspace_bytes,
            },
        }

    def _selected(
        self,
        request_ids: Sequence[str] | None,
    ) -> list[RequestState]:
        if request_ids is None:
            return list(self._requests.values())
        seen = set()
        output = []
        for request_id in request_ids:
            if request_id in seen:
                raise ValueError(f"duplicate request_id in batch: {request_id}")
            seen.add(request_id)
            output.append(self.request(request_id))
        return output

    def _batches(
        self,
        requests: list[RequestState],
    ) -> Iterable[list[RequestState]]:
        limit = self.config.max_batch_size
        for start in range(0, len(requests), limit):
            yield requests[start : start + limit]

    @staticmethod
    def _active_prefill(
        requests: list[RequestState],
    ) -> list[RequestState]:
        return [
            request
            for request in requests
            if not request.cancelled and request.remaining > 0
        ]

    def _run_prefill_wave(self, requests: list[RequestState]) -> None:
        # Preserve absolute quantum boundaries across persistent-state
        # continuations. A continuation can begin after a short tail; feeding a
        # fresh full quantum from that unaligned recurrent position is not
        # numerically equivalent for the Native vector prefill cache.
        groups: dict[int, list[RequestState]] = defaultdict(list)
        for request in requests:
            groups[self._next_prefill_length(request)].append(request)
        for length in sorted(groups, reverse=True):
            for batch in self._batches(groups[length]):
                self._run_exact_chunk(batch, length)

    @staticmethod
    def _mark_prefill_complete(requests: list[RequestState]) -> None:
        now = time.monotonic()
        for request in requests:
            if request.remaining == 0 and request.completed_at is None:
                request.completed_at = now

    @staticmethod
    def _available_logits(
        requests: list[RequestState],
    ) -> dict[str, torch.Tensor]:
        return {
            request.request_id: request.logits
            for request in requests
            if request.logits is not None
        }

    def _next_prefill_length(self, request: RequestState) -> int:
        if request.remaining <= 0:
            raise ValueError("request has no remaining prefill tokens")
        quantum = self.config.prefill_chunk_size
        offset = request.seen_tokens % quantum
        until_boundary = quantum - offset if offset else quantum
        return min(request.remaining, until_boundary)

    def _token_tensor(
        self,
        rows: list[list[int]] | list[int],
    ) -> torch.Tensor:
        return torch.tensor(
            rows,
            dtype=torch.long,
            device=self.token_device,
        )

    def _run_exact_chunk(
        self,
        requests: list[RequestState],
        length: int,
    ) -> None:
        if not requests or length < 1:
            raise ValueError("exact chunk batch must be non-empty")
        rows = []
        for request in requests:
            end = request.offset + length
            if end > len(request.token_ids):
                raise ValueError("chunk exceeds request token count")
            rows.append(list(request.token_ids[request.offset:end]))
        handles = [request.handle for request in requests]
        state, borrowed = self.pool.checkout(handles)
        tokens = self._token_tensor(rows)
        logits = self.model.forward(tokens, state)
        self.pool.checkin(handles, state, borrowed=borrowed)
        now = time.monotonic()
        for row, request in enumerate(requests):
            request.offset += length
            request.seen_tokens += length
            request.logits = logits[row].detach().clone()
            if request.first_service_at is None:
                request.first_service_at = now
        self._metrics["prefill_calls"] += 1
        self._metrics["prefill_tokens"] += len(requests) * length
        self._shape_counts[(len(requests), length)] += 1

    def _run_token_batch(
        self,
        requests: list[RequestState],
        tokens: list[int],
    ) -> None:
        handles = [request.handle for request in requests]
        state, borrowed = self.pool.checkout(handles)
        token_tensor = self._token_tensor(tokens).view(-1, 1)
        logits = self.model.forward(token_tensor, state)
        self.pool.checkin(handles, state, borrowed=borrowed)
        for row, request in enumerate(requests):
            request.seen_tokens += 1
            request.logits = logits[row].detach().clone()
        self._metrics["decode_calls"] += 1
        self._shape_counts[(len(requests), 1)] += 1
