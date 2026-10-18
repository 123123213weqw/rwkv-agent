"""Persistent recurrent-state lifecycle for bounded Agent branches.

This module deliberately sits beside ``ContinuousBatchEngine``.  Ordinary
completion requests remain short-lived jobs; state-native Agent turns retain
explicit scheduler requests between model/tool/model steps and release them as
a group when the turn finishes.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Callable, Sequence

from rwkv_runtime.classification import finite_label_scores
from rwkv_runtime.decode import append_greedy_token, decode_text_stops
from rwkv_runtime.protocols import SchedulerProtocol, TokenizerProtocol
from rwkv_runtime.state_snapshot import (
    DEFAULT_MAX_SNAPSHOT_BYTES,
    decode_state_snapshot,
    encode_state_snapshot,
)

from .persistent_state import PersistentState, PersistentStateRegistry
from .state_batching import StateContinuationItem


class PersistentStateRuntime:
    """Own persistent RWKV states on one Sidecar and one GPU state slab."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerProtocol,
        scheduler: SchedulerProtocol,
        context_limit: int,
        eos_token_id: int = 0,
        capacity: int = 8,
        ttl_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        decode_engine: Any | None = None,
        max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
    ) -> None:
        if context_limit < 1:
            raise ValueError("context_limit must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.context_limit = int(context_limit)
        self.eos_token_id = int(eos_token_id)
        self.capacity = int(capacity)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self.decode_engine = decode_engine
        self.max_snapshot_bytes = int(max_snapshot_bytes)
        if self.max_snapshot_bytes < 1:
            raise ValueError("max_snapshot_bytes must be positive")
        self.registry = PersistentStateRegistry(
            scheduler=scheduler,
            capacity=self.capacity,
            ttl_seconds=self.ttl_seconds,
            clock=clock,
        )
        self._lock = threading.RLock()
        self._busy: set[str] = set()
        self._metrics = {
            "created": 0,
            "batch_prefill_calls": 0,
            "batch_prefill_states": 0,
            "forked": 0,
            "continued": 0,
            "classified": 0,
            "released": 0,
            "snapshots": 0,
            "restores": 0,
            "snapshot_bytes": 0,
            "expired": 0,
            "failed": 0,
        }

    def classify_many(
        self,
        *,
        owner_id: str,
        items: Sequence[dict[str, Any]],
        labels: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Append inputs to persistent leaves and return exact next-token logits."""

        owner = self.registry.clean_owner(owner_id)
        normalized_labels = {str(name): int(token) for name, token in labels.items()}
        if not items:
            raise ValueError("items must not be empty")
        if len(normalized_labels) < 2 or any(
            token < 0 for token in normalized_labels.values()
        ):
            raise ValueError("classification requires at least two token labels")
        with self._lock:
            self._cleanup_expired_locked()
            records: list[PersistentState] = []
            continuations: list[tuple[str, tuple[int, ...]]] = []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("state classification item must be an object")
                record = self.registry.require(
                    str(item.get("state_id") or ""),
                    owner,
                )
                tokens = self._encode(str(item.get("input") or ""))
                request = self.scheduler.request(record.state_id)
                if request.seen_tokens + len(tokens) > self.context_limit:
                    raise ValueError(
                        f"state {record.state_id} classification would exceed "
                        f"context limit {self.context_limit}"
                    )
                records.append(record)
                continuations.append((record.state_id, tokens))
            state_ids = [record.state_id for record in records]
            if len(set(state_ids)) != len(state_ids):
                raise ValueError("duplicate state_id in batch")
            self._ensure_available(records)
            try:
                self.scheduler.continue_many(continuations)
                output: list[dict[str, Any]] = []
                for record in records:
                    request = self.scheduler.request(record.state_id)
                    if request.logits is None:
                        raise RuntimeError("persistent state has no next-token logits")
                    scores = finite_label_scores(
                        request.logits,
                        normalized_labels,
                        error_message=(
                            "persistent state produced non-finite "
                            "classification logits"
                        ),
                    )
                    output.append(
                        {
                            "state_id": record.state_id,
                            "branch": record.branch,
                            "scores": scores,
                            "seen_tokens": request.seen_tokens,
                        }
                    )
            except Exception:
                self._metrics["failed"] += 1
                raise
            now = self._clock()
            for record in records:
                record.last_used_at = now
            self._metrics["classified"] += len(records)
            return output

    def _encode(self, text: str) -> tuple[int, ...]:
        value = str(text or "")
        if not value:
            raise ValueError("state input must not be empty")
        tokens = tuple(int(token) for token in self.tokenizer.encode(value))
        if not tokens:
            raise ValueError("state input encoded to no tokens")
        if len(tokens) > self.context_limit:
            raise ValueError(
                f"state input has {len(tokens)} tokens, limit is "
                f"{self.context_limit}"
            )
        return tokens

    def _cleanup_expired_locked(self) -> list[str]:
        expired = self.registry.cleanup_expired(exclude=self._busy)
        if expired:
            self._metrics["expired"] += len(expired)
            self._metrics["released"] += len(expired)
        return expired

    @staticmethod
    def _describe(record: PersistentState, seen_tokens: int) -> dict[str, Any]:
        return {
            "state_id": record.state_id,
            "owner_id": record.owner_id,
            "parent_state_id": record.parent_state_id,
            "branch": record.branch,
            "seen_tokens": int(seen_tokens),
        }

    def _ensure_available(self, records: Sequence[PersistentState]) -> None:
        busy = [
            record.state_id
            for record in records
            if record.state_id in self._busy
        ]
        if busy:
            raise RuntimeError(f"persistent states are busy: {','.join(busy)}")

    def has_state(
        self,
        *,
        owner_id: str,
        state_id: str,
        touch: bool = False,
    ) -> bool:
        """Check a State under owner isolation, optionally refreshing its TTL."""

        owner = self.registry.clean_owner(owner_id)
        with self._lock:
            self._cleanup_expired_locked()
            try:
                record = self.registry.require(state_id, owner)
            except KeyError:
                return False
            if touch:
                record.last_used_at = self._clock()
            return True

    def snapshot(
        self,
        *,
        owner_id: str,
        state_id: str,
    ) -> dict[str, Any]:
        """Create a safe CPU snapshot while retaining the hot source State."""

        owner = self.registry.clean_owner(owner_id)
        with self._lock:
            self._cleanup_expired_locked()
            record = self.registry.require(state_id, owner)
            self._ensure_available([record])
            try:
                manifest, tensors = self.scheduler.export_state(record.state_id)
                manifest = {
                    **manifest,
                    "owner_id": owner,
                    "source_state_id": record.state_id,
                    "branch": record.branch,
                    "parent_state_id": record.parent_state_id,
                }
                payload = encode_state_snapshot(manifest=manifest, tensors=tensors)
                if len(payload) > self.max_snapshot_bytes:
                    raise ValueError("snapshot exceeds configured byte limit")
            except Exception:
                self._metrics["failed"] += 1
                raise
            record.last_used_at = self._clock()
            checksum = hashlib.sha256(payload).hexdigest()
            self._metrics["snapshots"] += 1
            self._metrics["snapshot_bytes"] += len(payload)
            return {
                "payload": payload,
                "checksum": f"sha256:{checksum}",
                "size_bytes": len(payload),
                "seen_tokens": int(manifest["seen_tokens"]),
                "branch": record.branch,
            }

    def restore(
        self,
        *,
        owner_id: str,
        payload: bytes,
        branch: str | None = None,
    ) -> dict[str, Any]:
        """Restore an exact snapshot as a fresh owner-bound State identity."""

        owner = self.registry.clean_owner(owner_id)
        with self._lock:
            self._cleanup_expired_locked()
            self.registry.ensure_capacity(1)
            try:
                manifest, tensors = decode_state_snapshot(
                    payload,
                    max_bytes=self.max_snapshot_bytes,
                )
                if manifest.get("owner_id") != owner:
                    raise PermissionError("snapshot owner mismatch")
                source_state_id = str(manifest.get("source_state_id") or "")
                if not source_state_id:
                    raise ValueError("snapshot source_state_id is missing")
                try:
                    self.registry.require(source_state_id, owner)
                except KeyError:
                    pass
                else:
                    raise RuntimeError(
                        "snapshot source remains live; release it before restore"
                    )
                state_id = self.registry.new_state_id()
                self.scheduler.import_state(state_id, manifest, tensors)
            except Exception:
                self._metrics["failed"] += 1
                raise
            now = self._clock()
            restored_branch = str(
                branch if branch is not None else manifest.get("branch") or "restored"
            )[:80]
            record = PersistentState(
                state_id=state_id,
                owner_id=owner,
                parent_state_id=None,
                branch=restored_branch,
                created_at=now,
                last_used_at=now,
            )
            try:
                self.registry.add(record)
            except Exception:
                self.scheduler.release(state_id)
                raise
            self._metrics["restores"] += 1
            request = self.scheduler.request(state_id)
            return self._describe(record, request.seen_tokens)

    def prefill(
        self,
        *,
        owner_id: str,
        prompt: str,
        branch: str = "root",
    ) -> dict[str, Any]:
        owner = self.registry.clean_owner(owner_id)
        tokens = self._encode(prompt)
        with self._lock:
            self._cleanup_expired_locked()
            self.registry.ensure_capacity(1)
            state_id = self.registry.new_state_id()
            now = self._clock()
            try:
                self.scheduler.admit(state_id, tokens)
                self.scheduler.prefill([state_id])
            except Exception:
                try:
                    self.scheduler.release(state_id)
                except KeyError:
                    pass
                self._metrics["failed"] += 1
                raise
            record = PersistentState(
                state_id=state_id,
                owner_id=owner,
                parent_state_id=None,
                branch=str(branch or "root")[:80],
                created_at=now,
                last_used_at=now,
            )
            self.registry.add(record)
            self._metrics["created"] += 1
            request = self.scheduler.request(state_id)
            return self._describe(record, request.seen_tokens)

    def prefill_many(
        self,
        *,
        items: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create independently owned States with one vectorized prefill.

        Owner isolation remains per row; only model weights and the scheduler
        call are shared.  This is intentionally different from Root Fork.
        """

        if not items:
            raise ValueError("batch prefill items must not be empty")
        normalized: list[tuple[str, tuple[int, ...], str]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("batch prefill item must be an object")
            normalized.append(
                (
                    self.registry.clean_owner(str(item.get("owner_id") or "")),
                    self._encode(str(item.get("prompt") or "")),
                    str(item.get("branch") or "root")[:80],
                )
            )
        with self._lock:
            self._cleanup_expired_locked()
            self.registry.ensure_capacity(len(normalized))
            state_ids = [self.registry.new_state_id() for _item in normalized]
            admitted: list[str] = []
            try:
                for state_id, (_owner, tokens, _branch) in zip(
                    state_ids, normalized, strict=True
                ):
                    self.scheduler.admit(state_id, tokens)
                    admitted.append(state_id)
                self.scheduler.prefill(state_ids)
            except Exception:
                for state_id in admitted:
                    try:
                        self.scheduler.release(state_id)
                    except KeyError:
                        pass
                self._metrics["failed"] += 1
                raise
            now = self._clock()
            records = [
                PersistentState(
                    state_id=state_id,
                    owner_id=owner,
                    parent_state_id=None,
                    branch=branch,
                    created_at=now,
                    last_used_at=now,
                )
                for state_id, (owner, _tokens, branch) in zip(
                    state_ids, normalized, strict=True
                )
            ]
            for record in records:
                self.registry.add(record)
            self._metrics["created"] += len(records)
            self._metrics["batch_prefill_calls"] += 1
            self._metrics["batch_prefill_states"] += len(records)
            return [
                self._describe(
                    record,
                    self.scheduler.request(record.state_id).seen_tokens,
                )
                for record in records
            ]

    def fork(
        self,
        *,
        owner_id: str,
        parent_state_id: str,
        branches: Sequence[str],
    ) -> list[dict[str, Any]]:
        owner = self.registry.clean_owner(owner_id)
        labels = [str(value or "").strip() for value in branches]
        if not labels or any(not value for value in labels):
            raise ValueError("branches must contain non-empty labels")
        if len(set(labels)) != len(labels):
            raise ValueError("branch labels must be unique")
        with self._lock:
            self._cleanup_expired_locked()
            parent = self.registry.require(parent_state_id, owner)
            self._ensure_available([parent])
            self.registry.ensure_capacity(len(labels))
            state_ids = [
                self.registry.new_state_id()
                for _label in labels
            ]
            try:
                children = self.scheduler.fork(parent.state_id, state_ids)
            except Exception:
                self._metrics["failed"] += 1
                raise
            now = self._clock()
            output = []
            for label, child in zip(labels, children, strict=True):
                record = PersistentState(
                    state_id=child.request_id,
                    owner_id=owner,
                    parent_state_id=parent.state_id,
                    branch=label[:80],
                    created_at=now,
                    last_used_at=now,
                )
                self.registry.add(record)
                output.append(self._describe(record, child.seen_tokens))
            parent.last_used_at = now
            self._metrics["forked"] += len(output)
            return output

    def continue_many(
        self,
        *,
        owner_id: str,
        items: Sequence[dict[str, Any]],
        stops: Sequence[str],
        max_tokens: int,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        owner = self.registry.clean_owner(owner_id)
        if not items:
            raise ValueError("items must not be empty")
        if max_tokens < 1 or max_tokens > 1024:
            raise ValueError("max_tokens out of range")
        stop_values = tuple(str(value) for value in stops if str(value))
        with self._lock:
            self._cleanup_expired_locked()
            records: list[PersistentState] = []
            continuations: list[tuple[str, tuple[int, ...]]] = []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("state continuation item must be an object")
                record = self.registry.require(
                    str(item.get("state_id") or ""),
                    owner,
                )
                tokens = self._encode(str(item.get("input") or ""))
                request = self.scheduler.request(record.state_id)
                if request.seen_tokens + len(tokens) + max_tokens > self.context_limit:
                    raise ValueError(
                        f"state {record.state_id} continuation plus decode "
                        f"would exceed context limit {self.context_limit}"
                    )
                records.append(record)
                continuations.append((record.state_id, tokens))
            state_ids = [record.state_id for record in records]
            if len(set(state_ids)) != len(state_ids):
                raise ValueError("duplicate state_id in batch")
            self._ensure_available(records)
            self._busy.update(state_ids)

        succeeded = False
        try:
            if self.decode_engine is not None:
                results = self.decode_engine.continue_many(
                    [
                        StateContinuationItem(
                            state_id=record.state_id,
                            branch=record.branch,
                            token_ids=tokens,
                        )
                        for record, (_state_id, tokens) in zip(
                            records,
                            continuations,
                            strict=True,
                        )
                    ],
                    stops=stop_values,
                    max_tokens=max_tokens,
                    event_sink=event_sink,
                )
            else:
                self.scheduler.continue_many(continuations)
                results = self._decode_locked(
                    records,
                    stops=stop_values,
                    max_tokens=max_tokens,
                )
            succeeded = True
            return results
        except Exception:
            with self._lock:
                self._metrics["failed"] += 1
            raise
        finally:
            with self._lock:
                self._busy.difference_update(state_ids)
                if succeeded:
                    now = self._clock()
                    for record in records:
                        record.last_used_at = now
                    self._metrics["continued"] += len(records)

    def _decode_locked(
        self,
        records: Sequence[PersistentState],
        *,
        stops: tuple[str, ...],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        by_id = {record.state_id: record for record in records}
        active = list(by_id)
        output_ids = {state_id: [] for state_id in active}
        output_text = {state_id: "" for state_id in active}
        stop_reason = {state_id: "" for state_id in active}

        while active:
            sampled = self.scheduler.sample_next(active)
            advance: dict[str, int] = {}
            finished: set[str] = set()
            for state_id in active:
                token_status = append_greedy_token(
                    output_ids[state_id],
                    sampled[state_id],
                    eos_token_id=self.eos_token_id,
                    max_tokens=max_tokens,
                )
                if token_status.eos:
                    stop_reason[state_id] = "</s>"
                    finished.add(state_id)
                    continue
                # Commit every non-EOS token, including the token which
                # completes a stop string.  A later tool observation must
                # resume after the exact text the branch generated.
                advance[state_id] = token_status.token
            if advance:
                self.scheduler.advance_tokens(advance)

            for state_id in active:
                if state_id in finished:
                    continue
                decoded = decode_text_stops(
                    self.tokenizer,
                    output_ids[state_id],
                    previous_text=output_text[state_id],
                    stops=stops,
                )
                output_text[state_id] = decoded.text
                if decoded.stop_reason:
                    stop_reason[state_id] = decoded.stop_reason
                    finished.add(state_id)
                    continue
                if len(output_ids[state_id]) >= max_tokens:
                    stop_reason[state_id] = "max_tokens"
                    finished.add(state_id)
            active = [state_id for state_id in active if state_id not in finished]

        output = []
        for record in records:
            request = self.scheduler.request(record.state_id)
            output.append(
                {
                    "state_id": record.state_id,
                    "branch": record.branch,
                    "text": output_text[record.state_id],
                    "token_ids": output_ids[record.state_id],
                    "stop_reason": stop_reason[record.state_id],
                    "seen_tokens": request.seen_tokens,
                }
            )
        return output

    def release(
        self,
        *,
        owner_id: str,
        state_ids: Sequence[str],
    ) -> dict[str, Any]:
        owner = self.registry.clean_owner(owner_id)
        ids = [str(value or "").strip() for value in state_ids]
        if not ids or any(not value for value in ids):
            raise ValueError("state_ids must contain non-empty values")
        if len(set(ids)) != len(ids):
            raise ValueError("state_ids must be unique")
        with self._lock:
            self._cleanup_expired_locked()
            records = [
                self.registry.require(state_id, owner)
                for state_id in ids
            ]
            self._ensure_available(records)
            self.registry.release_records(records)
            self._metrics["released"] += len(records)
            return {"released": len(records), "state_ids": ids}

    def health(self) -> dict[str, Any]:
        with self._lock:
            expired = self._cleanup_expired_locked()
            return {
                "enabled": True,
                "capacity": self.capacity,
                "allocated": self.registry.allocated,
                "free": self.registry.free,
                "ttl_seconds": self.ttl_seconds,
                "expired_on_health": len(expired),
                "oldest_idle_seconds": round(
                    self.registry.oldest_idle_seconds(),
                    3,
                ),
                "busy": len(self._busy),
                "batching": (
                    (
                        self.decode_engine.state_health()
                        if hasattr(self.decode_engine, "state_health")
                        else self.decode_engine.health()
                    )
                    if self.decode_engine is not None
                    else {"enabled": False}
                ),
                "metrics": dict(self._metrics),
            }

    def close(self) -> None:
        with self._lock:
            if self._busy:
                raise RuntimeError("cannot close persistent states while busy")
            self.registry.clear()
