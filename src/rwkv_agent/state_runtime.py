"""Persistent recurrent-state lifecycle for bounded Agent branches.

This module deliberately sits beside ``ContinuousBatchEngine``.  Ordinary
completion requests remain short-lived jobs; state-native Agent turns retain
explicit scheduler requests between model/tool/model steps and release them as
a group when the turn finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Sequence
import uuid


@dataclass(slots=True)
class PersistentState:
    state_id: str
    owner_id: str
    parent_state_id: str | None
    branch: str
    created_at: float
    last_used_at: float


class PersistentStateRuntime:
    """Own persistent RWKV states on one Sidecar and one GPU state slab."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        scheduler: Any,
        context_limit: int,
        eos_token_id: int = 0,
        capacity: int = 8,
        ttl_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
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
        self._records: dict[str, PersistentState] = {}
        self._lock = threading.RLock()
        self._metrics = {
            "created": 0,
            "forked": 0,
            "continued": 0,
            "classified": 0,
            "released": 0,
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

        owner = self._clean_owner(owner_id)
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
                record = self._require(str(item.get("state_id") or ""), owner)
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
            try:
                self.scheduler.continue_many(continuations)
                output: list[dict[str, Any]] = []
                for record in records:
                    request = self.scheduler.request(record.state_id)
                    if request.logits is None:
                        raise RuntimeError("persistent state has no next-token logits")
                    scores = {
                        name: float(request.logits[token].item())
                        for name, token in normalized_labels.items()
                    }
                    if not all(math.isfinite(score) for score in scores.values()):
                        raise RuntimeError(
                            "persistent state produced non-finite "
                            "classification logits"
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

    @staticmethod
    def _clean_owner(owner_id: str) -> str:
        value = str(owner_id or "").strip()
        if not value:
            raise ValueError("owner_id must not be empty")
        if len(value) > 128:
            raise ValueError("owner_id is too long")
        return value

    @staticmethod
    def _new_state_id() -> str:
        return "state-" + uuid.uuid4().hex

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

    def _require(self, state_id: str, owner_id: str) -> PersistentState:
        clean_id = str(state_id or "").strip()
        try:
            record = self._records[clean_id]
        except KeyError as exc:
            raise KeyError(f"unknown state_id: {clean_id}") from exc
        if record.owner_id != owner_id:
            raise PermissionError("state owner mismatch")
        return record

    def _cleanup_expired_locked(self) -> list[str]:
        now = self._clock()
        expired = [
            state_id
            for state_id, record in self._records.items()
            if now - record.last_used_at >= self.ttl_seconds
        ]
        for state_id in expired:
            self._records.pop(state_id, None)
            try:
                self.scheduler.release(state_id)
            except KeyError:
                pass
        if expired:
            self._metrics["expired"] += len(expired)
            self._metrics["released"] += len(expired)
        return expired

    def _ensure_capacity_locked(self, additional: int) -> None:
        if additional < 1:
            return
        if len(self._records) + additional > self.capacity:
            raise RuntimeError(
                f"persistent state capacity exceeded: "
                f"{len(self._records)}+{additional}>{self.capacity}"
            )

    @staticmethod
    def _describe(record: PersistentState, seen_tokens: int) -> dict[str, Any]:
        return {
            "state_id": record.state_id,
            "owner_id": record.owner_id,
            "parent_state_id": record.parent_state_id,
            "branch": record.branch,
            "seen_tokens": int(seen_tokens),
        }

    def prefill(
        self,
        *,
        owner_id: str,
        prompt: str,
        branch: str = "root",
    ) -> dict[str, Any]:
        owner = self._clean_owner(owner_id)
        tokens = self._encode(prompt)
        with self._lock:
            self._cleanup_expired_locked()
            self._ensure_capacity_locked(1)
            state_id = self._new_state_id()
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
            self._records[state_id] = record
            self._metrics["created"] += 1
            request = self.scheduler.request(state_id)
            return self._describe(record, request.seen_tokens)

    def fork(
        self,
        *,
        owner_id: str,
        parent_state_id: str,
        branches: Sequence[str],
    ) -> list[dict[str, Any]]:
        owner = self._clean_owner(owner_id)
        labels = [str(value or "").strip() for value in branches]
        if not labels or any(not value for value in labels):
            raise ValueError("branches must contain non-empty labels")
        if len(set(labels)) != len(labels):
            raise ValueError("branch labels must be unique")
        with self._lock:
            self._cleanup_expired_locked()
            parent = self._require(parent_state_id, owner)
            self._ensure_capacity_locked(len(labels))
            state_ids = [self._new_state_id() for _label in labels]
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
                self._records[record.state_id] = record
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
    ) -> list[dict[str, Any]]:
        owner = self._clean_owner(owner_id)
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
                record = self._require(str(item.get("state_id") or ""), owner)
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

            try:
                self.scheduler.continue_many(continuations)
                results = self._decode_locked(
                    records,
                    stops=stop_values,
                    max_tokens=max_tokens,
                )
            except Exception:
                self._metrics["failed"] += 1
                raise
            now = self._clock()
            for record in records:
                record.last_used_at = now
            self._metrics["continued"] += len(records)
            return results

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
                token = int(sampled[state_id])
                if token == self.eos_token_id:
                    stop_reason[state_id] = "</s>"
                    finished.add(state_id)
                    continue
                output_ids[state_id].append(token)
                # Commit every non-EOS token, including the token which
                # completes a stop string.  A later tool observation must
                # resume after the exact text the branch generated.
                advance[state_id] = token
            if advance:
                self.scheduler.advance_tokens(advance)

            for state_id in active:
                if state_id in finished:
                    continue
                decoded = self.tokenizer.decode(output_ids[state_id])
                if "\ufffd" not in decoded:
                    output_text[state_id] = decoded
                    hits = [
                        (decoded.find(stop), stop)
                        for stop in stops
                        if stop in decoded
                    ]
                    if hits:
                        index, reason = min(hits)
                        output_text[state_id] = decoded[:index]
                        stop_reason[state_id] = reason
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
        owner = self._clean_owner(owner_id)
        ids = [str(value or "").strip() for value in state_ids]
        if not ids or any(not value for value in ids):
            raise ValueError("state_ids must contain non-empty values")
        if len(set(ids)) != len(ids):
            raise ValueError("state_ids must be unique")
        with self._lock:
            self._cleanup_expired_locked()
            records = [self._require(state_id, owner) for state_id in ids]
            for record in records:
                self.scheduler.release(record.state_id)
                self._records.pop(record.state_id, None)
            self._metrics["released"] += len(records)
            return {"released": len(records), "state_ids": ids}

    def health(self) -> dict[str, Any]:
        with self._lock:
            expired = self._cleanup_expired_locked()
            now = self._clock()
            return {
                "enabled": True,
                "capacity": self.capacity,
                "allocated": len(self._records),
                "free": self.capacity - len(self._records),
                "ttl_seconds": self.ttl_seconds,
                "expired_on_health": len(expired),
                "oldest_idle_seconds": round(
                    max(
                        (now - record.last_used_at for record in self._records.values()),
                        default=0.0,
                    ),
                    3,
                ),
                "metrics": dict(self._metrics),
            }

    def close(self) -> None:
        with self._lock:
            for state_id in list(self._records):
                try:
                    self.scheduler.release(state_id)
                except KeyError:
                    pass
                self._records.pop(state_id, None)
