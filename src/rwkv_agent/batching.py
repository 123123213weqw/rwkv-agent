"""One ready queue for ephemeral and persistent-State RWKV inference."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rwkv_runtime.classification import finite_label_scores
from rwkv_runtime.decode import append_greedy_token, decode_text_stops
from rwkv_runtime.protocols import SchedulerProtocol, TokenizerProtocol

from .state_batching import StateContinuationItem


@dataclass(slots=True)
class InferenceJob:
    job_id: str
    kind: str
    prompt: str = ""
    prefix_token_ids: tuple[int, ...] = ()
    state_items: tuple[StateContinuationItem, ...] = ()
    stops: tuple[str, ...] = ()
    max_tokens: int = 0
    labels: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    admitted_at: float | None = None
    future: Future[Any] = field(default_factory=Future)
    cancelled: threading.Event = field(default_factory=threading.Event)
    state_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def persistent(self) -> bool:
        return self.kind == "state_continuation"


@dataclass(slots=True)
class _ActiveRow:
    request_id: str
    job: InferenceJob
    branch: str = ""
    output_ids: list[int] = field(default_factory=list)
    output_text: str = ""

    @property
    def persistent(self) -> bool:
        return self.job.persistent


class ContinuousBatchEngine:
    """Advance all ready inference classes in one scheduler thread.

    Ordinary completion/classification jobs and persistent-State continuation
    jobs enter the same bounded FIFO queue. Their exact-length prefill rows and
    ready decode rows are advanced together, so two independent HTTP paths can
    no longer alternate separate B1/B2 model calls behind the scheduler lock.

    Persistent states are externally owned and are therefore never released by
    this engine. Ephemeral completion/classification slots are always released
    on success, failure, cancellation, or shutdown.
    """

    def __init__(
        self,
        *,
        tokenizer: TokenizerProtocol,
        scheduler: SchedulerProtocol,
        context_limit: int,
        eos_token_id: int = 0,
        batch_window_ms: float = 4.0,
        max_waiting_jobs: int = 256,
        request_timeout_seconds: float = 300.0,
        max_state_rows: int = 8,
        thread_name: str = "rwkv-unified-batcher",
    ) -> None:
        if context_limit < 1:
            raise ValueError("context_limit must be positive")
        if batch_window_ms < 0:
            raise ValueError("batch_window_ms must not be negative")
        if max_waiting_jobs < 1:
            raise ValueError("max_waiting_jobs must be positive")
        if max_state_rows < 1:
            raise ValueError("max_state_rows must be positive")
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.context_limit = int(context_limit)
        self.eos_token_id = int(eos_token_id)
        self.batch_window_ms = float(batch_window_ms)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_state_rows = int(max_state_rows)
        self._queue: queue.Queue[InferenceJob] = queue.Queue(max_waiting_jobs)
        self._closing = threading.Event()
        self._state_lock = threading.Lock()
        self._metrics = Counter()
        self._worker_error = ""
        self._waiting = 0
        self._prefilling = 0
        self._decoding = 0
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    def complete(
        self,
        prompt: str,
        *,
        stops: Sequence[str],
        max_tokens: int,
        prefix_token_ids: Sequence[int] = (),
    ) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return self._submit(
            InferenceJob(
                job_id=uuid.uuid4().hex,
                kind="completion",
                prompt=prompt,
                prefix_token_ids=tuple(
                    int(value) for value in prefix_token_ids
                ),
                stops=tuple(str(value) for value in stops if value),
                max_tokens=int(max_tokens),
            )
        )

    def classify(
        self,
        prompt: str,
        *,
        labels: Mapping[str, int],
    ) -> dict[str, Any]:
        normalized = {str(name): int(token) for name, token in labels.items()}
        if not prompt:
            raise ValueError("prompt must not be empty")
        if len(normalized) < 2 or any(token < 0 for token in normalized.values()):
            raise ValueError("classification requires at least two token labels")
        return self._submit(
            InferenceJob(
                job_id=uuid.uuid4().hex,
                kind="classification",
                prompt=prompt,
                labels=normalized,
            )
        )

    def continue_many(
        self,
        items: Sequence[StateContinuationItem],
        *,
        stops: Sequence[str],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        normalized = tuple(items)
        if not normalized:
            raise ValueError("state continuation items must not be empty")
        if len(normalized) > self.max_state_rows:
            raise ValueError(
                f"state continuation rows exceed batch limit "
                f"{self.max_state_rows}"
            )
        state_ids = [item.state_id for item in normalized]
        if any(not state_id for state_id in state_ids):
            raise ValueError("state_id must not be empty")
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("duplicate state_id in continuation job")
        if any(not item.token_ids for item in normalized):
            raise ValueError("continuation token_ids must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return self._submit(
            InferenceJob(
                job_id=uuid.uuid4().hex,
                kind="state_continuation",
                state_items=normalized,
                stops=tuple(str(value) for value in stops if str(value)),
                max_tokens=int(max_tokens),
            )
        )

    def close(self, *, timeout: float = 10.0) -> None:
        self._closing.set()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            raise TimeoutError("unified batch worker did not stop")

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            state = {
                "waiting": self._waiting,
                "prefilling": self._prefilling,
                "decoding": self._decoding,
            }
            metrics = dict(self._metrics)
        return {
            "mode": "unified_ready_queue",
            "worker_alive": self._thread.is_alive(),
            "worker_error": self._worker_error,
            "batch_window_ms": self.batch_window_ms,
            "request_timeout_seconds": self.request_timeout_seconds,
            "queue_capacity": self._queue.maxsize,
            "max_state_rows": self.max_state_rows,
            **state,
            "metrics": metrics,
            "scheduler": self.scheduler.metrics(),
        }

    def state_health(self) -> dict[str, Any]:
        health = self.health()
        metrics = dict(health["metrics"])
        return {
            "enabled": True,
            "mode": "shared_unified_ready_queue",
            "worker_alive": health["worker_alive"],
            "worker_error": health["worker_error"],
            "batch_window_ms": health["batch_window_ms"],
            "max_batch_size": self.max_state_rows,
            "waiting": health["waiting"],
            "active_rows": health["prefilling"] + health["decoding"],
            "metrics": {
                key: value
                for key, value in metrics.items()
                if "state" in key
            },
        }

    def _submit(self, job: InferenceJob) -> Any:
        if self._closing.is_set():
            raise RuntimeError("unified batch engine is closed")
        try:
            self._queue.put(job, timeout=1.0)
        except queue.Full as exc:
            with self._state_lock:
                self._metrics["rejected_queue_full"] += 1
            raise RuntimeError("inference queue is full") from exc
        with self._state_lock:
            self._metrics["submitted"] += 1
            self._metrics[f"submitted_{job.kind}"] += 1
            if job.persistent:
                self._metrics["submitted_state_rows"] += len(job.state_items)
        try:
            return job.future.result(timeout=self.request_timeout_seconds)
        except FutureTimeout as exc:
            if job.persistent and job.admitted_at is not None:
                # Once a persistent continuation has been installed, returning
                # early would let the Runtime clear its per-State Busy guard
                # while this worker is still mutating the recurrent state.
                # Finish the bounded decode instead and report the deadline
                # overrun as telemetry.
                with self._state_lock:
                    self._metrics["active_state_deadline_overruns"] += 1
                return job.future.result()
            job.cancelled.set()
            with self._state_lock:
                self._metrics["request_timeouts"] += 1
            raise TimeoutError(
                f"inference request exceeded {self.request_timeout_seconds}s"
            ) from exc

    def _run(self) -> None:
        waiting: deque[InferenceJob] = deque()
        prefilling: dict[str, _ActiveRow] = {}
        decoding: dict[str, _ActiveRow] = {}
        try:
            while (
                not self._closing.is_set()
                or waiting
                or prefilling
                or decoding
                or not self._queue.empty()
            ):
                self._collect(
                    waiting,
                    idle=not waiting and not prefilling and not decoding,
                )
                self._admit(waiting, prefilling)
                self._publish_counts(waiting, prefilling, decoding)

                if prefilling:
                    ids = list(prefilling)
                    try:
                        self.scheduler.prefill_round(ids)
                    except Exception as exc:
                        self._fail_rows(
                            [prefilling.pop(row_id) for row_id in ids],
                            exc,
                        )
                        continue
                    for row_id in list(prefilling):
                        row = prefilling[row_id]
                        if (
                            row.job.cancelled.is_set()
                            and not row.persistent
                        ):
                            prefilling.pop(row_id)
                            self._cancel_row(row)
                        elif self.scheduler.request(row_id).remaining == 0:
                            prefilling.pop(row_id)
                            if row.job.kind == "classification":
                                try:
                                    self._finish_classification(row)
                                except Exception as exc:
                                    self._fail_rows([row], exc)
                            else:
                                decoding[row_id] = row

                if decoding:
                    self._decode_tick(decoding)

                self._publish_counts(waiting, prefilling, decoding)
                if not waiting and not prefilling and not decoding:
                    time.sleep(0.001)
        except Exception as exc:
            self._worker_error = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            error = RuntimeError("unified batch engine closed")
            active = list(prefilling.values()) + list(decoding.values())
            self._fail_rows(active, error)
            while waiting:
                self._fail_without_state(waiting.popleft(), error)
            while True:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._fail_without_state(job, error)

    def _collect(self, waiting: deque[InferenceJob], *, idle: bool) -> None:
        if idle and not waiting:
            try:
                waiting.append(self._queue.get(timeout=0.05))
            except queue.Empty:
                return
            deadline = time.monotonic() + self.batch_window_ms / 1000.0
            while time.monotonic() < deadline:
                try:
                    waiting.append(self._queue.get_nowait())
                except queue.Empty:
                    time.sleep(0.00025)
        while True:
            try:
                waiting.append(self._queue.get_nowait())
            except queue.Empty:
                break

    def _admit(
        self,
        waiting: deque[InferenceJob],
        prefilling: dict[str, _ActiveRow],
    ) -> None:
        # Try every waiting job once. Persistent rows need no new pool slot and
        # must not starve behind an ephemeral job waiting for capacity.
        attempts = len(waiting)
        for _index in range(attempts):
            job = waiting.popleft()
            if job.cancelled.is_set():
                self._cancel_without_state(job)
                continue
            if job.persistent:
                self._admit_state_job(job, prefilling)
                continue
            if self.scheduler.pool.free <= 0:
                waiting.append(job)
                continue
            try:
                values = (
                    list(job.prefix_token_ids)
                    + list(self.tokenizer.encode(job.prompt))
                )[-self.context_limit :]
                if not values:
                    raise ValueError("prompt encoded to no tokens")
                self.scheduler.admit(job.job_id, values)
                job.admitted_at = time.monotonic()
                prefilling[job.job_id] = _ActiveRow(
                    request_id=job.job_id,
                    job=job,
                )
                with self._state_lock:
                    self._metrics["admitted"] += 1
            except Exception as exc:
                self._fail_without_state(job, exc)

    def _admit_state_job(
        self,
        job: InferenceJob,
        prefilling: dict[str, _ActiveRow],
    ) -> None:
        try:
            # Publish ownership before installing tokens so a simultaneous
            # caller deadline cannot make the Runtime drop its Busy guard.
            job.admitted_at = time.monotonic()
            rows = [
                _ActiveRow(
                    request_id=item.state_id,
                    job=job,
                    branch=item.branch,
                )
                for item in job.state_items
            ]
            if any(row.request_id in prefilling for row in rows):
                raise RuntimeError("persistent State is already prefilling")
            self.scheduler.install_continuations(
                [
                    (item.state_id, item.token_ids)
                    for item in job.state_items
                ]
            )
            prefilling.update({row.request_id: row for row in rows})
            with self._state_lock:
                self._metrics["admitted_state_jobs"] += 1
                self._metrics["admitted_state_rows"] += len(rows)
        except Exception as exc:
            self._fail_without_state(job, exc)

    def _finish_classification(self, row: _ActiveRow) -> None:
        request = self.scheduler.request(row.request_id)
        if request.logits is None:
            raise RuntimeError("classification request has no logits")
        scores = finite_label_scores(request.logits, row.job.labels)
        self._finish_ephemeral(row, {"scores": scores})

    def _decode_tick(self, decoding: dict[str, _ActiveRow]) -> None:
        ids = list(decoding)
        try:
            sampled = self.scheduler.sample_next(ids)
        except Exception as exc:
            rows = [decoding.pop(row_id) for row_id in ids]
            self._fail_rows(rows, exc)
            return

        advance: dict[str, int] = {}
        finished: list[tuple[_ActiveRow, str]] = []
        for row_id in ids:
            row = decoding.get(row_id)
            if row is None:
                continue
            if row.job.cancelled.is_set() and not row.persistent:
                decoding.pop(row_id)
                self._cancel_row(row)
                continue
            status = append_greedy_token(
                row.output_ids,
                sampled[row_id],
                eos_token_id=self.eos_token_id,
                max_tokens=row.job.max_tokens,
            )
            if status.eos:
                finished.append((row, "</s>"))
                continue
            try:
                decoded = decode_text_stops(
                    self.tokenizer,
                    row.output_ids,
                    previous_text=row.output_text,
                    stops=row.job.stops,
                )
            except Exception as exc:
                affected = [
                    candidate
                    for candidate in decoding.values()
                    if candidate.job is row.job
                ]
                for candidate in affected:
                    decoding.pop(candidate.request_id, None)
                self._fail_rows(affected, exc)
                continue
            row.output_text = decoded.text
            stop_reason = (
                decoded.stop_reason
                or ("max_tokens" if status.budget_reached else "")
            )
            # A persistent recurrent state must include its terminal stop or
            # budget token. Ephemeral jobs keep the historical behavior and
            # do not spend a forward on a state which is about to be released.
            if row.persistent or not stop_reason:
                advance[row_id] = status.token
            if stop_reason:
                finished.append((row, stop_reason))

        if advance:
            try:
                self.scheduler.advance_tokens(advance)
            except Exception as exc:
                rows = [
                    decoding.pop(row_id)
                    for row_id in ids
                    if row_id in decoding
                ]
                self._fail_rows(rows, exc)
                return

        for row, stop_reason in finished:
            if decoding.pop(row.request_id, None) is None:
                continue
            if row.persistent:
                self._finish_state_row(row, stop_reason)
            else:
                self._finish_ephemeral(
                    row,
                    {
                        "text": row.output_text,
                        "stop_reason": stop_reason,
                        "token_ids": list(row.output_ids),
                    },
                )

    def _finish_state_row(self, row: _ActiveRow, stop_reason: str) -> None:
        job = row.job
        job.state_results[row.request_id] = {
            "state_id": row.request_id,
            "branch": row.branch,
            "text": row.output_text,
            "token_ids": list(row.output_ids),
            "stop_reason": stop_reason,
            "seen_tokens": self.scheduler.request(row.request_id).seen_tokens,
        }
        if len(job.state_results) != len(job.state_items):
            return
        elapsed = (time.monotonic() - job.created_at) * 1000.0
        admitted = job.admitted_at or job.created_at
        queue_ms = (admitted - job.created_at) * 1000.0
        output = []
        for item in job.state_items:
            value = dict(job.state_results[item.state_id])
            value["elapsed_ms"] = round(elapsed, 3)
            value["queue_ms"] = round(queue_ms, 3)
            value["batch_mode"] = "unified"
            output.append(value)
        if not job.future.done():
            job.future.set_result(output)
        with self._state_lock:
            self._metrics["completed"] += 1
            self._metrics["completed_state_continuation"] += 1
            self._metrics["completed_state_rows"] += len(output)

    def _finish_ephemeral(
        self,
        row: _ActiveRow,
        result: dict[str, Any],
    ) -> None:
        job = row.job
        elapsed = (time.monotonic() - job.created_at) * 1000.0
        admitted = job.admitted_at or job.created_at
        queue_ms = (admitted - job.created_at) * 1000.0
        payload = {
            **result,
            "elapsed_ms": round(elapsed, 3),
            "queue_ms": round(queue_ms, 3),
            "batch_mode": "unified",
        }
        self._release(row.request_id)
        if not job.future.done():
            job.future.set_result(payload)
        with self._state_lock:
            self._metrics["completed"] += 1
            self._metrics[f"completed_{job.kind}"] += 1

    def _fail_rows(self, rows: Sequence[_ActiveRow], exc: Exception) -> None:
        jobs: dict[str, InferenceJob] = {}
        for row in rows:
            jobs[row.job.job_id] = row.job
            if not row.persistent:
                self._release(row.request_id)
        for job in jobs.values():
            if not job.future.done():
                job.future.set_exception(exc)
        if jobs:
            with self._state_lock:
                self._metrics["failed"] += len(jobs)

    def _fail_without_state(self, job: InferenceJob, exc: Exception) -> None:
        if not job.future.done():
            job.future.set_exception(exc)
        with self._state_lock:
            self._metrics["failed"] += 1

    def _cancel_row(self, row: _ActiveRow) -> None:
        if not row.persistent:
            self._release(row.request_id)
        if not row.job.future.done():
            row.job.future.cancel()
        with self._state_lock:
            self._metrics["cancelled"] += 1

    def _cancel_without_state(self, job: InferenceJob) -> None:
        if not job.future.done():
            job.future.cancel()
        with self._state_lock:
            self._metrics["cancelled"] += 1

    def _release(self, request_id: str) -> None:
        try:
            self.scheduler.release(request_id)
        except KeyError:
            pass

    def _publish_counts(
        self,
        waiting: deque[InferenceJob],
        prefilling: dict[str, _ActiveRow],
        decoding: dict[str, _ActiveRow],
    ) -> None:
        with self._state_lock:
            self._waiting = len(waiting) + self._queue.qsize()
            self._prefilling = len(prefilling)
            self._decoding = len(decoding)
