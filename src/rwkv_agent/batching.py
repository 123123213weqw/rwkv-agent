"""Continuous batching facade for the recurrent-state scheduler."""

from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class InferenceJob:
    job_id: str
    kind: str
    prompt: str
    prefix_token_ids: tuple[int, ...] = ()
    stops: tuple[str, ...] = ()
    max_tokens: int = 0
    labels: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    admitted_at: float | None = None
    output_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    future: Future[dict[str, Any]] = field(default_factory=Future)
    cancelled: threading.Event = field(default_factory=threading.Event)


class ContinuousBatchEngine:
    """Collect concurrent requests and advance them in one scheduler thread.

    New prompts can join between decode ticks. Long prompts receive one exact
    prefill quantum per round, so ready-to-decode requests are not blocked until
    every newly admitted prompt has completed its full prefill.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        scheduler: Any,
        context_limit: int,
        eos_token_id: int = 0,
        batch_window_ms: float = 4.0,
        max_waiting_jobs: int = 256,
        request_timeout_seconds: float = 300.0,
        thread_name: str = "rwkv-continuous-batcher",
    ) -> None:
        if context_limit < 1:
            raise ValueError("context_limit must be positive")
        if batch_window_ms < 0:
            raise ValueError("batch_window_ms must not be negative")
        if max_waiting_jobs < 1:
            raise ValueError("max_waiting_jobs must be positive")
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.context_limit = int(context_limit)
        self.eos_token_id = int(eos_token_id)
        self.batch_window_ms = float(batch_window_ms)
        self.request_timeout_seconds = float(request_timeout_seconds)
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
        job = InferenceJob(
            job_id=uuid.uuid4().hex,
            kind="completion",
            prompt=prompt,
            prefix_token_ids=tuple(int(value) for value in prefix_token_ids),
            stops=tuple(str(value) for value in stops if value),
            max_tokens=int(max_tokens),
        )
        return self._submit(job)

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
        job = InferenceJob(
            job_id=uuid.uuid4().hex,
            kind="classification",
            prompt=prompt,
            labels=normalized,
        )
        return self._submit(job)

    def close(self, *, timeout: float = 10.0) -> None:
        self._closing.set()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            raise TimeoutError("continuous batch worker did not stop")

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            state = {
                "waiting": self._waiting,
                "prefilling": self._prefilling,
                "decoding": self._decoding,
            }
            metrics = dict(self._metrics)
        return {
            "mode": "continuous_batch",
            "worker_alive": self._thread.is_alive(),
            "worker_error": self._worker_error,
            "batch_window_ms": self.batch_window_ms,
            "request_timeout_seconds": self.request_timeout_seconds,
            "queue_capacity": self._queue.maxsize,
            **state,
            "metrics": metrics,
            "scheduler": self.scheduler.metrics(),
        }

    def _submit(self, job: InferenceJob) -> dict[str, Any]:
        if self._closing.is_set():
            raise RuntimeError("continuous batch engine is closed")
        try:
            self._queue.put(job, timeout=1.0)
        except queue.Full as exc:
            with self._state_lock:
                self._metrics["rejected_queue_full"] += 1
            raise RuntimeError("inference queue is full") from exc
        with self._state_lock:
            self._metrics["submitted"] += 1
        try:
            return job.future.result(timeout=self.request_timeout_seconds)
        except FutureTimeout as exc:
            job.cancelled.set()
            with self._state_lock:
                self._metrics["request_timeouts"] += 1
            raise TimeoutError(
                f"inference request exceeded {self.request_timeout_seconds}s"
            ) from exc

    def _run(self) -> None:
        waiting: deque[InferenceJob] = deque()
        prefilling: dict[str, InferenceJob] = {}
        decoding: dict[str, InferenceJob] = {}
        while (
            not self._closing.is_set()
            or waiting
            or prefilling
            or decoding
            or not self._queue.empty()
        ):
            self._collect(waiting, idle=not waiting and not prefilling and not decoding)
            self._admit(waiting, prefilling)
            self._publish_counts(waiting, prefilling, decoding)

            if prefilling:
                ids = list(prefilling)
                try:
                    self.scheduler.prefill_round(ids)
                except Exception as exc:
                    for job_id in ids:
                        self._fail(prefilling.pop(job_id), exc)
                    continue
                for job_id in list(prefilling):
                    job = prefilling[job_id]
                    if job.cancelled.is_set():
                        prefilling.pop(job_id)
                        self._cancel(job)
                    elif self.scheduler.request(job_id).remaining == 0:
                        prefilling.pop(job_id)
                        if job.kind == "classification":
                            try:
                                self._finish_classification(job)
                            except Exception as exc:
                                self._fail(job, exc)
                        else:
                            decoding[job_id] = job

            if decoding:
                self._decode_tick(decoding)

            self._publish_counts(waiting, prefilling, decoding)
            if not waiting and not prefilling and not decoding:
                time.sleep(0.001)

        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            self._fail(job, RuntimeError("continuous batch engine closed"))

    def _collect(self, waiting: deque[InferenceJob], *, idle: bool) -> None:
        if idle and not waiting:
            try:
                first = self._queue.get(timeout=0.05)
            except queue.Empty:
                return
            waiting.append(first)
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
        prefilling: dict[str, InferenceJob],
    ) -> None:
        while waiting and self.scheduler.pool.free > 0:
            job = waiting.popleft()
            if job.cancelled.is_set():
                self._cancel_without_state(job)
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
                prefilling[job.job_id] = job
                with self._state_lock:
                    self._metrics["admitted"] += 1
            except Exception as exc:
                self._fail_without_state(job, exc)

    def _finish_classification(self, job: InferenceJob) -> None:
        request = self.scheduler.request(job.job_id)
        assert request.logits is not None
        scores = {
            name: float(request.logits[token].item())
            for name, token in job.labels.items()
        }
        if not all(math.isfinite(score) for score in scores.values()):
            raise RuntimeError("non-finite classification logits")
        self._succeed(job, {"scores": scores})

    def _decode_tick(self, decoding: dict[str, InferenceJob]) -> None:
        ids = list(decoding)
        try:
            sampled = self.scheduler.sample_next(ids)
        except Exception as exc:
            for job_id in ids:
                self._fail(decoding.pop(job_id), exc)
            return

        advance: dict[str, int] = {}
        for job_id in ids:
            job = decoding.get(job_id)
            if job is None:
                continue
            if job.cancelled.is_set():
                decoding.pop(job_id)
                self._cancel(job)
                continue
            token = int(sampled[job_id])
            if token == self.eos_token_id:
                decoding.pop(job_id)
                self._succeed(
                    job,
                    {
                        "text": job.output_text,
                        "stop_reason": "</s>",
                        "token_ids": list(job.output_ids),
                    },
                )
                continue

            job.output_ids.append(token)
            try:
                decoded = self.tokenizer.decode(job.output_ids)
            except Exception as exc:
                decoding.pop(job_id)
                self._fail(job, exc)
                continue
            stop_reason = ""
            if "\ufffd" not in decoded:
                job.output_text = decoded
                hits = [
                    (job.output_text.find(stop), stop)
                    for stop in job.stops
                    if stop in job.output_text
                ]
                if hits:
                    index, stop_reason = min(hits)
                    job.output_text = job.output_text[:index]
            if stop_reason or len(job.output_ids) >= job.max_tokens:
                decoding.pop(job_id)
                self._succeed(
                    job,
                    {
                        "text": job.output_text,
                        "stop_reason": stop_reason or "max_tokens",
                        "token_ids": list(job.output_ids),
                    },
                )
            else:
                advance[job_id] = token

        if advance:
            try:
                self.scheduler.advance_tokens(advance)
            except Exception as exc:
                for job_id in list(advance):
                    job = decoding.pop(job_id, None)
                    if job is not None:
                        self._fail(job, exc)

    def _succeed(self, job: InferenceJob, result: dict[str, Any]) -> None:
        elapsed = (time.monotonic() - job.created_at) * 1000.0
        admitted = job.admitted_at or job.created_at
        queue_ms = (admitted - job.created_at) * 1000.0
        payload = {
            **result,
            "elapsed_ms": round(elapsed, 3),
            "queue_ms": round(queue_ms, 3),
            "batch_mode": "continuous",
        }
        self._release(job.job_id)
        if not job.future.done():
            job.future.set_result(payload)
        with self._state_lock:
            self._metrics["completed"] += 1
            self._metrics[f"completed_{job.kind}"] += 1

    def _fail(self, job: InferenceJob, exc: Exception) -> None:
        self._release(job.job_id)
        if not job.future.done():
            job.future.set_exception(exc)
        with self._state_lock:
            self._metrics["failed"] += 1

    def _fail_without_state(self, job: InferenceJob, exc: Exception) -> None:
        if not job.future.done():
            job.future.set_exception(exc)
        with self._state_lock:
            self._metrics["failed"] += 1

    def _cancel(self, job: InferenceJob) -> None:
        self._release(job.job_id)
        if not job.future.done():
            job.future.cancel()
        with self._state_lock:
            self._metrics["cancelled"] += 1

    def _cancel_without_state(self, job: InferenceJob) -> None:
        if not job.future.done():
            job.future.cancel()
        with self._state_lock:
            self._metrics["cancelled"] += 1

    def _release(self, job_id: str) -> None:
        try:
            self.scheduler.release(job_id)
        except KeyError:
            pass

    def _publish_counts(
        self,
        waiting: deque[InferenceJob],
        prefilling: dict[str, InferenceJob],
        decoding: dict[str, InferenceJob],
    ) -> None:
        with self._state_lock:
            self._waiting = len(waiting) + self._queue.qsize()
            self._prefilling = len(prefilling)
            self._decoding = len(decoding)
