from __future__ import annotations

import threading
import unittest
from dataclasses import dataclass
import time
from typing import Any

from rwkv_agent.batching import ContinuousBatchEngine
from rwkv_agent.state_batching import StateContinuationItem


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class Logits:
    def __getitem__(self, token: int) -> Scalar:
        return Scalar({1: 5.0, 2: 2.0}.get(token, -10.0))


class NonFiniteLogits:
    def __getitem__(self, token: int) -> Scalar:
        return Scalar(float("nan") if token == 1 else 2.0)


@dataclass
class FakeRequest:
    token_ids: list[int]
    remaining: int
    output: list[int]
    cursor: int = 0
    logits: Any = None
    seen_tokens: int = 0


class FakePool:
    def __init__(self, scheduler: "FakeScheduler", capacity: int) -> None:
        self.scheduler = scheduler
        self.capacity = capacity

    @property
    def free(self) -> int:
        return self.capacity - len(self.scheduler.requests)


class FakeTokenizer:
    scripts = {
        10: [ord("O"), ord("K"), 0],
        11: [ord("A"), ord("1"), 0],
        12: [ord("B"), ord("2"), 0],
        13: [ord("C"), ord("3"), 0],
        20: [ord("a"), ord("!"), ord("x"), 0],
        30: [0],
    }

    def encode(self, prompt: str) -> list[int]:
        marker = int(prompt.split(":", 1)[0])
        return [marker, marker + 1, marker + 2, marker + 3, marker + 4]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token) for token in token_ids)


class FakeScheduler:
    def __init__(self, *, capacity: int = 8, quantum: int = 2) -> None:
        self.requests: dict[str, FakeRequest] = {}
        self.pool = FakePool(self, capacity)
        self.quantum = quantum
        self.prefill_batches: list[list[str]] = []
        self.advance_batches: list[dict[str, int]] = []

    def admit(self, request_id: str, token_ids: list[int]) -> None:
        marker = token_ids[0]
        self.requests[request_id] = FakeRequest(
            list(token_ids),
            len(token_ids),
            list(FakeTokenizer.scripts[marker]),
            logits=Logits(),
        )

    def prefill_round(self, request_ids: list[str]) -> None:
        self.prefill_batches.append(list(request_ids))
        for request_id in request_ids:
            request = self.requests[request_id]
            consumed = min(request.remaining, self.quantum)
            request.remaining -= consumed
            request.seen_tokens += consumed

    def install_continuations(self, rows) -> None:
        for request_id, token_ids in rows:
            request = self.requests[request_id]
            request.token_ids = list(token_ids)
            request.remaining = len(token_ids)

    def request(self, request_id: str) -> FakeRequest:
        return self.requests[request_id]

    def sample_next(self, request_ids: list[str]) -> dict[str, int]:
        output = {}
        for request_id in request_ids:
            request = self.requests[request_id]
            output[request_id] = request.output[request.cursor]
            request.cursor += 1
        return output

    def advance_tokens(self, values: dict[str, int]) -> None:
        self.advance_batches.append(dict(values))

    def release(self, request_id: str) -> None:
        del self.requests[request_id]

    def metrics(self) -> dict[str, Any]:
        return {"allocated": len(self.requests)}


class BlockingScheduler(FakeScheduler):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_started = threading.Event()
        self.prefill_continue = threading.Event()

    def prefill_round(self, request_ids: list[str]) -> None:
        self.prefill_started.set()
        self.prefill_continue.wait(timeout=2)
        super().prefill_round(request_ids)


class ContinuousBatchEngineTests(unittest.TestCase):
    def make_engine(
        self,
        *,
        batch_window_ms: float = 20,
    ) -> tuple[FakeScheduler, ContinuousBatchEngine]:
        scheduler = FakeScheduler()
        engine = ContinuousBatchEngine(
            tokenizer=FakeTokenizer(),
            scheduler=scheduler,
            context_limit=32,
            batch_window_ms=batch_window_ms,
            request_timeout_seconds=2,
        )
        self.addCleanup(engine.close)
        return scheduler, engine

    def test_concurrent_completions_share_prefill_and_decode_batches(self) -> None:
        scheduler, engine = self.make_engine()
        barrier = threading.Barrier(4)
        outputs: dict[str, dict[str, Any]] = {}

        def run(marker: int) -> None:
            barrier.wait()
            outputs[str(marker)] = engine.complete(
                f"{marker}:prompt",
                stops=[],
                max_tokens=2,
            )

        threads = [
            threading.Thread(target=run, args=(marker,))
            for marker in (10, 11, 12, 13)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(
            {key: value["text"] for key, value in outputs.items()},
            {"10": "OK", "11": "A1", "12": "B2", "13": "C3"},
        )
        self.assertTrue(any(len(batch) == 4 for batch in scheduler.prefill_batches))
        self.assertTrue(any(len(batch) == 4 for batch in scheduler.advance_batches))
        self.assertTrue(
            all(value["batch_mode"] == "unified" for value in outputs.values())
        )
        self.assertTrue(all(value["input_tokens"] > 0 for value in outputs.values()))
        self.assertEqual(scheduler.pool.free, scheduler.pool.capacity)

    def test_completion_and_persistent_state_share_one_ready_queue(self) -> None:
        scheduler, engine = self.make_engine()
        scheduler.requests["state-a"] = FakeRequest(
            token_ids=[],
            remaining=0,
            output=list(FakeTokenizer.scripts[11]),
        )
        barrier = threading.Barrier(2)
        outputs: dict[str, Any] = {}

        def complete() -> None:
            barrier.wait()
            outputs["completion"] = engine.complete(
                "10:prompt",
                stops=[],
                max_tokens=2,
            )

        def state() -> None:
            barrier.wait()
            outputs["state"] = engine.continue_many(
                [
                    StateContinuationItem(
                        state_id="state-a",
                        branch="chat",
                        token_ids=(40, 41, 42, 43, 44),
                    )
                ],
                stops=[],
                max_tokens=2,
            )[0]

        threads = [
            threading.Thread(target=complete),
            threading.Thread(target=state),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(outputs["completion"]["text"], "OK")
        self.assertEqual(outputs["state"]["text"], "A1")
        self.assertEqual(outputs["state"]["input_tokens"], 5)
        self.assertTrue(
            any(
                len(batch) == 2 and "state-a" in batch
                for batch in scheduler.prefill_batches
            )
        )
        self.assertTrue(
            any(len(batch) == 2 for batch in scheduler.advance_batches)
        )
        health = engine.health()
        self.assertEqual(health["mode"], "unified_ready_queue")
        self.assertEqual(
            health["metrics"]["completed_state_continuation"],
            1,
        )
        del scheduler.requests["state-a"]

    def test_persistent_terminal_stop_token_is_committed(self) -> None:
        scheduler, engine = self.make_engine(batch_window_ms=0)
        scheduler.requests["state-stop"] = FakeRequest(
            token_ids=[],
            remaining=0,
            output=list(FakeTokenizer.scripts[20]),
        )
        result = engine.continue_many(
            [
                StateContinuationItem(
                    state_id="state-stop",
                    branch="chat",
                    token_ids=(40,),
                )
            ],
            stops=["!"],
            max_tokens=8,
        )[0]
        self.assertEqual(result["text"], "a")
        self.assertEqual(result["stop_reason"], "!")
        self.assertEqual(
            scheduler.advance_batches[-2:],
            [{"state-stop": ord("a")}, {"state-stop": ord("!")}],
        )
        del scheduler.requests["state-stop"]

    def test_persistent_continuation_emits_real_decode_deltas(self) -> None:
        scheduler, engine = self.make_engine(batch_window_ms=0)
        scheduler.requests["state-stream"] = FakeRequest(
            token_ids=[],
            remaining=0,
            output=list(FakeTokenizer.scripts[11]),
        )
        events: list[dict[str, Any]] = []
        result = engine.continue_many(
            [
                StateContinuationItem(
                    state_id="state-stream",
                    branch="chat",
                    token_ids=(40,),
                )
            ],
            stops=[],
            max_tokens=2,
            event_sink=events.append,
        )[0]
        self.assertEqual(result["text"], "A1")
        self.assertEqual([event["text"] for event in events], ["A", "A1"])
        self.assertEqual([event["delta"] for event in events], ["A", "1"])
        self.assertTrue(all(event["type"] == "delta" for event in events))
        del scheduler.requests["state-stream"]

    def test_timed_out_ephemeral_job_is_cancelled_and_released(self) -> None:
        scheduler = BlockingScheduler()
        engine = ContinuousBatchEngine(
            tokenizer=FakeTokenizer(),
            scheduler=scheduler,
            context_limit=32,
            batch_window_ms=0,
            request_timeout_seconds=0.05,
        )
        try:
            with self.assertRaisesRegex(TimeoutError, "exceeded"):
                engine.complete("10:prompt", stops=[], max_tokens=2)
            scheduler.prefill_continue.set()
            deadline = time.monotonic() + 1
            while scheduler.pool.free != scheduler.pool.capacity:
                if time.monotonic() >= deadline:
                    self.fail("cancelled request did not release its state")
                time.sleep(0.005)
            self.assertEqual(engine.health()["metrics"]["request_timeouts"], 1)
        finally:
            scheduler.prefill_continue.set()
            engine.close()

    def test_shared_queue_applies_one_backpressure_budget(self) -> None:
        scheduler = BlockingScheduler()
        engine = ContinuousBatchEngine(
            tokenizer=FakeTokenizer(),
            scheduler=scheduler,
            context_limit=32,
            batch_window_ms=0,
            max_waiting_jobs=1,
            request_timeout_seconds=2,
        )
        outputs: list[dict[str, Any]] = []

        def submit(marker: int) -> None:
            outputs.append(
                engine.complete(
                    f"{marker}:prompt",
                    stops=[],
                    max_tokens=1,
                )
            )

        first = threading.Thread(target=submit, args=(10,))
        second = threading.Thread(target=submit, args=(11,))
        try:
            first.start()
            self.assertTrue(scheduler.prefill_started.wait(timeout=1))
            second.start()
            deadline = time.monotonic() + 1
            while engine._queue.qsize() != 1:  # noqa: SLF001
                if time.monotonic() >= deadline:
                    self.fail("second request did not enter the bounded queue")
                time.sleep(0.005)
            with self.assertRaisesRegex(RuntimeError, "queue is full"):
                engine.complete("12:prompt", stops=[], max_tokens=1)
            self.assertEqual(
                engine.health()["metrics"]["rejected_queue_full"],
                1,
            )
        finally:
            scheduler.prefill_continue.set()
            first.join(timeout=2)
            second.join(timeout=2)
            engine.close()
        self.assertEqual(len(outputs), 2)

    def test_stop_string_is_trimmed_and_state_is_released(self) -> None:
        scheduler, engine = self.make_engine(batch_window_ms=0)
        result = engine.complete(
            "20:stop",
            stops=["!"],
            max_tokens=8,
        )
        self.assertEqual(result["text"], "a")
        self.assertEqual(result["stop_reason"], "!")
        self.assertEqual(result["token_ids"], [ord("a"), ord("!")])
        self.assertEqual(scheduler.pool.free, scheduler.pool.capacity)

    def test_classification_uses_prefill_logits_without_decode(self) -> None:
        scheduler, engine = self.make_engine(batch_window_ms=0)
        result = engine.classify(
            "30:gate",
            labels={"tool": 1, "chat": 2},
        )
        self.assertEqual(result["scores"], {"tool": 5.0, "chat": 2.0})
        self.assertEqual(scheduler.advance_batches, [])
        self.assertEqual(scheduler.pool.free, scheduler.pool.capacity)

    def test_classification_rejects_non_finite_logits_and_releases_state(self) -> None:
        scheduler, engine = self.make_engine(batch_window_ms=0)
        original_admit = scheduler.admit

        def admit_with_nan(request_id: str, token_ids: list[int]) -> None:
            original_admit(request_id, token_ids)
            scheduler.requests[request_id].logits = NonFiniteLogits()

        scheduler.admit = admit_with_nan  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "non-finite classification logits"):
            engine.classify("30:gate", labels={"tool": 1, "chat": 2})
        self.assertEqual(scheduler.pool.free, scheduler.pool.capacity)

    def test_health_reports_bounded_continuous_mode(self) -> None:
        _, engine = self.make_engine(batch_window_ms=3)
        health = engine.health()
        self.assertEqual(health["mode"], "unified_ready_queue")
        self.assertTrue(health["worker_alive"])
        self.assertEqual(health["batch_window_ms"], 3)


if __name__ == "__main__":
    unittest.main()
