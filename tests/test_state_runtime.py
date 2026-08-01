from __future__ import annotations

import threading
import unittest

import pytest

torch = pytest.importorskip("torch")

from rwkv7_scheduler import (  # noqa: E402
    AlbatrossChunkScheduler,
    AlbatrossStatePool,
    SchedulerConfig,
)
from rwkv_agent.state_runtime import PersistentStateRuntime  # noqa: E402
from rwkv_agent.batching import ContinuousBatchEngine  # noqa: E402


class FakeAlbatross:
    vocab_size = 257

    def zero_state(self, batch: int) -> list[torch.Tensor]:
        return [
            torch.zeros((2, 2, batch, 3), dtype=torch.float32),
            torch.zeros((2, batch, 1, 2, 2), dtype=torch.float32),
            torch.zeros((batch,), dtype=torch.int32),
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
    ) -> torch.Tensor:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        batch, length = tokens.shape
        values = tokens.float().sum(dim=1)
        state[0].add_(values.view(1, 1, batch, 1))
        state[1].add_(values.view(1, batch, 1, 1, 1))
        state[2].add_(length)
        target = (
            state[0][0, 0, :, 0].long()
            + state[1][0, :, 0, 0, 0].long()
            + state[2].long()
        ) % self.vocab_size
        logits = torch.full((batch, self.vocab_size), -1000.0)
        logits[torch.arange(batch), target] = 1000.0
        return logits


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) % 251 + 1 for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(65 + token % 26) for token in token_ids)


class PersistentStateRuntimeTests(unittest.TestCase):
    def make_runtime(self, *, clock=lambda: 0.0):
        model = FakeAlbatross()
        pool = AlbatrossStatePool(model, capacity=8, max_batch_size=8)
        scheduler = AlbatrossChunkScheduler(
            model,
            pool=pool,
            config=SchedulerConfig(
                prefill_chunk_size=4,
                max_batch_size=8,
                max_queue_size=8,
                max_input_tokens=128,
            ),
        )
        runtime = PersistentStateRuntime(
            tokenizer=FakeTokenizer(),
            scheduler=scheduler,
            context_limit=128,
            eos_token_id=999,
            capacity=5,
            ttl_seconds=5,
            clock=clock,
        )
        return scheduler, runtime

    def test_prefill_fork_batch_resume_and_release(self) -> None:
        scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="turn-a", prompt="abc")
        branches = runtime.fork(
            owner_id="turn-a",
            parent_state_id=root["state_id"],
            branches=["official", "verify"],
        )
        results = runtime.continue_many(
            owner_id="turn-a",
            items=[
                {"state_id": branches[0]["state_id"], "input": "de"},
                {"state_id": branches[1]["state_id"], "input": "fg"},
            ],
            stops=[],
            max_tokens=2,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["stop_reason"] == "max_tokens" for result in results))
        self.assertEqual(scheduler.request(root["state_id"]).seen_tokens, 3)
        self.assertEqual(
            [scheduler.request(item["state_id"]).seen_tokens for item in branches],
            [7, 7],
        )
        self.assertIn("B2T1", scheduler.metrics()["shape_counts"])
        self.assertNotIn("B2T2", scheduler.metrics()["shape_counts"])
        released = runtime.release(
            owner_id="turn-a",
            state_ids=[root["state_id"]]
            + [item["state_id"] for item in branches],
        )
        self.assertEqual(released["released"], 3)
        self.assertEqual(scheduler.pool.allocated, 0)

    def test_concurrent_runtime_calls_use_unified_ready_queue(self) -> None:
        scheduler, _unused = self.make_runtime()
        tokenizer = FakeTokenizer()
        engine = ContinuousBatchEngine(
            tokenizer=tokenizer,
            scheduler=scheduler,
            context_limit=128,
            eos_token_id=999,
            max_state_rows=8,
            batch_window_ms=20,
            request_timeout_seconds=2,
        )
        self.addCleanup(engine.close)
        runtime = PersistentStateRuntime(
            tokenizer=tokenizer,
            scheduler=scheduler,
            context_limit=128,
            eos_token_id=999,
            capacity=5,
            ttl_seconds=5,
            decode_engine=engine,
        )
        roots = [
            runtime.prefill(owner_id=f"turn-{index}", prompt="abc")
            for index in range(2)
        ]
        barrier = threading.Barrier(2)
        results = {}

        def run(index: int) -> None:
            barrier.wait()
            results[index] = runtime.continue_many(
                owner_id=f"turn-{index}",
                items=[
                    {
                        "state_id": roots[index]["state_id"],
                        "input": "de",
                    }
                ],
                stops=[],
                max_tokens=2,
            )

        threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 2)
        self.assertEqual(engine.health()["metrics"]["completed_state_rows"], 2)
        self.assertIn("B2T1", scheduler.metrics()["shape_counts"])
        for index, root in enumerate(roots):
            runtime.release(
                owner_id=f"turn-{index}",
                state_ids=[root["state_id"]],
            )

    def test_persistent_batch_classification_returns_label_logits(self) -> None:
        scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="turn-a", prompt="abc")
        branches = runtime.fork(
            owner_id="turn-a",
            parent_state_id=root["state_id"],
            branches=["left", "right"],
        )
        results = runtime.classify_many(
            owner_id="turn-a",
            items=[
                {"state_id": branches[0]["state_id"], "input": "de"},
                {"state_id": branches[1]["state_id"], "input": "fg"},
            ],
            labels={"A": 1, "B": 2},
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(set(results[0]["scores"]), {"A", "B"})
        self.assertEqual(runtime.health()["metrics"]["classified"], 2)
        runtime.release(
            owner_id="turn-a",
            state_ids=[root["state_id"]]
            + [item["state_id"] for item in branches],
        )
        self.assertEqual(scheduler.pool.allocated, 0)

    def test_persistent_batch_classification_rejects_non_finite_logits(self) -> None:
        scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="turn-a", prompt="abc")
        branch = runtime.fork(
            owner_id="turn-a",
            parent_state_id=root["state_id"],
            branches=["left"],
        )[0]
        original_continue_many = scheduler.continue_many

        def continue_with_nan(items):
            original_continue_many(items)
            request = scheduler.request(branch["state_id"])
            request.logits = request.logits.clone()
            request.logits[1] = float("nan")

        scheduler.continue_many = continue_with_nan
        with self.assertRaisesRegex(RuntimeError, "non-finite classification logits"):
            runtime.classify_many(
                owner_id="turn-a",
                items=[{"state_id": branch["state_id"], "input": "de"}],
                labels={"A": 1, "B": 2},
            )
        self.assertEqual(runtime.health()["metrics"]["failed"], 1)
        runtime.release(
            owner_id="turn-a",
            state_ids=[root["state_id"], branch["state_id"]],
        )
        self.assertEqual(scheduler.pool.allocated, 0)

    def test_owner_isolation_and_ttl_cleanup(self) -> None:
        now = [0.0]
        scheduler, runtime = self.make_runtime(clock=lambda: now[0])
        root = runtime.prefill(owner_id="turn-a", prompt="abc")
        with self.assertRaises(PermissionError):
            runtime.fork(
                owner_id="turn-b",
                parent_state_id=root["state_id"],
                branches=["steal"],
            )
        now[0] = 6.0
        health = runtime.health()
        self.assertEqual(health["allocated"], 0)
        self.assertEqual(health["expired_on_health"], 1)
        self.assertEqual(scheduler.pool.allocated, 0)

    def test_capacity_is_bounded(self) -> None:
        _scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="turn-a", prompt="abc")
        runtime.fork(
            owner_id="turn-a",
            parent_state_id=root["state_id"],
            branches=["a", "b", "c", "d"],
        )
        with self.assertRaises(RuntimeError):
            runtime.prefill(owner_id="turn-b", prompt="def")


if __name__ == "__main__":
    unittest.main()
