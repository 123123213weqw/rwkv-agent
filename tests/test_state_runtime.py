from __future__ import annotations

import threading
from types import SimpleNamespace
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

    def test_batch_prefill_vectorizes_independent_owners_without_fork(self) -> None:
        scheduler, runtime = self.make_runtime()
        states = runtime.prefill_many(
            items=[
                {"owner_id": "owner-a", "prompt": "abcd", "branch": "site-a"},
                {"owner_id": "owner-b", "prompt": "wxyz", "branch": "site-b"},
            ]
        )
        self.assertEqual(len(states), 2)
        self.assertEqual({row["owner_id"] for row in states}, {"owner-a", "owner-b"})
        self.assertTrue(all(row["parent_state_id"] is None for row in states))
        self.assertIn("B2T4", scheduler.metrics()["shape_counts"])
        self.assertEqual(runtime.health()["metrics"]["batch_prefill_states"], 2)
        for state in states:
            runtime.release(
                owner_id=state["owner_id"],
                state_ids=[state["state_id"]],
            )
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

    def test_cached_classification_root_survives_fork_classify_release(self) -> None:
        scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="gate", prompt="static examples")
        child = runtime.fork(
            owner_id="gate",
            parent_state_id=root["state_id"],
            branches=["request-1"],
        )[0]
        result = runtime.classify_many(
            owner_id="gate",
            items=[{"state_id": child["state_id"], "input": "current request"}],
            labels={"tool": 1, "chat": 2},
        )[0]
        self.assertEqual(set(result["scores"]), {"tool", "chat"})
        runtime.release(owner_id="gate", state_ids=[child["state_id"]])
        self.assertTrue(
            runtime.has_state(
                owner_id="gate",
                state_id=root["state_id"],
                touch=True,
            )
        )
        self.assertEqual(scheduler.pool.allocated, 1)
        runtime.release(owner_id="gate", state_ids=[root["state_id"]])

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

    def test_safe_snapshot_release_restore_continue_roundtrip(self) -> None:
        scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="turn-a", prompt="abc", branch="primary")
        snapshot = runtime.snapshot(
            owner_id="turn-a",
            state_id=root["state_id"],
        )
        self.assertTrue(snapshot["checksum"].startswith("sha256:"))
        self.assertEqual(snapshot["size_bytes"], len(snapshot["payload"]))
        with self.assertRaisesRegex(RuntimeError, "source remains live"):
            runtime.restore(owner_id="turn-a", payload=snapshot["payload"])
        with self.assertRaises(PermissionError):
            runtime.restore(owner_id="turn-b", payload=snapshot["payload"])

        original = runtime.continue_many(
            owner_id="turn-a",
            items=[{"state_id": root["state_id"], "input": "de"}],
            stops=[],
            max_tokens=3,
        )[0]
        runtime.release(owner_id="turn-a", state_ids=[root["state_id"]])
        self.assertEqual(scheduler.pool.allocated, 0)

        restored = runtime.restore(
            owner_id="turn-a",
            payload=snapshot["payload"],
        )
        self.assertNotEqual(restored["state_id"], root["state_id"])
        self.assertEqual(restored["branch"], "primary")
        resumed = runtime.continue_many(
            owner_id="turn-a",
            items=[{"state_id": restored["state_id"], "input": "de"}],
            stops=[],
            max_tokens=3,
        )[0]
        self.assertEqual(resumed["token_ids"], original["token_ids"])
        self.assertEqual(resumed["text"], original["text"])
        self.assertEqual(runtime.health()["metrics"]["snapshots"], 1)
        self.assertEqual(runtime.health()["metrics"]["restores"], 1)

    def test_snapshot_rejects_corruption_without_allocating_state(self) -> None:
        scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="turn-a", prompt="abc")
        snapshot = runtime.snapshot(owner_id="turn-a", state_id=root["state_id"])
        runtime.release(owner_id="turn-a", state_ids=[root["state_id"]])
        damaged = bytearray(snapshot["payload"])
        damaged[-1] ^= 0xFF
        with self.assertRaises(ValueError):
            runtime.restore(owner_id="turn-a", payload=bytes(damaged))
        self.assertEqual(scheduler.pool.allocated, 0)

    def test_sidecar_snapshot_restore_transport_enforces_identity(self) -> None:
        from fastapi.testclient import TestClient
        import rwkv_agent.sidecar as sidecar

        _scheduler, runtime = self.make_runtime()
        root = runtime.prefill(owner_id="turn-a", prompt="abc")
        previous = sidecar.service
        sidecar.service = SimpleNamespace(states=runtime)
        try:
            client = TestClient(sidecar.create_app())
            identity = {
                "model_id": sidecar.MODEL_ID,
                "revision": sidecar.MODEL_REVISION,
                "tokenizer": sidecar.TOKENIZER_ID,
                "state_abi": sidecar.STATE_ABI,
            }
            wrong = client.post(
                f"/v1/states/{root['state_id']}/snapshot",
                json={"owner_id": "turn-a", "model_ref": {**identity, "revision": "wrong"}},
            )
            self.assertEqual(wrong.status_code, 409)
            response = client.post(
                f"/v1/states/{root['state_id']}/snapshot",
                json={
                    "owner_id": "turn-a",
                    "model_ref": identity,
                    "target_tier": "cpu",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            checkpoint = response.json()
            live = client.post(
                "/v1/states/restore",
                json={
                    "owner_id": "turn-a",
                    "model_ref": identity,
                    "checksum": checkpoint["checkpoint"]["checksum"],
                    "payload_base64": checkpoint["payload_base64"],
                },
            )
            self.assertEqual(live.status_code, 409)
            runtime.release(owner_id="turn-a", state_ids=[root["state_id"]])
            restored = client.post(
                "/v1/states/restore",
                json={
                    "owner_id": "turn-a",
                    "model_ref": identity,
                    "checksum": checkpoint["checkpoint"]["checksum"],
                    "payload_base64": checkpoint["payload_base64"],
                },
            )
            self.assertEqual(restored.status_code, 200, restored.text)
            self.assertNotEqual(
                restored.json()["state"]["state_id"],
                root["state_id"],
            )
        finally:
            sidecar.service = previous


if __name__ == "__main__":
    unittest.main()
