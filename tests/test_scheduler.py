from __future__ import annotations

import unittest

import pytest

torch = pytest.importorskip("torch")

from rwkv7_scheduler import (
    AlbatrossChunkScheduler,
    AlbatrossStatePool,
    SchedulerConfig,
    StaleStateHandle,
)


class FakeAlbatross:
    """CPU model with the same state batch dimensions as faster3a."""

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
        assert state[0].shape[2] == batch
        values = tokens.float().sum(dim=1)
        state[0].add_(values.view(1, 1, batch, 1))
        state[1].add_(values.view(1, batch, 1, 1, 1))
        state[2].add_(length)
        score = (
            state[0][0, 0, :, 0].long()
            + state[1][0, :, 0, 0, 0].long()
            + state[2].long()
        ) % self.vocab_size
        logits = torch.full((batch, self.vocab_size), -1000.0)
        logits[torch.arange(batch), score] = 1000.0
        return logits


class FakePipelineAlbatross:
    """Two-stage CPU stand-in for faster3a pipeline-parallel state layout."""

    def zero_state(self, batch: int):
        return [
            [torch.zeros((2, batch, 3)) for _ in range(2)],
            [torch.zeros((batch, 1, 2, 2)) for _ in range(2)],
            [torch.zeros((batch,), dtype=torch.int32) for _ in range(2)],
        ]


def state_values(state):
    for group in state:
        if isinstance(group, list):
            yield from group
        else:
            yield group


def serial(
    model: FakeAlbatross,
    tokens: list[int],
    *,
    continuation: list[int] | None = None,
    decode_tokens: int = 0,
) -> tuple[list[torch.Tensor], torch.Tensor, list[int]]:
    state = model.zero_state(1)
    logits = model.forward(torch.tensor(tokens).view(1, -1), state)
    if continuation:
        logits = model.forward(
            torch.tensor(continuation).view(1, -1),
            state,
        )
    output = []
    for _ in range(decode_tokens):
        token = int(torch.argmax(logits[0]).item())
        if token == 0:
            break
        output.append(token)
        logits = model.forward(torch.tensor([[token]]), state)
    return state, logits, output


class StatePoolTests(unittest.TestCase):
    def test_pipeline_parallel_state_pool_lifecycle(self) -> None:
        pool = AlbatrossStatePool(
            FakePipelineAlbatross(), capacity=4, max_batch_size=4
        )
        handles = pool.allocate_many(["a", "b", "c", "d"])
        full, borrowed = pool.checkout(handles)
        self.assertTrue(borrowed)
        for value in state_values(full):
            value.add_(5)
        pool.checkin(handles, full, borrowed=borrowed)

        pool.release(handles[3])
        child = pool.clone_many(handles[0], ["child"])[0]
        self.assertTrue(
            all(
                torch.all(value == 5)
                for value in state_values(pool.snapshot(child))
            )
        )
        selected = [handles[0], handles[2]]
        gathered, borrowed = pool.checkout(selected)
        self.assertFalse(borrowed)
        for value in gathered[2]:
            value.copy_(torch.tensor([11, 13], dtype=value.dtype))
        pool.checkin(selected, gathered, borrowed=borrowed)
        self.assertTrue(
            all(
                int(value.item()) == 11
                for value in pool.snapshot(handles[0])[2]
            )
        )
        self.assertTrue(
            all(
                int(value.item()) == 13
                for value in pool.snapshot(handles[2])[2]
            )
        )

    def test_stale_handle_and_zero_on_reuse(self) -> None:
        model = FakeAlbatross()
        pool = AlbatrossStatePool(model, capacity=2, max_batch_size=2)
        first = pool.allocate("first")
        state, borrowed = pool.checkout([first])
        state[0].fill_(9)
        pool.checkin([first], state, borrowed=borrowed)
        pool.release(first)
        with self.assertRaises(StaleStateHandle):
            pool.snapshot(first)
        second = pool.allocate("second")
        self.assertEqual(second.slot, first.slot)
        self.assertGreater(second.generation, first.generation)
        self.assertTrue(all(torch.count_nonzero(v) == 0 for v in pool.snapshot(second)))

    def test_only_full_contiguous_slab_is_borrowed(self) -> None:
        model = FakeAlbatross()
        pool = AlbatrossStatePool(model, capacity=4, max_batch_size=4)
        handles = pool.allocate_many(["a", "b", "c", "d"])
        partial_state, partial_borrowed = pool.checkout(handles[:3])
        self.assertFalse(partial_borrowed)
        partial_state[2].add_(3)
        pool.checkin(handles[:3], partial_state, borrowed=partial_borrowed)
        state, borrowed = pool.checkout(handles)
        self.assertTrue(borrowed)
        state[2].add_(7)
        pool.checkin(handles, state, borrowed=borrowed)
        self.assertEqual(
            [int(pool.snapshot(handle)[2].item()) for handle in handles],
            [10, 10, 10, 7],
        )

    def test_noncontiguous_checkout_scatter(self) -> None:
        model = FakeAlbatross()
        pool = AlbatrossStatePool(model, capacity=4, max_batch_size=4)
        handles = pool.allocate_many(["a", "b", "c"])
        selected = [handles[0], handles[2]]
        state, borrowed = pool.checkout(selected)
        self.assertFalse(borrowed)
        state[2].copy_(torch.tensor([3, 5]))
        pool.checkin(selected, state, borrowed=borrowed)
        self.assertEqual(int(pool.snapshot(handles[0])[2].item()), 3)
        self.assertEqual(int(pool.snapshot(handles[1])[2].item()), 0)
        self.assertEqual(int(pool.snapshot(handles[2])[2].item()), 5)

    def test_clone_many_copies_source_and_isolates_children(self) -> None:
        model = FakeAlbatross()
        pool = AlbatrossStatePool(model, capacity=4, max_batch_size=4)
        source = pool.allocate("source")
        source_state, borrowed = pool.checkout([source])
        source_state[0].fill_(7)
        source_state[1].fill_(11)
        source_state[2].fill_(13)
        pool.checkin([source], source_state, borrowed=borrowed)

        children = pool.clone_many(source, ["child-a", "child-b"])
        expected = pool.snapshot(source)
        for child in children:
            for actual, reference in zip(
                pool.snapshot(child),
                expected,
                strict=True,
            ):
                self.assertTrue(torch.equal(actual, reference))

        child_state, child_borrowed = pool.checkout([children[0]])
        child_state[2].add_(5)
        pool.checkin([children[0]], child_state, borrowed=child_borrowed)
        self.assertEqual(int(pool.snapshot(source)[2].item()), 13)
        self.assertEqual(int(pool.snapshot(children[0])[2].item()), 18)
        self.assertEqual(int(pool.snapshot(children[1])[2].item()), 13)


class SchedulerTests(unittest.TestCase):
    def make_scheduler(
        self,
        *,
        chunk: int = 4,
    ) -> tuple[FakeAlbatross, AlbatrossChunkScheduler]:
        model = FakeAlbatross()
        pool = AlbatrossStatePool(model, capacity=8, max_batch_size=8)
        scheduler = AlbatrossChunkScheduler(
            model,
            pool=pool,
            config=SchedulerConfig(
                prefill_chunk_size=chunk,
                max_batch_size=8,
                max_queue_size=8,
                max_input_tokens=128,
            ),
        )
        return model, scheduler

    def test_variable_prefill_matches_serial(self) -> None:
        model, scheduler = self.make_scheduler(chunk=4)
        rows = {
            "a": [1, 2, 3, 4, 5],
            "b": [6, 7, 8, 9, 10, 11, 12, 13, 14],
            "c": list(range(20, 32)),
        }
        scheduler.admit_many(rows.items())
        scheduler.prefill()
        for request_id, tokens in rows.items():
            reference_state, reference_logits, _ = serial(model, tokens)
            actual = scheduler.request(request_id)
            self.assertTrue(torch.equal(actual.logits, reference_logits[0]))
            snapshot = scheduler.pool.snapshot(actual.handle)
            for left, right in zip(snapshot, reference_state, strict=True):
                self.assertTrue(torch.equal(left, right))
        metrics = scheduler.metrics()
        self.assertEqual(metrics["shape_counts"]["B3T4"], 1)
        self.assertEqual(metrics["shape_counts"]["B2T4"], 1)
        self.assertEqual(metrics["shape_counts"]["B1T4"], 1)
        self.assertEqual(metrics["shape_counts"]["B1T1"], 2)

    def test_continuation_and_decode_match_serial(self) -> None:
        model, scheduler = self.make_scheduler(chunk=4)
        prompt = [1, 2, 3, 4, 5, 6]
        continuation = [30, 31, 32, 33, 34]
        scheduler.admit("request", prompt)
        scheduler.prefill()
        scheduler.continue_tokens("request", continuation)
        actual = scheduler.greedy_decode(
            ["request"],
            max_new_tokens=6,
        )["request"]
        reference_state, reference_logits, expected = serial(
            model,
            prompt,
            continuation=continuation,
            decode_tokens=6,
        )
        self.assertEqual(actual, expected)
        request = scheduler.request("request")
        self.assertTrue(torch.equal(request.logits, reference_logits[0]))
        snapshot = scheduler.pool.snapshot(request.handle)
        for left, right in zip(snapshot, reference_state, strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_context_limit_and_cancel(self) -> None:
        _, scheduler = self.make_scheduler(chunk=4)
        scheduler.admit("request", list(range(125)))
        scheduler.prefill()
        with self.assertRaises(ValueError):
            scheduler.continue_tokens("request", [1, 2, 3, 4])
        scheduler.cancel("request")
        with self.assertRaises(RuntimeError):
            scheduler.continue_tokens("request", [1])

    def test_prefill_round_and_explicit_decode_steps_match_serial(self) -> None:
        model, scheduler = self.make_scheduler(chunk=4)
        prompt = list(range(1, 12))
        scheduler.admit("request", prompt)

        scheduler.prefill_round(["request"])
        self.assertEqual(scheduler.request("request").remaining, 7)
        scheduler.prefill_round(["request"])
        self.assertEqual(scheduler.request("request").remaining, 3)
        scheduler.prefill_round(["request"])
        self.assertEqual(scheduler.request("request").remaining, 0)

        output = []
        for _ in range(6):
            token = scheduler.sample_next(["request"])["request"]
            if token == 0:
                break
            output.append(token)
            scheduler.advance_tokens({"request": token})

        reference_state, reference_logits, expected = serial(
            model,
            prompt,
            decode_tokens=6,
        )
        self.assertEqual(output, expected)
        request = scheduler.request("request")
        self.assertTrue(torch.equal(request.logits, reference_logits[0]))
        snapshot = scheduler.pool.snapshot(request.handle)
        for left, right in zip(snapshot, reference_state, strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_fork_and_batched_continuations_match_serial(self) -> None:
        model, scheduler = self.make_scheduler(chunk=4)
        prompt = [1, 2, 3, 4, 5]
        continuations = {
            "child-a": [21, 22, 23, 24, 25],
            "child-b": [31, 32, 33],
        }
        scheduler.admit("root", prompt)
        scheduler.prefill(["root"])
        root_before = scheduler.pool.snapshot(
            scheduler.request("root").handle
        )
        children = scheduler.fork("root", list(continuations))
        self.assertEqual([child.seen_tokens for child in children], [5, 5])

        scheduler.continue_many(list(continuations.items()))
        for request_id, continuation in continuations.items():
            reference_state, reference_logits, _ = serial(
                model,
                prompt,
                continuation=continuation,
            )
            request = scheduler.request(request_id)
            self.assertTrue(torch.equal(request.logits, reference_logits[0]))
            for actual, expected in zip(
                scheduler.pool.snapshot(request.handle),
                reference_state,
                strict=True,
            ):
                self.assertTrue(torch.equal(actual, expected))

        for actual, expected in zip(
            scheduler.pool.snapshot(scheduler.request("root").handle),
            root_before,
            strict=True,
        ):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(scheduler.metrics()["forked_states"], 2)


if __name__ == "__main__":
    unittest.main()
