from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # The CPU-only public test job omits model extras.
    raise unittest.SkipTest("torch is required for HF scheduler tests") from exc

from rwkv7_scheduler import HFRecurrentScheduler, SchedulerConfig


class FakeCache:
    def __init__(self, state, xpa, xpf, v_first, seen_tokens: int = 0):
        self.state = state
        self.xpa = xpa
        self.xpf = xpf
        self.v_first = v_first
        self.seen_tokens = int(seen_tokens)

    def __iter__(self):
        yield self.state
        yield self.xpa
        yield self.xpf
        yield self.v_first

    def clone(self):
        return type(self)(
            [value.clone() for value in self.state],
            [value.clone() for value in self.xpa],
            [value.clone() for value in self.xpf],
            self.v_first.clone(),
            seen_tokens=self.seen_tokens,
        )

    def select_batch(self, indices, *, inplace=False):
        del inplace
        indices = indices.to(dtype=torch.long)
        return type(self)(
            [value.index_select(0, indices) for value in self.state],
            [value.index_select(0, indices) for value in self.xpa],
            [value.index_select(0, indices) for value in self.xpf],
            self.v_first.index_select(0, indices),
            seen_tokens=self.seen_tokens,
        )


class FakeCanonicalCache:
    """Shape-compatible stand-in for the published RWKV7Cache 0.10 API."""

    def __init__(
        self,
        recurrent_state,
        attention_shift,
        ffn_shift,
        *,
        seen_tokens: int = 0,
    ):
        self.recurrent_state = recurrent_state
        self.attention_shift = attention_shift
        self.ffn_shift = ffn_shift
        self.seen_tokens = int(seen_tokens)

    def __iter__(self):
        return iter(
            zip(
                self.recurrent_state,
                self.attention_shift,
                self.ffn_shift,
                strict=True,
            )
        )

    def clone(self):
        return type(self)(
            [value.clone() for value in self.recurrent_state],
            [value.clone() for value in self.attention_shift],
            [value.clone() for value in self.ffn_shift],
            seen_tokens=self.seen_tokens,
        )

    def select_batch(self, indices, *, inplace=False):
        del inplace
        indices = indices.to(dtype=torch.long)
        return type(self)(
            [value.index_select(0, indices) for value in self.recurrent_state],
            [value.index_select(0, indices) for value in self.attention_shift],
            [value.index_select(0, indices) for value in self.ffn_shift],
            seen_tokens=self.seen_tokens,
        )


class FakeHFModel:
    """Small deterministic recurrent model with the Native cache shape."""

    def __init__(self, vocab_size: int = 257):
        self.vocab_size = vocab_size
        self.calls = []
        self.returned_caches = []

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
        logits_to_keep,
        return_dict,
    ):
        assert use_cache and logits_to_keep == 1 and return_dict
        batch, width = input_ids.shape
        if past_key_values is None:
            value = torch.zeros(batch, dtype=torch.long)
            seen = 0
        else:
            value = tuple(past_key_values)[0][0][:, 0].to(dtype=torch.long)
            seen = int(past_key_values.seen_tokens)
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        for index in range(width):
            advanced = (value * 31 + input_ids[:, index] + 1) % self.vocab_size
            value = torch.where(mask[:, index].bool(), advanced, value)
        logits = torch.full((batch, 1, self.vocab_size), -1000.0)
        logits[torch.arange(batch), 0, value] = 10.0
        encoded = value.view(batch, 1).float()
        cache = FakeCache(
            [encoded],
            [encoded + 1],
            [encoded + 2],
            encoded + 3,
            seen_tokens=seen + width,
        )
        self.calls.append((batch, width, tuple(mask.sum(dim=1).tolist())))
        self.returned_caches.append(cache)
        return SimpleNamespace(logits=logits, past_key_values=cache)


class FakeCanonicalHFModel(FakeHFModel):
    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
        logits_to_keep,
        return_dict,
    ):
        assert use_cache and logits_to_keep == 1 and return_dict
        batch, width = input_ids.shape
        if past_key_values is None:
            value = torch.zeros(batch, dtype=torch.long)
            seen = 0
        else:
            value = past_key_values.recurrent_state[0][:, 0].to(dtype=torch.long)
            seen = int(past_key_values.seen_tokens)
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        for index in range(width):
            advanced = (value * 31 + input_ids[:, index] + 1) % self.vocab_size
            value = torch.where(mask[:, index].bool(), advanced, value)
        logits = torch.full((batch, 1, self.vocab_size), -1000.0)
        logits[torch.arange(batch), 0, value] = 10.0
        encoded = value.view(batch, 1).float()
        cache = FakeCanonicalCache(
            [encoded],
            [encoded + 1],
            [encoded + 2],
            seen_tokens=seen + width,
        )
        self.calls.append((batch, width, tuple(mask.sum(dim=1).tolist())))
        self.returned_caches.append(cache)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def make_scheduler(*, capacity=8, max_batch=4, chunk=2):
    return HFRecurrentScheduler(
        FakeHFModel(),
        config=SchedulerConfig(
            prefill_chunk_size=chunk,
            max_batch_size=max_batch,
            max_queue_size=capacity,
            max_input_tokens=64,
        ),
        device="cpu",
        capacity=capacity,
    )


def make_canonical_scheduler(*, capacity=8, max_batch=4, chunk=2):
    return HFRecurrentScheduler(
        FakeCanonicalHFModel(),
        config=SchedulerConfig(
            prefill_chunk_size=chunk,
            max_batch_size=max_batch,
            max_queue_size=capacity,
            max_input_tokens=64,
        ),
        device="cpu",
        capacity=capacity,
    )


class HFRecurrentSchedulerTests(unittest.TestCase):
    def test_variable_prompt_batch_and_release(self):
        scheduler = make_scheduler()
        scheduler.admit("a", [1, 2, 3])
        scheduler.admit("b", [9])
        scheduler.prefill(["a", "b"])

        self.assertEqual(scheduler.request("a").seen_tokens, 3)
        self.assertEqual(scheduler.request("b").seen_tokens, 1)
        self.assertEqual(scheduler.pool.allocated, 2)
        self.assertIn("B2T2", scheduler.metrics()["shape_counts"])

        scheduler.release("a")
        scheduler.release("b")
        self.assertEqual(scheduler.pool.allocated, 0)
        self.assertEqual(scheduler.pool.free, scheduler.pool.capacity)

    def test_batched_continuation_matches_isolated_rows(self):
        batched = make_scheduler()
        isolated = make_scheduler()
        for scheduler in (batched, isolated):
            scheduler.admit("a", [1, 2, 3])
            scheduler.admit("b", [7, 8])
            scheduler.prefill(["a", "b"])

        batched.continue_many([("a", [4, 5]), ("b", [9])])
        isolated.continue_many([("a", [4, 5])])
        isolated.continue_many([("b", [9])])

        self.assertTrue(torch.equal(batched.request("a").logits, isolated.request("a").logits))
        self.assertTrue(torch.equal(batched.request("b").logits, isolated.request("b").logits))
        self.assertEqual(batched.sample_next(["a", "b"]), isolated.sample_next(["a", "b"]))

    def test_single_row_reuses_returned_cache_without_gather_or_scatter_copy(self):
        scheduler = make_scheduler()
        scheduler.admit("a", [1, 2, 3])
        scheduler.prefill(["a"])
        self.assertIs(scheduler.request("a").cache, scheduler.model.returned_caches[-1])

        scheduler.continue_many([("a", [4])])
        self.assertIs(scheduler.request("a").cache, scheduler.model.returned_caches[-1])
        self.assertEqual(scheduler.pool.workspace_bytes, 0)

    def test_fork_is_independent(self):
        scheduler = make_scheduler()
        scheduler.admit("root", [3, 4])
        scheduler.prefill(["root"])
        scheduler.fork("root", ["left", "right"])

        scheduler.continue_many([("left", [1]), ("right", [2])])
        self.assertNotEqual(
            scheduler.sample_next(["left"])["left"],
            scheduler.sample_next(["right"])["right"],
        )
        self.assertEqual(scheduler.request("root").seen_tokens, 2)

    def test_capacity_and_atomic_continuation_validation(self):
        scheduler = make_scheduler(capacity=2, max_batch=2)
        scheduler.admit("a", [1])
        scheduler.admit("b", [2])
        with self.assertRaises(RuntimeError):
            scheduler.admit("c", [3])
        scheduler.prefill()
        with self.assertRaises(ValueError):
            scheduler.install_continuations([("a", [4]), ("a", [5])])
        self.assertEqual(scheduler.request("a").remaining, 0)

    def test_canonical_010_cache_batches_and_reports_memory(self):
        scheduler = make_canonical_scheduler()
        scheduler.admit("a", [1, 2, 3])
        scheduler.admit("b", [7, 8])
        scheduler.prefill(["a", "b"])
        scheduler.continue_many([("a", [4]), ("b", [9])])

        self.assertIsInstance(
            scheduler.request("a").cache,
            FakeCanonicalCache,
        )
        self.assertGreater(scheduler.metrics()["pool"]["slab_bytes"], 0)
        self.assertEqual(scheduler.metrics()["shape_counts"]["B2T1"], 1)


if __name__ == "__main__":
    unittest.main()
