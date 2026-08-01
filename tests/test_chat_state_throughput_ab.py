from __future__ import annotations

import unittest

from benchmarks.run_chat_state_throughput_ab import (
    CHAT_STOPS,
    DIRECT_SYSTEM_PROMPT,
    build_conversations,
    compare_outputs,
    percentile,
    require_persistent_capacity,
    run_state_workload,
    run_transcript_workload,
    summarize_shape_delta,
)
from rwkv_agent.chat_prompts import (
    CHAT_STOPS as PRODUCTION_CHAT_STOPS,
    DIRECT_SYSTEM_PROMPT as PRODUCTION_DIRECT_SYSTEM_PROMPT,
)


class FakeSidecar:
    endpoint = "fake://sidecar"

    def __init__(self) -> None:
        self.states: dict[str, dict[str, str]] = {}
        self.counter = 0

    def get(self, _path: str) -> dict:
        return {
            "inference": {
                "scheduler": {
                    "shape_counts": {
                        "B1T1": self.counter,
                        "B2T64": self.counter // 2,
                    }
                }
            },
            "persistent_states": {"allocated": len(self.states)},
        }

    def post(self, path: str, payload: dict) -> dict:
        self.counter += 1
        if path == "/v1/completions":
            prompt = str(payload["prompt"])
            token = len(prompt) % 65535
            return {
                "g1i": {
                    "text": "x",
                    "token_ids": [token],
                    "stop_reason": "max_tokens",
                    "queue_ms": 0,
                }
            }
        if path == "/v1/states/prefill":
            state_id = f"state-{len(self.states) + 1}"
            self.states[state_id] = {
                "owner_id": str(payload["owner_id"]),
                "text": str(payload["prompt"]),
            }
            return {"state": {"state_id": state_id, "seen_tokens": 1}}
        if path == "/v1/states/batch_continue":
            item = payload["items"][0]
            state = self.states[str(item["state_id"])]
            self.assert_owner(state, str(payload["owner_id"]))
            state["text"] += str(item["input"])
            token = len(state["text"]) % 65535
            state["text"] += "x"
            return {
                "results": [
                    {
                        "state_id": item["state_id"],
                        "text": "x",
                        "token_ids": [token],
                        "stop_reason": "max_tokens",
                        "seen_tokens": len(state["text"]),
                    }
                ]
            }
        if path == "/v1/states/release":
            released = 0
            for state_id in payload["state_ids"]:
                state = self.states[state_id]
                self.assert_owner(state, str(payload["owner_id"]))
                self.states.pop(state_id)
                released += 1
            return {"released": released}
        raise AssertionError(path)

    @staticmethod
    def assert_owner(state: dict[str, str], owner_id: str) -> None:
        if state["owner_id"] != owner_id:
            raise AssertionError("owner mismatch")


class ChatStateThroughputABTests(unittest.TestCase):
    def test_frozen_protocol_matches_production_chat_protocol(self) -> None:
        self.assertEqual(DIRECT_SYSTEM_PROMPT, PRODUCTION_DIRECT_SYSTEM_PROMPT)
        self.assertEqual(CHAT_STOPS, PRODUCTION_CHAT_STOPS)

    def test_frozen_conversations_are_bounded_and_deterministic(self) -> None:
        first = build_conversations(4, turns=3)
        second = build_conversations(4, turns=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual([len(item.messages) for item in first], [3] * 4)
        self.assertGreater(len(first[0].messages[0]), 1000)

    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(percentile([4, 1, 3, 2], 0.50), 2)
        self.assertEqual(percentile([4, 1, 3, 2], 0.95), 4)
        self.assertEqual(percentile([], 0.95), 0)

    def test_shape_delta_reports_decode_and_prefill_batch_fill(self) -> None:
        before = {
            "inference": {
                "scheduler": {"shape_counts": {"B1T1": 2, "B2T64": 1}}
            }
        }
        after = {
            "inference": {
                "scheduler": {
                    "shape_counts": {
                        "B1T1": 4,
                        "B4T1": 3,
                        "B2T64": 3,
                    }
                }
            }
        }
        result = summarize_shape_delta(before, after)
        self.assertEqual(result["shape_counts"], {"B1T1": 2, "B2T64": 2, "B4T1": 3})
        self.assertEqual(result["decode_average_batch_fill"], 2.8)
        self.assertEqual(result["prefill_average_batch_fill"], 2.0)

    def test_fake_transport_proves_both_arms_and_release_contract(self) -> None:
        transport = FakeSidecar()
        conversations = build_conversations(4, turns=3)
        transcript = run_transcript_workload(
            transport,
            conversations,
            max_tokens=1,
        )
        state = run_state_workload(
            transport,
            conversations,
            max_tokens=1,
        )
        comparison = compare_outputs(transcript, state)
        self.assertTrue(comparison["all_exact"])
        self.assertEqual(comparison["compared_turns"], 12)
        self.assertEqual(state["released_states"], 4)
        self.assertEqual(transport.states, {})

    def test_capacity_preflight_refuses_partial_state_allocation(self) -> None:
        health = {"persistent_states": {"capacity": 8, "allocated": 0}}
        self.assertEqual(require_persistent_capacity(health, (1, 4, 8)), 8)
        with self.assertRaisesRegex(RuntimeError, "below required concurrency 16"):
            require_persistent_capacity(health, (1, 4, 8, 16))
        with self.assertRaisesRegex(RuntimeError, "must be empty"):
            require_persistent_capacity(
                {"persistent_states": {"capacity": 16, "allocated": 1}},
                (16,),
            )


if __name__ == "__main__":
    unittest.main()
