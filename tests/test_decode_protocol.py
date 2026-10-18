from __future__ import annotations

import unittest

from rwkv_runtime.decode import append_greedy_token, decode_text_stops


class Tokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token) for token in token_ids)


class GreedyDecodeProtocolTests(unittest.TestCase):
    def test_eos_is_not_appended_and_budget_is_exact(self) -> None:
        output: list[int] = []
        first = append_greedy_token(
            output,
            ord("a"),
            eos_token_id=0,
            max_tokens=2,
        )
        second = append_greedy_token(
            output,
            ord("b"),
            eos_token_id=0,
            max_tokens=2,
        )
        eos = append_greedy_token(
            output,
            0,
            eos_token_id=0,
            max_tokens=2,
        )
        self.assertFalse(first.finished)
        self.assertTrue(second.budget_reached)
        self.assertTrue(eos.eos)
        self.assertEqual(output, [ord("a"), ord("b")])

    def test_earliest_stop_wins_and_replacement_keeps_previous_text(self) -> None:
        result = decode_text_stops(
            Tokenizer(),
            [ord("a"), ord("!"), ord("?")],
            previous_text="",
            stops=["?", "!"],
        )
        self.assertEqual(result.text, "a")
        self.assertEqual(result.stop_reason, "!")

        class ReplacementTokenizer:
            def decode(self, _token_ids: list[int]) -> str:
                return "partial\ufffd"

        replacement = decode_text_stops(
            ReplacementTokenizer(),
            [1],
            previous_text="stable",
            stops=[],
        )
        self.assertEqual(replacement.text, "stable")
        self.assertEqual(replacement.stop_reason, "")


if __name__ == "__main__":
    unittest.main()
