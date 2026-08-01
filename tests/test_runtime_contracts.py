from __future__ import annotations

import unittest

from rwkv_runtime.classification import finite_label_scores


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class Logits:
    def __init__(self, values: dict[int, float]) -> None:
        self.values = values

    def __getitem__(self, token: int) -> Scalar:
        return Scalar(self.values[token])


class RuntimeContractTests(unittest.TestCase):
    def test_finite_label_scores_preserves_labels(self) -> None:
        self.assertEqual(
            finite_label_scores(
                Logits({1: 4.0, 2: -1.5}),
                {"tool": 1, "chat": 2},
            ),
            {"tool": 4.0, "chat": -1.5},
        )

    def test_non_finite_scores_use_caller_error_contract(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "persistent bad logits"):
            finite_label_scores(
                Logits({1: float("nan"), 2: 0.0}),
                {"tool": 1, "chat": 2},
                error_message="persistent bad logits",
            )


if __name__ == "__main__":
    unittest.main()
