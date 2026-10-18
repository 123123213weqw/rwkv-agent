from __future__ import annotations

from rwkv_agent.longbench_state import (
    build_state_evidence_views,
    run_state_longbench_chunk_ensemble,
    run_state_longbench_permutation_reader,
    run_state_longbench_reader,
)
from rwkv_agent.tools.long_text import TextChunk


class FakeStateModel:
    def __init__(self) -> None:
        self.released: list[str] = []

    def state_prefill(self, *, owner_id: str, prompt: str):
        assert "Question" in prompt
        return {"state_id": "root", "home_url": "fake://model"}

    def state_fork(self, *, branches: list[str], **kwargs):
        return [
            {"state_id": f"branch-{index}", "branch": branch}
            for index, branch in enumerate(branches)
        ]

    def state_batch_continue(self, *, items, **kwargs):
        if items[0]["state_id"] == "root":
            return [{"text": "B", "output_tokens": 1}]
        return [
            {
                "text": choice + '\",\"support\":\"quoted\"}',
                "branch": item["state_id"],
                "output_tokens": 4,
            }
            for item, choice in zip(items, "ABBC", strict=True)
        ]

    def state_release(self, *, state_ids, **kwargs):
        self.released = list(state_ids)
        return {"status": "ok", "released": len(state_ids)}

    def state_batch_classify(self, *, items, **kwargs):
        winners = ("BADC" * ((len(items) + 3) // 4))[: len(items)]
        return [
            {
                "branch": item["state_id"],
                "scores": {
                    label: (10.0 if label == winner else 0.0)
                    for label in "ABCD"
                },
            }
            for item, winner in zip(items, winners, strict=True)
        ]


def test_state_reader_partitions_excerpts_and_releases_all_states() -> None:
    model = FakeStateModel()
    selected = [
        (float(12 - index), TextChunk(index, f"chunk {index}", index, index + 1))
        for index in range(12)
    ]

    result = run_state_longbench_reader(
        model,
        question="Question with A, B, C, D?",
        selected=selected,
        session_id="case-1",
    )

    assert result["choice"] == "B"
    assert len(result["reports"]) == 4
    assert sorted(
        chunk_id
        for report in result["reports"]
        for chunk_id in report["chunk_ids"]
    ) == list(range(6, 12))
    assert model.released == [
        "root",
        "branch-0",
        "branch-1",
        "branch-2",
        "branch-3",
    ]


def test_state_views_keep_lexical_baseline_and_add_complementary_packets() -> None:
    chunks = [
        TextChunk(index, text, index * 10, index * 10 + len(text))
        for index, text in enumerate(
            [
                "unrelated introduction",
                "project launch date was 2025",
                "nearby launch context",
                "model B was selected",
                "model A was rejected",
                "closing appendix",
            ]
        )
    ]
    question = "When did the project launch?\nA. 2024\nB. 2025\nC. 2026\nD. 2027"
    views = build_state_evidence_views(question, chunks, chunks_per_view=3)
    assert len(views) == 4
    assert all(view for view in views)
    assert any(chunk.chunk_id == 1 for _score, chunk in views[0])
    assert len({tuple(chunk.chunk_id for _score, chunk in view) for view in views}) > 1


def test_state_permutation_reader_maps_all_label_positions_back_to_content() -> None:
    model = FakeStateModel()
    selected = [
        (float(6 - index), TextChunk(index, f"evidence {index}", index, index + 1))
        for index in range(6)
    ]
    result = run_state_longbench_permutation_reader(
        model,
        question="Which is supported?\nA. alpha\nB. beta\nC. gamma\nD. delta",
        selected=selected,
        session_id="case-permute",
    )
    assert result["choice"] == "B"
    assert result["choice_scores"]["B"] == 40.0
    assert result["state_leak_count"] == 0
    assert len(model.released) == 5


def test_state_chunk_ensemble_reads_eight_chunks_and_cancels_label_positions() -> None:
    model = FakeStateModel()
    selected = [
        (float(8 - index), TextChunk(index, f"evidence {index}", index, index + 1))
        for index in range(8)
    ]
    result = run_state_longbench_chunk_ensemble(
        model,
        question="Which is supported?\nA. alpha\nB. beta\nC. gamma\nD. delta",
        selected=selected,
        session_id="case-chunks",
    )
    assert result["choice"] == "B"
    assert result["choice_scores"]["B"] == 80.0
    assert len(result["reports"]) == 8
    assert result["state_leak_count"] == 0
    assert len(model.released) == 9
