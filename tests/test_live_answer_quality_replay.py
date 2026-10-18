import importlib.util
from pathlib import Path

import pytest

CAPTURE_DIR = Path("benchmarks/scheduler/unified_live_smoke_v1")
REQUIRED_CAPTURES = ("search.json", "research.json", "research-direct.json")
pytestmark = pytest.mark.skipif(
    not all((CAPTURE_DIR / name).is_file() for name in REQUIRED_CAPTURES),
    reason="live replay captures are local-only and are not part of the public release",
)
MODULE_PATH = Path("benchmarks/replay_live_answer_quality.py")
SPEC = importlib.util.spec_from_file_location("replay_live_answer_quality", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_live_answer_quality_replay_schema_and_positive_control() -> None:
    result = MODULE.replay(CAPTURE_DIR)

    assert result["schema_version"] == "rwkv-agent-live-answer-quality-replay.v1"
    assert result["metrics"] == {
        "search_unsafe_term_accept_count": 0,
        "invented_relation_accepted": False,
        "grounded_relation_accepted": True,
        "standalone_heading_output_count": 0,
        "unrelated_medical_evidence_selected": False,
        "ordinary_search_evidence_rejected": 1,
        "live_compound_claim_accepted": False,
        "live_irrelevant_research_accepted": False,
    }
    assert len(result["details"]["merged_evidence"]) == 8
