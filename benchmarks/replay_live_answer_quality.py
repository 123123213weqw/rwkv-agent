#!/usr/bin/env python3
"""Replay the frozen 2026-08-01 live-smoke answer-quality failures.

This benchmark never calls the network or the model.  It reuses the captured
CLI/Controller artifacts and applies the current deterministic Evidence merge
and claim gate so answer-safety changes can be compared with the same inputs.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from rwkv_agent.evidence_admission import EntityEvidenceAdmission
from rwkv_agent.state_answer import coordinate_answer_output
from rwkv_agent.state_evidence import _merge_evidence


DEFAULT_CAPTURE_DIR = Path(
    "benchmarks/scheduler/unified_live_smoke_v1"
)
QUESTION = "搜索 RWKV 的作者和维护组织，给出简短答案并标注来源。"
UNRELATED_MEDICAL_URI = "https://doi.org/10.1109/jbhi.2025.3588555"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _coordinate(
    answer: str,
    evidence: list[dict[str, Any]],
    *,
    question: str,
) -> dict[str, Any]:
    kwargs = (
        {"question": question}
        if "question" in inspect.signature(coordinate_answer_output).parameters
        else {}
    )
    return coordinate_answer_output(answer, evidence, **kwargs)


def replay(capture_dir: Path = DEFAULT_CAPTURE_DIR) -> dict[str, Any]:
    captures = {
        name: _load(capture_dir / name)
        for name in ("search.json", "research.json", "research-direct.json")
    }
    search = captures["search.json"]
    search_evidence = list(search["tool_result"]["evidence"])
    search_validation = _coordinate(
        str(search.get("answer") or ""),
        search_evidence,
        question=str(search.get("message") or QUESTION),
    )
    live_search_path = capture_dir / "quality-live-search.json"
    live_search = _load(live_search_path) if live_search_path.is_file() else None
    live_search_validation = (
        _coordinate(
            str(live_search.get("answer") or ""),
            list(live_search["tool_result"]["evidence"]),
            question=str(live_search.get("message") or QUESTION),
        )
        if live_search is not None
        else None
    )
    live_research_path = capture_dir / "quality-live-research-v2.json"
    live_research = _load(live_research_path) if live_research_path.is_file() else None
    live_research_validation = (
        _coordinate(
            str(live_research.get("answer") or ""),
            list(live_research["tool_result"]["evidence"]),
            question=str(live_research.get("message") or QUESTION),
        )
        if live_research is not None
        else None
    )

    false_positive_evidence = [
        {
            "id": "W1",
            "title": "PENG Bo (@BlinkDL)",
            "content": (
                "RWKV is all you need. GitHub user: BlinkDL. "
                "Public repositories: 35."
            ),
            "uri": "https://github.com/BlinkDL",
        }
    ]
    invented_relation = coordinate_answer_output(
        "BlinkDL拥有35个公开仓库，因此正式收购了RWKV。[W1]",
        false_positive_evidence,
    )
    grounded_relation = coordinate_answer_output(
        "BlinkDL拥有35个公开仓库。[W1]",
        false_positive_evidence,
    )

    heading_salvage = coordinate_answer_output(
        "维护组织：[W1]\nRWKV由Example Foundation维护。[W1]\n相关公司：[W2]",
        [
            {
                "id": "W1",
                "title": "RWKV maintenance",
                "content": "维护组织：Example Foundation 维护 RWKV。",
            },
            {"id": "W2", "title": "Weather", "content": "Rain tomorrow."},
        ],
    )

    merged = _merge_evidence(
        [
            {
                "query": str(value.get("message") or QUESTION),
                "evidence": list(value["tool_result"]["evidence"]),
            }
            for value in captures.values()
        ],
        question=QUESTION,
        limit=8,
    )
    admitted, admission_trace = EntityEvidenceAdmission().admit(
        QUESTION,
        search_evidence,
    )
    heading_units = [
        str(claim.get("text") or "")
        for claim in heading_salvage.get("claim_verification") or []
        if claim.get("claim_shape") == "heading"
        and claim.get("supported")
        and str(claim.get("text") or "")
        in str(heading_salvage.get("answer") or "")
    ]
    bad_search_terms = ("中国人", "RWKV-Vibe 组织中统一入口点")
    accepted_bad_search_terms = [
        term
        for term in bad_search_terms
        if search_validation.get("valid")
        and term in str(search_validation.get("answer") or "")
    ]
    metrics = {
        "search_unsafe_term_accept_count": len(accepted_bad_search_terms),
        "invented_relation_accepted": bool(invented_relation.get("valid")),
        "grounded_relation_accepted": bool(grounded_relation.get("valid")),
        "standalone_heading_output_count": len(heading_units),
        "unrelated_medical_evidence_selected": any(
            item.get("uri") == UNRELATED_MEDICAL_URI for item in merged
        ),
        "ordinary_search_evidence_rejected": len(search_evidence) - len(admitted),
        "live_compound_claim_accepted": bool(
            live_search_validation and live_search_validation.get("valid")
        ),
        "live_irrelevant_research_accepted": bool(
            live_research_validation and live_research_validation.get("valid")
        ),
    }
    return {
        "schema_version": "rwkv-agent-live-answer-quality-replay.v1",
        "capture_dir": str(capture_dir),
        "metrics": metrics,
        "details": {
            "accepted_bad_search_terms": accepted_bad_search_terms,
            "search_validation": search_validation,
            "invented_relation": invented_relation,
            "grounded_relation": grounded_relation,
            "heading_salvage": heading_salvage,
            "standalone_heading_units": heading_units,
            "merged_evidence": merged,
            "ordinary_search_admitted_uris": [item.get("uri") for item in admitted],
            "ordinary_search_admission": admission_trace.to_dict(),
            "live_search_validation": live_search_validation,
            "live_research_validation": live_research_validation,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.capture_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
