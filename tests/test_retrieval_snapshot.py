from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

from benchmarks.analyze_retrieval_funnel import build_funnel
from benchmarks.retrieval_snapshot import (
    RetrievalSnapshotRecorder,
    freeze_config_snapshot,
    load_snapshots,
)
from rwkv_agent.tools.web import WebSearchAdapter


def _call(**stages):
    value = {
        "call_index": 1,
        "status": "ok",
        "query": "query",
        "original_query": "query",
        "effective_query": "query",
        "compiled_query": {},
        "scope_root": "https://example.org/",
        "scope_mode": "strict",
        "scope_rejected": {},
        "raw_candidates": [],
        "initial_candidates": [],
        "post_pivot_candidates": [],
        "candidates": [],
        "rejected_candidates": [],
        "results": [],
        "evidence": [],
        "fetches": [],
        "warnings": [],
        "stats": {},
        "latency_ms": 1.0,
        "evidence_stage": "result",
    }
    value.update(stages)
    return value


def _snapshot(case_id: str, **stages):
    return {
        "schema_version": "rwkv-agent-retrieval-snapshot.v1",
        "case_id": case_id,
        "calls": [_call(**stages)],
    }


def test_config_snapshot_redacts_values_and_keeps_environment_names() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "config.json"
        target = root / "config.snapshot.json"
        source.write_text(
            json.dumps(
                {
                    "realtime": {
                        "tavily_api_key_env": "TAVILY_API_KEY",
                        "api_key": "placeholder-private-value",
                        "engines": ["dogpile", "baidu"],
                    }
                }
            ),
            encoding="utf-8",
        )
        binding = freeze_config_snapshot(source, target)
        snapshot = json.loads(target.read_text(encoding="utf-8"))
        realtime = snapshot["config"]["realtime"]
        assert realtime["tavily_api_key_env"] == "TAVILY_API_KEY"
        assert realtime["api_key"] == "<redacted>"
        assert realtime["engines"] == ["dogpile", "baidu"]
        assert binding["artifact"] == "config.snapshot.json"
        assert len(binding["sha256"]) == 64


def test_recorder_writes_content_light_sorted_snapshots() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "retrieval.jsonl"
        recorder = RetrievalSnapshotRecorder(output)
        for case_id in ("case-b", "case-a"):
            with recorder.capture_case(case_id):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(
                        recorder.observe,
                        case_id,
                        {
                            "status": "ok",
                            "evidence": [
                                {
                                    "id": "W1",
                                    "uri": "https://example.org/a",
                                    "title": "A",
                                    "content": "must not be persisted",
                                }
                            ],
                        },
                        {
                            "status": "ok",
                            "query": "q",
                            "raw_candidates": [
                                {
                                    "url": "https://example.org/a",
                                    "title": "A",
                                    "snippet": "search excerpt",
                                }
                            ],
                            "results": [
                                {
                                    "url": "https://example.org/a",
                                    "title": "A",
                                    "content": "must not be persisted",
                                    "content_length": 21,
                                }
                            ],
                        },
                    ).result()
        recorder.finalize()
        rows = load_snapshots(output)
        assert [row["case_id"] for row in rows] == ["case-a", "case-b"]
        serialized = output.read_text(encoding="utf-8")
        assert "must not be persisted" not in serialized
        assert rows[0]["calls"][0]["evidence"][0]["uri"].endswith("/a")


def test_web_adapter_exposes_raw_candidates_to_trace_observer() -> None:
    observed = []

    class Engine:
        def search_events(self, *_args, **_kwargs):
            yield {
                "type": "discovery_progress",
                "progress": {
                    "raw_candidates": [
                        {
                            "url": "https://example.org/raw",
                            "title": "Raw",
                            "snippet": "raw candidate excerpt long enough for replay",
                            "engine": "demo",
                        }
                    ],
                    "candidates": [
                        {
                            "url": "https://example.org/kept",
                            "title": "Kept",
                            "snippet": "kept candidate excerpt long enough for replay",
                            "engine": "demo",
                        }
                    ],
                    "rejected_candidates": [
                        {
                            "url": "https://example.org/rejected",
                            "title": "Rejected",
                            "snippet": "rejected excerpt",
                            "engine": "demo",
                        }
                    ],
                },
            }
            yield {"type": "realtime_result", "results": [], "stats": {}}

        def close(self):
            return None

    adapter = WebSearchAdapter(
        engine=Engine(),
        shadow=False,
        trace_observer=lambda case_id, public, trace: observed.append(
            (case_id, public, trace)
        ),
    )
    with adapter.scoped("https://example.org/"), adapter.capture_trace_case("case"):
        adapter.execute("target page")
    assert observed[0][0] == "case"
    assert observed[0][2]["raw_candidates"][0]["url"].endswith("/raw")
    assert observed[0][2]["rejected_candidates"][0]["url"].endswith(
        "/rejected"
    )


def test_funnel_assigns_the_first_observed_loss_stage() -> None:
    target = "https://example.org/target"
    other = "https://example.org/other"
    cases = []
    results = []
    evaluations = []
    snapshots = []

    def add(case_id: str, snapshot, evidence=(), f1=0.0, citation=0.0):
        cases.append(
            {
                "id": case_id,
                "language": "en",
                "gold": {"source_uris": [target]},
            }
        )
        results.append(
            {"case_id": case_id, "evidence": [{"uri": uri} for uri in evidence]}
        )
        evaluations.append(
            {
                "case_id": case_id,
                "metrics": {
                    "answer_token_f1": f1,
                    "citation_exact_page_recall": citation,
                },
            }
        )
        snapshots.append(snapshot)

    add(
        "not-invoked",
        {
            "schema_version": "rwkv-agent-retrieval-snapshot.v1",
            "case_id": "not-invoked",
            "calls": [],
        },
    )

    add(
        "domain",
        _snapshot("domain", raw_candidates=[{"url": "https://other.test/a"}]),
    )
    add(
        "discovery",
        _snapshot("discovery", raw_candidates=[{"url": other}]),
    )
    add(
        "admission",
        _snapshot(
            "admission",
            raw_candidates=[{"url": target}],
            initial_candidates=[{"url": target}],
            candidates=[{"url": other}],
        ),
    )
    add(
        "fetch",
        _snapshot(
            "fetch",
            raw_candidates=[{"url": target}],
            initial_candidates=[{"url": target}],
            candidates=[{"url": target}],
        ),
    )
    add(
        "evidence",
        _snapshot(
            "evidence",
            raw_candidates=[{"url": target}],
            initial_candidates=[{"url": target}],
            candidates=[{"url": target}],
            results=[{"url": target}],
            evidence=[{"uri": target}],
        ),
        evidence=(other,),
    )
    add(
        "answer",
        _snapshot(
            "answer",
            raw_candidates=[{"url": target}],
            initial_candidates=[{"url": target}],
            candidates=[{"url": target}],
            results=[{"url": target}],
            evidence=[{"uri": target}],
        ),
        evidence=(target,),
    )
    add(
        "citation",
        _snapshot(
            "citation",
            raw_candidates=[{"url": target}],
            initial_candidates=[{"url": target}],
            candidates=[{"url": target}],
            results=[{"url": target}],
            evidence=[{"uri": target}],
        ),
        evidence=(target,),
        f1=1.0,
    )
    add(
        "pass",
        _snapshot(
            "pass",
            raw_candidates=[{"url": target}],
            initial_candidates=[{"url": target}],
            candidates=[{"url": target}],
            results=[{"url": target}],
            evidence=[{"uri": target}],
        ),
        evidence=(target,),
        f1=1.0,
        citation=1.0,
    )

    summary, rows = build_funnel(cases, results, evaluations, snapshots)
    blockers = {row["case_id"]: row["primary_blocker"] for row in rows}
    assert blockers == {
        "answer": "answer_synthesis_failure",
        "citation": "citation_binding_failure",
        "discovery": "exact_page_discovery_miss",
        "domain": "domain_discovery_miss",
        "evidence": "evidence_selection_loss",
        "fetch": "fetch_or_extraction_loss",
        "admission": "candidate_admission_or_rerank_loss",
        "not-invoked": "search_not_invoked",
        "pass": "partial_or_pass",
    }
    assert summary["cases"] == 9
    assert summary["primary_blockers"]["exact_page_discovery_miss"] == 1
