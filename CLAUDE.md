# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

- Install for local development:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e '.[realtime,dev]'
  ```
- Optional extras:
  ```bash
  pip install -e '.[model]'      # Hugging Face RWKV support
  pip install -e '.[browser]'    # Playwright fallback for a small number of JS pages
  ```
- Initialize the SQLite schema and run the local Web UI with extractive fallback:
  ```bash
  rwkv-search --config configs/default.json init
  rwkv-search --config configs/default.json serve --host 127.0.0.1 --port 8765 --no-model
  ```
- Run the full test suite:
  ```bash
  PYTHONPATH=src python -m unittest discover -s tests
  ```
- Run one test module or one test case:
  ```bash
  PYTHONPATH=src python -m unittest tests.test_realtime_search
  PYTHONPATH=src python -m unittest tests.test_realtime_search.TestRealtimeSearch.test_name
  ```
- Run lint before submitting changes:
  ```bash
  ruff check src bench tests
  ```
- Run a small realtime retrieval benchmark smoke test:
  ```bash
  PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
    --config configs/benchmark.json \
    --case-id retrieval-zh-001 \
    --case-id retrieval-en-006
  ```
- Run the full 50-case retrieval benchmark:
  ```bash
  PYTHONPATH=src python bench/run_realtime_retrieval_bench.py --config configs/benchmark.json
  ```

## Architecture overview

RWKV Agent is the user-facing local-first chat and research product. Its internal RWKV Search subsystem separates search gating, query/tool-call formation, URL discovery, candidate filtering, fetching/extraction, evidence construction, and answer generation so each stage can be benchmarked independently.

Two request paths are central:

- Ordinary chat path: `SearchService` routes the current user turn through `RuleRouter`; if no retrieval is needed, the request goes directly to the configured RWKV answerer or an extractive/chat fallback.
- Search path: G1I/P4 generates a strict single `web_search` tool call, `SearchRequest`-style deterministic code merges hard constraints from the original question, discovery finds URLs, candidate admission and precision discovery refine them, pages are fetched/extracted, `EvidenceBuilder` creates citations, then RWKV or fallback answer generation consumes the evidence.

Important implementation areas:

- `src/rwkv_search/cli.py` wires CLI commands (`init`, `crawl`, `search`, `ask`, `serve`, `cc-import`, `stats`) to configuration, database, crawler, service, and model loading.
- `src/rwkv_search/service.py` is the main orchestration layer for routing, local/realtime retrieval, FineWiki shadow search, evidence construction, streaming answer events, and fallback behavior.
- `src/rwkv_search/api.py` provides the local HTTP server, static web UI, health/search/crawl endpoints, and SSE chat endpoints. The v1 stream endpoint wraps service events into protocol events from `protocol.py` and supports cancellation/debug traces.
- `src/rwkv_search/realtime/` contains realtime web retrieval: `engine.py` owns the persistent asyncio runtime; `discovery.py` talks to SearXNG/HTML fallback; `candidate_ranker.py`, `precision_discovery.py`, and `ranker.py` handle candidate admission, source channels/domain pivot/one-hop expansion, and final ranking; `fetcher.py` and `extractor.py` handle bounded network fetch and page text extraction.
- Local indexed search lives around `db.py`, `crawler.py`, `search.py`, `evidence.py`, plus importers such as `commoncrawl.py`, `wikipedia.py`, and `finewiki.py`.
- RWKV integration and tool-call handling are in `rwkv_answerer.py`, `g1i_native.py`, `g1i_tool_call.py`, `g1i_types.py`, and `p4_search.py`.
- `contracts/` holds JSON schemas for chat requests/events/errors/sources/evidence; keep API changes consistent with these contracts and the tests.
- `bench/` contains public datasets, runners, metrics, and frozen baselines. Benchmark run outputs belong in ignored `bench/runs/`; only reviewed summaries should be added under `bench/baselines/`.

## Project constraints from docs

- Python 3.10+ is required; package source is under `src/` and tests expect `PYTHONPATH=src` when not installed.
- Realtime retrieval features such as candidate admission, source channels, domain pivot, and one-hop link expansion are controlled by config and are not all enabled in `configs/default.json`. Use benchmarks or shadow validation before changing production defaults.
- Retrieval quality changes must compare against the same `bench/realtime_web_retrieval.jsonl`, configuration, and metrics; do not collapse URL discovery, crawling/extraction, evidence quality, and model answer quality into one number.
- New source strategies should be based on general page/source characteristics, not industry-specific routing tables.
- Do not commit model weights, crawled page bodies, debug token traces, private server addresses, secrets, or local absolute paths.
