# RWKV Agent

Local-first RWKV chat, tool use, Web research and evidence-grounded answers.

- **Current release:** `v0.3.0-beta.1`
- **Primary interface:** `rwkv` terminal client + RWKV Agent HTTP backend
- **Verified model:** RWKV-7 G1I Preview4922 13.3B, context 12,288

RWKV Agent keeps ordinary chat fast and enters retrieval only when the user
explicitly requests search or the semantic Search Gate selects a tool. Search,
URL discovery, page extraction, Evidence selection and answer generation remain
separate and auditable stages.

**RWKV Agent** is the user-facing product. **RWKV Search** is its internal
retrieval subsystem. The repository and Python package retain the `rwkv-search`
compatibility name in this Beta.

> This Beta is suitable for local use and controlled internal deployment. The
> HTTP service has no public authentication or rate limiting; keep it on
> loopback or behind your own authenticated gateway.

## What works

- ordinary multi-turn RWKV chat;
- strict greedy `web_search`, `knowledge_search` and `long_text_qa` Tool Calls;
- Tavily, GitHub, MediaWiki and Crossref discovery, with bounded Bing fallback;
- low-resource static page extraction and evidence-based answers;
- explicit B1-B4, one-to-three-round state-native Web research;
- session transcript, pasted long-text QA, citations and safe abstention;
- reproducible retrieval and 200-case Agent regression benchmarks.

The older `rwkv-search` Web UI remains available as a **Legacy Web Preview**.
It does not yet expose every capability of the current Agent backend and is not
the recommended first-run experience.

## Five-minute setup

### 1. Install

Requirements: Linux CUDA host, Python 3.10+, Rust toolchain, `curl`, an RWKV G1I
checkpoint and a compatible Albatross runtime.

```bash
git clone https://github.com/123123213weqw/rwkv-search.git
cd rwkv-search

python -m venv .venv
source .venv/bin/activate
pip install -e '.[realtime,agent]'

cd cli
./install.sh
cd ..
```

### 2. Configure the model

```bash
rwkv-agent-service init
$EDITOR ~/.config/rwkv-agent/rwkv-agent.env
rwkv-agent-service doctor
```

Set absolute values for `RWKV_AGENT_PROJECT_ROOT`, `RWKV_AGENT_PYTHON`,
`G1I_MODEL_PATH` and `G1I_RUNTIME_DIR`. The complete template is
[`.env.example`](.env.example). Model and runtime details are in
[Model setup](docs/MODEL_SETUP.md).

### 3. Start and chat

```bash
rwkv
```

The `rwkv` launcher checks the local Controller, starts the configured RWKV
model backend when needed, and opens an interactive conversation.

Useful commands:

```bash
rwkv ask "你好"
rwkv tool web-search "Python latest stable release"
rwkv research --branches 4 --rounds 2 \
  "Who created RWKV and what did the official repositories update recently?"
```

Administrator commands:

```bash
rwkv-agent-service status
rwkv-agent-service logs
rwkv-agent-service stop
```

See [Quickstart](docs/QUICKSTART.md) for SearXNG, remote GPU hosts and failure
recovery.

## Architecture

```mermaid
flowchart LR
    U["User / rwkv"] --> API["Agent HTTP"]
    API --> G["Semantic Search Gate"]
    G -->|"chat"| M["RWKV 13.3B"]
    G -->|"tool"| T["Strict Tool Call"]
    T --> D["Structured providers / SearXNG / fallback"]
    D --> F["Bounded fetch + extraction"]
    F --> E["Evidence selection + claim checks"]
    E --> M
    M --> A["Answer + citations"]
    API --> R["Opt-in state research"]
    R --> D
```

The model produces a small tool request, not a large Planner JSON document.
Time, source and explicit-site constraints are merged deterministically. The
retrieval path uses general source/page features rather than topic-specific
finance, software or policy routers.

## Verified quality

The current 13.3B stack was evaluated on 200 fixed cases: 40 each from BFCL,
WebWalkerQA, FRAMES, LongBench v2 and ALCE.

| Metric | 7.2B P0 | 13.3B P0 |
|---|---:|---:|
| BFCL official AST | 67.5% | **87.5%** |
| BFCL Tool Call exact | 45.0% | **57.5%** |
| FRAMES answer F1 | 1.00% | **10.90%** |
| FRAMES domain recall | 0% | **57.5%** |
| FRAMES exact-page recall | 0% | **14.79%** |
| LongBench v2 choice accuracy | 25.0% | **27.5%** |
| ALCE citation exact-page recall | 35.5% | 35.5% |

The run had no HTTP 409, route, state-leak or budget-overrun failures. It also
found three malformed BFCL Tool Calls, weak WebWalker coverage and increased
unsupported claims on answered FRAMES cases. These are tracked explicitly in
[Known issues](docs/KNOWN_ISSUES.md); this table is not a claim of
Doubao/DeepSeek-level search quality.

Machine-readable summaries:

- [`e2e-p0-13b-preview4922-summary.json`](bench/baselines/agent-unified-regression-v1/e2e-p0-13b-preview4922-summary.json)
- [`e2e-p0-13b-vs-7b-comparison.json`](bench/baselines/agent-unified-regression-v1/e2e-p0-13b-vs-7b-comparison.json)
- [`public-results-manifest.json`](bench/baselines/agent-unified-regression-v1/public-results-manifest.json)

## Repository layout

```text
src/rwkv_agent/       Current Agent Controller, tools and Sidecar
src/rwkv7_scheduler/  Recurrent state pool and continuous batching
src/rwkv_search/      RWKV Search retrieval subsystem and Legacy Web Preview
cli/                  Rust terminal client and local service lifecycle
configs/              Default, production example and benchmark configs
deploy/               Agent host and optional SearXNG examples
contracts/            Chat, Evidence, source and error schemas
benchmarks/            Agent evaluation runners
bench/baselines/      Reviewed, publishable benchmark summaries
docs/                 User, architecture and development documentation
tests/                Unit and regression tests
```

## Documentation

- [Quickstart](docs/QUICKSTART.md)
- [Model setup](docs/MODEL_SETUP.md)
- [Local knowledge service](docs/KNOWLEDGE_SETUP.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Configuration](docs/CONFIGURATION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Known issues and optimization backlog](docs/KNOWN_ISSUES.md)
- [Release checklist](docs/RELEASE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Benchmark method](docs/AGENT_BENCHMARK.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## Data and privacy

The repository does not include model weights, API keys, crawled page bodies,
private server configuration, complete token traces or license-restricted
benchmark cases. Runtime state belongs under the configured state directory and
is ignored by Git.

## License

[MIT](LICENSE)
