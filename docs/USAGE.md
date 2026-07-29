# Usage

## Current Agent path

The supported `v0.3.0-beta.1` experience is the Rust CLI connected to the
state-native Python Agent backend.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[realtime,agent]'

cd cli && ./install.sh && cd ..
rwkv-agent-service init
$EDITOR ~/.config/rwkv-agent/rwkv-agent.env
rwkv-agent-service doctor
rwkv
```

Detailed instructions:

- [Quickstart](QUICKSTART.md)
- [Model setup](MODEL_SETUP.md)
- [Deployment](DEPLOYMENT.md)
- [Configuration](CONFIGURATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## Agent commands

```bash
rwkv ask "hello"
rwkv tool web-search "latest official RWKV update"
rwkv tool knowledge-search "RWKV architecture"
rwkv research --branches 4 --rounds 2 "compare official RWKV projects"
rwkv --json health
```

Interactive chat supports `/web`, `/knowledge`, `/research`, `/longtext`,
`/session`, `/status` and `/json`.

## Legacy Web Preview

The earlier local Web UI remains available for retrieval development and
Hugging Face-format RWKV checkpoints:

```bash
pip install -e '.[realtime,model]'
rwkv-search --config configs/default.json init
rwkv-search --config configs/default.json serve \
  --host 127.0.0.1 --port 8765 \
  --model /path/to/rwkv-hf --device cuda:0 --dtype fp16
```

Extractive fallback without a model can be started with `--no-model`, but it is
not representative of the current 13.3B Agent quality.

## SearXNG

```bash
cd deploy/searxng
# Replace settings.yml secret_key first.
docker compose up -d
curl 'http://127.0.0.1:8888/search?q=rwkv&format=json'
```

SearXNG performs URL discovery. Page fetching, extraction, Evidence selection
and answer generation remain in the internal RWKV Search retrieval subsystem.

## Benchmarks

Run ordinary tests before live-network benchmarks:

```bash
pytest -q
python scripts/check_public_release.py
```

Realtime retrieval smoke:

```bash
PYTHONPATH=src python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --case-id retrieval-zh-001 \
  --case-id retrieval-en-006
```

Agent benchmark methodology and license boundaries are documented in
[AGENT_BENCHMARK.md](AGENT_BENCHMARK.md). Local outputs go to ignored `runs/`
directories. Only reviewed summaries belong under `bench/baselines/`.
