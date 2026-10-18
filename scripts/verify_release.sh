#!/usr/bin/env bash
# RWKV State Agent — reproducible source and runtime verification.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE_SOURCE=1
MODE_FULL=0
MODE_AMD=0
PYTHON="${PYTHON:-python3}"
CONTROLLER_URL="${CONTROLLER_URL:-http://127.0.0.1:18120}"
SIDECAR_URL="${SIDECAR_URL:-http://127.0.0.1:18118}"
DATA_PLANE_URL="${DATA_PLANE_URL:-http://127.0.0.1:18121}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/root/rwkv-agent/evidence}"

usage() {
  cat <<'TXT'
Usage: scripts/verify_release.sh [--source] [--full] [--amd-live]

  --source    Static source, documentation and Web asset checks (default).
  --full      Also run Python regressions and `cargo test --workspace`.
  --amd-live  Verify the running 13.3B ROCm Sidecar, data plane, Controller,
              streaming API, task wall, and frozen AMD evidence summaries.
Modes may be combined. Endpoint and evidence paths can be overridden through
CONTROLLER_URL, SIDECAR_URL, DATA_PLANE_URL, and EVIDENCE_ROOT.
TXT
}

while (($#)); do
  case "$1" in
    --source) MODE_SOURCE=1 ;;
    --full) MODE_FULL=1 ;;
    --amd-live) MODE_AMD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
step=0
check() { step=$((step + 1)); printf '\n[%02d] %s\n' "$step" "$1"; }
fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -s "$1" ]] || fail "required file is missing or empty: $1"; }

if ((MODE_SOURCE)); then
  check "Required project documents"
  for file in \
    README.md \
    docs/PROJECT_SPECIFICATION.md \
    docs/CODEMAP.md \
    LICENSE \
    scripts/verify_release.sh; do
    require_file "$file"
  done
  grep -qi "RWKV State Agent" README.md || fail "README lacks project identity"
  grep -qi "application scenarios" docs/PROJECT_SPECIFICATION.md || fail "specification lacks application scenarios"
  grep -qi "system architecture" docs/PROJECT_SPECIFICATION.md || fail "specification lacks architecture"
  grep -qi "AMD Radeon inference optimization" docs/PROJECT_SPECIFICATION.md || fail "specification lacks AMD optimization"

  check "Release-facing source paths"
  for path in \
    crates/agent-core/src/lib.rs \
    crates/agent-runtime/src/service.rs \
    crates/agent-server/src/lib.rs \
    src/rwkv7_scheduler/hf_scheduler.py \
    src/rwkv_agent/sidecar.py \
    web/index.html \
    web/app.js \
    web/app.css \
    demos/run_agent_website_swarm.py; do
    require_file "$path"
  done

  check "No tracked credentials, environments, model weights or build products"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    tracked="$(git ls-files -co --exclude-standard)"
  else
    # Remote verification mirrors intentionally exclude .git and build output.
    tracked="$(find . -type f \
      -not -path './target/*' \
      -not -path './node_modules/*' \
      -not -path './.venv/*' \
      -not -path '*/__pycache__/*' \
      -print)"
  fi
  if printf '%s\n' "$tracked" | grep -E '(^|/)\.env($|\.(local|dev|development|test|staging|prod|production)$)|(^|/)id_(rsa|ed25519)$|\.(pem|key|pth|safetensors)$|(^|/)(target|node_modules|\.venv)/' >/tmp/rwkv-release-forbidden.txt; then
    cat /tmp/rwkv-release-forbidden.txt >&2
    fail "forbidden release files found"
  fi

  check "Embedded Web UI and task-wall static contract"
  command -v node >/dev/null || fail "node is required for JavaScript syntax validation"
  node --check web/app.js
  grep -q 'href="/tasks"' web/index.html || fail "Web UI has no /tasks entry"
  grep -q '"/v1/tasks"' crates/agent-server/src/lib.rs || fail "Rust Controller has no task API"
  grep -q 'run_stream' crates/agent-server/src/lib.rs || fail "Rust Controller has no stream API"

  check "Python source syntax"
  "$PYTHON" -m compileall -q src benchmarks demos tests
fi

if ((MODE_FULL)); then
  check "Focused Python regressions"
  "$PYTHON" -m pytest -q \
    tests/test_hf_scheduler.py \
    tests/test_batching.py \
    tests/test_state_runtime.py \
    tests/test_agent.py \
    tests/test_long_text.py

  check "Rust workspace regression"
  command -v cargo >/dev/null || fail "cargo is required for --full"
  cargo test --workspace
fi

if ((MODE_AMD)); then
  check "Live ROCm Sidecar, data plane and Rust Controller"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsS "$SIDECAR_URL/health" >"$tmp/sidecar.json"
  curl -fsS "$DATA_PLANE_URL/health" >"$tmp/data-plane.json"
  curl -fsS "$CONTROLLER_URL/health" >"$tmp/controller.json"
  curl -fsS "$CONTROLLER_URL/v1/tasks" >"$tmp/tasks.json"
  curl -fsS "$CONTROLLER_URL/tasks" >"$tmp/tasks.html"
  "$PYTHON" - "$tmp" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sidecar = json.loads((root / "sidecar.json").read_text())
data = json.loads((root / "data-plane.json").read_text())
controller = json.loads((root / "controller.json").read_text())
tasks = json.loads((root / "tasks.json").read_text())
page = (root / "tasks.html").read_text()
assert sidecar["status"] == "ready", sidecar
assert sidecar["backend"] == "hf_recurrent", sidecar
assert "preview4922-13.3b" in sidecar["model"].lower(), sidecar
assert sidecar["context"] == 12288, sidecar
assert sidecar["tool_gate_state"]["root_available"] is True, sidecar
assert sidecar["inference"]["worker_alive"] is True, sidecar
assert controller["status"] == "ready", controller
models = controller.get("model") or []
assert any("preview4922-13.3b" in str(row.get("model", "")).lower() for row in models), models
assert {"run_command", "knowledge_search", "long_text_qa", "web_search"}.issubset(set(controller["tools"])), controller["tools"]
assert tasks["status"] == "ok" and "counts" in tasks and isinstance(tasks["tasks"], list), tasks
assert "Live task wall" in page, "task page content missing"
assert data["status"] == "ready", data
print("live AMD services: PASS")
PY

  check "True Sidecar → Rust streaming and live task transition"
  CONTROLLER_URL="$CONTROLLER_URL" "$PYTHON" <<'PY'
import json, os, threading, time, urllib.request, uuid
base = os.environ["CONTROLLER_URL"].rstrip("/")
session = "verify-release-" + uuid.uuid4().hex[:12]
events, errors = [], []
def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.load(response)
def run():
    request = urllib.request.Request(
        base + "/v1/agent/run_stream",
        data=json.dumps({"session_id": session, "message": "Hello. Reply only: hello"}).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                if line.strip():
                    events.append(json.loads(line))
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
worker = threading.Thread(target=run)
worker.start()
running = False
while worker.is_alive():
    running = running or get("/v1/tasks")["counts"]["running"] > 0
    time.sleep(0.2)
worker.join()
board = get("/v1/tasks")
row = next(task for task in board["tasks"] if task["session_id"] == session)
assert not errors, errors
assert running, "running task transition was not observed"
assert any(event.get("type") == "delta" for event in events), events
assert any(event.get("type") == "final" for event in events), events
assert row["status"] == "complete", row
print("live stream and task transition: PASS", row["id"], row["elapsed_ms"])
PY

  check "Frozen AMD evidence summaries"
  for file in \
    "$EVIDENCE_ROOT/gate3/gate3-summary.json" \
    "$EVIDENCE_ROOT/gate4/final-100-v2/metrics.json" \
    "$EVIDENCE_ROOT/gate4/final-100-v2/SHA256SUMS" \
    "$EVIDENCE_ROOT/gate5/b1-zero-copy-stream-parity.json" \
    "$EVIDENCE_ROOT/gate5/chat-stream-two-turn-final.json" \
    "$EVIDENCE_ROOT/gate5/semantic-tool-gate-40.json" \
    "$EVIDENCE_ROOT/gate5/b32-throughput-regression-final.json" \
    "$EVIDENCE_ROOT/gate5/tasks-live-smoke.json"; do
    require_file "$file"
  done
  "$PYTHON" - "$EVIDENCE_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
gate3 = json.loads((root / "gate3/gate3-summary.json").read_text())
gate4 = json.loads((root / "gate4/final-100-v2/metrics.json").read_text())
b32 = json.loads((root / "gate5/b32-throughput-regression-final.json").read_text())
tasks = json.loads((root / "gate5/tasks-live-smoke.json").read_text())
assert gate3["status"] == "pass", gate3.get("status")
assert gate4["status"] == "pass", gate4.get("status")
assert b32["status"] == "pass" and all(b32["acceptance"].values()), b32
assert tasks["status"] == "pass" and tasks["running_observed"], tasks
print("frozen AMD evidence: PASS")
PY
fi

printf '\nRWKV State Agent release verification: PASS\n'
