#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=${RWKV_AGENT_MOCK_PORT:-18121}
BIN="$ROOT/target/release/rwkv-agent"
mkdir -p "$ROOT/runs"

python3 "$ROOT/tests/mock_server.py" "$PORT" &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 0.1
done

ENDPOINT="http://127.0.0.1:$PORT"
"$BIN" --endpoint "$ENDPOINT" health | grep -q "status: ready"
"$BIN" --endpoint "$ENDPOINT" doctor | grep -q "\[ok\] model_sidecar"
"$BIN" --endpoint "$ENDPOINT" --json doctor | grep -q '"status": "ready"'
"$BIN" --endpoint "$ENDPOINT" ask "hello world" | grep -q "mock answer"
"$BIN" --endpoint "$ENDPOINT" research --branches 4 --rounds 2 \
  "Who created RWKV?" | grep -q "mock research"
"$BIN" --endpoint "$ENDPOINT" --json tool web-search "latest release" \
  | grep -q '"status": "ok"'
"$BIN" --endpoint "$ENDPOINT" --session pasted-demo --json \
  tool long-text-qa "What does it say?" \
  | grep -q '"status": "ok"'
printf 'hello chat\n/exit\n' \
  | "$BIN" --endpoint "$ENDPOINT" chat \
  | grep -q "mock answer"
printf '/status\n/tools\n/web latest\n/research Who created RWKV?\n/session next-session\n/json on\nhello default chat\n/json off\n/exit\n' \
  | "$BIN" --endpoint "$ENDPOINT" \
  >"$ROOT/runs/mock_claude_chat.txt"
grep -q "status: ready" "$ROOT/runs/mock_claude_chat.txt"
grep -q "web_search(query)" "$ROOT/runs/mock_claude_chat.txt"
grep -q "Mock evidence" "$ROOT/runs/mock_claude_chat.txt"
grep -q "Parallel state research" "$ROOT/runs/mock_claude_chat.txt"
grep -q "mock research" "$ROOT/runs/mock_claude_chat.txt"
grep -q "session switched: next-session" "$ROOT/runs/mock_claude_chat.txt"
grep -q '"answer": "mock answer' "$ROOT/runs/mock_claude_chat.txt"

"$BIN" --endpoint "$ENDPOINT" --json ask first >"$ROOT/runs/fresh_session_1.json"
"$BIN" --endpoint "$ENDPOINT" --json ask second >"$ROOT/runs/fresh_session_2.json"
python3 - "$ROOT/runs/fresh_session_1.json" "$ROOT/runs/fresh_session_2.json" <<'PY'
import json
import sys

first = json.load(open(sys.argv[1]))
second = json.load(open(sys.argv[2]))
assert first["session_id"].startswith("cli-")
assert second["session_id"].startswith("cli-")
assert first["session_id"] != second["session_id"]
PY

echo "CLI mock smoke passed"
