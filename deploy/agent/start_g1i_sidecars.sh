#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${G1I_PYTHON:-python3}
CUDA_ROOT=${G1I_CUDA_HOME:-$(dirname "$(dirname "$PYTHON")")}
COUNT=${1:-1}
BASE_PORT=${G1I_BASE_PORT:-8118}
mkdir -p "$ROOT/var"
for ((i=0;i<COUNT;i++)); do
  port=$((BASE_PORT+i))
  pidfile="$ROOT/var/g1i-$port.pid"
  logfile="$ROOT/var/g1i-$port.log"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "already running port=$port pid=$(cat "$pidfile")"
    continue
  fi
  env PYTHONPATH="$ROOT/src" \
    PATH="$(dirname "$PYTHON"):$PATH" \
    CUDA_HOME="$CUDA_ROOT" \
    CUDA_VISIBLE_DEVICES=$i \
    nohup "$PYTHON" -m rwkv_agent.sidecar \
      --host 127.0.0.1 --port "$port" >"$logfile" 2>&1 &
  echo $! >"$pidfile"
  echo "started port=$port gpu=$i pid=$! log=$logfile"
done
