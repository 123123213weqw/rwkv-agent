#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for pidfile in "$ROOT"/var/g1i-*.pid; do
  [[ -e "$pidfile" ]] || continue
  pid=$(cat "$pidfile")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "stopped pid=$pid"
  fi
  : >"$pidfile"
  python3 - "$pidfile" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1]).unlink(missing_ok=True)
PY
done
