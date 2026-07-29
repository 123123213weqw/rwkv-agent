#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "stop_g1i_sidecars.sh is deprecated; stopping the public Beta service PID files." >&2
RWKV_AGENT_PROJECT_ROOT=${RWKV_AGENT_PROJECT_ROOT:-"$ROOT"} \
  exec "$ROOT/cli/scripts/rwkv-agent-service" stop
