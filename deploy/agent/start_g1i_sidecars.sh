#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "start_g1i_sidecars.sh is deprecated; starting the public single-Sidecar Beta service." >&2
RWKV_AGENT_PROJECT_ROOT=${RWKV_AGENT_PROJECT_ROOT:-"$ROOT"} \
  exec "$ROOT/cli/scripts/rwkv-agent-service" start
