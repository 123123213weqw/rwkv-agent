#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

CLIENT_PREFIX="$WORK/client"
FULL_PREFIX="$WORK/full"

"$ROOT/install.sh" --client-only --skip-build --prefix "$CLIENT_PREFIX" >/dev/null
test -x "$CLIENT_PREFIX/bin/rwkv-agent"
test ! -e "$CLIENT_PREFIX/bin/rwkv"
test ! -e "$CLIENT_PREFIX/bin/rwkv-agent-service"
"$CLIENT_PREFIX/bin/rwkv-agent" --version | grep -q '^rwkv-agent '

"$ROOT/install.sh" --skip-build --prefix "$FULL_PREFIX" >/dev/null
test -x "$FULL_PREFIX/bin/rwkv-agent"
test -x "$FULL_PREFIX/bin/rwkv"
test -x "$FULL_PREFIX/bin/rwkv-agent-service"

if "$ROOT/install.sh" --unknown-option >/dev/null 2>&1; then
  echo "installer accepted an unknown option" >&2
  exit 1
fi

echo "CLI install smoke passed"
