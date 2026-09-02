#!/usr/bin/env bash
set -euo pipefail

CLI_DIR=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$CLI_DIR/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

PREFIX="$WORK/client"

"$CLI_DIR/install.sh" --skip-build --prefix "$PREFIX" >/dev/null
test -x "$PREFIX/bin/rwkv-agent"
test ! -e "$PREFIX/bin/rwkv"
test ! -e "$PREFIX/bin/rwkv-agent-service"
"$PREFIX/bin/rwkv-agent" --version | grep -q '^rwkv-agent '

if "$CLI_DIR/install.sh" --unknown-option >/dev/null 2>&1; then
  echo "installer accepted an unknown option" >&2
  exit 1
fi

echo "CLI install smoke passed"
