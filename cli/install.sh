#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PREFIX=${PREFIX:-"$HOME/.local"}

cd "$ROOT"
cargo build --release --locked
install -d "$PREFIX/bin"
install -m 0755 target/release/rwkv-agent "$PREFIX/bin/rwkv-agent"
install -m 0755 scripts/rwkv "$PREFIX/bin/rwkv"
install -m 0755 scripts/rwkv-agent-service "$PREFIX/bin/rwkv-agent-service"
echo "installed $PREFIX/bin/rwkv-agent"
echo "installed $PREFIX/bin/rwkv"
echo "installed $PREFIX/bin/rwkv-agent-service"
