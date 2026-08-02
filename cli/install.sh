#!/usr/bin/env bash
set -euo pipefail

CLI_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$CLI_DIR/.." && pwd)
PREFIX=${PREFIX:-"$HOME/.local"}
CLIENT_ONLY=0
SKIP_BUILD=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [--client-only] [--prefix PATH] [--skip-build]

  --client-only  Install only the rwkv-agent terminal client. Use this on a
                 laptop that connects to an existing remote Agent backend.
  --prefix PATH  Install under PATH/bin instead of $HOME/.local/bin.
  --skip-build   Install an existing target/release/rwkv-agent binary.
  -h, --help     Show this help.

Environment: PREFIX remains supported for backward compatibility.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --client-only)
      CLIENT_ONLY=1
      shift
      ;;
    --prefix)
      [[ $# -ge 2 ]] || { echo "--prefix requires a path" >&2; exit 2; }
      PREFIX=$2
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
if [[ $SKIP_BUILD -eq 0 ]]; then
  command -v cargo >/dev/null 2>&1 || {
    echo "cargo is required to build rwkv-agent" >&2
    exit 1
  }
  cargo build --release --locked -p rwkv-agent-cli
fi

[[ -x target/release/rwkv-agent ]] || {
  echo "missing executable: $ROOT/target/release/rwkv-agent" >&2
  exit 1
}

install -d "$PREFIX/bin"
install -m 0755 target/release/rwkv-agent "$PREFIX/bin/rwkv-agent"
echo "installed $PREFIX/bin/rwkv-agent"

if [[ $CLIENT_ONLY -eq 0 ]]; then
  install -m 0755 "$CLI_DIR/scripts/rwkv" "$PREFIX/bin/rwkv"
  install -m 0755 "$CLI_DIR/scripts/rwkv-agent-service" "$PREFIX/bin/rwkv-agent-service"
  echo "installed $PREFIX/bin/rwkv"
  echo "installed $PREFIX/bin/rwkv-agent-service"
fi

case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) echo "add $PREFIX/bin to PATH before using rwkv-agent" ;;
esac
