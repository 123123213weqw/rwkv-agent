#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:-}
OUTPUT_DIR=${2:-"$ROOT/dist"}

cd "$ROOT"

if [[ -z "$TARGET" ]]; then
  TARGET=$(rustc -vV | awk '/^host:/ {print $2}')
fi

VERSION=$(python3 - "$ROOT/Cargo.toml" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
if not match:
    raise SystemExit("cannot read package version from Cargo.toml")
print(match.group(1))
PY
)

cargo build --release --locked --target "$TARGET"

BINARY="$ROOT/target/$TARGET/release/rwkv-agent"
[[ -x "$BINARY" ]] || { echo "missing release binary: $BINARY" >&2; exit 1; }

NAME="rwkv-agent-v$VERSION-$TARGET"
mkdir -p "$OUTPUT_DIR"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/$NAME"
install -m 0755 "$BINARY" "$WORK/$NAME/rwkv-agent"
install -m 0644 "$ROOT/README.md" "$WORK/$NAME/README.md"
install -m 0644 "$ROOT/../LICENSE" "$WORK/$NAME/LICENSE"

tar -C "$WORK" -czf "$OUTPUT_DIR/$NAME.tar.gz" "$NAME"
(cd "$OUTPUT_DIR" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

echo "created $OUTPUT_DIR/$NAME.tar.gz"
echo "created $OUTPUT_DIR/$NAME.tar.gz.sha256"
