#!/bin/bash
# Build Flux2App (+ mlx.metallib), deploy to Tart Virtual Mac, launch, screencapture.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VM="${VM:-tart-virtual-mac}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/flux2-smoke}"
OUT="${OUT:-/tmp/flux2-smoke.png}"
CONFIG="${CONFIG:-Debug}"

case "$CONFIG" in
  Debug) BUILD_DIR="$ROOT/.build/arm64-apple-macosx/debug" ;;
  Release) BUILD_DIR="$ROOT/.build/arm64-apple-macosx/release" ;;
  *) echo "CONFIG must be Debug or Release" >&2; exit 1 ;;
esac

cd "$ROOT"
swift build --product Flux2App -c "$CONFIG"
"$ROOT/bin/build-mlx-metallib.sh"

ssh -o ControlPath=none "$VM" "mkdir -p '$REMOTE_DIR'"
scp "$BUILD_DIR/Flux2App" "$BUILD_DIR/mlx.metallib" "$VM:$REMOTE_DIR/"
ssh -o ControlPath=none "$VM" "pkill -x Flux2App 2>/dev/null || true"
ssh -o ControlPath=none "$VM" "cd '$REMOTE_DIR' && ./Flux2App >/tmp/flux2-smoke.log 2>&1 &"
sleep 6
ssh -o ControlPath=none "$VM" 'pgrep -l Flux2App || { echo "Flux2App not running"; cat /tmp/flux2-smoke.log; exit 1; }'
ssh -o ControlPath=none "$VM" 'screencapture -x /tmp/flux2-smoke.png'
scp "$VM:/tmp/flux2-smoke.png" "$OUT"
echo "Screenshot: $OUT"
