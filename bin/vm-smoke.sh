#!/bin/bash
# Build Flux2App (+ mlx.metallib), deploy to Tart Virtual Mac, open VMSmoke project, verify, screencapture.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VM="${VM:-tart-virtual-mac}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/flux2-smoke}"
FIXTURE="${FIXTURE:-$ROOT/Tests/Fixtures/VMSmoke/project.json}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_DIR/VMSmoke/project.json}"
MARKER="${MARKER:-/tmp/flux2-smoke-ready}"
OUT="${OUT:-/tmp/flux2-smoke.png}"
CONFIG="${CONFIG:-debug}"
MARKER_WAIT_SECONDS="${MARKER_WAIT_SECONDS:-30}"

case "$CONFIG" in
  [Dd]ebug) CONFIG_LC=debug; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/debug" ;;
  [Rr]elease) CONFIG_LC=release; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/release" ;;
  *) echo "CONFIG must be debug or release" >&2; exit 1 ;;
esac

if [[ ! -f "$FIXTURE" ]]; then
  echo "Fixture not found: $FIXTURE" >&2
  exit 1
fi

cd "$ROOT"
swift build --product Flux2App -c "$CONFIG_LC"
"$ROOT/bin/build-mlx-metallib.sh"

ssh -o ControlPath=none "$VM" "mkdir -p '$(dirname "$REMOTE_PROJECT")'"
scp "$BUILD_DIR/Flux2App" "$BUILD_DIR/mlx.metallib" "$VM:$REMOTE_DIR/"
scp "$FIXTURE" "$VM:$REMOTE_PROJECT"

ssh -o ControlPath=none "$VM" "pkill -x Flux2App 2>/dev/null || true; rm -f '$MARKER'"

ssh -o ControlPath=none "$VM" "cd '$REMOTE_DIR' && F2SM_PROJECT='$REMOTE_PROJECT' F2SM_SMOKE_MARKER='$MARKER' ./Flux2App >/tmp/flux2-smoke.log 2>&1 &"

deadline=$((SECONDS + MARKER_WAIT_SECONDS))
while (( SECONDS < deadline )); do
  if ssh -o ControlPath=none "$VM" "test -f '$MARKER'"; then
    break
  fi
  sleep 1
done

if ! ssh -o ControlPath=none "$VM" "test -f '$MARKER'"; then
  echo "Smoke marker not written within ${MARKER_WAIT_SECONDS}s: $MARKER" >&2
  ssh -o ControlPath=none "$VM" 'pgrep -l Flux2App || true; tail -50 /tmp/flux2-smoke.log 2>/dev/null || true' >&2
  exit 1
fi

marker="$(ssh -o ControlPath=none "$VM" "cat '$MARKER'")"
if [[ "$(printf '%s' "$marker" | head -1)" != "ok" ]]; then
  echo "Smoke project load failed:" >&2
  printf '%s\n' "$marker" >&2
  exit 1
fi

ssh -o ControlPath=none "$VM" 'pgrep -l Flux2App || { echo "Flux2App not running"; cat /tmp/flux2-smoke.log; exit 1; }'
ssh -o ControlPath=none "$VM" 'screencapture -x /tmp/flux2-smoke.png'
scp "$VM:/tmp/flux2-smoke.png" "$OUT"

echo "Smoke marker:"
printf '%s\n' "$marker"
echo "Screenshot: $OUT"
