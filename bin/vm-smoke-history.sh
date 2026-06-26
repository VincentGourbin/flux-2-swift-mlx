#!/bin/bash
# VM smoke: open a v3 bundle with one Import history step; verify marker + on-disk history JXL.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=vm-smoke-circus.sh
source "$ROOT/bin/vm-smoke-circus.sh"

REMOTE_DIR="${REMOTE_DIR:-/tmp/flux2-smoke-history}"
BUNDLE_NAME="${BUNDLE_NAME:-VMSmokeHistory.flux2project}"
COMMITTED_FIXTURE="$ROOT/Tests/Fixtures/VMSmokeHistory/$BUNDLE_NAME"
HOST_BUNDLE="${HOST_BUNDLE:-$COMMITTED_FIXTURE}"
REMOTE_BUNDLE="${REMOTE_BUNDLE:-$REMOTE_DIR/$BUNDLE_NAME}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_BUNDLE/project.json}"
MARKER="${MARKER:-/tmp/flux2-smoke-history-ready}"
OUT="${OUT:-/tmp/flux2-smoke-history.png}"
CONFIG="${CONFIG:-debug}"
MARKER_WAIT_SECONDS="${MARKER_WAIT_SECONDS:-90}"
GUEST_SCREENSHOT="/tmp/flux2-smoke-history.png"
TARBALL="/tmp/flux2-smoke-history-bundle.tgz"

case "$CONFIG" in
  [Dd]ebug) CONFIG_LC=debug; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/debug" ;;
  [Rr]elease) CONFIG_LC=release; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/release" ;;
  *) echo "CONFIG must be debug or release" >&2; exit 1 ;;
esac

cd "$ROOT"
swift build --product Flux2App -c "$CONFIG_LC"
"$ROOT/bin/build-mlx-metallib.sh"

if [[ ! -f "$HOST_BUNDLE/project.json" ]]; then
  echo "Building history bundle fixture → $HOST_BUNDLE"
  mkdir -p "$(dirname "$HOST_BUNDLE")"
  swift build --product Flux2SmokeFixture -c "$CONFIG_LC"
  "$BUILD_DIR/Flux2SmokeFixture" export-history-bundle "$HOST_BUNDLE"
fi

if [[ ! -f "$HOST_BUNDLE/project.json" ]]; then
  echo "History bundle missing: $HOST_BUNDLE" >&2
  echo "Run: swift build --product Flux2SmokeFixture && .build/.../Flux2SmokeFixture export-history-bundle $COMMITTED_FIXTURE" >&2
  echo "(requires JPEG XL encode: brew install jpeg-xl)" >&2
  exit 1
fi
echo "Using history bundle: $HOST_BUNDLE"

circus_ensure_ready 180

rm -f "$TARBALL"
tar -czf "$TARBALL" -C "$(dirname "$HOST_BUNDLE")" "$(basename "$HOST_BUNDLE")"

circus_exec --timeout 30 -- "mkdir -p $(printf '%q' "$REMOTE_DIR") && rm -rf $(printf '%q' "$REMOTE_BUNDLE")"
circus_put "$BUILD_DIR/Flux2App" "$REMOTE_DIR/Flux2App"
circus_put "$BUILD_DIR/mlx.metallib" "$REMOTE_DIR/mlx.metallib"
circus_put "$TARBALL" "$TARBALL"
circus_exec --timeout 60 -- "tar -xzf $(printf '%q' "$TARBALL") -C $(printf '%q' "$REMOTE_DIR")"

circus_exec --timeout 30 -- "test -f $(printf '%q' "$REMOTE_PROJECT")"
circus_exec --timeout 30 -- "test -f $(printf '%q' "$REMOTE_BUNDLE/history/0001.jxl")"
circus_exec --timeout 30 -- "test -f $(printf '%q' "$REMOTE_BUNDLE/thumbs/0001.jxl")"

circus_exec --timeout 30 -- "pkill -x Flux2App 2>/dev/null || true; rm -f $(printf '%q' "$MARKER")"

circus_exec --timeout 30 -- \
  "export F2SM_PROJECT=$(printf '%q' "$REMOTE_PROJECT") F2SM_SMOKE_MARKER=$(printf '%q' "$MARKER"); cd $(printf '%q' "$REMOTE_DIR") && open -n ./Flux2App"

if ! circus_wait_file "$MARKER" "$MARKER_WAIT_SECONDS"; then
  echo "History smoke marker not written within ${MARKER_WAIT_SECONDS}s" >&2
  circus_exec --timeout 30 -- 'pgrep -l Flux2App || true' >&2
  exit 1
fi

marker="$(circus_exec --timeout 30 -- "cat $(printf '%q' "$MARKER")")"
if [[ "$(printf '%s' "$marker" | head -1)" != "ok" ]]; then
  echo "History project load failed:" >&2
  printf '%s\n' "$marker" >&2
  exit 1
fi

history_steps="$(printf '%s\n' "$marker" | sed -n 's/^history_steps=//p' | head -1)"
if [[ -z "$history_steps" ]] || [[ "$history_steps" -lt 1 ]]; then
  echo "Expected history_steps >= 1, got: ${history_steps:-<missing>}" >&2
  printf '%s\n' "$marker" >&2
  exit 1
fi

circus_exec --timeout 30 -- 'pgrep -l Flux2App || { echo "Flux2App not running"; exit 1; }'
circus_exec --timeout 30 -- "screencapture -x $(printf '%q' "$GUEST_SCREENSHOT")"
circus_get "$GUEST_SCREENSHOT" "$OUT"

echo "History smoke marker:"
printf '%s\n' "$marker"
echo "Screenshot: $OUT"
