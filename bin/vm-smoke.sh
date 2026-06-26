#!/bin/bash
# Build Flux2App (+ mlx.metallib), deploy via Circus, open VMSmoke project, verify, screencapture.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=vm-smoke-circus.sh
source "$ROOT/bin/vm-smoke-circus.sh"

REMOTE_DIR="${REMOTE_DIR:-/tmp/flux2-smoke}"
FIXTURE="${FIXTURE:-$ROOT/Tests/Fixtures/VMSmoke/project.json}"
REMOTE_PROJECT="${REMOTE_PROJECT:-$REMOTE_DIR/VMSmoke/project.json}"
MARKER="${MARKER:-/tmp/flux2-smoke-ready}"
VM_MODELS_DIR="${VM_MODELS_DIR:-/Volumes/My Shared Files/flux2-model-cache}"
OUT="${OUT:-/tmp/flux2-smoke.png}"
CONFIG="${CONFIG:-debug}"
MARKER_WAIT_SECONDS="${MARKER_WAIT_SECONDS:-60}"
GUEST_SCREENSHOT="/tmp/flux2-smoke.png"

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

circus_ensure_ready 180

if ! circus_path_is_dir "$VM_MODELS_DIR/black-forest-labs"; then
  echo "Models mount not visible in guest: $VM_MODELS_DIR" >&2
  echo "Profile $CIRCUS_PROFILE must include flux2-model-cache; restart the VM from Circus after changing shares." >&2
  exit 1
fi

circus_exec --timeout 30 -- "mkdir -p $(printf '%q' "$(dirname "$REMOTE_PROJECT")")"
circus_put "$BUILD_DIR/Flux2App" "$REMOTE_DIR/Flux2App"
circus_put "$BUILD_DIR/mlx.metallib" "$REMOTE_DIR/mlx.metallib"
circus_put "$FIXTURE" "$REMOTE_PROJECT"

circus_exec --timeout 30 -- "pkill -x Flux2App 2>/dev/null || true; rm -f $(printf '%q' "$MARKER")"

circus_exec --timeout 30 -- \
  "export F2SM_PROJECT=$(printf '%q' "$REMOTE_PROJECT") F2SM_SMOKE_MARKER=$(printf '%q' "$MARKER"); cd $(printf '%q' "$REMOTE_DIR") && open -n ./Flux2App"

if ! circus_wait_file "$MARKER" "$MARKER_WAIT_SECONDS"; then
  echo "Smoke marker not written within ${MARKER_WAIT_SECONDS}s: $MARKER" >&2
  circus_exec --timeout 30 -- 'pgrep -l Flux2App || true; tail -50 /tmp/flux2-smoke.log 2>/dev/null || true' >&2
  exit 1
fi

marker="$(circus_exec --timeout 30 -- "cat $(printf '%q' "$MARKER")")"
if [[ "$(printf '%s' "$marker" | head -1)" != "ok" ]]; then
  echo "Smoke project load failed:" >&2
  printf '%s\n' "$marker" >&2
  exit 1
fi

if ! printf '%s\n' "$marker" | grep -q '^history_steps='; then
  echo "Smoke marker missing history_steps (edit history hook):" >&2
  printf '%s\n' "$marker" >&2
  exit 1
fi

circus_exec --timeout 30 -- 'pgrep -l Flux2App || { echo "Flux2App not running"; cat /tmp/flux2-smoke.log; exit 1; }'
circus_exec --timeout 30 -- "screencapture -x $(printf '%q' "$GUEST_SCREENSHOT")"
circus_get "$GUEST_SCREENSHOT" "$OUT"

echo "Smoke marker:"
printf '%s\n' "$marker"
echo "Screenshot: $OUT"
