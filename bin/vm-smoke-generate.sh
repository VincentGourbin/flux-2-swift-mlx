#!/bin/bash
# Run a minimal Klein 4B I2I in the Tart VM when weights exist on the shared host cache.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VM="${VM:-tart-virtual-mac}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/flux2-smoke-gen}"
VM_MODELS_DIR="${VM_MODELS_DIR:-/Volumes/My Shared Files/flux2-model-cache}"
IMAGE="${IMAGE:-$ROOT/Tests/Fixtures/VMSmoke/reference.png}"
OUT="${OUT:-/tmp/flux2-smoke-i2i.png}"
CONFIG="${CONFIG:-debug}"
GENERATE_TIMEOUT_SECONDS="${GENERATE_TIMEOUT_SECONDS:-900}"

# Paths relative to VM_MODELS_DIR (must exist on the host cache / virtio share).
REQUIRED=(
  "black-forest-labs/FLUX.2-klein-4B-klein4b-8bit"
  "black-forest-labs/FLUX.2-klein-4B-vae"
)

case "$CONFIG" in
  [Dd]ebug) CONFIG_LC=debug; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/debug" ;;
  [Rr]elease) CONFIG_LC=release; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/release" ;;
  *) echo "CONFIG must be debug or release" >&2; exit 1 ;;
esac

if [[ ! -f "$IMAGE" ]]; then
  echo "Reference image not found: $IMAGE" >&2
  exit 1
fi

"$ROOT/bin/vm-smoke-models.sh" || exit 1

missing=()
for rel in "${REQUIRED[@]}"; do
  if ! ssh -o ControlPath=none "$VM" "test -d '$VM_MODELS_DIR/$rel'"; then
    missing+=("$rel")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Generate smoke skipped — missing on shared cache:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "Download Klein 4B to the host cache (e.g. flux2 download --model klein-4b), then retry." >&2
  exit 2
fi

cd "$ROOT"
swift build --product Flux2CLI -c "$CONFIG_LC"
"$ROOT/bin/build-mlx-metallib.sh"

ssh -o ControlPath=none "$VM" "mkdir -p '$REMOTE_DIR'"
scp "$BUILD_DIR/Flux2CLI" "$BUILD_DIR/mlx.metallib" "$IMAGE" "$VM:$REMOTE_DIR/"

remote_out="$REMOTE_DIR/output.png"
ssh -o ControlPath=none "$VM" "rm -f '$remote_out' '$REMOTE_DIR/generate.log'"

echo "Running Klein 4B I2I in VM (qint8 / 4 steps; timeout ${GENERATE_TIMEOUT_SECONDS}s)…"

ssh -o ControlPath=none "$VM" "cd '$REMOTE_DIR' && \
  F2SM_MODELS_DIR='$VM_MODELS_DIR' nohup ./Flux2CLI i2i 'VM smoke i2i' \
  -i reference.png \
  -o output.png \
  --model klein-4b \
  --transformer-quant qint8 \
  --text-quant 4bit \
  --vae-variant standard \
  --steps 4 \
  -g 1.0 \
  -w 512 -h 384 \
  >generate.log 2>&1 < /dev/null &"

deadline=$((SECONDS + GENERATE_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if ssh -o ControlPath=none "$VM" "test -s '$remote_out'"; then
    break
  fi
  if ! ssh -o ControlPath=none "$VM" "pgrep -x Flux2CLI >/dev/null 2>&1"; then
    break
  fi
  sleep 10
done

if ! ssh -o ControlPath=none "$VM" "test -s '$remote_out'"; then
  echo "No output image produced at $remote_out" >&2
  ssh -o ControlPath=none "$VM" "tail -80 '$REMOTE_DIR/generate.log' 2>/dev/null || true" >&2
  exit 1
fi

scp "$VM:$remote_out" "$OUT"
bytes="$(wc -c <"$OUT" | tr -d ' ')"
echo "I2I output: $OUT ($bytes bytes)"
