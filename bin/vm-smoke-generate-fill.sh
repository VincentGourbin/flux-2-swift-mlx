#!/bin/bash
# Generative-fill smoke: Klein 4B inpaint + Qwen3.5 4-bit VLM enrichment in the Tart VM.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=vm-smoke-circus.sh
source "$ROOT/bin/vm-smoke-circus.sh"

REMOTE_DIR="${REMOTE_DIR:-/tmp/flux2-smoke-fill}"
VM_MODELS_DIR="${VM_MODELS_DIR:-/Volumes/My Shared Files/flux2-model-cache}"
HOST_MODELS_DIR="${HOST_MODELS_DIR:-$HOME/Library/Caches/models}"
IMAGE="${IMAGE:-$ROOT/Tests/Fixtures/VMSmoke/reference.png}"
MASK="${MASK:-$ROOT/Tests/Fixtures/VMSmoke/fill-mask.png}"
OUT="${OUT:-/tmp/flux2-smoke-fill.png}"
CONFIG="${CONFIG:-debug}"
GENERATE_TIMEOUT_SECONDS="${GENERATE_TIMEOUT_SECONDS:-1200}"

KLEIN_TRANSFORMER="black-forest-labs/FLUX.2-klein-4B-klein4b-8bit"
KLEIN_VAE="black-forest-labs/FLUX.2-klein-4B-vae"
SMALL_DECODER="black-forest-labs/FLUX.2-small-decoder"
QWEN35_4BIT="mlx-community/Qwen3.5-4B-MLX-4bit"

case "$CONFIG" in
  [Dd]ebug) CONFIG_LC=debug; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/debug" ;;
  [Rr]elease) CONFIG_LC=release; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/release" ;;
  *) echo "CONFIG must be debug or release" >&2; exit 1 ;;
esac

if [[ ! -f "$IMAGE" ]]; then
  echo "Reference image not found: $IMAGE" >&2
  exit 1
fi
if [[ ! -f "$MASK" ]]; then
  echo "Fill mask not found: $MASK" >&2
  exit 1
fi

"$ROOT/bin/vm-smoke-models.sh" || exit 1

missing=()
for rel in "$KLEIN_TRANSFORMER" "$KLEIN_VAE" "$SMALL_DECODER"; do
  if ! circus_path_is_dir "$VM_MODELS_DIR/$rel"; then
    missing+=("$rel")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Fill smoke skipped — missing on shared cache:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "Download Klein 4B to the host cache (e.g. flux2 download --model klein-4b), then retry." >&2
  exit 2
fi

cd "$ROOT"
swift build --product Flux2CLI -c "$CONFIG_LC"
"$ROOT/bin/build-mlx-metallib.sh"

if ! circus_path_is_dir "$VM_MODELS_DIR/$QWEN35_4BIT"; then
  echo "Qwen3.5 4-bit not on shared cache — downloading on host to $HOST_MODELS_DIR …" >&2
  if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) to auto-download Qwen3.5, or pre-cache at:" >&2
    echo "  $HOST_MODELS_DIR/$QWEN35_4BIT" >&2
    echo "Hint: flux2 test-qwen35 smoke --variant 4bit --no-think --max-tokens 8" >&2
    exit 2
  fi
  F2SM_MODELS_DIR="$HOST_MODELS_DIR" \
    "$BUILD_DIR/Flux2CLI" test-qwen35 "smoke" \
    --variant 4bit \
    --no-think \
    --max-tokens 8 \
    >/tmp/flux2-qwen35-download.log 2>&1
  if ! circus_path_is_dir "$VM_MODELS_DIR/$QWEN35_4BIT"; then
    echo "Qwen3.5 download finished but guest still does not see $QWEN35_4BIT" >&2
    echo "Restart the VM from Circus if the share was empty at launch." >&2
    tail -30 /tmp/flux2-qwen35-download.log >&2 || true
    exit 2
  fi
fi

circus_ensure_ready 180

circus_exec --timeout 30 -- "mkdir -p $(printf '%q' "$REMOTE_DIR")"
circus_put "$BUILD_DIR/Flux2CLI" "$REMOTE_DIR/Flux2CLI"
circus_put "$BUILD_DIR/mlx.metallib" "$REMOTE_DIR/mlx.metallib"
circus_put "$IMAGE" "$REMOTE_DIR/reference.png"
circus_put "$MASK" "$REMOTE_DIR/fill-mask.png"

remote_out="$REMOTE_DIR/output.png"
circus_exec --timeout 30 -- "rm -f $(printf '%q' "$remote_out") $(printf '%q' "$REMOTE_DIR/fill.log")"

echo "Running Klein 4B generative fill in VM (VLM 4-bit, modify intent, 4 steps; timeout ${GENERATE_TIMEOUT_SECONDS}s)…"

remote_cmd="cd $(printf '%q' "$REMOTE_DIR") && \
  export F2SM_MODELS_DIR=$(printf '%q' "$VM_MODELS_DIR") && \
  ./Flux2CLI inpaint 'fill it in' \
  -i reference.png \
  -m fill-mask.png \
  -o output.png \
  --flux-model klein-4b \
  --models-dir $(printf '%q' "$VM_MODELS_DIR") \
  --enrich-prompt-with-vlm \
  --qwen35-variant 4bit \
  --intent modify \
  --steps 4 \
  --guidance 1.0 \
  --max-pixels 196608 \
  >fill.log 2>&1"

if ! circus_exec --timeout "$GENERATE_TIMEOUT_SECONDS" -- "$remote_cmd"; then
  echo "Flux2CLI inpaint failed" >&2
  circus_exec --timeout 30 -- "tail -100 $(printf '%q' "$REMOTE_DIR/fill.log") 2>/dev/null || true" >&2
  exit 1
fi

if ! circus_exec --timeout 30 -- "test -s $(printf '%q' "$remote_out")"; then
  echo "No output image at $remote_out" >&2
  circus_exec --timeout 30 -- "tail -100 $(printf '%q' "$REMOTE_DIR/fill.log") 2>/dev/null || true" >&2
  exit 1
fi

circus_get "$remote_out" "$OUT"
bytes="$(wc -c <"$OUT" | tr -d ' ')"
echo "Generative fill output: $OUT ($bytes bytes)"
echo "--- VLM / inpaint log (tail) ---"
circus_exec --timeout 30 -- "tail -25 $(printf '%q' "$REMOTE_DIR/fill.log") 2>/dev/null || true"
