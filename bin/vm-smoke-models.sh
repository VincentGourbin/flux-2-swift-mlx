#!/bin/bash
# Verify the Tart guest sees the host FLUX model cache via flux2-model-cache.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=vm-smoke-circus.sh
source "$ROOT/bin/vm-smoke-circus.sh"

VM_MODELS_DIR="${VM_MODELS_DIR:-/Volumes/My Shared Files/flux2-model-cache}"

circus_ensure_ready 60

if ! circus_path_is_dir "$VM_MODELS_DIR"; then
  echo "Mount missing: $VM_MODELS_DIR" >&2
  echo "Pick profile $CIRCUS_PROFILE in Circus and restart the VM if shares changed." >&2
  exit 1
fi

echo "Mount: $VM_MODELS_DIR (profile $CIRCUS_PROFILE)"
circus_exec --timeout 30 -- "ls -1 $(printf '%q' "$VM_MODELS_DIR")"

probe() {
  local label="$1"
  local rel="$2"
  if circus_path_is_dir "$VM_MODELS_DIR/$rel"; then
    echo "  ok  $label"
    return 0
  fi
  echo "  —   $label (missing: $rel)"
  return 1
}

echo
echo "Weight probes:"
missing=0
probe "FLUX.2 Dev transformer (bf16)" "black-forest-labs/FLUX.2-dev-transformer-bf16" || ((missing++)) || true
probe "FLUX.2 Klein 4B transformer (8bit)" "black-forest-labs/FLUX.2-klein-4B-klein4b-8bit" || ((missing++)) || true
probe "FLUX.2 Klein 4B VAE" "black-forest-labs/FLUX.2-klein-4B-vae" || ((missing++)) || true
probe "Mistral MLX 8bit (dev text encoder)" "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit" || ((missing++)) || true

echo
if (( missing > 0 )); then
  echo "$missing optional/missing component(s). Mount is up; generate smoke needs Klein 4B transformer + Qwen weights on the host cache."
  exit 0
fi

echo "All probed weights present."
