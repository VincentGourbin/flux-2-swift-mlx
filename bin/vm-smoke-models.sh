#!/bin/bash
# Verify the Tart guest sees the host FLUX model cache via flux2-model-cache.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VM="${VM:-tart-virtual-mac}"
VM_MODELS_DIR="${VM_MODELS_DIR:-/Volumes/My Shared Files/flux2-model-cache}"

check_guest() {
  ssh -o ControlPath=none "$VM" "$@"
}

if ! check_guest "test -d '$VM_MODELS_DIR'"; then
  echo "Mount missing: $VM_MODELS_DIR" >&2
  exit 1
fi

echo "Mount: $VM_MODELS_DIR"
check_guest "ls -1 '$VM_MODELS_DIR'"

probe() {
  local label="$1"
  local rel="$2"
  if check_guest "test -d '$VM_MODELS_DIR/$rel'"; then
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
probe "FLUX.2 Klein 4B transformer (8bit)" "black-forest-labs/FLUX.2-klein-4B-8bit" || ((missing++)) || true
probe "FLUX.2 Klein 4B VAE" "black-forest-labs/FLUX.2-klein-4B-vae" || ((missing++)) || true
probe "Mistral MLX 8bit (dev text encoder)" "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit" || ((missing++)) || true

echo
if (( missing > 0 )); then
  echo "$missing optional/missing component(s). Mount is up; generate smoke needs Klein 4B transformer + Qwen weights on the host cache."
  exit 0
fi

echo "All probed weights present."
