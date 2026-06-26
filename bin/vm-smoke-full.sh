#!/bin/bash
# Full Circus VM battery: environment, UI, history bundle, CLI generate (when weights exist).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "═══════════════════════════════════════════════════════════════"
echo " Flux2 VM smoke — full battery (Circus profile: ${CIRCUS_PROFILE:-tart-av-dev})"
echo "═══════════════════════════════════════════════════════════════"

failures=0
skips=0

run_step() {
  local name="$1"
  shift
  echo ""
  echo "── $name ──"
  if "$@"; then
    echo "✓ $name"
    return 0
  fi
  local code=$?
  if [[ "$code" -eq 2 ]]; then
    echo "⊘ $name (skipped)"
    skips=$((skips + 1))
    return 0
  fi
  echo "✗ $name (exit $code)" >&2
  failures=$((failures + 1))
  return 0
}

circus --api-version
circus ensure-ready --profile "${CIRCUS_PROFILE:-tart-av-dev}" --json 2>/dev/null | head -20 || circus ensure-ready --profile "${CIRCUS_PROFILE:-tart-av-dev}"

run_step "Model cache probe" "$ROOT/bin/vm-smoke-models.sh"
run_step "UI load (VMSmoke JSON)" "$ROOT/bin/vm-smoke.sh"
run_step "History bundle (v3 + Import step)" "$ROOT/bin/vm-smoke-history.sh"
run_step "CLI I2I generate (Klein 4B)" "$ROOT/bin/vm-smoke-generate.sh"
run_step "CLI generative fill" "$ROOT/bin/vm-smoke-generate-fill.sh"

echo ""
echo "═══════════════════════════════════════════════════════════════"
if (( failures > 0 )); then
  echo "FAILED: $failures step(s); skipped: $skips"
  exit 1
fi
echo "PASSED (skipped: $skips)"
echo ""
echo "Artifacts on host:"
echo "  /tmp/flux2-smoke.png           — UI load"
echo "  /tmp/flux2-smoke-history.png   — history bundle"
echo "  /tmp/flux2-smoke-i2i.png       — CLI generate (if run)"
echo "  /tmp/flux2-smoke-fill.png      — generative fill (if run)"
echo "═══════════════════════════════════════════════════════════════"
