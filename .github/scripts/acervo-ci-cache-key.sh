#!/usr/bin/env bash
#
# acervo-ci-cache-key.sh — emit a deterministic actions/cache key derived from
# the live CDN manifests of the requested models.
#
# The key is sha256(concat of each model's manifestChecksum). When a model is
# re-shipped (its manifest changes), the key changes, so actions/cache misses
# and acervo-ci-prime.sh re-fetches the fresh files. No brittle static "v2"
# keys, no manual bumping.
#
# Usage (in a workflow step):
#   echo "value=$(ACERVO_CI_MODELS="$ACERVO_CI_MODELS" bash acervo-ci-cache-key.sh)" >> "$GITHUB_OUTPUT"
#
# Env:
#   ACERVO_CI_MODELS  whitespace/newline-separated slugs (or pass as args).
#   ACERVO_CDN_BASE   optional CDN base override.
#
set -euo pipefail

CDN_BASE="${ACERVO_CDN_BASE:-https://pub-8e049ed02be340cbb18f921765fd24f3.r2.dev/models}"
read -r -a SLUGS <<< "$(printf '%s %s' "${ACERVO_CI_MODELS:-}" "$*" | tr '\n' ' ')"

acc=""
for slug in "${SLUGS[@]}"; do
  [[ -z "$slug" ]] && continue
  body="$(curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors "$CDN_BASE/$slug/manifest.json" 2>/dev/null || true)"
  # Prefer manifestChecksum; fall back to a hash of the whole manifest body so
  # the key is still stable+unique even for a manifest missing the field.
  ck="$(printf '%s' "$body" | jq -r '.manifestChecksum // empty' 2>/dev/null || true)"
  if [[ -z "$ck" ]]; then
    ck="$(printf '%s' "$body" | shasum -a 256 | cut -d' ' -f1)"
  fi
  acc="${acc}${slug}:${ck};"
done

printf '%s' "$acc" | shasum -a 256 | cut -d' ' -f1
