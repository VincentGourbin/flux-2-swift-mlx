#!/usr/bin/env bash
#
# acervo-ci-prime.sh — credential-free CDN model primer for CI.
#
# Mirrors SwiftAcervo's AcervoDownloader exactly, using only curl + jq:
#   * Reads each model's CDNManifest (manifest.json) from the public R2 CDN.
#   * Downloads every file listed in `.files[].path` to its relative location.
#   * Writes the manifest byte-equal to <slug>/manifest.json so SwiftAcervo's
#     local validity check (size-per-file + checksum-of-checksums) passes and
#     the test can run with ACERVO_OFFLINE=1 (no network during the test).
#
# It deliberately does NOT use the `acervo` CLI or the Python `hf` tool, so it
# adds zero Swift/Python build cost to a consumer's test job. The downloads are
# public reads — no HF_TOKEN, no R2 credentials.
#
# Usage:
#   ACERVO_MODELS_DIR=/path/to/cache \
#   ACERVO_CI_MODELS="slug-1 slug-2" \
#   bash acervo-ci-prime.sh [extra-slug ...]
#
# Env:
#   ACERVO_MODELS_DIR   (required) destination cache root. Each model lands in
#                       $ACERVO_MODELS_DIR/<slug>/...  where <slug> is the CDN
#                       directory name == SwiftAcervo's slugify(modelId).
#   ACERVO_CI_MODELS    (optional) whitespace/newline-separated list of slugs.
#                       Merged with any slugs passed as positional args.
#   ACERVO_CDN_BASE     (optional) override CDN base. Default matches the value
#                       hardcoded in AcervoDownloader.cdnBaseURL.
#   ACERVO_PRIME_JOBS   (optional) parallel file downloads per model (default 4,
#                       matching AcervoDownloader.maxConcurrentDownloads).
#
set -euo pipefail

CDN_BASE="${ACERVO_CDN_BASE:-https://pub-8e049ed02be340cbb18f921765fd24f3.r2.dev/models}"
JOBS="${ACERVO_PRIME_JOBS:-4}"

if [[ -z "${ACERVO_MODELS_DIR:-}" ]]; then
  echo "::error::ACERVO_MODELS_DIR is not set. Point it at the cache root." >&2
  exit 2
fi

command -v jq   >/dev/null || { echo "::error::jq not found on PATH"   >&2; exit 2; }
command -v curl >/dev/null || { echo "::error::curl not found on PATH" >&2; exit 2; }

# Collect slugs from $ACERVO_CI_MODELS plus positional args.
read -r -a SLUGS <<< "$(printf '%s %s' "${ACERVO_CI_MODELS:-}" "$*" | tr '\n' ' ')"
if [[ ${#SLUGS[@]} -eq 0 ]]; then
  echo "::error::No model slugs supplied (set ACERVO_CI_MODELS or pass args)." >&2
  exit 2
fi

# curl with retries; -f makes HTTP errors non-zero, -L follows redirects.
fetch() { curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors "$@"; }

# Emits the (slug<TAB>path<TAB>size) work lines on stdout; all human-readable
# progress and errors go to stderr so they never pollute the work list.
prime_one() {
  local slug="$1"
  local model_url="$CDN_BASE/$slug"
  local dest="$ACERVO_MODELS_DIR/$slug"
  mkdir -p "$dest"

  echo "→ $slug" >&2
  local manifest="$dest/manifest.json"
  if ! fetch -o "$manifest" "$model_url/manifest.json"; then
    echo "::error::Cannot fetch manifest for '$slug' at $model_url/manifest.json." >&2
    echo "::error::Has this model been shipped? Run /acervo-download-ship or /acervo-cdn-setup first." >&2
    return 1
  fi

  # Reject legacy hand-rolled manifests ({"files":[{"name","size"}]}).
  # SwiftAcervo's CDNManifest requires .files[].path + .manifestChecksum; a
  # legacy manifest on disk would be rejected by the validator at test time.
  if [[ "$(jq -r '.files[0].path // empty' "$manifest")" == "" ]]; then
    echo "::error::'$slug' has a LEGACY manifest (no .files[].path / .manifestChecksum)." >&2
    echo "::error::SwiftAcervo cannot validate it. Re-ship with: acervo ship <org/repo> --slug $slug" >&2
    return 1
  fi

  # One work line per file: "slug path sizeBytes" (space-separated). HF/CDN file
  # paths never contain spaces, so three columns survive xargs -n3 cleanly —
  # and -n3 sidesteps the tab-handling quirks of BSD xargs -I.
  jq -r --arg slug "$slug" '.files[] | "\($slug) \(.path) \(.sizeBytes)"' "$manifest"
}

# Build the full work list (slug path size) first, then fetch in parallel.
WORKLIST="$(mktemp)"
trap 'rm -f "$WORKLIST"' EXIT
for slug in "${SLUGS[@]}"; do
  [[ -z "$slug" ]] && continue
  prime_one "$slug" >> "$WORKLIST"
done

export ACERVO_MODELS_DIR CDN_BASE
fetch_file() {
  local slug="$1" path="$2" size="$3" dest url have
  dest="$ACERVO_MODELS_DIR/$slug/$path"
  url="$CDN_BASE/$slug/$path"
  if [[ -f "$dest" ]]; then
    have="$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null || echo -1)"
    if [[ "$have" == "$size" ]]; then
      echo "  ✓ $slug/$path (cached)" >&2
      return 0
    fi
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  ↓ $slug/$path ($size bytes)" >&2
  curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors -o "$dest" "$url"
}
export -f fetch_file

if [[ -s "$WORKLIST" ]]; then
  # -P fans out JOBS workers; -n3 hands each worker exactly slug/path/size.
  xargs -P "$JOBS" -n3 bash -c 'fetch_file "$@"' _ < "$WORKLIST"
fi

echo "✅ Primed ${#SLUGS[@]} model(s) into $ACERVO_MODELS_DIR"
