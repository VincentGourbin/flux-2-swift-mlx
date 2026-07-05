#!/bin/bash
# Build mlx.metallib for SwiftPM Flux2App builds.
#
# swift build does not compile MLX Metal kernels (upstream limitation). This runs
# xcodebuild on mlx-swift's Cmlx scheme and copies default.metallib beside
# Flux2App as mlx.metallib (@loader_path lookup).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLX_ROOT="${MLX_ROOT:-$ROOT/.build/checkouts/mlx-swift}"
CONFIG="${CONFIG:-Debug}"
DERIVED_DATA="${DERIVED_DATA:-/tmp/mlx-metallib-dd}"

case "$CONFIG" in
  Debug) BUILD_DIR="${BUILD_DIR:-$ROOT/.build/arm64-apple-macosx/debug}" ;;
  Release) BUILD_DIR="${BUILD_DIR:-$ROOT/.build/arm64-apple-macosx/release}" ;;
  *) echo "CONFIG must be Debug or Release (got: $CONFIG)" >&2; exit 1 ;;
esac

if [[ ! -d "$MLX_ROOT/xcode/MLX.xcodeproj" ]]; then
  echo "mlx-swift checkout not found at $MLX_ROOT — run: swift package resolve" >&2
  exit 1
fi

echo "Building Cmlx ($CONFIG) → $BUILD_DIR/mlx.metallib"
if ! xcodebuild build \
  -project "$MLX_ROOT/xcode/MLX.xcodeproj" \
  -scheme Cmlx \
  -configuration "$CONFIG" \
  -destination 'platform=macOS' \
  -derivedDataPath "$DERIVED_DATA" \
  ONLY_ACTIVE_ARCH=YES 2>&1 | tee /tmp/build-mlx-metallib.log | tail -8; then
  if grep -q 'missing Metal Toolchain' /tmp/build-mlx-metallib.log; then
    echo "Install the Metal Toolchain: xcodebuild -downloadComponent MetalToolchain" >&2
  fi
  exit 1
fi

METALLIB="$DERIVED_DATA/Build/Products/$CONFIG/Cmlx.framework/Versions/A/Resources/default.metallib"
if [[ ! -f "$METALLIB" ]]; then
  echo "Expected metallib missing: $METALLIB" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cp "$METALLIB" "$BUILD_DIR/mlx.metallib"
ls -lh "$BUILD_DIR/mlx.metallib"
