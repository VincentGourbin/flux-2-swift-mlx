#!/bin/bash
# Build Flux2App, stage as build/Flux2App.app (gitignored). Install via Project Builder relocate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-debug}"
BUNDLE_NAME="${BUNDLE_NAME:-Flux2App.app}"
STAGING_DIR="$ROOT/build"
BUNDLE="$STAGING_DIR/$BUNDLE_NAME"
INFO_TEMPLATE="$ROOT/Supporting/Flux2App/Info.plist"

read_version() {
  local app_info="$ROOT/Sources/Flux2Core/Flux2Core.swift"
  local full_version build_number short_version
  full_version="$(sed -n 's/^[[:space:]]*public static let version = "\(.*\)".*/\1/p' "$app_info" | head -1)"
  if [[ -z "$full_version" ]]; then
    echo "Could not read Flux2Core.version from $app_info" >&2
    exit 1
  fi
  build_number="${full_version##*.}"
  short_version="${full_version%.*}"
  if [[ -z "$build_number" || -z "$short_version" || "$short_version" == "$full_version" ]]; then
    echo "Flux2Core.version must be Major.Minor.Revision.Build (got: $full_version)" >&2
    exit 1
  fi
  SHORT_VERSION="$short_version"
  BUILD_NUMBER="$build_number"
}

case "$CONFIG" in
  [Dd]ebug) CONFIG_LC=debug; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/debug" ;;
  [Rr]elease) CONFIG_LC=release; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/release" ;;
  *) echo "CONFIG must be debug or release" >&2; exit 1 ;;
esac

cd "$ROOT"
read_version
swift build --product Flux2App -c "$CONFIG_LC"
"$ROOT/bin/build-mlx-metallib.sh"
"$ROOT/bin/build-app-icon.sh"

ICNS="$ROOT/Assets/AppIcon/AppIcon.icns"
if [[ ! -f "$ICNS" ]]; then
  echo "App icon missing after build-app-icon.sh" >&2
  exit 1
fi
if [[ ! -f "$INFO_TEMPLATE" ]]; then
  echo "Missing Info.plist template: $INFO_TEMPLATE" >&2
  exit 1
fi

rm -rf "$BUNDLE"
mkdir -p "$STAGING_DIR"
touch "$STAGING_DIR/.metadata_never_index"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"

cp "$BUILD_DIR/Flux2App" "$BUNDLE/Contents/MacOS/Flux2App"
cp "$BUILD_DIR/mlx.metallib" "$BUNDLE/Contents/MacOS/mlx.metallib"
cp "$ICNS" "$BUNDLE/Contents/Resources/AppIcon.icns"
cp "$INFO_TEMPLATE" "$BUNDLE/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $SHORT_VERSION" "$BUNDLE/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $BUILD_NUMBER" "$BUNDLE/Contents/Info.plist"

echo "Packaged $BUNDLE ($SHORT_VERSION build $BUILD_NUMBER)"
echo "Canonical install: /Applications/Flux2App.app (Project Builder relocate)"
