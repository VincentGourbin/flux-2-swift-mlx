#!/bin/bash
# Build Assets/AppIcon/AppIcon.icns from Flux2App-1024.png (macOS iconutil).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/Assets/AppIcon/Flux2App-1024.png"
ICONSET="$ROOT/Assets/AppIcon/AppIcon.iconset"
ICNS="$ROOT/Assets/AppIcon/AppIcon.icns"

if [[ ! -f "$SRC" ]]; then
  echo "Missing source icon: $SRC" >&2
  exit 1
fi

rm -rf "$ICONSET"
mkdir -p "$ICONSET"

make_icon() {
  local name="$1"
  local size="$2"
  sips -z "$size" "$size" "$SRC" --out "$ICONSET/$name" >/dev/null
}

make_icon icon_16x16.png 16
make_icon icon_16x16@2x.png 32
make_icon icon_32x32.png 32
make_icon icon_32x32@2x.png 64
make_icon icon_128x128.png 128
make_icon icon_128x128@2x.png 256
make_icon icon_256x256.png 256
make_icon icon_256x256@2x.png 512
make_icon icon_512x512.png 512
make_icon icon_512x512@2x.png 1024

iconutil -c icns "$ICONSET" -o "$ICNS"
echo "Wrote $ICNS"
