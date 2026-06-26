#!/bin/bash
# Build Flux2App + mlx.metallib, then wrap as Flux2App.app with icon for Dock/Finder.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-debug}"
BUNDLE_NAME="${BUNDLE_NAME:-Flux2App.app}"
BUNDLE="$ROOT/$BUNDLE_NAME"

case "$CONFIG" in
  [Dd]ebug) CONFIG_LC=debug; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/debug" ;;
  [Rr]elease) CONFIG_LC=release; BUILD_DIR="$ROOT/.build/arm64-apple-macosx/release" ;;
  *) echo "CONFIG must be debug or release" >&2; exit 1 ;;
esac

cd "$ROOT"
swift build --product Flux2App -c "$CONFIG_LC"
"$ROOT/bin/build-mlx-metallib.sh"
"$ROOT/bin/build-app-icon.sh"

ICNS="$ROOT/Assets/AppIcon/AppIcon.icns"
if [[ ! -f "$ICNS" ]]; then
  echo "App icon missing after build-app-icon.sh" >&2
  exit 1
fi

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"

cp "$BUILD_DIR/Flux2App" "$BUNDLE/Contents/MacOS/Flux2App"
cp "$BUILD_DIR/mlx.metallib" "$BUNDLE/Contents/MacOS/mlx.metallib"
cp "$ICNS" "$BUNDLE/Contents/Resources/AppIcon.icns"

cat >"$BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>Flux2App</string>
	<key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundleIdentifier</key>
	<string>com.realnotsteve.flux2app</string>
	<key>CFBundleName</key>
	<string>FLUX.2</string>
	<key>CFBundleDisplayName</key>
	<string>FLUX.2</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>2.4.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>15.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>CFBundleDocumentTypes</key>
	<array>
		<dict>
			<key>CFBundleTypeExtensions</key>
			<array>
				<string>flux2project</string>
			</array>
			<key>CFBundleTypeName</key>
			<string>FLUX.2 Project</string>
			<key>CFBundleTypeRole</key>
			<string>Editor</string>
			<key>LSHandlerRank</key>
			<string>Owner</string>
			<key>LSItemContentTypes</key>
			<array>
				<string>com.realnotsteve.flux2-project</string>
			</array>
		</dict>
	</array>
	<key>UTExportedTypeDeclarations</key>
	<array>
		<dict>
			<key>UTTypeConformsTo</key>
			<array>
				<string>com.apple.package</string>
			</array>
			<key>UTTypeDescription</key>
			<string>FLUX.2 Project</string>
			<key>UTTypeIdentifier</key>
			<string>com.realnotsteve.flux2-project</string>
			<key>UTTypeTagSpecification</key>
			<dict>
				<key>public.filename-extension</key>
				<array>
					<string>flux2project</string>
				</array>
			</dict>
		</dict>
	</array>
</dict>
</plist>
PLIST

echo "Packaged $BUNDLE"
echo "Open with: open \"$BUNDLE\""
