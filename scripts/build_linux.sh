#!/usr/bin/env bash
# Build a Linux one-folder bundle and (optionally) an AppImage.
#
# Usage:
#   bash scripts/build_linux.sh            # bundle only
#   bash scripts/build_linux.sh appimage   # bundle + AppImage
#
# Run from an activated venv that has the dev extras installed:
#   pip install -e '.[dev]'
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

if ! command -v pyinstaller >/dev/null 2>&1; then
    echo "pyinstaller not found — run: pip install -e '.[dev]'" >&2
    exit 1
fi

rm -rf build dist
pyinstaller --noconfirm scripts/folder1004.spec
echo
echo "Built bundle: $ROOT/dist/folder1004/"
echo "Run:           $ROOT/dist/folder1004/folder1004"

if [[ "${1:-}" == "appimage" ]]; then
    if ! command -v appimagetool >/dev/null 2>&1; then
        echo "appimagetool not on PATH — skipping AppImage step." >&2
        exit 0
    fi
    APPDIR="$ROOT/dist/Folder1004.AppDir"
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
    cp -r "$ROOT/dist/folder1004/." "$APPDIR/usr/bin/"
    cp "$ROOT/assets/icon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/folder1004.png" 2>/dev/null || true
    cp "$ROOT/assets/icon.png" "$APPDIR/folder1004.png" 2>/dev/null || true
    cat > "$APPDIR/folder1004.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Folder1004
Exec=folder1004
Icon=folder1004
Categories=Utility;FileTools;
Comment=LLM-powered folder auto-organizer
Terminal=false
EOF
    cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/folder1004" "$@"
EOF
    chmod +x "$APPDIR/AppRun"
    appimagetool "$APPDIR" "$ROOT/dist/Folder1004-x86_64.AppImage"
    echo
    echo "AppImage:      $ROOT/dist/Folder1004-x86_64.AppImage"
fi
