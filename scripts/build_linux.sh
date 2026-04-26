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
pyinstaller --noconfirm scripts/folderangel.spec
echo
echo "Built bundle: $ROOT/dist/folderangel/"
echo "Run:           $ROOT/dist/folderangel/folderangel"

if [[ "${1:-}" == "appimage" ]]; then
    if ! command -v appimagetool >/dev/null 2>&1; then
        echo "appimagetool not on PATH — skipping AppImage step." >&2
        exit 0
    fi
    APPDIR="$ROOT/dist/FolderAngel.AppDir"
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
    cp -r "$ROOT/dist/folderangel/." "$APPDIR/usr/bin/"
    cp "$ROOT/assets/icon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/folderangel.png" 2>/dev/null || true
    cp "$ROOT/assets/icon.png" "$APPDIR/folderangel.png" 2>/dev/null || true
    cat > "$APPDIR/folderangel.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=FolderAngel
Exec=folderangel
Icon=folderangel
Categories=Utility;FileTools;
Comment=LLM-powered folder auto-organizer
Terminal=false
EOF
    cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/folderangel" "$@"
EOF
    chmod +x "$APPDIR/AppRun"
    appimagetool "$APPDIR" "$ROOT/dist/FolderAngel-x86_64.AppImage"
    echo
    echo "AppImage:      $ROOT/dist/FolderAngel-x86_64.AppImage"
fi
