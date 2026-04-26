#!/usr/bin/env bash
# Build the macOS .app bundle (and optionally a .dmg) for FolderAngel.
#
# Usage (run from the repo root, inside an activated venv with dev extras):
#   bash scripts/build_macos.sh           # .app only
#   bash scripts/build_macos.sh dmg       # .app + .dmg
#
# Requires:
#   pip install -e '.[dev]'
#   (for dmg) brew install create-dmg
#
# Code signing / notarisation is OUT OF SCOPE here — see Apple's docs.
# The unsigned .app will run on the developer's own Mac with a
# Gatekeeper override (right-click → Open the first time).
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

APP="$ROOT/dist/FolderAngel.app"
if [[ ! -d "$APP" ]]; then
    echo "Expected $APP after build but it's missing." >&2
    exit 1
fi
echo
echo "Built: $APP"
echo "First run: right-click → Open (Gatekeeper) until you sign + notarise."

if [[ "${1:-}" == "dmg" ]]; then
    if ! command -v create-dmg >/dev/null 2>&1; then
        echo "create-dmg not found — run: brew install create-dmg" >&2
        exit 0
    fi
    DMG="$ROOT/dist/FolderAngel-1.0.0.dmg"
    rm -f "$DMG"
    create-dmg \
        --volname "FolderAngel" \
        --window-pos 200 200 \
        --window-size 640 400 \
        --icon-size 110 \
        --icon "FolderAngel.app" 175 200 \
        --hide-extension "FolderAngel.app" \
        --app-drop-link 465 200 \
        "$DMG" \
        "$APP"
    echo
    echo "DMG: $DMG"
fi
