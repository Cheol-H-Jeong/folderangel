#!/usr/bin/env bash
# Build a single-file Linux binary with PyInstaller.
#
# Run from the repo root, inside an activated virtualenv that has the `dev`
# extra installed.  The resulting executable lives under ``dist/folderangel``.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

if ! command -v pyinstaller >/dev/null 2>&1; then
    echo "pyinstaller not found; install with: pip install -e '.[dev]'" >&2
    exit 1
fi

rm -rf build dist folderangel.spec

pyinstaller \
    --noconfirm \
    --name folderangel \
    --windowed \
    --onefile \
    --icon "$ROOT/assets/icon.png" 2>/dev/null || true

pyinstaller \
    --noconfirm \
    --name folderangel \
    --windowed \
    --onefile \
    --paths "$ROOT/src" \
    --collect-submodules folderangel \
    --collect-submodules PySide6 \
    "$ROOT/src/folderangel/__main__.py"

echo "Built: $ROOT/dist/folderangel"
