# Build a single-file Windows executable with PyInstaller.
#
# Run from the repo root:
#   .\scripts\build_windows.ps1
# Requires the `dev` extra to be installed in the active venv:
#   pip install -e ".[dev,windows]"
param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Error "pyinstaller not found. Run: pip install -e `".[dev,windows]`""
}

Remove-Item -Recurse -Force build, dist, folderangel.spec -ErrorAction SilentlyContinue

pyinstaller `
    --noconfirm `
    --name FolderAngel `
    --windowed `
    --onefile `
    --paths "$root\src" `
    --collect-submodules folderangel `
    --collect-submodules PySide6 `
    "$root\src\folderangel\__main__.py"

Write-Host "Built: $root\dist\FolderAngel.exe"
