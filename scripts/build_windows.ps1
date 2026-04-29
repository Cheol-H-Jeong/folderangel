# Build a Windows one-folder bundle (and optionally an Inno Setup installer).
#
# Usage (PowerShell, repo root, with an activated venv that has dev + windows extras):
#   pip install -e ".[dev,windows]"
#   .\scripts\build_windows.ps1                # bundle only
#   .\scripts\build_windows.ps1 -Installer     # bundle + .exe installer (requires Inno Setup 6 'iscc')
#
# Code signing / SmartScreen reputation is OUT OF SCOPE.  The unsigned
# bundle still runs locally; SmartScreen may warn on first run.
param(
    [switch]$Installer
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Error 'pyinstaller not found — run: pip install -e ".[dev,windows]"'
}

Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

pyinstaller --noconfirm "scripts\folder1004.spec"

$bundle = "$root\dist\folder1004"
if (-not (Test-Path $bundle)) {
    Write-Error "Build did not produce $bundle"
}
Write-Host ""
Write-Host "Built bundle: $bundle"
Write-Host "Run:          $bundle\folder1004.exe"

if ($Installer) {
    $iscc = Get-Command iscc -ErrorAction SilentlyContinue
    if (-not $iscc) {
        Write-Warning "Inno Setup 'iscc' not on PATH — skipping installer step."
        Write-Warning "Install from https://jrsoftware.org/isdl.php and re-run."
        exit 0
    }
    iscc "$root\scripts\folder1004.iss"
    Write-Host ""
    Write-Host "Installer: $root\dist\Folder1004-Setup.exe"
}
