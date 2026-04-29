# -*- mode: python ; coding: utf-8 -*-
"""Unified PyInstaller spec — works on Linux, macOS, and Windows.

Build with::

    pyinstaller scripts/folder1004.spec

Output:
  Linux:    dist/folder1004/folder1004             (one-folder bundle)
  Windows:  dist/folder1004/folder1004.exe         (one-folder bundle)
  macOS:    dist/folder1004/folder1004             (one-folder bundle)
            dist/Folder1004.app                    (.app bundle, BUNDLE)

We ship as one-folder rather than one-file because PySide6 plugin
discovery is more reliable that way and start-up time is faster.
"""
from pathlib import Path
import sys

block_cipher = None
ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
ENTRY = SRC / "folder1004" / "__main__.py"
ICON_LINUX = ROOT / "assets" / "icon.png"
ICON_MAC   = ROOT / "assets" / "icon.icns"
ICON_WIN   = ROOT / "assets" / "icon.ico"

icon = None
if sys.platform.startswith("win") and ICON_WIN.exists():
    icon = str(ICON_WIN)
elif sys.platform == "darwin" and ICON_MAC.exists():
    icon = str(ICON_MAC)
elif ICON_LINUX.exists():
    icon = str(ICON_LINUX)

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "folder1004",
        "folder1004.ui",
        "folder1004.parsers",
        "folder1004.llm",
        # PyPDF / python-docx / python-pptx / openpyxl / olefile and
        # their internal lazy imports — PyInstaller occasionally misses.
        "pypdf", "docx", "pptx", "openpyxl", "olefile",
        "PySide6.QtSvg",  # for crisp SVG icons
        # kiwipiepy ships its model as a package data file; the C
        # extension is found automatically but the model resource path
        # needs the package itself in the bundle.
        "kiwipiepy", "kiwipiepy_model",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="folder1004",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,         # GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="folder1004",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Folder1004.app",
        icon=icon,
        bundle_identifier="app.folder1004",
        info_plist={
            "CFBundleName": "Folder1004",
            "CFBundleDisplayName": "Folder1004",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSRequiresAquaSystemAppearance": False,
        },
    )
