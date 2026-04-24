# FolderAngel

LLM-powered folder auto-organizer for Linux and Windows.

FolderAngel scans a folder you point it at, reads file names, metadata and
the first page of document bodies (PDF/DOCX/PPTX/HWP/HWPX/TXT/…), then asks
Google Gemini to design a set of human-friendly Korean/English folders and
file-by-file assignments.  Files with ambiguous placement are moved into the
best-fit folder and shortcuts (symlink on Linux, `.lnk` on Windows) are left
in the runners-up.  Every run is recorded in a local SQLite index so you can
find moved files later by name or reason.

- LLM is used **only** for naming folders and planning assignments.  All file
  IO stays on your machine.
- **Mock planner** kicks in when you don't have an API key, so the app is
  usable offline with sensible heuristics.
- **Rollback** any past run from the History tab (or via `IndexDB.rollback`).

## Install

```bash
git clone https://github.com/Cheol-H-Jeong/folderangel.git
cd folderangel

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Python 3.11+ is required.  On Windows, `pip install -e ".[windows]"` pulls in
`pywin32` so that real Windows shell shortcuts are produced.

## Run

```bash
folderangel                       # launches the GUI
folderangel --cli --path ~/Downloads --recursive --dry-run
```

Put your Gemini API key in the **Settings** tab (stored in OS keyring if
available, otherwise in `~/.folderangel/config.json`).  Alternatively export
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) before launch.

Without a key FolderAngel still works — it just falls back to a deterministic
heuristic planner (extension + filename keyword) so you can preview the flow.

## Features

- Scan with optional recursion; safe patterns skip hidden/system files and
  never follow symlinks.
- Extracts the first ~1,800 characters of body text from PDF, DOCX, PPTX,
  XLSX, ODT, RTF, HWP, HWPX, HTML, and plain-text files.
- Two-stage planner (per-batch candidates → merge → per-batch assignment) so
  thousands of files still fit LLM context limits.
- Dry-run mode that shows the plan without touching files.
- Markdown report written to the target folder on every run.
- SQLite FTS5 index (+ LIKE fallback) for fast search by filename, folder,
  category, or original path.
- Rollback restores files to their original paths and removes now-empty
  category folders.

## CLI options

```
usage: folderangel [-h] [--cli] [--path PATH] [--recursive] [--dry-run] [--mock] [--quiet]

  --cli         run headless without launching the UI
  --path PATH   target folder
  --recursive   include subfolders
  --dry-run     plan only, do not move files
  --mock        force mock planner (skip Gemini even if key is set)
  --quiet       suppress progress output
```

## Project layout

```
docs/SPEC.md      — full functional & UI specification
docs/MODULES.md   — per-module contracts
src/folderangel/  — Python package
tests/            — pytest suite (unit + pipeline smoke)
scripts/          — cross-platform PyInstaller build scripts
```

## Build standalone binaries

```bash
bash scripts/build_linux.sh
powershell scripts\build_windows.ps1
```

Each script produces a single-file executable in `dist/`.  PyInstaller ≥ 6.5
is required (installed via the `dev` extra).

## License

MIT — see `LICENSE`.
