# Folder1004

LLM-powered folder auto-organizer for **Linux · macOS · Windows**.

Folder1004 scans a folder you point it at, reads file names + metadata
+ the first page of document bodies (PDF/DOCX/PPTX/HWP/HWPX/XLSX/TXT/…),
asks an LLM to design human-friendly folders grouped by **project /
business / period**, then moves each file into the right folder.
Ambiguous files get a hardlink in their secondary folder so you can
find them either way.  Every run is recorded in a local SQLite +
FTS5 index so you can search any past file by name, content, or
project.

It is designed for the folders people actually abandon: KakaoTalk
downloads, browser Downloads, Desktop, Documents, class folders,
studio shoot folders, admin paperwork, and semester-long student
archives.

- **Pluggable LLM**: Google Gemini *or* any OpenAI-compatible endpoint
  (OpenAI · OpenRouter · Together · Groq · Anthropic-via-gateway ·
  Ollama · vLLM · LM Studio · llama.cpp's HTTP server).  You only fill
  in API URL + API key; the model list is auto-discovered.
- **Period-aware folders**: each category gets a duration tag —
  `burst` (1 month) / `short` (Q1) / `annual` / `multi-year` —
  reflected in folder names.  E.g. `1. 범정부 초거대 AI 공통기반
  〈2023–2025〉` for a multi-year programme.
- **Folder profile detection**: Folder1004 can infer the kind of mess it
  is looking at — Downloads, KakaoTalk downloads, Desktop, Documents,
  photo/design studio, teacher/classroom, student/semester, admin/public
  office, research, or business paperwork — and choose a matching
  taxonomy instead of using one generic folder style everywhere.
- **User sorting principles**: type natural-language guidance like
  “고객명과 촬영일을 우선해줘” or click presets such as 프로젝트 중심,
  날짜/기간 중심, 보수적으로 정리, 버림 후보 분리. The guidance is saved
  locally and sent with each LLM planning prompt.
- **Review-only trash candidates**: broken downloads, stale installers,
  OS leftovers, empty files, and obvious temporary files are separated as
  “검토 필요” candidates.  Folder1004 does not silently delete them.
- **Folder health score**: abandoned folders get a clean-up score with
  concrete reasons like old installer count, loose root files, unsorted
  screenshots, duplicate candidates, and recent file growth.
- **Tray angel mode**: on Windows, Folder1004 is planned to run quietly in
  the system tray after reboot, watch user-selected messy folders, and
  suggest new or incremental clean-ups when the computer appears idle.
- **Cross-platform shortcuts**: hardlink on Linux, symlink on macOS,
  `.lnk` on Windows.  Double-click opens the file.
- **Privacy by default**: only filenames, dates, and ≤ 1,800 chars of
  body are sent to the LLM.  Logs auto-redact API keys.
- **Search**: Korean / English search across filename, folder name,
  category, reason, original path, and parsed body.  Live-as-you-type.

---

## Install

### Pre-built packages (recommended)

Grab a release for your OS from the [Releases](https://github.com/Cheol-H-Jeong/folder1004/releases)
page.  CI builds them on every tag (Linux bundle / macOS .app / Windows
one-folder bundle):

- **Linux** — `folder1004-linux-…` archive, extract anywhere, run
  `./folder1004/folder1004`.  An AppImage build is also produced when
  `appimagetool` is on the build host.
- **macOS** — open the `.app`.  First launch: right-click → Open
  (Gatekeeper warning; signing/notarisation is pending).
- **Windows** — extract the bundle and run `folder1004.exe`, or use
  the Inno-Setup `Folder1004-Setup.exe` if available.  SmartScreen
  may warn on first launch (unsigned).

### From source (any OS, Python ≥ 3.11)

```bash
git clone https://github.com/Cheol-H-Jeong/folder1004.git
cd folder1004

python3 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"          # add ',windows' on Windows for pywin32
folder1004                       # launches the GUI
```

---

## First run

1. Open the **Settings** tab.
2. Fill in **API 엔드포인트** and **API 키**:
   - Gemini: `https://generativelanguage.googleapis.com/v1beta` + your AI Studio key.
   - OpenAI:  `https://api.openai.com/v1` + `sk-…`.
   - Local Ollama / vLLM / llama.cpp: `http://localhost:11434/v1` (or 8080/v1 etc.) + the key your server expects (often empty).
3. The model list auto-fills from the endpoint.  Single-model servers lock the field; multi-model providers show a drop-down.
4. Click **설정 저장**.  The status line under the connection card turns green ("● 연결 준비 — …").
5. Switch to **Organize**, drag a folder in (or click **폴더 선택…**), choose Dry-Run if you want to preview, click **정리 시작**.

You can also leave the API key blank — Folder1004 falls back to a
deterministic heuristic ("Mock 모드") so the rest of the app stays
usable offline.

---

## Search

Switch to the **Search** tab.  Typing queries the local index live —
filename, folder, category, reason, original path, parsed body
content all included.  Double-click a row to open the file's current
location.

If the index is empty (you haven't organised yet, or organised a
folder a previous run didn't record), click **폴더 다시 인덱싱…** and
pick the folder — its files become searchable in seconds without
any LLM call.

---

## CLI

```
folder1004 --cli --path ~/Downloads --recursive --dry-run
folder1004 --cli --path /work/docs --provider openai_compat \
            --base-url http://localhost:11434/v1 --model qwen2.5 \
            --reasoning off
```

Flags: `--path PATH` `--recursive` `--dry-run` `--mock`
`--no-economy` `--provider {gemini,openai_compat}` `--base-url URL`
`--model NAME` `--reasoning {off,on,auto}` `--quiet`.

---

## Where data lives

| OS | Data dir |
| --- | --- |
| Linux   | `$XDG_DATA_HOME/folder1004` or `~/.local/share/folder1004` (legacy: `~/.folder1004`) |
| macOS   | `~/Library/Application Support/Folder1004` |
| Windows | `%LOCALAPPDATA%\Folder1004` |

Override with `FOLDER1004_HOME=/path/to/dir` for portable installs / tests.

The directory holds:
- `config.json` — non-secret settings.
- `index.db`    — SQLite + FTS5 search index.
- `logs/`       — per-run timestamped logs (DEBUG+INFO + tracebacks).
   API keys and Bearer tokens are redacted at the formatter; logs are
   never committed.

API keys go to the OS keyring (libsecret on Linux / Keychain on macOS
/ Credential Manager on Windows).  If keyring is unavailable, they
fall back to `config.json` with a clear warning.

---

## Build packages

```bash
# Linux (one-folder bundle in dist/folder1004/)
bash scripts/build_linux.sh
bash scripts/build_linux.sh appimage   # also build AppImage if appimagetool is installed

# macOS (.app in dist/Folder1004.app/)
bash scripts/build_macos.sh
bash scripts/build_macos.sh dmg        # also build a .dmg via create-dmg

# Windows (one-folder bundle in dist\folder1004\)
.\scripts\build_windows.ps1
.\scripts\build_windows.ps1 -Installer # also build Inno Setup .exe
```

Requires `pip install -e ".[dev]"` (Windows: add `,windows`).
PyInstaller spec is shared across all three OSes
(`scripts/folder1004.spec`).

---

## Cross-platform notes

- **Long paths on Windows**: works out of the box on Win 10 1607+ when
  `LongPathsEnabled` is on.  Otherwise paths > 260 chars may fail —
  enable via `gpedit.msc` → Computer Configuration → Administrative
  Templates → System → Filesystem → "Enable Win32 long paths".
- **macOS Gatekeeper**: the unsigned `.app` requires a right-click →
  Open the first time.  Notarised builds will land once a
  developer-id signing profile is attached.
- **Linux .desktop trust**: secondary-folder shortcuts use **hardlinks**
  by default — no Gatekeeper / "Allow Launching" toggle needed.
- **Korean filenames**: tested on NTFS, APFS, ext4 / Btrfs.  All
  filename + folder operations go through NFC normalisation.

---

## Project layout

```
docs/SPEC.md          full functional & UI specification
docs/MODULES.md       per-module contracts
docs/SELF_REVIEW.md   shared QA checklist (read before commits)
src/folder1004/      Python package
  ui/                 PySide6 views + worker
  parsers/            PDF / DOCX / PPTX / XLSX / HWP / legacy Office
  llm/                Gemini + OpenAI-compat clients
  …
tests/                pytest suite (60+ cases incl. cross-platform)
scripts/              PyInstaller spec + per-OS build scripts
.github/workflows/    CI matrix (Linux/macOS/Windows × py3.11/3.12)
```

---

## License

MIT — see `LICENSE`.
