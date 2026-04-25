# Self-review checklist — applies to every code change in any project

This is a generalised QA manual derived from real bugs the user caught in
my work.  Each rule maps a concrete past failure to a class of mistakes,
so I don't repeat them in different shapes across other projects.

**How to use it:** before declaring any task "done" — and especially
before committing — walk this checklist and explicitly verify each
applicable item.  Do not skip silently.  When a rule does not apply to
the current change, say so.  Treat unverified items as defects.

**Self-update loop (permanent rule):** Every time the user points out a
defect in any project I work on, I (1) fix it, (2) abstract the lesson
into a project-agnostic rule, (3) append or sharpen the relevant item
in this manual *before* closing the conversation, (4) commit/push the
manual update along with the fix, (5) note in the response which item
was added.  No user-reported bug is allowed to leave only a one-off
fix behind — every one becomes a permanent guard.

---

## A. End-to-end UX completeness

The user repeatedly caught me reporting "done" when the result still
looked broken from their seat.  Test the *user-visible* outcome, not
just the unit tests.

- **A1. Try the actual interface I changed.**  If I touched a CLI flag,
  run the CLI.  If I changed a UI label, instantiate the view (offscreen
  Qt is fine).  Code-level tests don't catch wrong copy or wrong layouts.
- **A2. Verify the outcome a real user double-clicks.**  Shortcuts /
  launchers / opener handlers must actually open the target on the
  target OS — write a `.desktop`/`.lnk`/symlink and confirm with the
  platform's open mechanism, don't just assume `os.symlink` "works".
- **A3. Empty / partial states must be tidy.**  After any operation
  that creates folders or files: sweep empties, dedupe near-duplicates,
  unify naming conventions.  Half-done outputs are user-visible bugs.
- **A4. Cancel / abort paths must be near-instant.**  Long-running work
  needs a cancel token threaded all the way through (HTTP socket close,
  loop checkpoints, worker termination).  ≥ 1 s wall-clock between
  click and acknowledgement is a defect.
- **A5. Long operations must show liveness, not just a percentage.**
  Heartbeat lines (`… 7s 경과`), per-file status, or token counters —
  silence > 1 s during a single sub-step looks broken.
- **A6. Error messages reach the user in their language and tone.**
  Don't surface raw English exception strings or modal "Critical
  Error" dialogs for routine failures (cancel / timeout / 401).  Map
  them to friendly inline copy in the user's UI language.

## B. External I/O correctness (network / files / subprocess)

- **B1. Force the encoding I expect.**  `requests.iter_lines
  (decode_unicode=True)` and similar APIs default to *Latin-1* when the
  server omits charset.  Set `resp.encoding = "utf-8"` and decode bytes
  explicitly.  Symptom: `ì¤`, `ë³´`, `Ã©` in any user-visible string =
  encoding bug, almost always on the *client* side first.
- **B2. Don't trust LLM/server output to be complete.**  Detect
  truncation (`finish_reason == "length"`, missing closing quote /
  bracket).  Bump the budget and retry once; on second failure raise a
  clean error instead of feeding half-data to a parser.
- **B3. Recover gracefully from "almost-valid" structured output.**  If
  JSON parsing fails, try once more after stripping code fences, leading
  reasoning blocks (`</think>` etc.), and balancing brackets.  But never
  silently accept garbage — if the recovery still fails, raise.
- **B4. Sanitise anything the LLM/server gave us before it touches the
  filesystem or shell.**  Strip control chars, replacement char (`�`),
  BOM, JSON-key leakage (`"name":"…`), Windows reserved names.  Reject
  values with fewer than ~2 visible characters.
- **B5. Pin auth / config sources explicitly.**  When a CLI flag changes
  the provider/host, also reset stale fields that belonged to the old
  provider (e.g. base_url) unless explicitly carried over.
- **B6. Path quoting on every `Exec=` / shell command.**  Use
  `shlex.quote` (or `subprocess` arg-list form) so spaces / Korean /
  special chars don't break.
- **B7. Detect and recover from transient errors with bounded retries
  and growing timeouts.**  Don't infinite-loop.  Don't retry on cancel.

## C. LLM-specific defaults

- **C1. Default `max_tokens` on local servers (llama.cpp / Ollama) is
  often 256 — far too small.**  Pass an explicit, generous value
  (≥ 4 096 for plans).  Detect `finish_reason=length` and bump on retry.
- **C2. Reasoning ("thinking") models — Qwen3 / DeepSeek-R1 / Magistral
  / Phi-4-mini-reasoning — burn hundreds of tokens before any visible
  content.  For pure structured output tasks default to *off*
  (`chat_template_kwargs.enable_thinking=false`, `/no_think` prefix).
  Strip residual `</think>` from the response.  Expose a user toggle —
  do not hard-disable.
- **C3. Pick the call topology by *prompt size*, not by provider name.**
  Fits in `context_window − response_budget` ⇒ single call.  Doesn't
  fit ⇒ chunk it.  Detect ctx via `GET /v1/models` (llama.cpp / Ollama
  / vLLM / OpenAI all expose it); fall back to a conservative assumed
  default.
- **C4. Stream when possible.**  Token-by-token feedback (a) keeps the
  read timeout from firing during long generations, (b) lets the user
  cancel within ~1 RTT, (c) gives a live "still working" signal.
- **C5. Don't display raw stream chunks in user-facing logs.**
  Multi-byte boundaries split inside a single SSE chunk and look like
  mojibake; JSON escapes look like noise out of context.  Show byte /
  char counts and stage labels instead, plus a *single* mojibake
  warning when the cumulative buffer trips the heuristic.
- **C6. Per-call latency / TTFT / tok-s must be measured and surfaced.**
  Without it, "feels slow" debates have no resolution.  Log + report
  every call's duration_s, ttft_s, prompt_chars, response_chars.
- **C7. Model output cost ≠ free.**  Surface a token + currency
  estimate per run.  Use a public per-model price table; treat unknown
  models as $0 only when the URL is local (Ollama / vLLM).

## D. Naming & filesystem hygiene

- **D1. Canonicalise folder/file names before writing.**  One template
  per category (e.g. `"{n}. {name} ({period})"`), applied to *every*
  category — including ones the LLM forgot to tag.  Pre-existing
  folders that fuzzy-match a new category get *renamed* to the
  canonical form, not duplicated.
- **D2. Always strip mojibake / replacement chars / BOM from any
  externally-supplied name** (LLM, scraped data, user paste).  Fall
  back to a safe placeholder if cleaning leaves < 2 visible chars.
- **D3. Order on disk should match the LLM's grouping intent.**  If the
  model proposed `group` numbers (relevance buckets), the prefix lets
  file managers sort by name and reproduce that grouping.
- **D4. Empty-folder sweep at the end of every batch operation.**
  Including pre-existing empty folders — they may be leftovers from
  previous runs.
- **D5. Never rely on host file manager to "follow" symlinks-to-files.**
  Some Linux file managers don't.  Use `.desktop` `Type=Link` URI
  (preferred) or `Type=Application` with `gio open`/`xdg-open`; on
  Windows use `.lnk` via pywin32 or PowerShell, with `.url` fallback.

## E. Diagnostics & evidence-based fixes

- **E1. Probe before guessing.**  When a service "feels slow / weird":
  run an actual measurement against the live endpoint (curl, raw HTTP
  client) before theorising about VRAM / context / network.  Past
  example: I claimed "VRAM offload" for a Qwen run when the user's GPU
  had plenty — actual cause was reasoning mode burning 250 tokens per
  call.  Always check the cheap hypothesis first.
- **E2. Open a per-run log file**, stream INFO+DEBUG into it, install
  `sys.excepthook` to capture tracebacks.  Surface the path in error
  toasts so the user can attach it.  Don't print blobs to stdout.
- **E3. When the user reports "still broken", reproduce the exact
  symptom they saw before changing anything.**  Probe the live system,
  read raw bytes, dump headers, then form a hypothesis.
- **E4. Distinguish "looks broken" from "is broken".**  Open WebUI
  hides reasoning latency in a collapsible block — same hardware, same
  model, same delay.  Don't conflate UX polish with speed.

## F. Process discipline

- **F1. Auto-commit and push without being asked, in projects where
  that's the user's stated preference.**  Use the project-local memory
  to track which projects opt in.
- **F2. Use the right git identity per machine.**  See the global
  "Git identity" section in `~/.claude/CLAUDE.md`.  No `-c
  user.email=…` overrides.
- **F3. Never tell the user to `git pull` for a project they run from
  the same machine I edit on — push is just a backup, not a delivery
  channel.**  Editable installs (`pip install -e .`, `npm link`) take
  effect immediately; new sessions read the latest code without any
  user action.
- **F4. Track real progress, not promises.**  Use TaskCreate /
  TaskUpdate for multi-step work; close tasks only when the user-facing
  outcome is verified, not when "the diff compiles".

## G. When applying this checklist

For every non-trivial change, write (mentally or in the response) one
line per applicable section letter explaining the verification, e.g.:

```
A: re-instantiated SettingsView headless, label updated to "OpenAI API 키"
B1: forced resp.encoding="utf-8", verified raw bytes decode cleanly
C1: max_tokens bumped to 8192, finish_reason=length retry path tested
D4: empty-folder sweep test green
F1: committed `abc123` and pushed
```

If a section doesn't apply, name it: "C: no LLM call touched in this
patch."  Skipping the checklist silently is itself a defect.
