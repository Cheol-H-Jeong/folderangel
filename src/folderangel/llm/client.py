"""LLM clients used by the planner.

Two transports are supported:

- :class:`GeminiClient` — Google AI Studio native REST endpoint.  Uses the
  ``response_mime_type = application/json`` knob for structured JSON.
- :class:`OpenAICompatClient` — any service that speaks OpenAI's
  ``POST /v1/chat/completions`` (OpenAI itself, Together, Groq,
  OpenRouter, Ollama with ``/v1``, vLLM, LM Studio, Gemini via the
  OpenAI-compat proxy at ``…/v1beta/openai``, etc.).  Asks the model for
  JSON via ``response_format = {"type": "json_object"}`` and falls back
  to plain text parsing.

Both clients expose the same surface — ``generate_json(prompt, *,
heartbeat=None)`` plus ``request_count`` / ``prompt_chars`` /
``response_chars`` — so the planner doesn't need to know which one is in
use.  :func:`make_llm_client` picks the right one from a :class:`Config`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

import requests

log = logging.getLogger(__name__)

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_ENDPOINT = _GEMINI_ENDPOINT  # legacy alias (kept for any external callers)


class LLMError(RuntimeError):
    pass


def list_models(base_url: str, api_key: str) -> list[str]:
    """Probe ``GET {base_url}/models`` and return the model ids the
    endpoint advertises.  Empty list on any failure.

    Works for OpenAI, OpenRouter, Together, Groq, Anthropic-via-gateway,
    Ollama, vLLM, LM Studio, llama.cpp's HTTP server — every shape we
    ship: each returns a list of objects with either ``id`` or
    ``name``.  Single-model endpoints typically return exactly one id,
    in which case the UI can populate the field automatically.
    """
    if not base_url:
        return []
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(url, headers=headers, timeout=(5.0, 5.0))
        if r.status_code != 200:
            return []
        data = r.json()
    except (requests.RequestException, ValueError):
        return []
    raw = data.get("data") or data.get("models") or []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name") or ""
            if mid:
                out.append(str(mid))
        elif isinstance(item, str):
            out.append(item)
    return out


def infer_provider_from_url(base_url: str, model: str = "") -> str:
    """Pick the right transport from the user-supplied API endpoint.

    The user only ever has to fill in two fields: API endpoint URL and
    API key.  We treat any URL that hits Google's native generative-
    language host as Gemini (which has its own request shape) and
    everything else as OpenAI-compatible.  An empty URL with a
    Gemini-style model name also routes to Gemini.
    """
    u = (base_url or "").lower()
    m = (model or "").lower()
    if not u:
        # No URL given — fall back to the model name to decide.
        return "gemini" if m.startswith(("gemini-", "models/gemini-")) else "openai_compat"
    if "generativelanguage.googleapis.com" in u and "/openai" not in u:
        # Google's *native* Gemini endpoint.  The OpenAI-compat proxy
        # at .../v1beta/openai is intentionally NOT matched here.
        return "gemini"
    return "openai_compat"


def make_llm_client(config, api_key: Optional[str]):
    """Build the appropriate LLM client from a :class:`Config`.

    Returns ``None`` when no key is available so the caller can fall
    back to the mock planner.  The ``llm_provider`` config field is
    derived from the URL when missing — the user is never asked to
    pick a provider explicitly.
    """
    if not api_key:
        return None
    base_url = (getattr(config, "llm_base_url", "") or "").strip()
    model = getattr(config, "model", "")
    provider = (getattr(config, "llm_provider", "") or "").lower()
    if provider not in ("gemini", "openai_compat"):
        provider = infer_provider_from_url(base_url, model)
    if provider == "gemini":
        client = GeminiClient(api_key=api_key, model=model or "gemini-2.5-flash")
        if base_url and ("googleapis.com" in base_url or "google" in base_url):
            client.base_url = base_url.rstrip("/")
        return client
    # Default: OpenAI-compatible.
    if not base_url:
        base_url = "https://api.openai.com/v1"
    return OpenAICompatClient(
        api_key=api_key,
        model=model or "gpt-4o-mini",
        base_url=base_url,
        # Reasoning is always handled automatically: detect reasoning
        # models by name and disable thinking, since for our JSON task
        # it's pure overhead.  No user knob.
        reasoning_mode="off",
    )


def _gemini_ctx_for(model: str) -> int:
    """Best-effort context length for known Gemini models (in tokens)."""
    m = (model or "").lower()
    if "gemini-2.5-pro" in m or "gemini-1.5-pro" in m:
        return 2_000_000
    if "gemini-1.5" in m:
        return 1_000_000
    if "gemini-2.5-flash" in m:
        return 1_000_000
    return 32_000


class GeminiClient:
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise LLMError("empty api key")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # Usage counters so the user can see how much the run cost.
        self.request_count: int = 0
        self.prompt_chars: int = 0
        self.response_chars: int = 0
        self.total_duration_s: float = 0.0
        self.calls: list = []  # list[LLMCall]

    def context_length(self) -> int:
        return _gemini_ctx_for(self.model)

    # ------------------------------------------------------------------
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        heartbeat: Optional[Callable[[float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """Send *prompt* to Gemini and parse the JSON reply.

        ``heartbeat`` is called from a background thread roughly once a
        second with the elapsed seconds while the HTTP call is in flight.
        Use it to keep a UI progress log moving so users don't think the
        app is hung when a single long inference is running.
        """
        url = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        }

        self.prompt_chars += len(prompt)

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            if cancel_check is not None and cancel_check():
                raise LLMError("canceled by user")
            # Lengthen the read timeout on each retry — the API often just
            # needs more wall-clock time on long batches.
            current_timeout = self.timeout * (1.0 + 0.5 * attempt)
            stop_evt = threading.Event()
            start_ts = time.monotonic()

            def _beat():
                while not stop_evt.is_set():
                    elapsed = time.monotonic() - start_ts
                    if heartbeat is not None:
                        try:
                            heartbeat(elapsed)
                        except Exception:
                            pass
                    if stop_evt.wait(1.0):
                        break

            beat_thread = None
            if heartbeat is not None:
                beat_thread = threading.Thread(target=_beat, daemon=True)
                beat_thread.start()
            try:
                resp = requests.post(
                    url,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=(15.0, current_timeout),  # (connect, read)
                )
                if resp.status_code >= 400:
                    # Retry on transient 429 / 5xx; otherwise fail fast.
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise LLMError(
                            f"gemini transient http {resp.status_code}: {resp.text[:200]}"
                        )
                    raise LLMError(
                        f"gemini http {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()
                text = _extract_text(data)
                duration = time.monotonic() - start_ts
                self.request_count += 1
                self.response_chars += len(text)
                self.total_duration_s += duration
                from ..models import LLMCall
                self.calls.append(
                    LLMCall(
                        label="gemini",
                        prompt_chars=len(prompt),
                        response_chars=len(text),
                        duration_s=duration,
                        success=True,
                    )
                )
                log.info(
                    "gemini call ok in %.2fs — prompt=%d chars, response=%d chars (~%.1f tok/s)",
                    duration, len(prompt), len(text),
                    (len(text) / 3) / duration if duration > 0 else 0.0,
                )
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Some models wrap JSON in code fences; try to strip them.
                    stripped = _strip_code_fence(text)
                    return json.loads(stripped)
            except (requests.RequestException, LLMError, json.JSONDecodeError) as exc:
                last_exc = exc
                duration = time.monotonic() - start_ts
                from ..models import LLMCall
                self.calls.append(
                    LLMCall(
                        label="gemini",
                        prompt_chars=len(prompt),
                        duration_s=duration,
                        success=False,
                        error=str(exc)[:200],
                    )
                )
                log.warning(
                    "gemini call failed in %.2fs (attempt %d/%d, timeout=%.0fs): %s",
                    duration, attempt + 1, self.max_retries + 1, current_timeout, exc,
                )
                attempt += 1
                if attempt <= self.max_retries:
                    time.sleep(min(20.0, 1.5 * (2 ** attempt)))
                continue
            finally:
                stop_evt.set()
                if beat_thread is not None:
                    beat_thread.join(timeout=0.1)
        raise LLMError(f"gemini failed after retries: {last_exc}")


def _extract_text(response: dict) -> str:
    try:
        candidates = response["candidates"]
        parts = candidates[0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected gemini response shape: {exc}") from exc


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    # Qwen3 / DeepSeek-R1 style models often leak a leading <think>…</think>
    # block (or just a trailing "</think>") even when reasoning is disabled.
    # Drop everything up to and including the closing think tag.
    low = t.lower()
    if "</think>" in low:
        idx = low.rfind("</think>")
        t = t[idx + len("</think>"):].lstrip()
    if t.startswith("```"):
        # Remove first fence line and optional language tag
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return t


class OpenAICompatClient:
    """Client for any OpenAI Chat Completions compatible API.

    Works against OpenAI, OpenRouter, Together, Groq, Ollama (``/v1``),
    vLLM, LM Studio, TGI, and Gemini's OpenAI-compatible proxy.  Local
    backends typically take much longer to first-token than hosted ones,
    so we (a) default to a *very* long read timeout (10 minutes), and
    (b) request server-sent-event streaming so the connection is kept
    alive by token chunks — read-timeout only fires if the server stops
    emitting tokens completely.

    The caller can supply a ``cancel_check`` callable.  We poll it during
    the streaming loop and abort the HTTP call as soon as it returns
    True, so the user's "Cancel" button reflects in real time instead of
    waiting for the entire timeout.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 600.0,
        max_retries: int = 3,
        stream: bool = True,
        reasoning_mode: str = "off",
    ) -> None:
        if not api_key:
            raise LLMError("empty api key")
        if not base_url:
            raise LLMError("empty base url")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.stream = stream
        # "on" / "off" / "auto" — see config.Config.reasoning_mode
        self.reasoning_mode = (reasoning_mode or "off").lower()
        self.request_count: int = 0
        self.prompt_chars: int = 0
        self.response_chars: int = 0
        self.total_duration_s: float = 0.0
        self.calls: list = []  # list[LLMCall]
        self._cached_ctx: Optional[int] = None

    def context_length(self, default: int = 8192) -> int:
        """Best-effort detection of the model's max context window.

        Tries ``GET {base_url}/models`` (OpenAI / llama.cpp / Ollama /
        vLLM all expose model metadata there).  Falls back to ``default``.
        """
        if self._cached_ctx is not None:
            return self._cached_ctx
        ctx = default
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=(5.0, 5.0),
            )
            if r.status_code == 200:
                data = r.json()
                items = data.get("data") or data.get("models") or []
                target_lc = (self.model or "").lower()
                for item in items:
                    name = str(item.get("id") or item.get("name") or "").lower()
                    if name == target_lc or name.endswith(target_lc) or target_lc in name:
                        meta = item.get("meta") or {}
                        for key in ("n_ctx_train", "context_length", "ctx", "max_input_tokens"):
                            v = meta.get(key) or item.get(key)
                            if isinstance(v, int) and v > 0:
                                ctx = v
                                break
                        break
        except (requests.RequestException, ValueError) as exc:
            log.debug("context_length probe failed: %s", exc)
        self._cached_ctx = ctx
        return ctx

    # ------------------------------------------------------------------
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        heartbeat: Optional[Callable[[float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        stream_text: Optional[Callable[[str, int], None]] = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Always respond with a single valid JSON object, no prose.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            # Local backends (llama.cpp / Ollama) often default ``max_tokens``
            # to 256, which truncates a 30-file plan halfway and leaves the
            # JSON unterminated.  Ask for plenty of headroom.
            "max_tokens": 8192,
        }
        if self.stream:
            body["stream"] = True
        # Reasoning ("thinking") models: Qwen3 / DeepSeek-R1 / Magistral /
        # Phi-4-mini-reasoning generate hundreds of internal reasoning
        # tokens before the actual answer.  For our task (return a JSON
        # plan) those tokens are pure overhead, but for users who want
        # them — Open WebUI shows them as a collapsed "💭 Thought for…"
        # block — the model still works either way.  Honour the user's
        # ``reasoning_mode`` config knob:
        #   "off"  → actively disable via every knob the major backends
        #            recognise (default; ~5–10× faster on our task).
        #   "on"   → leave thinking enabled; we transparently strip the
        #            ``…</think>`` prefix from the response before parsing.
        #   "auto" → same as "off" today; reserved for future heuristics.
        model_lc = (self.model or "").lower()
        is_reasoning_model = any(
            tag in model_lc
            for tag in ("qwen3", "qwen-3", "deepseek-r1", "magistral", "phi-4-mini-reasoning")
        )
        mode = (self.reasoning_mode or "off").lower()
        want_thinking = (mode == "on")
        if is_reasoning_model and not want_thinking:
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            # llama.cpp also accepts a `/no_think` prefix on the user turn.
            if body["messages"] and isinstance(body["messages"][-1].get("content"), str):
                if not body["messages"][-1]["content"].lstrip().startswith("/no_think"):
                    body["messages"][-1]["content"] = "/no_think " + body["messages"][-1]["content"]

        self.prompt_chars += len(prompt)
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            if cancel_check is not None and cancel_check():
                raise LLMError("canceled by user")
            current_timeout = self.timeout * (1.0 + 0.25 * attempt)
            stop_evt = threading.Event()
            start_ts = time.monotonic()

            def _beat():
                while not stop_evt.is_set():
                    elapsed = time.monotonic() - start_ts
                    if heartbeat is not None:
                        try:
                            heartbeat(elapsed)
                        except Exception:
                            pass
                    if stop_evt.wait(1.0):
                        break

            beat_thread = None
            if heartbeat is not None:
                beat_thread = threading.Thread(target=_beat, daemon=True)
                beat_thread.start()
            resp = None
            cancel_watcher_stop = threading.Event()

            def _watch_cancel():
                # Aggressively close the live response as soon as the
                # user cancels so the call returns within ~50 ms instead
                # of waiting for the next SSE chunk.
                while not cancel_watcher_stop.is_set():
                    if cancel_check is not None and cancel_check():
                        try:
                            if resp is not None:
                                resp.close()
                        except Exception:
                            pass
                        try:
                            # Also yank the underlying socket for good measure.
                            if resp is not None and hasattr(resp, "raw"):
                                raw = resp.raw
                                if raw is not None:
                                    try:
                                        raw.close()
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        return
                    if cancel_watcher_stop.wait(0.05):
                        return

            cancel_thread = None
            if cancel_check is not None:
                cancel_thread = threading.Thread(target=_watch_cancel, daemon=True)
                cancel_thread.start()
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=(15.0, current_timeout),
                    stream=bool(body.get("stream")),
                )
                # Force UTF-8 decoding regardless of what the server
                # advertises.  llama.cpp / Ollama / vLLM all return UTF-8
                # JSON, but ``requests`` falls back to Latin-1 when the
                # ``Content-Type`` header omits charset — that turns
                # Korean (multi-byte UTF-8) into mojibake like ``ì¤`` (the
                # bytes 0xEC 0xA4 0x91 of "중" decoded as Latin-1).
                resp.encoding = "utf-8"
                if resp.status_code >= 400:
                    text_preview = ""
                    try:
                        text_preview = resp.text[:300]
                    except Exception:
                        pass
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise LLMError(
                            f"openai-compat transient http {resp.status_code}: {text_preview[:200]}"
                        )
                    # Some self-hosted backends reject either the
                    # ``response_format`` knob or the ``stream`` knob — drop
                    # them once and retry.
                    dropped = False
                    if "response_format" in text_preview:
                        body.pop("response_format", None)
                        dropped = True
                    if "stream" in text_preview and body.get("stream"):
                        body.pop("stream", None)
                        dropped = True
                    if dropped and attempt == 0:
                        attempt += 1
                        continue
                    raise LLMError(
                        f"openai-compat http {resp.status_code}: {text_preview}"
                    )
                ttft_box: list = []
                finish_box: list = []
                if body.get("stream"):
                    text = _consume_openai_stream(
                        resp, cancel_check, stream_text,
                        start_ts=start_ts, ttft_box=ttft_box,
                        finish_box=finish_box,
                    )
                else:
                    data = resp.json()
                    text = _extract_openai_text(data)
                    try:
                        fr = data.get("choices", [{}])[0].get("finish_reason")
                        if fr:
                            finish_box.append(fr)
                    except Exception:
                        pass
                # Truncated by the model's output cap → the JSON is almost
                # certainly unterminated.  Bumping max_tokens once and
                # retrying is the right move; if even that fails we surface
                # a clean error instead of feeding a half-string into
                # json.loads.
                if finish_box and finish_box[0] == "length":
                    if attempt < self.max_retries and body.get("max_tokens", 0) < 32768:
                        log.warning(
                            "openai-compat response truncated (finish_reason=length, %d chars). "
                            "raising max_tokens and retrying…",
                            len(text),
                        )
                        body["max_tokens"] = max(16384, body.get("max_tokens", 8192) * 2)
                        attempt += 1
                        continue
                    raise LLMError(
                        "model output was truncated (finish_reason=length); "
                        "the JSON is incomplete"
                    )
                duration = time.monotonic() - start_ts
                self.request_count += 1
                self.response_chars += len(text)
                self.total_duration_s += duration
                from ..models import LLMCall
                ttft = ttft_box[0] if ttft_box else 0.0
                self.calls.append(
                    LLMCall(
                        label="openai-compat",
                        prompt_chars=len(prompt),
                        response_chars=len(text),
                        duration_s=duration,
                        ttft_s=ttft,
                        success=True,
                    )
                )
                tps = (len(text) / 3) / duration if duration > 0 else 0.0
                log.info(
                    "openai-compat call ok in %.2fs (ttft %.2fs) — "
                    "prompt=%d chars, response=%d chars (~%.1f tok/s)",
                    duration, ttft, len(prompt), len(text), tps,
                )
                # Repair Latin-1 ↔ UTF-8 mojibake if it happened upstream.
                text = _try_repair_mojibake(text)
                # If the response is *still* full of mojibake, the model
                # itself produced garbage (e.g. a Q-quant template
                # mismatch).  Fail loudly instead of writing nonsense to
                # disk.
                if _looks_like_mojibake(text):
                    raise LLMError(
                        "model output appears to be mojibake (likely an "
                        "encoding or chat-template mismatch in the local "
                        "server).  Try a different quant or check the "
                        "server's chat template."
                    )
                cleaned = _strip_code_fence(text)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    # Best-effort recovery: strip trailing junk (truncated
                    # mid-string) and try once more before raising.
                    try:
                        return json.loads(_recover_truncated_json(cleaned))
                    except json.JSONDecodeError:
                        return json.loads(_strip_code_fence(text))
            except (requests.RequestException, LLMError, json.JSONDecodeError) as exc:
                last_exc = exc
                duration = time.monotonic() - start_ts
                from ..models import LLMCall
                self.calls.append(
                    LLMCall(
                        label="openai-compat",
                        prompt_chars=len(prompt),
                        duration_s=duration,
                        success=False,
                        error=str(exc)[:200],
                    )
                )
                log.warning(
                    "openai-compat call failed in %.2fs (attempt %d/%d, timeout=%.0fs): %s",
                    duration, attempt + 1, self.max_retries + 1, current_timeout, exc,
                )
                # Don't keep retrying after an explicit user cancel.
                if isinstance(exc, LLMError) and "canceled" in str(exc):
                    raise
                attempt += 1
                if attempt <= self.max_retries:
                    time.sleep(min(20.0, 1.5 * (2 ** attempt)))
                continue
            finally:
                stop_evt.set()
                cancel_watcher_stop.set()
                if beat_thread is not None:
                    beat_thread.join(timeout=0.1)
                if cancel_thread is not None:
                    cancel_thread.join(timeout=0.1)
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            # If we got here via a user cancel, raise immediately.
            if cancel_check is not None and cancel_check():
                raise LLMError("canceled by user")
        raise LLMError(f"openai-compat failed after retries: {last_exc}")


def _consume_openai_stream(
    resp: "requests.Response",
    cancel_check: Optional[Callable[[], bool]],
    stream_text: Optional[Callable[[str, int], None]] = None,
    *,
    start_ts: Optional[float] = None,
    ttft_box: Optional[list] = None,
    finish_box: Optional[list] = None,
) -> str:
    """Read an OpenAI SSE stream and return the joined text content.

    Each line is ``data: { "choices": [...] }``.  We stop on the special
    ``data: [DONE]`` marker, on ``finish_reason`` non-null, on user
    cancel, or on connection close.

    When ``stream_text`` is given it is called with
    ``(latest_chunk, total_chars_so_far)`` for every text delta — the UI
    layer uses this to display the partial response live so the user
    sees that the model is actively generating.
    """
    out: list[str] = []
    last_emit = 0.0
    total = 0
    # Use bytes mode so we control the decoding — never trust the server's
    # advertised charset (often missing or wrong for self-hosted backends).
    for raw_bytes in resp.iter_lines(decode_unicode=False):
        if cancel_check is not None and cancel_check():
            try:
                resp.close()
            except Exception:
                pass
            raise LLMError("canceled by user")
        if not raw_bytes:
            continue
        try:
            raw_line = raw_bytes.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            try:
                raw_line = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
            except Exception:
                continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        try:
            choice = obj["choices"][0]
        except (KeyError, IndexError, TypeError):
            continue
        delta = choice.get("delta") or {}
        chunk = delta.get("content")
        chunk_text = ""
        if isinstance(chunk, list):
            for part in chunk:
                if isinstance(part, dict):
                    chunk_text += part.get("text", "")
        elif isinstance(chunk, str):
            chunk_text = chunk
        if chunk_text:
            if ttft_box is not None and not ttft_box:
                ttft_box.append(time.monotonic() - (start_ts or time.monotonic()))
            out.append(chunk_text)
            total += len(chunk_text)
            if stream_text is not None:
                # Throttle UI updates to ~5 Hz so the log doesn't churn.
                now = time.monotonic()
                if now - last_emit > 0.2:
                    try:
                        stream_text(chunk_text, total)
                    except Exception:
                        pass
                    last_emit = now
        if choice.get("finish_reason"):
            if finish_box is not None:
                finish_box.append(choice.get("finish_reason"))
            # Final flush so the user sees the complete count.
            if stream_text is not None and chunk_text:
                try:
                    stream_text("", total)
                except Exception:
                    pass
            break
    return "".join(out)


_MOJIBAKE_HINTS = ("ì", "ë", "Ã", "â\x80", "â\x82", "ï¿½", "ê", "í", "ï")


def _looks_like_mojibake(text: str, *, strict: bool = False) -> bool:
    """Heuristic: did UTF-8 get decoded as Latin-1 somewhere upstream?

    Two threshold modes:
      * default — applied to long bodies (full response).  Requires 3+
        markers AND density ≥ 1% of total length, so a short proper-name
        with a couple of high-Latin chars doesn't false-trigger.
      * strict — applied to short fields (e.g. a single category name).
        Lower bar so a 30-char folder name with 4 mojibake markers
        trips even though the same density vs a 2 000-char body would
        not.  Required when scanning per-field, not whole-document.
    """
    if not text:
        return False
    hits = sum(text.count(t) for t in _MOJIBAKE_HINTS)
    if strict:
        # Short fields: any 3 markers within 60 chars, OR ≥ 25 % density,
        # is enough.
        if hits >= 3:
            return True
        return len(text) > 0 and (hits / max(1, len(text))) >= 0.25
    if hits < 3:
        return False
    return hits >= max(3, len(text) * 0.01)


def _try_repair_mojibake(text: str, *, strict: bool = False) -> str:
    """Best-effort: re-encode as Latin-1, decode as UTF-8.  No-op on failure.

    With ``strict=True`` we additionally accept a relaxed UTF-8 decode
    (errors="replace") for short fields where strict round-trip fails
    on a single bad continuation byte but most of the field is
    salvageable.
    """
    if not _looks_like_mojibake(text, strict=strict):
        return text
    try:
        repaired = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
        if not _looks_like_mojibake(repaired, strict=strict):
            log.info("repaired latin-1/UTF-8 mojibake (strict round-trip)")
            return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    if strict:
        # Last resort for short fields: tolerate a couple of replacement
        # chars instead of throwing the whole name away.
        try:
            relaxed = text.encode("latin-1", errors="replace").decode(
                "utf-8", errors="replace"
            )
            # Accept only if the relaxed repair actually reduced markers.
            if (
                not _looks_like_mojibake(relaxed, strict=strict)
                and "�" not in relaxed
            ):
                log.info("repaired latin-1/UTF-8 mojibake (relaxed)")
                return relaxed
        except Exception:  # pragma: no cover
            pass
    return text


def _recover_truncated_json(text: str) -> str:
    """Best-effort fixup for JSON that was cut off mid-stream.

    Handles the common case: the model ran out of ``max_tokens`` while
    inside a string in an array, leaving us with something like::

        {"categories":[{"id":"a","name":"한�

    We:
      * drop a trailing replacement / BOM char,
      * close any open string literal,
      * pop the deepest open ``[`` / ``{`` until the parser is happy.

    Imperfect, but recovers a usable plan from a truncated response.
    """
    s = text.rstrip().rstrip("�﻿�")
    # If we ended inside a string (odd number of unescaped quotes), close it.
    in_str = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
    if in_str:
        s = s + '"'
    # Drop a trailing comma that we might be left with after string close.
    s = s.rstrip()
    if s.endswith(","):
        s = s[:-1]
    # Balance brackets / braces.
    opens = []
    in_str = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "[{":
            opens.append(ch)
        elif ch in "]}":
            if opens and {"[": "]", "{": "}"}[opens[-1]] == ch:
                opens.pop()
    closer = {"[": "]", "{": "}"}
    s += "".join(closer[o] for o in reversed(opens))
    return s


def _extract_openai_text(response: dict) -> str:
    try:
        choices = response["choices"]
        msg = choices[0]["message"]
        content = msg.get("content", "")
        if isinstance(content, list):
            # Some servers return an array of {"type": "text", "text": "..."}
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected openai-compat response shape: {exc}") from exc


def resolve_api_key() -> Optional[str]:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or None
    )
