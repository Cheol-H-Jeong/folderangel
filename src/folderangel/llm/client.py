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


def make_llm_client(config, api_key: Optional[str]):
    """Build the appropriate LLM client from a :class:`Config`.

    Returns ``None`` when no key is available so the caller can fall back
    to the mock planner.
    """
    if not api_key:
        return None
    provider = (getattr(config, "llm_provider", "gemini") or "gemini").lower()
    base_url = (getattr(config, "llm_base_url", "") or "").strip()
    model = getattr(config, "model", "gemini-2.5-flash")
    if provider in ("gemini", "google", "google_ai", ""):
        client = GeminiClient(api_key=api_key, model=model)
        if base_url:
            client.base_url = base_url.rstrip("/")
        return client
    if provider in ("openai", "openai_compat", "openai-compatible", "compat"):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        return OpenAICompatClient(api_key=api_key, model=model, base_url=base_url)
    raise LLMError(f"unknown llm provider: {provider}")


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

    # ------------------------------------------------------------------
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        heartbeat: Optional[Callable[[float], None]] = None,
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
                self.request_count += 1
                self.response_chars += len(text)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Some models wrap JSON in code fences; try to strip them.
                    stripped = _strip_code_fence(text)
                    return json.loads(stripped)
            except (requests.RequestException, LLMError, json.JSONDecodeError) as exc:
                last_exc = exc
                log.warning(
                    "gemini call failed (attempt %d/%d, timeout=%.0fs): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    current_timeout,
                    exc,
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

    Works against:
      * OpenAI itself  (base_url ``https://api.openai.com/v1``)
      * Together / Groq / OpenRouter  (each one's base_url)
      * Ollama         (``http://localhost:11434/v1``)
      * vLLM / LM Studio / TGI exposing ``/v1/chat/completions``
      * Gemini via the OpenAI-compatible proxy
        (``https://generativelanguage.googleapis.com/v1beta/openai``)

    Same shape as :class:`GeminiClient` so the planner doesn't care which
    one is in use.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 180.0,
        max_retries: int = 3,
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
        self.request_count: int = 0
        self.prompt_chars: int = 0
        self.response_chars: int = 0

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        heartbeat: Optional[Callable[[float], None]] = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
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
        }

        self.prompt_chars += len(prompt)
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
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
                    headers=headers,
                    json=body,
                    timeout=(15.0, current_timeout),
                )
                if resp.status_code >= 400:
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise LLMError(
                            f"openai-compat transient http {resp.status_code}: {resp.text[:200]}"
                        )
                    # Retry once without response_format if the server
                    # rejects that field (some self-hosted backends don't
                    # implement it yet).
                    if (
                        attempt == 0
                        and resp.status_code == 400
                        and "response_format" in (resp.text or "")
                    ):
                        body.pop("response_format", None)
                        attempt += 1
                        continue
                    raise LLMError(
                        f"openai-compat http {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()
                text = _extract_openai_text(data)
                self.request_count += 1
                self.response_chars += len(text)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return json.loads(_strip_code_fence(text))
            except (requests.RequestException, LLMError, json.JSONDecodeError) as exc:
                last_exc = exc
                log.warning(
                    "openai-compat call failed (attempt %d/%d, timeout=%.0fs): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    current_timeout,
                    exc,
                )
                attempt += 1
                if attempt <= self.max_retries:
                    time.sleep(min(20.0, 1.5 * (2 ** attempt)))
                continue
            finally:
                stop_evt.set()
                if beat_thread is not None:
                    beat_thread.join(timeout=0.1)
        raise LLMError(f"openai-compat failed after retries: {last_exc}")


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
