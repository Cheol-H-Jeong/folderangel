"""Thin Gemini REST client.

We call the raw HTTPS endpoint so that we don't depend on the google-genai
package (its install surface is larger and pinned to recent Python).  The
function here is specifically for *structured* JSON generation: we request
``response_mime_type = application/json`` and parse the body.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class LLMError(RuntimeError):
    pass


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        if not api_key:
            raise LLMError("empty api key")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    def generate_json(self, prompt: str, temperature: float = 0.2) -> dict[str, Any]:
        url = _ENDPOINT.format(model=self.model)
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

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                resp = requests.post(
                    url, params=params, headers=headers, json=body, timeout=self.timeout
                )
                if resp.status_code >= 400:
                    raise LLMError(
                        f"gemini http {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()
                text = _extract_text(data)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Some models wrap JSON in code fences; try to strip them.
                    stripped = _strip_code_fence(text)
                    return json.loads(stripped)
            except (requests.RequestException, LLMError, json.JSONDecodeError) as exc:
                last_exc = exc
                log.warning("gemini call failed (attempt %d): %s", attempt + 1, exc)
                attempt += 1
                time.sleep(0.5 * (2 ** attempt))
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


def resolve_api_key() -> Optional[str]:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or None
    )
