"""Secret-redaction guard for the per-run logger.

Any string we ship into the logger that contains a recognisable API
key, Bearer token, or ``?key=…`` query string MUST be replaced with
``[REDACTED]`` before it lands in the file.  These tests pin that
contract — they run against a fresh log file in a temp dir so they
never touch the user's real ``~/.folder1004/logs``.
"""
import logging
import os
import re
from pathlib import Path

from folder1004.runlog import _redact, start_session


def test_redact_google_api_key():
    s = "calling endpoint?key=AIzaSyBv67NPTBiQmtmQ8JY5zM6--9sH2KZDZl0&model=x"
    out = _redact(s)
    assert "AIzaSy" not in out
    assert "REDACTED" in out


def test_redact_bearer_header():
    s = "Authorization: Bearer 93bd56cb58fea8670971162df9ead5d34147bef473704d5a6c6ba0050258fbf2"
    out = _redact(s)
    assert "93bd56cb" not in out
    assert "Authorization: Bearer [REDACTED]" == out


def test_redact_openai_sk_token():
    s = "OPENAI_API_KEY=sk-abcdef0123456789ABCDEF0123456789"
    out = _redact(s)
    assert "sk-abcdef" not in out


def test_redact_long_hex_token():
    s = "x-api-key: 93bd56cb58fea8670971162df9ead5d34147bef473704d5a6c6ba0050258fbf2"
    out = _redact(s)
    assert "93bd56cb" not in out


def test_log_file_does_not_contain_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    log_path = start_session("pytest")
    log = logging.getLogger("folder1004.test")
    # The kind of line urllib3 / requests would produce verbatim:
    log.info("POST https://api.openai.com/v1/chat/completions key=sk-Reallyleak1234567890ABCDEF")
    log.info("Authorization: Bearer 93bd56cb58fea8670971162df9ead5d34147bef473704d5a6c6ba0050258fbf2 OK")
    log.info("Probing https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyBv67NPTBiQmtmQ8JY5zM6--9sH2KZDZl0")
    text = log_path.read_text(encoding="utf-8")
    for needle in (
        "AIzaSyBv67",
        "sk-Reallyleak",
        "93bd56cb58fea867",
    ):
        assert needle not in text, f"secret {needle!r} leaked into log file"
    assert "REDACTED" in text
