"""Auto micro-batch policy: small prompt → single call, large → micro-batch."""
from folder1004.config import Config
from folder1004.planner import Planner


class FakeClient:
    def __init__(self, ctx: int):
        self._ctx = ctx

    def context_length(self, default: int = 8192) -> int:
        return self._ctx


def _payloads(n: int, excerpt_chars: int = 200):
    return [
        {
            "path": f"/p/f_{i}.md",
            "name": f"f_{i}.md",
            "ext": ".md",
            "modified": "2024-08",
            "excerpt": "x" * excerpt_chars,
        }
        for i in range(n)
    ]


def test_force_on_overrides_size():
    cfg = Config(); cfg.local_microbatch_mode = "on"
    p = Planner(cfg, gemini=FakeClient(ctx=1_000_000))
    assert p._should_use_microbatch(_payloads(1)) is True


def test_force_off_overrides_size():
    cfg = Config(); cfg.local_microbatch_mode = "off"
    p = Planner(cfg, gemini=FakeClient(ctx=4096))
    assert p._should_use_microbatch(_payloads(500)) is False


def test_auto_small_prompt_uses_single_call():
    """120 tiny files into a 1M-token model → no micro-batch."""
    cfg = Config(); cfg.local_microbatch_mode = "auto"
    p = Planner(cfg, gemini=FakeClient(ctx=1_000_000))
    assert p._should_use_microbatch(_payloads(120, excerpt_chars=50)) is False


def test_auto_large_prompt_falls_to_microbatch():
    """500 fat files into an 8k-context local model → micro-batch."""
    cfg = Config(); cfg.local_microbatch_mode = "auto"
    p = Planner(cfg, gemini=FakeClient(ctx=8192))
    assert p._should_use_microbatch(_payloads(500, excerpt_chars=600)) is True


def test_auto_medium_local_with_huge_ctx_skips_microbatch():
    """User's actual setup: Qwen3.6-35B-A3B with 256K ctx, 120 files →
    single call, same as Gemini path."""
    cfg = Config(); cfg.local_microbatch_mode = "auto"
    p = Planner(cfg, gemini=FakeClient(ctx=262144))
    assert p._should_use_microbatch(_payloads(120, excerpt_chars=200)) is False
