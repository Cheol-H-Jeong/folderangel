"""Unified entry point.

Usage:
  python -m folder1004                       # launch GUI
  python -m folder1004 --cli --path PATH [opts]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _run_cli(args) -> int:
    from .config import default_paths, load_config
    from .index import IndexDB
    from .pipeline import run
    from .runlog import current_log_path, start_session

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_path = start_session("cli")
    print(f"log → {log_path}")

    paths = default_paths()
    config = load_config(paths)
    if args.no_economy:
        config.economy_mode = False
    if args.provider and args.provider != config.llm_provider:
        # Switching providers: drop any saved base_url that belonged to the
        # previous one unless the user also passes --base-url explicitly.
        config.llm_provider = args.provider
        if not args.base_url:
            config.llm_base_url = ""
    if args.base_url:
        config.llm_base_url = args.base_url
    if args.model:
        config.model = args.model
    if args.reasoning:
        config.reasoning_mode = args.reasoning
    force_mock = bool(args.mock)
    db = IndexDB(paths.index_db)

    def _print(stage: str, pct: float):
        if pct < 0:
            sys.stdout.write(f"[ ····  ] {stage}\n")
        else:
            sys.stdout.write(f"[{pct*100:5.1f}%] {stage}\n")
        sys.stdout.flush()

    try:
        op = run(
            target_root=Path(args.path).expanduser().resolve(),
            config=config,
            recursive=args.recursive,
            dry_run=args.dry_run,
            index_db=db,
            progress=_print if not args.quiet else None,
            force_mock=force_mock,
        )
    finally:
        db.close()
    sys.stdout.write("\n")
    print(
        f"Done — scanned={op.total_scanned} moved={op.total_moved} "
        f"shortcuts={op.total_shortcuts} skipped={op.total_skipped} "
        f"categories={len(op.categories)}"
    )
    if op.llm_usage is not None:
        u = op.llm_usage
        if u.model == "mock" or u.request_count == 0:
            print("LLM   — 0 calls (mock)")
        else:
            print(
                f"LLM   — {u.request_count} call(s) on {u.model}; "
                f"~{u.estimated_prompt_tokens:,} prompt tokens, "
                f"~{u.estimated_response_tokens:,} response tokens"
            )
            tps = u.avg_tokens_per_second()
            ttft = u.avg_ttft_s()
            ttft_str = f", TTFT≈{ttft:.2f}s" if ttft > 0 else ""
            print(
                f"Speed — {u.total_duration_s:.1f}s total, ≈{tps:.1f} tok/s avg{ttft_str}"
            )
            for i, c in enumerate(u.calls, 1):
                ok = "✓" if c.success else "✗"
                print(
                    f"  call {i:>2}: {ok} {c.duration_s:6.2f}s "
                    f"prompt={c.prompt_chars:>5} resp={c.response_chars:>5} "
                    + (f"({c.tokens_per_second:.1f} tok/s)" if c.success else f"err: {c.error}")
                )
            print(
                f"Cost  — ≈ ${u.estimate_cost_usd():.5f} USD "
                f"(≈ ₩{u.estimate_cost_krw():,.2f}, public list prices)"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="folder1004")
    parser.add_argument("--cli", action="store_true", help="run headless without launching UI")
    parser.add_argument("--path", type=str, help="target folder for --cli mode")
    parser.add_argument("--recursive", action="store_true", help="include subfolders")
    parser.add_argument("--dry-run", action="store_true", help="plan without moving files")
    parser.add_argument("--mock", action="store_true", help="force mock planner")
    parser.add_argument(
        "--no-economy",
        action="store_true",
        help="disable single-call economy mode (use per-batch staging)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai_compat"],
        help="override LLM provider (gemini | openai_compat)",
    )
    parser.add_argument(
        "--base-url",
        help="override LLM endpoint base URL (e.g. http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--model",
        help="override model name (e.g. gpt-4o-mini, qwen2.5-72b-instruct)",
    )
    parser.add_argument(
        "--reasoning",
        choices=["off", "on", "auto"],
        help="thinking mode for Qwen3 / DeepSeek-R1 style models "
             "(default off — much faster for our JSON task)",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    if args.cli:
        if not args.path:
            parser.error("--cli requires --path")
        return _run_cli(args)

    # Default: launch GUI
    from .ui import launch

    return launch(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
