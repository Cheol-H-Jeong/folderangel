"""Unified entry point.

Usage:
  python -m folderangel                       # launch GUI
  python -m folderangel --cli --path PATH [opts]
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    paths = default_paths()
    config = load_config(paths)
    if args.mock:
        force_mock = True
    else:
        force_mock = False
    db = IndexDB(paths.index_db)

    def _print(stage: str, pct: float):
        sys.stdout.write(f"\r[{stage:<10}] {pct*100:5.1f}% ")
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
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(prog="folderangel")
    parser.add_argument("--cli", action="store_true", help="run headless without launching UI")
    parser.add_argument("--path", type=str, help="target folder for --cli mode")
    parser.add_argument("--recursive", action="store_true", help="include subfolders")
    parser.add_argument("--dry-run", action="store_true", help="plan without moving files")
    parser.add_argument("--mock", action="store_true", help="force mock planner")
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
