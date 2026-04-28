"""Markdown report generation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from .models import OperationResult


def emit_markdown(op: OperationResult, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else op.target_root
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = op.finished_at.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"FolderAngel_Report_{stamp}.md"
    path.write_text(_build(op), encoding="utf-8")
    return path


def _build(op: OperationResult) -> str:
    lines: list[str] = []
    lines.append(f"# FolderAngel Report — {op.target_root}")
    lines.append("")
    lines.append(f"- 실행 시각: {op.started_at.isoformat(timespec='seconds')} → {op.finished_at.isoformat(timespec='seconds')}")
    dur = (op.finished_at - op.started_at).total_seconds()
    lines.append(f"- 소요: {dur:.1f} 초")
    lines.append(f"- 모드: {'Dry-Run (변경 없음)' if op.dry_run else '실행'}")
    lines.append(f"- 스캔: {op.total_scanned}개 / 이동: {op.total_moved}개 / 바로가기: {op.total_shortcuts}개 / 스킵: {op.total_skipped}개")
    if getattr(op, "dupes_removed", None):
        mb = (op.bytes_freed or 0) / (1 << 20)
        lines.append(
            f"- 중복 삭제: {len(op.dupes_removed)}개 파일 / ≈ {mb:.1f} MB 회수"
        )
    if op.operation_id is not None:
        lines.append(f"- 오퍼레이션 ID: {op.operation_id}")
    if op.llm_usage is not None:
        u = op.llm_usage
        if u.model == "mock" or u.request_count == 0:
            lines.append("- LLM 사용: 0 호출 (Mock 휴리스틱) — 비용 없음")
        else:
            usd = u.estimate_cost_usd()
            krw = u.estimate_cost_krw()
            lines.append(
                f"- LLM 사용: {u.request_count}회 호출 ({u.model}) — "
                f"입력 ≈ {u.estimated_prompt_tokens:,} 토큰 / 출력 ≈ {u.estimated_response_tokens:,} 토큰"
            )
            tps = u.avg_tokens_per_second()
            ttft = u.avg_ttft_s()
            lines.append(
                f"- LLM 응답 성능: 총 {u.total_duration_s:.1f}초 / "
                f"평균 처리량 ≈ {tps:.1f} tok/s"
                + (f" · 평균 TTFT {ttft:.2f}s" if ttft > 0 else "")
            )
            lines.append(
                f"- LLM 예상 비용(추정): ≈ ${usd:.5f} USD (≈ ₩{krw:,.1f}) "
                f"— 공개 단가 기반의 대략 평균치이며 실제 청구액은 다를 수 있습니다."
            )
    lines.append("")

    # Per-call breakdown so the user can see *which* call was slow.
    if op.llm_usage is not None and op.llm_usage.calls:
        lines.append("## LLM 호출 상세")
        lines.append("")
        lines.append("| # | 결과 | 소요 | TTFT | 입력(자) | 출력(자) | 처리량 | 비고 |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for i, c in enumerate(op.llm_usage.calls, 1):
            status = "✅" if c.success else "❌"
            ttft = f"{c.ttft_s:.2f}s" if c.ttft_s > 0 else "—"
            tps = f"{c.tokens_per_second:.1f} tok/s" if c.success and c.duration_s > 0 else "—"
            note = c.error if not c.success else c.label
            lines.append(
                f"| {i} | {status} | {c.duration_s:.2f}s | {ttft} | "
                f"{c.prompt_chars:,} | {c.response_chars:,} | {tps} | {note} |"
            )
        lines.append("")

    # Category distribution
    counter = Counter(m.category_id for m in op.moved)
    lines.append("## 카테고리 분포")
    lines.append("")
    lines.append("| 카테고리 | 폴더명 | 파일 수 |")
    lines.append("| --- | --- | ---: |")
    for cat in op.categories:
        n = counter.get(cat.id, 0)
        lines.append(f"| `{cat.id}` | {cat.name} | {n} |")
    lines.append("")

    lines.append("## 이동 목록")
    lines.append("")
    if op.moved:
        lines.append("| 카테고리 | 새 경로 | 원본 경로 | 사유 | 점수 |")
        lines.append("| --- | --- | --- | --- | ---: |")
        for mf in op.moved[:500]:
            lines.append(
                f"| {mf.category_id} | `{mf.new_path}` | `{mf.original_path}` | {mf.reason or '-'} | {mf.score:.2f} |"
            )
        if len(op.moved) > 500:
            lines.append(f"| … | (+{len(op.moved) - 500} more) | | | |")
    else:
        lines.append("_이동된 파일이 없습니다._")
    lines.append("")

    shortcut_lines = [mf for mf in op.moved if mf.shortcuts]
    if shortcut_lines:
        lines.append("## 바로가기")
        lines.append("")
        lines.append("| 대상 파일 | 바로가기 위치 |")
        lines.append("| --- | --- |")
        for mf in shortcut_lines[:500]:
            for sp in mf.shortcuts:
                lines.append(f"| `{mf.new_path}` | `{sp}` |")
        lines.append("")

    if op.skipped:
        lines.append("## 스킵된 파일")
        lines.append("")
        lines.append("| 경로 | 사유 |")
        lines.append("| --- | --- |")
        for sf in op.skipped[:500]:
            lines.append(f"| `{sf.path}` | {sf.reason} |")
        lines.append("")

    # Detailed dedup ledger.  Each row says "삭제된 파일 ↔ 남은 정본
    # ↔ 회수 용량" so the user can audit what disappeared and why.
    if getattr(op, "dupes_removed", None):
        total_mb = (op.bytes_freed or 0) / (1 << 20)
        lines.append("## 중복 삭제 내역")
        lines.append("")
        lines.append(
            f"_총 {len(op.dupes_removed)}개 파일 삭제 · "
            f"≈ {total_mb:.1f} MB 회수._  동일 내용 파일이 여러 곳에 있어, "
            "정본 1개만 분류 후 나머지를 삭제했습니다."
        )
        lines.append("")
        lines.append("| 삭제된 파일 | 남긴 정본 | 회수 용량 |")
        lines.append("| --- | --- | ---: |")
        for deleted, canonical, bytes_freed in op.dupes_removed[:500]:
            mb = bytes_freed / (1 << 20)
            size_label = (
                f"{mb:.1f} MB"
                if bytes_freed >= 1 << 20
                else f"{bytes_freed:,} B"
            )
            lines.append(f"| `{deleted}` | `{canonical}` | {size_label} |")
        if len(op.dupes_removed) > 500:
            lines.append(
                f"| … (+{len(op.dupes_removed) - 500} more) | | |"
            )
        lines.append("")

    lines.append(f"_Generated by FolderAngel at {datetime.now().astimezone().isoformat(timespec='seconds')}_")
    return "\n".join(lines)
