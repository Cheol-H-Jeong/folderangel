"""Archive parser: lists member names without extracting bodies."""
from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from folder1004.parsers import archive, registry


def test_zip_lists_member_names(tmp_path):
    z = tmp_path / "RTX_GPU_3대_구매계약_부속.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("세금계산서_2026_01.pdf", "ignored body")
        zf.writestr("발주서_v2.docx", "ignored body 2")
        zf.writestr("supporting/제품사양서.pdf", "ignored")
    out = archive.parse(z, max_chars=1000)
    assert "세금계산서_2026_01.pdf" in out
    assert "발주서_v2.docx" in out
    assert "supporting/제품사양서.pdf" in out
    # No actual body text from members leaks through.
    assert "ignored body" not in out


def test_tar_lists_member_names(tmp_path):
    t = tmp_path / "회의록_묶음.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        # Add real entries from a tiny in-memory blob.
        for n in ["회의록_2025-01.md", "회의록_2025-02.md"]:
            data = b"ignored"
            info = tarfile.TarInfo(n)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    out = archive.parse(t, max_chars=1000)
    assert "회의록_2025-01.md" in out
    assert "회의록_2025-02.md" in out


def test_registry_dispatches_zip_to_archive(tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("inside.pdf", "x")
    out = registry.extract_excerpt(z, max_chars=200, timeout=2.0)
    assert "inside.pdf" in out


def test_zip_extension_is_supported():
    assert ".zip" in registry.SUPPORTED_EXTENSIONS
    assert ".tgz" in registry.SUPPORTED_EXTENSIONS


def test_corrupt_archive_returns_empty(tmp_path):
    bad = tmp_path / "corrupt.zip"
    bad.write_bytes(b"not actually a zip file")
    out = archive.parse(bad, max_chars=1000)
    assert out == ""


def test_truncates_at_max_chars(tmp_path):
    z = tmp_path / "many.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for i in range(500):
            zf.writestr(f"file_{i:03}_긴_한글_파일명_for_truncation.pdf", "x")
    out = archive.parse(z, max_chars=300)
    assert len(out) <= 320  # ~300 + ellipsis tolerance
    assert "…" in out or "더" not in out
