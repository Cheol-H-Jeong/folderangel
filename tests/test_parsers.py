import io
import zipfile
from pathlib import Path

from folder1004.parsers import extract_excerpt


def test_plain_text(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("안녕하세요 Folder1004!", encoding="utf-8")
    excerpt = extract_excerpt(p)
    assert "Folder1004" in excerpt


def test_markdown(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# 제목\n본문입니다." * 3, encoding="utf-8")
    assert "본문" in extract_excerpt(p, max_chars=200)


def test_hwpx_zip_with_section(tmp_path):
    p = tmp_path / "doc.hwpx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("Contents/section0.xml", "<root><p>월간 보고서 초안</p></root>")
    excerpt = extract_excerpt(p, max_chars=500)
    assert "월간" in excerpt


def test_unsupported_returns_empty(tmp_path):
    p = tmp_path / "image.jpg"
    p.write_bytes(b"\xff\xd8\xff")
    assert extract_excerpt(p) == ""


def test_html_tags_stripped(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><script>var x=1;</script><body>안녕 <b>세상</b></body></html>", encoding="utf-8")
    out = extract_excerpt(p)
    assert "세상" in out
    assert "var x" not in out
