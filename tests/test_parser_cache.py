from pathlib import Path

from folder1004.parser_cache import ParserCache


def test_cache_hit_skips_cold_parse(tmp_path):
    cache = ParserCache(tmp_path / "cache.db")
    f = tmp_path / "a.txt"
    f.write_text("hello")
    st = f.stat()

    calls = {"n": 0}
    def cold():
        calls["n"] += 1
        return "EXCERPT"

    out1 = cache.get_or_parse(f, st.st_mtime, st.st_size, cold)
    out2 = cache.get_or_parse(f, st.st_mtime, st.st_size, cold)
    assert out1 == out2 == "EXCERPT"
    assert calls["n"] == 1, "cache should have prevented the second parse"
    cache.close()


def test_cache_miss_on_modified_file(tmp_path):
    cache = ParserCache(tmp_path / "cache.db")
    f = tmp_path / "a.txt"
    f.write_text("v1")
    st = f.stat()
    cache.get_or_parse(f, st.st_mtime, st.st_size, lambda: "v1-text")

    # bump mtime via a fresh write
    import time
    time.sleep(0.01)
    f.write_text("v2-longer")
    st2 = f.stat()
    out = cache.get_or_parse(f, st2.st_mtime, st2.st_size, lambda: "v2-text")
    assert out == "v2-text"
    cache.close()


def test_cache_evicts_missing_paths(tmp_path):
    cache = ParserCache(tmp_path / "cache.db")
    a = tmp_path / "a.txt"
    a.write_text("x")
    cache.get_or_parse(a, a.stat().st_mtime, a.stat().st_size, lambda: "X")

    a.unlink()
    n = cache.evict_missing()
    assert n == 1
    cache.close()
