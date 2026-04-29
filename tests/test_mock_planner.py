from folder1004.llm import mock


def _mkfile(name, ext, excerpt=""):
    return {"path": f"/tmp/{name}", "name": name, "ext": ext, "excerpt": excerpt}


def test_mock_plan_groups_by_extension():
    files = [
        _mkfile("2024 월간보고서.pdf", ".pdf", "월간보고서 요약"),
        _mkfile("재무제표.xlsx", ".xlsx"),
        _mkfile("회의록.docx", ".docx"),
        _mkfile("연구논문.pdf", ".pdf", "이 논문은..."),
        _mkfile("영수증_1월.pdf", ".pdf", "Receipt invoice"),
        _mkfile("사진.jpg", ".jpg"),
        _mkfile("사진2.jpg", ".jpg"),
    ]
    out = mock.plan(files)
    assert out["categories"], "categories should not be empty"
    assert len(out["assignments"]) == len(files)
    cat_ids = {c["id"] for c in out["categories"]}
    for a in out["assignments"]:
        assert a["primary"] in cat_ids
