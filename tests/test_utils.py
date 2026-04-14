from __future__ import annotations


def assert_validation_error_envelope(
    response_body: dict[str, object],
    *,
    expected_loc_fragment: str | None = None,
    expected_message_fragment: str | None = None,
) -> None:
    """共通 422 エラーフォーマットを検証する。"""
    assert response_body["ok"] is False
    error = response_body["error"]
    assert isinstance(error, dict)
    assert error["code"] == "VALIDATION_ERROR"
    assert error["message"] == "Validation error"
    details = error["details"]
    assert isinstance(details, dict)
    issues = details["issues"]
    assert isinstance(issues, list)
    assert issues

    for issue in issues:
        assert isinstance(issue, dict)
        assert "loc" in issue
        assert "msg" in issue
        loc = issue["loc"]
        msg = issue["msg"]
        assert isinstance(loc, (list, tuple))
        assert isinstance(msg, str)

    if expected_loc_fragment is not None:
        assert any(expected_loc_fragment in str(issue["loc"]) for issue in issues)
    if expected_message_fragment is not None:
        assert any(expected_message_fragment in issue["msg"] for issue in issues)
