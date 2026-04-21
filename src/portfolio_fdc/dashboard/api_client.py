from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NoReturn
from urllib.parse import quote

import requests

DEFAULT_TIMEOUT_SEC = 8.0


def _path_segment(segment: str) -> str:
    """URLパスセグメント用にID等をエンコードする（/や?等を安全化）"""
    return quote(str(segment), safe="")


@dataclass(frozen=True)
class APIError(Exception):
    """db_api 呼び出し失敗時の表示向けエラー情報。"""

    message: str
    code: str | None = None
    status_code: int | None = None


def parse_utc_millis(timestamp: str | None) -> str:
    """UTC ISO 8601 文字列を見やすい UTC 表示へ変換する。"""
    if not timestamp:
        return "-"

    raw = timestamp
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(raw).astimezone(UTC)
    except ValueError:
        return timestamp

    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def parse_api_error(payload: dict[str, Any] | None, status_code: int) -> APIError:
    """error envelope を表示可能なエラーへ正規化する。"""
    if not isinstance(payload, dict):
        return APIError(
            message=f"API error (status={status_code})",
            status_code=status_code,
        )

    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if isinstance(message, str) and message:
            return APIError(
                message=message,
                code=code if isinstance(code, str) else None,
                status_code=status_code,
            )

    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return APIError(message=detail, status_code=status_code)

    return APIError(message=f"API error (status={status_code})", status_code=status_code)


def _request_envelope(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT_SEC)
    except requests.RequestException as exc:
        raise APIError(message=f"Network error: {exc}") from exc

    payload: dict[str, Any] | None = None
    try:
        payload = response.json()
    except ValueError:
        if response.status_code >= 400:
            raise APIError(
                message=f"API error (status={response.status_code})",
                status_code=response.status_code,
            ) from None

    if response.status_code >= 400:
        raise parse_api_error(payload, response.status_code)

    if not isinstance(payload, dict):
        raise APIError(message="Invalid response envelope")

    if payload.get("ok") is not True:
        raise parse_api_error(payload, response.status_code)

    return payload.get("data")


def _raise_invalid_shape(endpoint: str, expected: str, actual: Any) -> NoReturn:
    logger = logging.getLogger(__name__)
    logger.error(
        "Malformed API response for %s: expected %s, got %s (full payload: %r)",
        endpoint,
        expected,
        type(actual).__name__,
        actual,
    )
    raise APIError(
        message=(
            f"Malformed API response for {endpoint}: expected {expected}, "
            f"got {type(actual).__name__}"
        )
    )


def get_charts(base_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = _request_envelope(base_url, "/charts", params=params)
    if not isinstance(data, list):
        _raise_invalid_shape("/charts", "list", data)
    return data


def get_active_charts(base_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    data = _request_envelope(base_url, "/charts/active", params=params)
    if not isinstance(data, dict):
        _raise_invalid_shape("/charts/active", "dict", data)
    return data


def get_charts_history(base_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = _request_envelope(base_url, "/charts/history", params=params)
    if not isinstance(data, list):
        _raise_invalid_shape("/charts/history", "list", data)
    return data


def get_chart_points(
    base_url: str,
    chart_id: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    seg = _path_segment(chart_id)
    data = _request_envelope(base_url, f"/charts/{seg}/points", params=params)
    if not isinstance(data, list):
        _raise_invalid_shape(f"/charts/{seg}/points", "list", data)
    return data


def get_process_waveform_preview(
    base_url: str,
    process_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seg = _path_segment(process_id)
    data = _request_envelope(
        base_url,
        f"/processes/{seg}/waveform-preview",
        params=params,
    )
    if not isinstance(data, dict):
        _raise_invalid_shape(f"/processes/{seg}/waveform-preview", "dict", data)
    return data


def get_judge_results(base_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = _request_envelope(base_url, "/judge/results", params=params)
    if not isinstance(data, list):
        _raise_invalid_shape("/judge/results", "list", data)
    return data


def get_judge_result(base_url: str, result_id: str) -> dict[str, Any]:
    seg = _path_segment(result_id)
    data = _request_envelope(base_url, f"/judge/results/{seg}")
    if not isinstance(data, dict):
        _raise_invalid_shape(f"/judge/results/{seg}", "dict", data)
    return data
