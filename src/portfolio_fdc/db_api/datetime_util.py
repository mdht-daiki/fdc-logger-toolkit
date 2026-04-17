"""Shared datetime normalization utilities."""

from datetime import UTC, datetime


def to_utc_millis(raw: str) -> str:
    """任意の ISO 8601 文字列を UTC ミリ秒固定の文字列へ正規化する。"""
    normalized = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
