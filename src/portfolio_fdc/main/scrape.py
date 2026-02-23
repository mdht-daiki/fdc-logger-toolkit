from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd

from portfolio_fdc.core.config import load_yaml
from portfolio_fdc.core.sensor_map import SensorMap

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
SENSOR_MAP_CSV_PATH = Path(__file__).resolve().parents[1] / "configs" / "sensor_map.csv"

# ------------------------------
# utilities
# ------------------------------


# DATA行の位置を探す
def find_data_header_line_no(path: Path, max_scan_lines: int = 2000) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip() == "DATA":
                return i
            if i >= max_scan_lines - 1:
                break
    raise RuntimeError("DATA marker not found within scan limit")


# 上から読む
def read_from_top(path: Path, data_line_no: int) -> pd.DataFrame:
    # DATA行の次が列ヘッダなので skiprows = data_line_no + 1
    df = pd.read_csv(path, skiprows=data_line_no + 1)
    return df


# 末尾から読む
def tail_lines(path: Path, n_lines: int) -> list[str]:
    # returns last n_lines lines as text (without header)
    # NOTE: this assumes '\n' newline
    chunk_size = 1024 * 1024
    data = b""
    with path.open("rb") as f:
        f.seek(0, 2)
        end = f.tell()
        pos = end
        lines = 0
        while pos > 0 and lines <= n_lines:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            buf = f.read(read_size)
            data = buf + data
            lines = data.count(b"\n")
        tail = data.splitlines()[-n_lines:]
    return [x.decode("utf-8", errors="ignore") for x in tail]


def read_columns_after_data(path: Path, data_line_no: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(data_line_no + 1):
            next(f)
        header = next(f).strip()
    return next(csv.reader(StringIO(header)))


def read_recent_tail_as_df(path: Path, columns: list[str], n_lines: int) -> pd.DataFrame:
    lines = tail_lines(path, n_lines)
    # drop empty and non-data lines
    rows = [ln for ln in lines if ln.strip() and ("," in ln)]

    csv_text = ",".join(columns) + "\n" + "\n".join(rows) + "\n"
    df = pd.read_csv(StringIO(csv_text))
    return df


# 前回以降30分だけ取り出す
def filter_by_time_window(df: pd.DataFrame, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    return df[(df["timestamp"] > start_ts) & (df["timestamp"] <= end_ts)].sort_values("timestamp")


# tool_id / chamber_id付与・列名置換
def apply_tool_mapping(
    df: pd.DataFrame, tool_id: str, chamber_id: str, channel_map: dict
) -> pd.DataFrame:
    out = df.copy()
    out["tool_id"] = tool_id
    out["chamber_id"] = chamber_id
    # rename value01 -> DC Bias etc
    safe_map = {k: v for k, v in channel_map.items() if k != "timestamp"}
    out = out.rename(columns=safe_map)
    return out


# ------------------------------
# main logic
# ------------------------------


def load_tool_channel_map(path: Path) -> dict:
    doc = load_yaml(path)
    return doc.get("tools", {})


def resolve_channel_map(tool_id: str, tool_cfg: dict) -> dict[str, str]:
    channels = tool_cfg.get("channels")
    if channels is not None:
        return channels

    sensor_map = SensorMap.from_csv(SENSOR_MAP_CSV_PATH.as_posix())
    channel_map = {
        sensor: parameter
        for (mapped_tool_id, sensor), parameter in sensor_map.mapping.items()
        if mapped_tool_id == tool_id
    }

    if not channel_map:
        raise ValueError(f"channel mapping not found for tool_id={tool_id}")
    return channel_map


def load_last_ts(tool_id: str) -> datetime | None:
    p = STATE_DIR / f"last_ts_{tool_id}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return datetime.fromisoformat(d["last_ts"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_last_ts(tool_id: str, last_ts: datetime) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = STATE_DIR / f"last_ts_{tool_id}.json"
    p.write_text(json.dumps({"last_ts": last_ts.isoformat()}, ensure_ascii=False), encoding="utf-8")


def scrape_logger_csv(
    raw_csv_path: Path,
    tool_id: str,
    tool_cfg: dict,
    now: datetime,
    lookback_minutes: int = 30,
    huge_threshold_mb: int = 500,
) -> pd.DataFrame:
    data_line_no = find_data_header_line_no(raw_csv_path)
    columns = read_columns_after_data(raw_csv_path, data_line_no)
    channel_map = resolve_channel_map(tool_id, tool_cfg)
    chamber_id = tool_cfg.get("chamber_id", "CH1")

    last = load_last_ts(tool_id)
    if last is None:
        start_ts = now.replace(tzinfo=None) - timedelta(minutes=lookback_minutes)
    else:
        start_ts = last.replace(tzinfo=None)
    end_ts = now.replace(tzinfo=None)

    size_mb = raw_csv_path.stat().st_size / (1024 * 1024)

    if size_mb < huge_threshold_mb:
        df = read_from_top(raw_csv_path, data_line_no)
    else:
        # grab tail lines enough to cover 30 minutes (1800 rows) + margin

        # Assume a conservative upper bound of ~10 rows/s
        # multiply by lookback seconds + 2x margin

        ROWS_PER_SECOND_ESTIMATE = 10
        tail_n = lookback_minutes * 60 * ROWS_PER_SECOND_ESTIMATE * 2
        df = read_recent_tail_as_df(raw_csv_path, columns, n_lines=tail_n)

    df = filter_by_time_window(df, start_ts, end_ts)

    # add tool/chamber + rename channels
    df = apply_tool_mapping(df, tool_id, chamber_id, channel_map)

    # update state (save newest timestamp if exists)
    if not df.empty:
        latest_ts = pd.Timestamp(df["timestamp"].max()).to_pydatetime()
        save_last_ts(tool_id, latest_ts)
    return df
