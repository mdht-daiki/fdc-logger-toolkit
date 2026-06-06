from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import aggregate, scrape

DEFAULT_AGGREGATE_CONFIG = Path("src/portfolio_fdc/configs/aggregate_tools.yaml")


@dataclass(frozen=True)
class RunOnceSummary:
    scraped_rows: int
    scrape_output_csv: str


def run_once(
    *,
    tool_id: str,
    raw_csv_path: Path,
    db_api: str,
    now: datetime | None = None,
    lookback_minutes: int = 30,
    config_path: Path = DEFAULT_AGGREGATE_CONFIG,
    scrape_output_csv: Path | None = None,
    detail_out_dir: Path = Path("data/detail"),
    dry_run: bool = False,
) -> RunOnceSummary:
    tools_cfg = scrape.load_tool_channel_map(config_path)
    if tool_id not in tools_cfg:
        raise ValueError(f"tool_id={tool_id} is not defined in {config_path.as_posix()}")

    scrape_df = scrape.scrape_logger_csv(
        raw_csv_path=raw_csv_path,
        tool_id=tool_id,
        tool_cfg=tools_cfg[tool_id],
        now=now or datetime.now(),
        lookback_minutes=lookback_minutes,
    )

    out_csv = scrape_output_csv or Path("data/scrape") / f"scrape_{tool_id}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scrape_df.to_csv(out_csv, index=False)

    if scrape_df.empty:
        return RunOnceSummary(scraped_rows=0, scrape_output_csv=out_csv.as_posix())

    argv = [
        "aggregate.py",
        "--input",
        out_csv.as_posix(),
        "--config",
        config_path.as_posix(),
        "--detail-out",
        detail_out_dir.as_posix(),
        "--db-api",
        db_api,
    ]
    if dry_run:
        argv.append("--dry-run")

    aggregate.main(argv[1:])

    return RunOnceSummary(scraped_rows=len(scrape_df), scrape_output_csv=out_csv.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--db-api", default="http://localhost:8000")
    parser.add_argument("--lookback-minutes", type=int, default=30)
    parser.add_argument("--config", default=DEFAULT_AGGREGATE_CONFIG.as_posix())
    parser.add_argument("--scrape-out", default="")
    parser.add_argument("--detail-out", default="data/detail")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    summary = run_once(
        tool_id=args.tool,
        raw_csv_path=Path(args.raw),
        db_api=args.db_api,
        lookback_minutes=args.lookback_minutes,
        config_path=Path(args.config),
        scrape_output_csv=Path(args.scrape_out) if args.scrape_out else None,
        detail_out_dir=Path(args.detail_out),
        dry_run=args.dry_run,
    )
    print(
        "run_once completed "
        f"tool={args.tool} scraped_rows={summary.scraped_rows} "
        f"scrape_out={summary.scrape_output_csv}"
    )


if __name__ == "__main__":
    main()
