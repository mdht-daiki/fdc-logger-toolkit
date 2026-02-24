# FDC Logger Toolkit (Portfolio Edition)

A portfolio-friendly Fault Detection & Classification (FDC) toolkit for manufacturing equipment data.

This project is a rebuilt version of an internal monitoring tool.
All sensitive information and proprietary logic are removed and replaced with:

- a synthetic logger data generator
- configurable mappings and rules
- a local SQLite-based pipeline

---

## What this project demonstrates

- Handling “messy” equipment data formats (large CSV logger streams)
- Incremental ingestion (run every 30 minutes)
- Process segmentation (edge-based / step-peak-based)
- Feature extraction (mean / max / min / std per step)
- SPC-style threshold monitoring (warn/crit)
- Interactive dashboard for:
  - selecting filters
  - plotting SPC charts
  - editing thresholds
  - drilling down into raw waveforms (click a point → show waveform, step-colored)
- Concurrency control with SQLite via a dedicated API process

---

## Components

This repository consists of four programs:

1. **main**

- `scrape`: read logger/device logs incrementally
- `aggregate`: segment processes, compute features, store results
- stores:
  - `ProcessInfo` (process metadata + raw detail CSV path)
  - `Parameters` (features)
  - `StepWindows` (step boundaries for visualization)
  - `ChartsV2` (thresholds via dashboard)

2. **dashboard** (Plotly Dash)

- filter conditions (tool/chamber/recipe/parameter/step/feature-type)
- SPC chart + threshold lines
- click a point → raw waveform viewer (step-colored)
- threshold editing UI → persisted into DB (ChartsV2)

3. **judge**

- fetch latest features and thresholds
- evaluate warn/crit
- alert (email)
- optional: equipment stop command interface (stub in portfolio edition)

4. **db_api** (FastAPI)

- SQLite read/write gateway with serialized write queue
- provides REST endpoints for main/dashboard/judge
- optional “Temp.db swap” strategy for read-during-write

---

## Architecture Overview

```text
Synthetic Logger CSV / Equipment Logs
        |
      scrape  (incremental read)
        |
    aggregate (segmentation + features + detail csv)
        |
      db_api   (SQLite gateway)
     /   |   \
dashboard judge  (future: exporter)
```

---

# Data flow (high level)

- Logger raw CSV:
  - huge 1-second sampling stream
  - contains only timestamp,value01,value02,...
  - generation may use an internal process-active mask for anomaly targeting, but it is not written to CSV
  - includes a header section and a DATA marker line

- scrape:
  - extracts only new rows since the previous run (~30 min)
  - does not load the entire huge CSV (tail reading supported)
  - adds tool_id / chamber_id
  - keeps wide format and renames channels using a mapping file

- aggregate:
  - segments process windows:
    - edge-based: detect “on” window(s) of a key channel
    - step-peak-based: detect multiple step windows and bundle into a single process
  - saves “detail” raw waveform as long CSV for dashboard drill-down
  - writes:
    - ProcessInfo, Parameters, StepWindows

---

# Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
```

---

# Quickstart (local)

## 1) Start db_api

```bash
python -m portfolio_fdc.db_api.app
# or uvicorn portfolio_fdc.db_api.app:app --host 0.0.0.0 --port 8000
```

## 2) Generate synthetic logger CSV

```bash
python -m portfolio_fdc.tools.generate_logger_csv --out data/raw/logger_raw.csv --seconds 86400 --scenario mix
```

## 3) Run main pipeline (scrape + aggregate)

```bash
python -m portfolio_fdc.main.run_once --tool TOOL_A --raw data/raw/logger_raw.csv --db-api http://localhost:8000
```

If DB API is not implemented yet, you can run aggregate in dry-run mode (local processing only, no POST):

```bash
python -m portfolio_fdc.main.aggregate \
  --input data/scrape/scrape_TOOL_A.csv \
  --config src/portfolio_fdc/configs/aggregate_tools.yaml \
  --detail-out data/detail \
  --dry-run
```

Makefile version (Mac/Linux):

```bash
make aggregate-dry-run
# optional override
make aggregate-dry-run AGG_INPUT=data/scrape/scrape_TOOL_B.csv AGG_DETAIL_OUT=data/detail_tmp
```

PowerShell task version (Windows):

```powershell
.\tasks.ps1 aggregate-dry-run
# optional override
.\tasks.ps1 aggregate-dry-run -AggInput data/scrape/scrape_TOOL_B.csv -AggDetailOut data/detail_tmp
```

## 4) Start dashboard

```bash
python -m portfolio_fdc.dashboard.app
# open http://localhost:8050
```

## 5) Run judge (manual)

```bash
python -m portfolio_fdc.judge.run_once --db-api http://localhost:8000
```

---

# Development Commands

```bash
make fmt
make lint
make type
make test
make all
```

---

# Quality Gate (CI)

All pull requests must pass:

- Ruff (lint + format)
- MyPy (type checking)
- Pytest
- (Optional) CodeRabbit review

---

# Disclaimer

This repository is a simplified portfolio version.
It does not include proprietary production logic, confidencial process conditions, or internal infrastructure details.
