.PHONY: all help venv install-dev pre-commit fmt lint type test test-fast test-slow test-db-api-aggregate check aggregate-dry-run clean demo-db-api demo-data demo-dashboard

all: help

PYTHON ?= python
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
PY := $(VENV_DIR)/bin/python
PRECOMMIT := $(VENV_DIR)/bin/pre-commit
AGG_INPUT ?= data/scrape/scrape_TOOL_A.csv
AGG_CONFIG ?= src/portfolio_fdc/configs/aggregate_tools.yaml
AGG_DETAIL_OUT ?= data/detail
DEMO_RAW ?= data/raw/logger_raw_demo.csv

help:
	@echo "Targets:"
	@echo "  make venv          Create venv in $(VENV_DIR)"
	@echo "  make install-dev   Install editable + dev deps"
	@echo "  make pre-commit    Install git hooks"
	@echo "  make fmt           Ruff format (apply)"
	@echo "  make lint          Ruff check"
	@echo "  make type          Mypy src"
	@echo "  make test          Pytest"
	@echo "  make test-fast     Pytest except slow tests"
	@echo "  make test-slow     Pytest only slow tests"
	@echo "  make test-db-api-aggregate  Pytest for db_api + aggregate connection"
	@echo "  make check         lint + type + test"
	@echo "  make aggregate-dry-run  Run aggregate without DB POST"
	@echo "  make demo-db-api   Start db_api for portfolio demo"
	@echo "  make demo-data     Generate sample data and run one pipeline cycle"
	@echo "  make demo-dashboard Start dashboard (http://localhost:8050)"
	@echo "  make clean         Remove caches/build artifacts"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -U pip

install-dev: venv
	$(PIP) install -e ".[dev]"

pre-commit: install-dev
	$(PRECOMMIT) install

fmt: install-dev
	$(PY) -m ruff format .

lint: install-dev
	$(PY) -m ruff check .

type: install-dev
	$(PY) -m mypy src

test: install-dev
	$(PY) -m pytest

test-fast: install-dev
	$(PY) -m pytest -m "not slow"

test-slow: install-dev
	$(PY) -m pytest -m slow

test-db-api-aggregate: install-dev
	$(PY) -m pytest tests/test_db_api_integration.py tests/test_aggregate_db_api_integration.py

check: lint type test

aggregate-dry-run: install-dev
	$(PY) -m portfolio_fdc.main.aggregate --input $(AGG_INPUT) --config $(AGG_CONFIG) --detail-out $(AGG_DETAIL_OUT) --dry-run

demo-db-api: install-dev
	$(PY) -m portfolio_fdc.db_api.app

demo-data: install-dev
	$(PY) -m portfolio_fdc.tools.generate_logger_csv --out $(DEMO_RAW) --seconds 7200 --scenario mix
	$(PY) -m portfolio_fdc.main.run_once --tool TOOL_A --raw $(DEMO_RAW) --db-api http://localhost:8000

demo-dashboard: install-dev
	$(PY) -m portfolio_fdc.dashboard.app

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info htmlcov .coverage
