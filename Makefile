.PHONY: help venv install-dev pre-commit fmt lint type test test-fast test-slow check clean

PYTHON ?= python
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
PY := $(VENV_DIR)/bin/python
PRECOMMIT := $(VENV_DIR)/bin/pre-commit

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
	@echo "  make check         lint + type + test"
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

check: lint type test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info htmlcov .coverage
