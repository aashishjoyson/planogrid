# Standard Unix Makefile — on Windows, run via Git Bash / WSL (or run the
# underlying commands directly; every target is a single pip/ruff/pytest
# invocation, see each line below).

.PHONY: install install-gpu install-ocr install-dev lint format typecheck test test-cov download-sample download-all clean

install:
	pip install -e .
	pip install -r requirements/base.txt

install-gpu:
	pip install -r requirements/gpu-cu130.txt

install-ocr:
	pip install -r requirements/ocr.txt

install-dev:
	pip install -r requirements/dev.txt
	pre-commit install

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

typecheck:
	mypy core datasets scripts

test:
	pytest

test-cov:
	pytest --cov --cov-report=term-missing --cov-report=html

download-sample:
	python scripts/download_datasets.py --datasets sample

download-all:
	python scripts/download_datasets.py --all

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
