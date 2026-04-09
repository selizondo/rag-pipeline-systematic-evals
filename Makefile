.PHONY: bootstrap test test-one lint format eval explore

bootstrap:
	uv sync --all-extras

test:
	uv run pytest

test-one:
	uv run pytest $(TEST)

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

eval:
	uv run python -m src.main

explore:
	uv run python scripts/explore_results.py
