.PHONY: setup test lint fmt reqs

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

reqs:
	uv export --no-hashes --no-dev -o requirements.txt
