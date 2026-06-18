.PHONY: run dev install build-index test lint format clean

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

install:
	uv sync

build-index:
	python scripts/build_index.py

build-index-reset:
	python scripts/build_index.py --reset

test:
	pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/ scripts/

format:
	ruff format app/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
