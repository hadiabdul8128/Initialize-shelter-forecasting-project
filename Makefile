.PHONY: install train test lint

install:
	python -m pip install -e ".[dev]"

train:
	shelter-forecast train

test:
	python -m pytest

lint:
	python -m ruff check .
