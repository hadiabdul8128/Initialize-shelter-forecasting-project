.PHONY: docker-build install serve train test lint

install:
	python -m pip install -e ".[dev]"

train:
	shelter-forecast train

serve:
	shelter-forecast serve

docker-build:
	docker build -t shelter-forecasting:local .

test:
	python -m pytest

lint:
	python -m ruff check .
