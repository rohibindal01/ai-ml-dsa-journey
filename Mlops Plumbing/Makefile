.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON        := poetry run python
PYTEST        := poetry run pytest
RUFF          := poetry run ruff
MYPY          := poetry run mypy
MLFLOW_URI    ?= http://localhost:5000

.PHONY: help
help:                       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:                    ## Install all dependencies (incl. dev)
	poetry install --no-interaction

.PHONY: install-prod
install-prod:               ## Install production dependencies only
	poetry install --no-interaction --without dev

.PHONY: lint
lint:                       ## Run ruff linter
	$(RUFF) check src tests

.PHONY: format
format:                     ## Auto-fix ruff lint issues and format
	$(RUFF) check --fix src tests
	$(RUFF) format src tests

.PHONY: typecheck
typecheck:                  ## Run mypy type checks
	$(MYPY) src

.PHONY: check
check: lint typecheck       ## Run all static checks

.PHONY: test
test:                       ## Run full test suite with coverage
	$(PYTEST) tests/ \
	  --cov=src \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  --cov-fail-under=80 \
	  -v

.PHONY: test-fast
test-fast:                  ## Run tests without coverage (faster)
	$(PYTEST) tests/ -v

.PHONY: preprocess
preprocess:                 ## Run DVC pipeline preprocessing stage
	dvc repro preprocess

.PHONY: train
train:                      ## Train the model via the DVC pipeline
	dvc repro train

.PHONY: evaluate
evaluate:                   ## Run evaluation stage
	dvc repro evaluate

.PHONY: pipeline
pipeline:                   ## Run full DVC pipeline end-to-end
	dvc repro

.PHONY: serve
serve:                      ## Start the FastAPI server (dev mode)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) \
	$(PYTHON) -m uvicorn src.serving.api:app \
	  --host 0.0.0.0 \
	  --port 8000 \
	  --reload

.PHONY: serve-prod
serve-prod:                 ## Start the FastAPI server (production mode)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) \
	$(PYTHON) -m uvicorn src.serving.api:app \
	  --host 0.0.0.0 \
	  --port 8000 \
	  --workers 4

.PHONY: docker-build
docker-build:               ## Build the Docker image
	docker build \
	  --build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	  --build-arg GIT_SHA=$$(git rev-parse --short HEAD) \
	  -t ml-project-template:latest .

.PHONY: docker-up
docker-up:                  ## Start MLflow + API with docker-compose
	docker compose up --build -d

.PHONY: docker-down
docker-down:                ## Stop all docker-compose services
	docker compose down

.PHONY: docker-logs
docker-logs:                ## Tail docker-compose logs
	docker compose logs -f

.PHONY: mlflow-ui
mlflow-ui:                  ## Launch MLflow tracking UI
	mlflow ui --host 0.0.0.0 --port 5000

.PHONY: hooks
hooks:                      ## Install pre-commit hooks
	poetry run pre-commit install

.PHONY: hooks-run
hooks-run:                  ## Run all pre-commit hooks against all files
	poetry run pre-commit run --all-files

.PHONY: clean
clean:                      ## Remove build artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build
	rm -f confusion_matrix_*.png
