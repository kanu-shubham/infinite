.PHONY: help install install-dev lint fmt type test train merge serve docker-build docker-train docker-serve clean

PYTHON ?= python
CONFIG ?= configs/qlora_llama3_8b.yaml
ADAPTER ?= outputs/adapter
MERGED ?= outputs/merged

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	$(PYTHON) -m pip install -e .

install-dev: ## Install with dev deps
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Ruff lint
	ruff check src tests

fmt: ## Ruff format
	ruff format src tests
	ruff check --fix src tests

type: ## Mypy
	mypy src

test: ## Run tests
	pytest

train: ## Train with CONFIG=path/to.yaml
	lora-finetune train --config $(CONFIG)

eval: ## Evaluate adapter
	lora-finetune evaluate --config $(CONFIG) --adapter $(ADAPTER)

merge: ## Merge adapter into base model
	lora-finetune merge --adapter $(ADAPTER) --output $(MERGED)

serve: ## Serve model via FastAPI
	lora-finetune serve --model $(MERGED)

docker-build: ## Build training/serving images
	docker build -f docker/Dockerfile.train -t lora-finetune:train .
	docker build -f docker/Dockerfile.serve -t lora-finetune:serve .

docker-train: docker-build ## Run training container
	docker run --gpus all --rm -v $(PWD):/workspace --env-file .env lora-finetune:train train --config $(CONFIG)

docker-serve: docker-build ## Run serving container
	docker run --gpus all --rm -p 8000:8000 -v $(PWD):/workspace --env-file .env lora-finetune:serve

clean: ## Remove caches and artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
