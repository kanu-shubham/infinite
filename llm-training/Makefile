SHELL := /bin/bash
PY ?= python
PIP ?= pip
ACCELERATE ?= accelerate launch

# Default config files
SFT_CONFIG ?= configs/sft/qwen_0_5b.yaml
DPO_CONFIG ?= configs/dpo/qwen_0_5b.yaml
DISTILL_CONFIG ?= configs/distill/qwen_7b_to_0_5b.yaml
ACCEL_CONFIG ?= accelerate_configs/single_gpu.yaml

.PHONY: help install install-dev lint format typecheck test test-fast \
        sft dpo distill merge-lora serve \
        docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  install         Install runtime deps"
	@echo "  install-dev     Install dev deps + package in editable mode"
	@echo "  lint            Run ruff lint"
	@echo "  format          Run ruff format"
	@echo "  typecheck       Run mypy"
	@echo "  test            Run full test suite"
	@echo "  test-fast       Run fast tests only (skip slow/gpu)"
	@echo "  sft             Run SFT training (CONFIG=$(SFT_CONFIG))"
	@echo "  dpo             Run DPO training (CONFIG=$(DPO_CONFIG))"
	@echo "  distill         Run distillation (CONFIG=$(DISTILL_CONFIG))"
	@echo "  merge-lora      Merge a LoRA adapter into its base model"
	@echo "  serve           Launch FastAPI inference server"
	@echo "  docker-build    Build training Docker image"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install || true

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

test:
	pytest

test-fast:
	pytest -m "not slow and not gpu"

sft:
	$(ACCELERATE) --config_file $(ACCEL_CONFIG) scripts/run_sft.py --config $(SFT_CONFIG)

dpo:
	$(ACCELERATE) --config_file $(ACCEL_CONFIG) scripts/run_dpo.py --config $(DPO_CONFIG)

distill:
	$(ACCELERATE) --config_file $(ACCEL_CONFIG) scripts/run_distill.py --config $(DISTILL_CONFIG)

merge-lora:
	$(PY) scripts/merge_lora.py --adapter $(ADAPTER) --base $(BASE) --output $(OUTPUT)

serve:
	$(PY) scripts/serve.py --model $(MODEL) --host 0.0.0.0 --port 8000

docker-build:
	docker build -f docker/Dockerfile -t llm-train:latest .

docker-run:
	docker run --gpus all --rm -it \
		-v $(PWD):/workspace \
		-v $(PWD)/hf_cache:/root/.cache/huggingface \
		--env-file .env \
		llm-train:latest bash

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
