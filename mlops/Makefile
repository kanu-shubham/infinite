.PHONY: install train evaluate serve test lint format docker-build docker-up docker-down clean

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements-dev.txt

# ── ML Pipeline ──────────────────────────────────────────────────────────────
train:
	PYTHONPATH=. python -m src.train

evaluate:
	PYTHONPATH=. python -m src.evaluate

# Full pipeline: train → evaluate (quality gate) → serve
pipeline: train evaluate serve

# ── API ───────────────────────────────────────────────────────────────────────
serve:
	PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=100
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t hotel-price-api:latest .

docker-up:
	docker-compose up -d api mlflow

docker-train:
	docker-compose --profile train up trainer

docker-down:
	docker-compose down

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache htmlcov .coverage
