"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import API_VERSION, router
from .config import CORS_ORIGINS, MODEL_CACHE_SIZE, RUNS_DIR
from .ml import model_cache, registry


@asynccontextmanager
async def lifespan(_: FastAPI):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    registry.load_from_disk()

    # Preload the most recent models so the first request after a deploy is not
    # the one that pays the cold-load cost. Without this you get a p99 spike
    # correlated with every release — and the request that eats it is usually a
    # health check or a canary.
    recent = [
        run["run_id"]
        for run in registry.list_summaries(status="succeeded", limit=MODEL_CACHE_SIZE)
    ]
    if recent:
        model_cache.warm(recent)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hotel Bookings ML Pipeline API",
        description=(
            "Trains and serves scikit-learn pipelines over a synthetic hotel "
            "bookings dataset. Runs execute in the background and expose "
            "per-stage progress for the React frontend."
        ),
        version=API_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> dict:
        return {"service": "hotel-bookings-ml", "version": API_VERSION, "docs": "/docs"}

    return app


app = create_app()
