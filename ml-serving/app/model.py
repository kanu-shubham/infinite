"""
Model loading and inference.

Two entry points are intentionally separate:
  predict_single(features)  — used by the synchronous and cached endpoints
  predict_batch(features[]) — used by the explicit-batch and micro-batcher endpoints

Keeping them separate makes it easy to swap the model (TorchServe, ONNX,
Triton) without touching the HTTP layer.

Thread safety
─────────────
scikit-learn's predict/predict_proba methods are safe to call from multiple
threads because the underlying numpy operations release Python's GIL.  We
rely on this when running inference inside a ThreadPoolExecutor via
asyncio.run_in_executor().
"""

import pickle
import time
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "model.pkl"


class HotelRanker:
    def __init__(self) -> None:
        self._model        = None
        self._scaler       = None
        self.model_version = "not-loaded"

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        with open(MODEL_PATH, "rb") as fh:
            artefacts = pickle.load(fh)
        self._model        = artefacts["model"]
        self._scaler       = artefacts["scaler"]
        self.model_version = artefacts["version"]
        print(f"[model] loaded {self.model_version}  "
              f"(train AUC {artefacts['train_auc']:.4f})")

    # ── Feature engineering ──────────────────────────────────────────────────

    def _to_matrix(self, hotels: list[dict]) -> np.ndarray:
        """Convert a list of feature dicts → scaled 2-D numpy array."""
        raw = np.array(
            [
                [
                    h["price"],
                    h["rating"],
                    h["review_count"],
                    h["amenity_count"],
                    h["location_score"],
                ]
                for h in hotels
            ],
            dtype=np.float64,
        )
        return self._scaler.transform(raw)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_single(self, hotel: dict) -> dict:
        """
        Inference for a single hotel.

        Overhead breakdown (approximate, depends on hardware):
          featurize  ~0.05 ms
          transform  ~0.10 ms   (StandardScaler — numpy ops)
          GBT infer  ~1–4  ms   (100 trees, depth 4)
          total      ~1–5  ms

        This is the baseline.  Everything else (batching, caching) is
        measured against this.
        """
        t0  = time.perf_counter()
        X   = self._to_matrix([hotel])
        prob = float(self._model.predict_proba(X)[0, 1])
        model_ms = (time.perf_counter() - t0) * 1_000

        return {
            "click_probability": prob,
            "rank_score":        prob,
            "model_version":     self.model_version,
            "model_latency_ms":  round(model_ms, 3),
        }

    def predict_batch(self, hotels: list[dict]) -> dict:
        """
        Inference for N hotels in a single model call.

        Why batching helps throughput
        ──────────────────────────────
        numpy / sklearn amortises fixed overhead (function dispatch, memory
        allocation, BLAS setup) across the whole batch.  A batch of 32 hotels
        typically takes only 2-3× the time of 1 hotel — not 32×.  So batching
        32 requests that each waited ≤10 ms gives ~10× the throughput of
        serving them individually.

        Why batching hurts individual latency
        ──────────────────────────────────────
        The first request in a batch must wait up to max_wait_ms for the
        batch window to fill before inference begins.  If the service is lightly
        loaded, most batches will be size 1 anyway (window expires before
        more arrive), so the penalty is small.  Under heavy load, the batch
        fills instantly, so the wait is negligible.
        """
        t0    = time.perf_counter()
        X     = self._to_matrix(hotels)
        probs = self._model.predict_proba(X)[:, 1]
        model_ms = (time.perf_counter() - t0) * 1_000

        return {
            "predictions": [
                {"click_probability": float(p), "rank_score": float(p)}
                for p in probs
            ],
            "batch_size":      len(hotels),
            "model_version":   self.model_version,
            "model_latency_ms": round(model_ms, 3),
        }
