"""End-to-end smoke: data -> train -> AUC clearly above random."""
import tempfile
from pathlib import Path

from ml.feed_ranking.config import TrainConfig
from ml.feed_ranking.data import GenerationConfig, generate_dataset
from ml.feed_ranking.inference import Ranker
from ml.feed_ranking.train import train


def test_train_mlp_beats_random():
    store = generate_dataset(
        GenerationConfig(n_users=120, n_posts=400, n_impressions=8_000, seed=11)
    )
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "model.pt"
        cfg = TrainConfig(epochs=4, batch_size=256, lr=2e-3, model_kind="mlp")
        result = train(store, cfg, checkpoint_path=ckpt)
        assert result.val_auc > 0.6, f"AUC too low: {result.val_auc}"
        assert result.test_nce < 1.0, f"NCE not below prior: {result.test_nce}"

        # Loading the checkpoint and ranking should work end to end.
        ranker = Ranker.load(ckpt)
        any_user = store.all_users()[0].user_id
        ranked = ranker.rank(any_user, store=store, k=5)
        assert len(ranked) == 5
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)


def test_train_logreg_runs():
    store = generate_dataset(
        GenerationConfig(n_users=80, n_posts=200, n_impressions=4_000, seed=12)
    )
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "lr.pt"
        result = train(store, TrainConfig(epochs=2, model_kind="logreg"), checkpoint_path=ckpt)
        assert result.model_kind == "logreg"
        assert result.n_train > 0
