import numpy as np

from ml.feed_ranking.config import FeatureSpec
from ml.feed_ranking.features import FeatureExtractor
from ml.feed_ranking.schema import Post, User
from ml.feed_ranking.store import Store


def _store_with_pair() -> tuple[Store, User, Post]:
    s = Store()
    u = User(user_id="u1", num_connections=10, num_followers=20,
             topic_affinities={"ml": 0.8, "engineering": 0.4})
    a = User(user_id="a1", num_connections=5, num_followers=100)
    s.upsert_user(u)
    s.upsert_user(a)
    p = Post(post_id="p1", author_id="a1", post_type="article", topic="ml",
             text_length=200, num_likes=10, num_comments=2, num_shares=1)
    s.upsert_post(p)
    return s, u, p


def test_feature_dim_matches_spec():
    spec = FeatureSpec()
    fx = FeatureExtractor(spec)
    s, u, p = _store_with_pair()
    v = fx.transform(u, p, store=s)
    assert v.shape == (spec.dim,)
    assert v.dtype == np.float32


def test_post_type_and_topic_one_hot():
    fx = FeatureExtractor()
    s, u, p = _store_with_pair()
    v = fx.transform(u, p, store=s)
    n_num = len(fx.spec.numeric)
    n_pt = len(fx.spec.post_type_vocab)
    post_type_block = v[n_num : n_num + n_pt]
    topic_block = v[n_num + n_pt :]
    assert post_type_block.sum() == 1.0
    assert topic_block.sum() == 1.0
    assert post_type_block[fx.spec.post_type_vocab.index("article")] == 1.0
    assert topic_block[fx.spec.topic_vocab.index("ml")] == 1.0


def test_is_connected_flag():
    fx = FeatureExtractor()
    s, u, p = _store_with_pair()
    v_unconn = fx.transform(u, p, store=s)
    s.connect("u1", "a1")
    v_conn = fx.transform(u, p, store=s)
    is_conn_idx = fx.spec.numeric.index("is_connected")
    assert v_unconn[is_conn_idx] == 0.0
    assert v_conn[is_conn_idx] == 1.0
