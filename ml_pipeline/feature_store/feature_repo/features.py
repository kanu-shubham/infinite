"""
features.py
-----------
Defines ALL features used by the hotel price prediction model.

HOW FEAST WORKS (3-minute mental model):
  1. You define an Entity  – the "key" that identifies a row (hotel_id).
  2. You define a DataSource – where the raw data lives (CSV / Parquet / BigQuery).
  3. You define a FeatureView – which columns from the source are "features",
     and for how long they stay valid (ttl).
  4. You run `feast apply`  – Feast registers everything in the registry.
  5. Training:  call get_historical_features() – point-in-time correct joins.
  6. Serving:   call get_online_features()     – millisecond retrieval.

WHY point-in-time correctness matters:
  Imagine you predict hotel price on 2024-03-01 using the hotel's review_score.
  If you accidentally use the review_score from 2024-06-01 (future!), your model
  looks great in training but fails in production. Feast prevents this leak.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# ── 1. Entity ─────────────────────────────────────────────────────────────────
# An entity is the primary key that links all feature tables.
# Think of it as the JOIN key in SQL.
hotel = Entity(
    name="hotel",
    join_keys=["hotel_id"],
    description="Unique hotel identifier",
)

# ── 2. Data Source ────────────────────────────────────────────────────────────
# Points Feast to the raw CSV/Parquet file.
# In production this would be a BigQuery table or Snowflake view.
_data_path = str(
    Path(__file__).parents[3] / "data" / "hotels.csv"
)

hotel_source = FileSource(
    path=_data_path,
    timestamp_field="event_timestamp",   # required for point-in-time joins
    name="hotel_data_source",
)

# ── 3. Feature View ───────────────────────────────────────────────────────────
# A FeatureView groups related features together.
# ttl (time-to-live): how old a feature value can be before it's considered stale.
hotel_features = FeatureView(
    name="hotel_features",
    entities=[hotel],
    ttl=timedelta(days=365),
    schema=[
        # Categorical
        Field(name="city",          dtype=String),
        Field(name="category",      dtype=String),
        # Numeric
        Field(name="star_rating",   dtype=Int64),
        Field(name="review_score",  dtype=Float32),
        Field(name="num_reviews",   dtype=Int64),
        Field(name="distance_km",   dtype=Float32),
        Field(name="amenities",     dtype=Int64),
        Field(name="rooms_available", dtype=Int64),
    ],
    source=hotel_source,
    description="Core hotel attributes used for price prediction",
)

# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE (run from the feature_repo/ directory):
#
#   feast apply                        # register features in the registry
#   feast materialize-incremental NOW  # push latest values to online store
#
# In Python (training):
#   store = FeatureStore(repo_path=".")
#   entity_df = pd.DataFrame({"hotel_id": [1,2,3], "event_timestamp": [...]})
#   training_df = store.get_historical_features(
#       entity_df=entity_df,
#       features=["hotel_features:star_rating", "hotel_features:review_score"],
#   ).to_df()
#
# In Python (serving / inference):
#   online_features = store.get_online_features(
#       features=["hotel_features:star_rating"],
#       entity_rows=[{"hotel_id": 42}],
#   ).to_dict()
# ─────────────────────────────────────────────────────────────────────────────
