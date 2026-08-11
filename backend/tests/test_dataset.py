import pandas as pd

from app.ml.dataset import (
    TARGETS,
    dataset_profile,
    feature_columns,
    feature_specs,
    generate_bookings,
    sample_rows,
)


def test_generation_is_deterministic():
    left = generate_bookings(300, seed=11)
    right = generate_bookings(300, seed=11)
    pd.testing.assert_frame_equal(left, right)


def test_different_seeds_produce_different_data():
    left = generate_bookings(300, seed=11)
    right = generate_bookings(300, seed=12)
    assert not left["is_canceled"].equals(right["is_canceled"])


def test_dataset_has_missing_values_for_the_imputers_to_handle():
    frame = generate_bookings(2000, seed=3)
    assert frame["children"].isna().sum() > 0
    assert frame["region"].isna().sum() > 0


def test_both_target_classes_are_present_and_not_degenerate():
    frame = generate_bookings(2000, seed=3)
    share = frame["is_canceled"].mean()
    assert 0.05 < share < 0.95


def test_target_column_is_never_offered_as_a_feature():
    for key, spec in TARGETS.items():
        columns = feature_columns(key)
        assert spec["column"] not in columns["numeric"] + columns["categorical"]


def test_feature_specs_cover_every_feature_column():
    columns = feature_columns("cancellation")
    names = {spec.name for spec in feature_specs("cancellation")}
    assert names == set(columns["numeric"] + columns["categorical"])


def test_categorical_specs_carry_options_and_numeric_specs_carry_bounds():
    for spec in feature_specs("cancellation"):
        if spec.kind == "categorical":
            assert spec.options and spec.default in spec.options
        else:
            assert spec.minimum is not None and spec.maximum >= spec.minimum


def test_profile_reports_class_balance_for_classification():
    profile = dataset_profile("cancellation")
    distribution = profile["target_distribution"]
    assert distribution["kind"] == "classes"
    assert round(sum(item["share"] for item in distribution["classes"]), 2) == 1.0


def test_profile_reports_summary_stats_for_regression():
    profile = dataset_profile("price")
    assert profile["target_distribution"]["kind"] == "numeric"
    assert profile["target"]["task"] == "regression"


def test_sample_rows_are_json_safe():
    for row in sample_rows(10):
        assert not any(value != value for value in row.values() if isinstance(value, float))
