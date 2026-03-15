"""
tests/test_features.py

Run with:  pytest tests/test_features.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from features import (
    FraudFeatureEngineer,
    build_feature_pipeline,
    apply_smote,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dummy_df(n=500, fraud_frac=0.02, seed=0):
    """Creates a minimal dummy dataset shaped like creditcard.csv."""
    rng = np.random.default_rng(seed)
    data = {f'V{i}': rng.normal(0, 1, n) for i in range(1, 29)}
    data['Time']   = rng.uniform(0, 172800, n)   # 0–48 hours in seconds
    data['Amount'] = rng.exponential(88, n)
    df = pd.DataFrame(data)
    y  = pd.Series(
        rng.choice([0, 1], size=n, p=[1 - fraud_frac, fraud_frac]),
        name='Class'
    )
    return df, y


@pytest.fixture
def raw_data():
    return make_dummy_df(n=1000, fraud_frac=0.02)


@pytest.fixture
def split_data():
    df, y = make_dummy_df(n=2000, fraud_frac=0.02)
    return train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)


# ── FraudFeatureEngineer ──────────────────────────────────────────────────────

class TestFraudFeatureEngineer:

    def test_output_shape_has_more_columns(self, raw_data):
        X, _ = raw_data
        eng  = FraudFeatureEngineer()
        out  = eng.fit_transform(X)
        assert out.shape[1] > X.shape[1], "Should add derived columns"

    def test_time_and_amount_dropped(self, raw_data):
        X, _ = raw_data
        eng  = FraudFeatureEngineer()
        out  = eng.fit_transform(X)
        assert 'Time'   not in out.columns
        assert 'Amount' not in out.columns

    def test_derived_columns_present(self, raw_data):
        X, _ = raw_data
        out  = FraudFeatureEngineer().fit_transform(X)
        for col in ['hour', 'is_night', 'log_amount', 'amount_bin',
                    'v14_x_amount', 'v17_x_amount']:
            assert col in out.columns, f"Missing column: {col}"

    def test_hour_range(self, raw_data):
        X, _ = raw_data
        out  = FraudFeatureEngineer().fit_transform(X)
        assert out['hour'].between(0, 24).all(), "Hour should be 0–24"

    def test_is_night_binary(self, raw_data):
        X, _ = raw_data
        out  = FraudFeatureEngineer().fit_transform(X)
        assert set(out['is_night'].unique()).issubset({0, 1})

    def test_log_amount_non_negative(self, raw_data):
        X, _ = raw_data
        out  = FraudFeatureEngineer().fit_transform(X)
        assert (out['log_amount'] >= 0).all()

    def test_no_nulls_in_output(self, raw_data):
        X, _ = raw_data
        out  = FraudFeatureEngineer().fit_transform(X)
        assert out.isnull().sum().sum() == 0

    def test_fit_returns_self(self, raw_data):
        X, _ = raw_data
        eng  = FraudFeatureEngineer()
        assert eng.fit(X) is eng

    def test_transform_does_not_modify_input(self, raw_data):
        X, _ = raw_data
        original_cols = X.columns.tolist()
        FraudFeatureEngineer().fit_transform(X)
        assert X.columns.tolist() == original_cols, "Input DataFrame was mutated"


# ── build_feature_pipeline ────────────────────────────────────────────────────

class TestFeaturePipeline:

    def test_pipeline_returns_numpy_array(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        pipe = build_feature_pipeline()
        out  = pipe.fit_transform(X_train)
        assert isinstance(out, np.ndarray)

    def test_no_leakage_transform_only_on_test(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        pipe = build_feature_pipeline()
        pipe.fit(X_train)
        # Should not raise — transform only, no refit
        out_test = pipe.transform(X_test)
        assert out_test.shape[0] == X_test.shape[0]

    def test_train_test_same_num_features(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        pipe = build_feature_pipeline()
        out_train = pipe.fit_transform(X_train)
        out_test  = pipe.transform(X_test)
        assert out_train.shape[1] == out_test.shape[1]

    def test_scaled_output_roughly_centered(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        pipe = build_feature_pipeline()
        out  = pipe.fit_transform(X_train)
        means = np.abs(np.median(out, axis=0))
        # RobustScaler centres on median — should be close to 0
        assert np.all(means < 1.0), "Scaled features not centred near 0"


# ── apply_smote ───────────────────────────────────────────────────────────────

class TestApplySmote:

    def test_output_larger_than_input(self, split_data):
        X_train, _, y_train, _ = split_data
        pipe    = build_feature_pipeline()
        X_sc    = pipe.fit_transform(X_train)
        X_sm, y_sm = apply_smote(X_sc, y_train.values)
        assert X_sm.shape[0] > X_sc.shape[0]

    def test_fraud_ratio_increases(self, split_data):
        X_train, _, y_train, _ = split_data
        pipe    = build_feature_pipeline()
        X_sc    = pipe.fit_transform(X_train)
        _, y_sm = apply_smote(X_sc, y_train.values, sampling_strategy=0.1)
        assert y_sm.mean() > y_train.mean(), "Fraud ratio should increase"

    def test_smote_only_on_train_not_test(self, split_data):
        """Test data shape must be unaffected — SMOTE is train-only."""
        X_train, X_test, y_train, y_test = split_data
        pipe   = build_feature_pipeline()
        pipe.fit(X_train)
        X_test_scaled = pipe.transform(X_test)
        # Test set must remain unchanged
        assert X_test_scaled.shape[0] == X_test.shape[0]