import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from pathlib import Path


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives new features from raw transaction data.

    Designed to work inside an sklearn Pipeline — safe to use
    at both training time and inference time.

    Input columns expected: V1-V28, Time, Amount
    """

    def fit(self, X, y=None):
        # Nothing to learn from data — transforms are deterministic
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Time → hour of day (0–24)
        X['hour'] = (X['Time'] % 86400) / 3600

        # 2. Night flag — fraud rate elevated between 10pm and 6am
        X['is_night'] = ((X['hour'] < 6) | (X['hour'] > 22)).astype(int)

        # 3. Log-transform Amount — reduces right-skew dramatically
        X['log_amount'] = np.log1p(X['Amount'])

        # 4. Amount bin — encodes transaction size category
        #    0: micro (<$1), 1: small ($1–10), 2: medium ($10–100),
        #    3: large ($100–500), 4: very large (>$500)
        X['amount_bin'] = pd.cut(
            X['Amount'],
            bins=[-1, 1, 10, 100, 500, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        # 5. Interaction features — top EDA predictors × log_amount
        #    Captures "was this a suspiciously large/small transaction
        #    given its PCA profile?"
        X['v14_x_amount'] = X['V14'] * X['log_amount']
        X['v17_x_amount'] = X['V17'] * X['log_amount']
        X['v12_x_amount'] = X['V12'] * X['log_amount']
        X['v10_x_amount'] = X['V10'] * X['log_amount']

        # 6. Drop raw columns — replaced by engineered versions
        X = X.drop(columns=['Time', 'Amount'])

        return X

    def get_feature_names_out(self, input_features=None):
        """Returns output feature names — required for SHAP compatibility."""
        base = [f'V{i}' for i in range(1, 29)]
        derived = [
            'hour', 'is_night', 'log_amount', 'amount_bin',
            'v14_x_amount', 'v17_x_amount', 'v12_x_amount', 'v10_x_amount'
        ]
        return base + derived


def build_feature_pipeline():
    """
    Returns a full sklearn Pipeline:
        FraudFeatureEngineer → RobustScaler

    Usage:
        pipeline = build_feature_pipeline()
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed  = pipeline.transform(X_test)   # never fit on test

    Note: fit only on training data to prevent leakage.
    """
    return Pipeline([
        ('engineer', FraudFeatureEngineer()),
        ('scaler',   RobustScaler()),
    ])


def apply_smote(X_train, y_train, sampling_strategy=0.1, random_state=42):
    """
    Applies SMOTE to training data only.

    sampling_strategy=0.1 means the minority class (fraud) will become
    10% of the majority class size — a good balance between signal
    and artificial noise.

    IMPORTANT: Never call this on test or validation data.

    Args:
        X_train: numpy array, already scaled
        y_train: numpy array of labels
        sampling_strategy: float, ratio of minority to majority after resampling
        random_state: int

    Returns:
        X_resampled, y_resampled (numpy arrays)
    """
    sm = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=5,
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print(f"  Before SMOTE: {Counter(y_train)}")
    print(f"  After SMOTE:  {Counter(y_res)}")
    print(f"  Fraud ratio after: {y_res.mean()*100:.2f}%")

    return X_res, y_res


def save_pipeline(pipeline, path='data/processed/feature_pipeline.pkl'):
    """Saves the fitted pipeline to disk for use at inference time."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to {path}")


def load_pipeline(path='data/processed/feature_pipeline.pkl'):
    """Loads the fitted pipeline from disk."""
    return joblib.load(path)


def run_feature_pipeline(
    X_train, X_test, y_train,
    smote_strategy=0.1,
    save_dir='data/processed'
):
    """
    Full Week 2 pipeline in one function call.

    1. Fits feature pipeline on X_train
    2. Transforms both train and test
    3. Applies SMOTE to training set only
    4. Saves all outputs + fitted pipeline

    Args:
        X_train, X_test: raw DataFrames (output of train_test_split)
        y_train: Series of training labels
        smote_strategy: SMOTE sampling ratio
        save_dir: where to write processed arrays

    Returns:
        dict with keys: X_train_sm, y_train_sm, X_test_scaled,
                        feature_names, pipeline
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Step 1/3 — Feature engineering + scaling...")
    pipeline = build_feature_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled  = pipeline.transform(X_test)
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape:  {X_test_scaled.shape}")

    print("\nStep 2/3 — Applying SMOTE to training set...")
    X_train_sm, y_train_sm = apply_smote(
        X_train_scaled, y_train.values,
        sampling_strategy=smote_strategy
    )

    print("\nStep 3/3 — Saving outputs...")
    np.save(save_path / 'X_train_scaled.npy', X_train_scaled)
    np.save(save_path / 'X_test_scaled.npy',  X_test_scaled)
    np.save(save_path / 'X_train_sm.npy',     X_train_sm)
    np.save(save_path / 'y_train_sm.npy',     y_train_sm)
    save_pipeline(pipeline, save_path / 'feature_pipeline.pkl')

    # Save feature names for SHAP and dashboard
    eng = FraudFeatureEngineer()
    feature_names = eng.get_feature_names_out()
    pd.Series(feature_names).to_csv(save_path / 'feature_names.csv', index=False)

    print(f"\nAll outputs saved to {save_dir}/")
    print("  X_train_scaled.npy")
    print("  X_test_scaled.npy")
    print("  X_train_sm.npy  (SMOTE-balanced)")
    print("  y_train_sm.npy")
    print("  feature_pipeline.pkl  (use this at inference)")
    print("  feature_names.csv")

    return {
        'X_train_sm':    X_train_sm,
        'y_train_sm':    y_train_sm,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'pipeline':      pipeline,
    }


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print("Running feature pipeline on creditcard.csv...\n")
    df = pd.read_csv('data/raw/creditcard.csv')

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = run_feature_pipeline(X_train, X_test, y_train)
    print("\nDone. Ready for Week 3 — model training.")