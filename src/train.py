"""Train supervised classifiers and an unsupervised anomaly detector."""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.config import (ANOMALY_MODEL_PATH, MODEL_PATH, MODELS_DIR,
                        RANDOM_STATE, TEST_SIZE)
from src.preprocess import build_preprocessor, clean, engineer_features, split_xy


def prepare(df: pd.DataFrame):
    """Full data preparation: clean -> engineer -> split. Returns train/test sets."""
    df = engineer_features(clean(df))
    X, y_bin, y_multi = split_xy(df)
    # stratify keeps the attack ratio identical in train and test (important: imbalance!)
    return train_test_split(X, y_bin, y_multi, test_size=TEST_SIZE,
                            stratify=y_multi, random_state=RANDOM_STATE)


def build_models(X_train):
    pre = build_preprocessor(X_train)
    return {
        "Logistic Regression": Pipeline([
            ("prep", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(X_train)),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           n_jobs=-1, random_state=RANDOM_STATE))]),
    }


def train_all(df: pd.DataFrame):
    X_train, X_test, yb_train, yb_test, ym_train, ym_test = prepare(df)
    models = build_models(X_train)
    for name, pipe in models.items():
        pipe.fit(X_train, yb_train)

    # Unsupervised anomaly detector: trained ONLY on normal traffic
    # (in real SOCs you rarely have attack labels, so this matters)
    normal_only = X_train[yb_train == 0]
    iforest = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", IsolationForest(n_estimators=200, contamination=0.05,
                                random_state=RANDOM_STATE, n_jobs=-1))])
    iforest.fit(normal_only)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["Random Forest"], MODEL_PATH)
    joblib.dump(iforest, ANOMALY_MODEL_PATH)
    return models, iforest, (X_test, yb_test, ym_test)
