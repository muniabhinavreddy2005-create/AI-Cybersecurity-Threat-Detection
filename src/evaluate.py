"""Metrics: accuracy, precision, recall, F1, confusion matrix."""
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compare_models(models: dict, iforest, X_test, y_test) -> pd.DataFrame:
    rows = {}
    for name, pipe in models.items():
        rows[name] = score(y_test, pipe.predict(X_test))
    # IsolationForest returns -1 (anomaly) / 1 (normal) -> convert to 1 (attack) / 0
    iso_pred = (iforest.predict(X_test) == -1).astype(int)
    rows["Isolation Forest (unsupervised)"] = score(y_test, iso_pred)
    return pd.DataFrame(rows).T.round(4)


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["normal", "attack"], digits=4)


def cm(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
