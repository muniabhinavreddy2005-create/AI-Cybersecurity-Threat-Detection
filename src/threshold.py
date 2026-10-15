"""Compare precision and recall at different alert thresholds."""
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def sweep(model, X_test, y_test, thresholds=(0.3, 0.5, 0.7, 0.9)) -> pd.DataFrame:
    """Higher threshold = fewer false alarms (precision up) but more missed attacks (recall down)."""
    prob = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "alerts": int(pred.sum()),
        })
    return pd.DataFrame(rows)