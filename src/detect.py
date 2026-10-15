"""Threat detection logic: turn model probabilities into SOC-style alerts."""
import pandas as pd
from src.config import ALERT_THRESHOLD, OUTPUTS, SEVERITY_BANDS


def severity(prob: float) -> str:
    for threshold, name in SEVERITY_BANDS:
        if prob >= threshold:
            return name
    return "LOW"


def generate_alerts(model, X: pd.DataFrame, meta: pd.DataFrame = None,
                    iforest=None, threshold: float = ALERT_THRESHOLD) -> pd.DataFrame:
    """
    Score each flow and return only the suspicious ones, sorted by risk.
    meta: optional DataFrame (same index) holding flow_id / src_ip for context.
    """
    out = X.copy()
    out["attack_probability"] = model.predict_proba(X)[:, 1]
    if iforest is not None:
        out["anomaly_score"] = -iforest.score_samples(X)   # higher = more unusual
    out["severity"] = out["attack_probability"].apply(severity)
    alerts = out[out["attack_probability"] >= threshold].copy()
    alerts["reason"] = alerts.apply(explain, axis=1)
    if meta is not None:
        alerts = meta.loc[alerts.index].join(alerts)
    return alerts.sort_values("attack_probability", ascending=False)


def explain(row) -> str:
    """Rule-of-thumb human explanation so an analyst knows WHY it fired."""
    reasons = []
    if row.get("failed_logins", 0) >= 5:
        reasons.append("many failed logins (possible brute force)")
    if row.get("unique_dst_ports", 0) >= 20:
        reasons.append("many ports touched (possible port scan)")
    if row.get("request_frequency", 0) > 150:
        reasons.append("very high request rate (possible DoS)")
    if row.get("src_bytes", 0) > 1e6 and row.get("is_night", 0) == 1:
        reasons.append("large night-time upload (possible exfiltration)")
    return "; ".join(reasons) or "unusual combination of traffic features"


def save_alerts(alerts: pd.DataFrame, name="alerts.csv"):
    path = OUTPUTS / "alerts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(path, index=False)
    return path
