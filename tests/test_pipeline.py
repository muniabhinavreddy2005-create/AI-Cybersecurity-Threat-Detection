from src.simulate import generate_traffic
from src.preprocess import engineer_features, split_xy
from src.detect import severity


def test_simulation_has_all_classes():
    df = generate_traffic(n_normal=1000)
    assert set(df["label"]) == {"normal", "dos", "brute_force", "port_scan", "exfiltration"}


def test_no_label_leakage():
    df = engineer_features(generate_traffic(n_normal=500))
    X, _, _ = split_xy(df)
    assert "label" not in X.columns and "is_attack" not in X.columns


def test_severity_bands():
    assert severity(0.95) == "CRITICAL"
    assert severity(0.8) == "HIGH"
    assert severity(0.6) == "MEDIUM"
    assert severity(0.1) == "LOW"
