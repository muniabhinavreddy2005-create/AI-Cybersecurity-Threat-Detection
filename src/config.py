"""Central configuration so no magic numbers are scattered across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "images"

SIMULATED_CSV = DATA_RAW / "simulated_traffic.csv"
MODEL_PATH = MODELS_DIR / "rf_pipeline.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "iforest_pipeline.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_COL = "label"            # attack type: normal, dos, brute_force, port_scan, exfiltration
BINARY_COL = "is_attack"       # 0 = normal, 1 = attack

# Alert severity thresholds applied on attack probability
SEVERITY_BANDS = [(0.90, "CRITICAL"), (0.75, "HIGH"), (0.50, "MEDIUM")]
