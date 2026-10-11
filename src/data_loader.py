"""Load either the simulated dataset or a real public dataset (NSL-KDD)."""
import pandas as pd
from src.config import SIMULATED_CSV, LABEL_COL, BINARY_COL
from src.simulate import save_simulated

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    LABEL_COL, "difficulty",
]


def load_simulated(path=SIMULATED_CSV, regenerate=False):
    """Load simulated traffic, generating it first if the file is missing."""
    if regenerate or not path.exists():
        return save_simulated(path)
    return pd.read_csv(path)


def load_nslkdd(path):
    """
    Load NSL-KDD (KDDTrain+.txt or KDDTest+.txt, no header row).
    Download: Kaggle 'NSL-KDD' or the UNB / UCI mirrors. Put it in data/raw/.
    """
    df = pd.read_csv(path, names=NSL_KDD_COLUMNS)
    df = df.drop(columns=["difficulty"])
    df[BINARY_COL] = (df[LABEL_COL] != "normal").astype(int)
    return df


def load_dataset(source="simulated", path=None):
    if source == "simulated":
        return load_simulated()
    if source == "nslkdd":
        if path is None:
            raise ValueError("Pass --path data/raw/KDDTrain+.txt for NSL-KDD")
        return load_nslkdd(path)
    raise ValueError(f"Unknown source: {source}")
