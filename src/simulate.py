"""
Virtual simulation of network traffic and cyber attacks.

Each row = one network connection/flow. We generate five behaviours:
  normal       - everyday browsing, API calls, file transfers
  dos          - Denial of Service: huge request rate, tiny packets, many SYNs
  brute_force  - repeated failed logins against SSH/FTP/RDP
  port_scan    - one source touching many ports with tiny, short flows
  exfiltration - very large outbound transfers at odd hours (data theft)
"""
import numpy as np
import pandas as pd
from src.config import SIMULATED_CSV, RANDOM_STATE, LABEL_COL, BINARY_COL

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "https", "ssh", "ftp", "dns", "smtp", "rdp"]


def _normal(n, rng):
    hour = np.clip(rng.normal(13, 4, n), 0, 23).astype(int)
    return pd.DataFrame({
        "duration": rng.exponential(5, n),
        "protocol": rng.choice(PROTOCOLS, n, p=[0.75, 0.2, 0.05]),
        "service": rng.choice(SERVICES, n, p=[0.3, 0.35, 0.05, 0.03, 0.17, 0.07, 0.03]),
        "src_bytes": rng.lognormal(6.5, 1.0, n),
        "dst_bytes": rng.lognormal(7.5, 1.2, n),
        "packet_count": rng.poisson(20, n) + 1,
        "request_frequency": rng.gamma(2, 5, n),
        "failed_logins": rng.choice([0, 1, 2], n, p=[0.96, 0.03, 0.01]),
        "unique_dst_ports": rng.choice([1, 2, 3], n, p=[0.8, 0.15, 0.05]),
        "syn_flag_ratio": rng.beta(2, 8, n),
        "hour_of_day": hour,
        LABEL_COL: "normal",
    })


def _dos(n, rng):
    return pd.DataFrame({
        "duration": rng.exponential(0.3, n),
        "protocol": rng.choice(PROTOCOLS, n, p=[0.5, 0.3, 0.2]),
        "service": rng.choice(["http", "https", "dns"], n),
        "src_bytes": rng.lognormal(4.0, 0.5, n),
        "dst_bytes": rng.lognormal(2.0, 0.8, n),
        "packet_count": rng.poisson(400, n) + 50,
        "request_frequency": rng.gamma(20, 25, n),
        "failed_logins": 0,
        "unique_dst_ports": rng.choice([1, 2], n),
        "syn_flag_ratio": rng.beta(9, 1.5, n),
        "hour_of_day": rng.integers(0, 24, n),
        LABEL_COL: "dos",
    })


def _brute_force(n, rng):
    return pd.DataFrame({
        "duration": rng.exponential(2, n),
        "protocol": "tcp",
        "service": rng.choice(["ssh", "ftp", "rdp"], n, p=[0.6, 0.2, 0.2]),
        "src_bytes": rng.lognormal(5.5, 0.4, n),
        "dst_bytes": rng.lognormal(5.0, 0.4, n),
        "packet_count": rng.poisson(30, n) + 5,
        "request_frequency": rng.gamma(8, 8, n),
        "failed_logins": rng.integers(5, 40, n),
        "unique_dst_ports": 1,
        "syn_flag_ratio": rng.beta(3, 5, n),
        "hour_of_day": rng.choice([1, 2, 3, 4, 22, 23], n),
        LABEL_COL: "brute_force",
    })


def _port_scan(n, rng):
    return pd.DataFrame({
        "duration": rng.exponential(0.1, n),
        "protocol": rng.choice(["tcp", "udp"], n, p=[0.9, 0.1]),
        "service": rng.choice(SERVICES, n),
        "src_bytes": rng.lognormal(3.5, 0.3, n),
        "dst_bytes": rng.lognormal(2.5, 0.5, n),
        "packet_count": rng.poisson(3, n) + 1,
        "request_frequency": rng.gamma(10, 10, n),
        "failed_logins": 0,
        "unique_dst_ports": rng.integers(20, 500, n),
        "syn_flag_ratio": rng.beta(8, 2, n),
        "hour_of_day": rng.integers(0, 24, n),
        LABEL_COL: "port_scan",
    })


def _exfil(n, rng):
    return pd.DataFrame({
        "duration": rng.exponential(60, n),
        "protocol": "tcp",
        "service": rng.choice(["https", "ftp", "smtp"], n, p=[0.6, 0.2, 0.2]),
        "src_bytes": rng.lognormal(13, 0.8, n),
        "dst_bytes": rng.lognormal(5, 0.8, n),
        "packet_count": rng.poisson(2000, n) + 200,
        "request_frequency": rng.gamma(2, 4, n),
        "failed_logins": 0,
        "unique_dst_ports": 1,
        "syn_flag_ratio": rng.beta(1.5, 9, n),
        "hour_of_day": rng.choice([0, 1, 2, 3, 4, 5], n),
        LABEL_COL: "exfiltration",
    })


NUM_COLS = ["duration", "src_bytes", "dst_bytes", "packet_count", "request_frequency",
            "failed_logins", "unique_dst_ports", "syn_flag_ratio"]


def _blend(base, other, rng, lo, hi):
    """Pull `base` rows toward `other` rows so classes overlap (makes the task realistic)."""
    base = base.copy().reset_index(drop=True)
    other = other.sample(len(base), replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    alpha = rng.uniform(lo, hi, len(base))
    for c in NUM_COLS:
        base[c] = (1 - alpha) * base[c] + alpha * other[c]
    base["failed_logins"] = base["failed_logins"].round()
    base["unique_dst_ports"] = base["unique_dst_ports"].round().clip(lower=1)
    return base


def generate_traffic(n_normal=20000, attack_share=0.12, stealth=0.20, hard_negatives=0.03,
                     seed=RANDOM_STATE):
    """Return a shuffled DataFrame of simulated traffic (imbalanced, like real life)."""
    rng = np.random.default_rng(seed)
    n_attack = int(n_normal * attack_share / (1 - attack_share))
    split = {"dos": 0.35, "brute_force": 0.25, "port_scan": 0.25, "exfiltration": 0.15}
    makers = {"dos": _dos, "brute_force": _brute_force,
              "port_scan": _port_scan, "exfiltration": _exfil}
    normal = _normal(n_normal, rng)
    attacks = [makers[name](int(n_attack * share), rng) for name, share in split.items()]
    attack_pool = pd.concat(attacks, ignore_index=True)

    # 1) "Low-and-slow" stealthy attacks: a fraction of attack rows look partly normal
    stealth_frames = []
    for a in attacks:
        k = int(len(a) * stealth)
        stealthy = _blend(a.iloc[:k], normal, rng, 0.35, 0.75)   # mostly normal-looking
        stealth_frames.append(pd.concat([stealthy, a.iloc[k:]], ignore_index=True))

    # 2) Hard negatives: legit-but-weird normal traffic (flash crowds, admins, typos)
    k = int(len(normal) * hard_negatives)
    weird = _blend(normal.iloc[:k], attack_pool, rng, 0.10, 0.45)
    normal = pd.concat([weird, normal.iloc[k:]], ignore_index=True)

    df = pd.concat([normal] + stealth_frames, ignore_index=True)

    # Real data is messy: inject a few missing values
    miss_idx = rng.choice(len(df), size=int(0.005 * len(df)), replace=False)
    df.loc[miss_idx, "duration"] = np.nan

    df.insert(0, "flow_id", np.arange(len(df)))
    df["src_ip"] = [f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for _ in range(len(df))]
    df[BINARY_COL] = (df[LABEL_COL] != "normal").astype(int)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def save_simulated(path=SIMULATED_CSV, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_traffic(**kwargs)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = save_simulated()
    print(d[LABEL_COL].value_counts())
    print(f"Saved to {SIMULATED_CSV}")
