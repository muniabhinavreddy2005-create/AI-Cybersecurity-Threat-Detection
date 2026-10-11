"""Cleaning, feature engineering and the sklearn preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import LABEL_COL, BINARY_COL

# Columns that identify a row but carry no behaviour signal -> must NOT be model inputs
ID_COLUMNS = ["flow_id", "src_ip", "Timestamp", "timestamp"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and infinite values (common in CICIDS2017)."""
    df = df.drop_duplicates().copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behaviour-based features. Each is only created if its inputs exist,
    so the same function works for simulated data AND real datasets."""
    df = df.copy()
    if {"src_bytes", "dst_bytes"} <= set(df.columns):
        df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
        df["bytes_ratio"] = (df["src_bytes"] + 1) / (df["dst_bytes"] + 1)
    if {"src_bytes", "packet_count"} <= set(df.columns):
        df["bytes_per_packet"] = df["src_bytes"] / df["packet_count"].clip(lower=1)
    if {"failed_logins", "duration"} <= set(df.columns):
        df["failed_login_rate"] = df["failed_logins"] / (df["duration"].fillna(1) + 1)
    if "hour_of_day" in df.columns:
        df["is_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
    return df


def split_xy(df: pd.DataFrame):
    """Separate model inputs (X) from targets. Labels are removed from X to avoid leakage."""
    drop_cols = [c for c in ID_COLUMNS + [LABEL_COL, BINARY_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y_binary = df[BINARY_COL]
    y_multi = df[LABEL_COL]
    return X, y_binary, y_multi


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute missing values, scale numbers, one-hot encode text columns."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
