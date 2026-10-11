"""All plots are saved to images/ so they can be embedded in the README."""
import matplotlib
matplotlib.use("Agg")  # works without a display (servers, CI)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.config import IMAGES, LABEL_COL

sns.set_theme(style="whitegrid")


def _save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def class_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df[LABEL_COL].value_counts().index
    sns.countplot(data=df, x=LABEL_COL, order=order, ax=ax, hue=LABEL_COL, legend=False)
    ax.set_title("Traffic Class Distribution (note the imbalance)")
    ax.set_yscale("log")
    return _save(fig, "01_class_distribution.png")


def feature_boxplots(df):
    cols = ["request_frequency", "failed_logins", "unique_dst_ports", "src_bytes"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    for ax, c in zip(np.atleast_1d(axes), cols):
        sns.boxplot(data=df, x=LABEL_COL, y=c, ax=ax, hue=LABEL_COL, legend=False)
        ax.set_yscale("symlog")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(c)
    fig.suptitle("How attacks differ from normal traffic", y=1.02)
    return _save(fig, "02_feature_behaviour.png")


def confusion_matrix_plot(cm, title="Confusion Matrix - Random Forest"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return _save(fig, "03_confusion_matrix.png")


def model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    results_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="lower right")
    return _save(fig, "04_model_comparison.png")


def feature_importance(pipe, top=12):
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    idx = np.argsort(imp)[-top:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([n.split("__")[-1] for n in names[idx]], imp[idx], color="#2a6f97")
    ax.set_title("Top Features Driving Threat Detection")
    return _save(fig, "05_feature_importance.png")


def anomaly_scatter(alerts_df, normal_df, x="request_frequency", y="unique_dst_ports"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(normal_df[x], normal_df[y], s=6, alpha=0.3, label="Normal", color="#999")
    ax.scatter(alerts_df[x], alerts_df[y], s=14, alpha=0.8, label="Flagged threat", color="#d62828")
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title("Anomaly Detection: Flagged Threats vs Normal Traffic")
    ax.legend()
    return _save(fig, "06_anomaly_scatter.png")


def alert_severity(alerts_df):
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["CRITICAL", "HIGH", "MEDIUM"]
    colors = {"CRITICAL": "#9d0208", "HIGH": "#e85d04", "MEDIUM": "#faa307"}
    counts = alerts_df["severity"].value_counts().reindex(order).fillna(0)
    ax.bar(counts.index, counts.values, color=[colors[c] for c in counts.index])
    ax.set_title("Generated Alerts by Severity")
    return _save(fig, "07_alert_severity.png")
