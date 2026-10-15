"""
Entry point.
  python main.py run                      -> full pipeline on simulated data
  python main.py run --source nslkdd --path data/raw/KDDTrain+.txt
  python main.py simulate                 -> only generate the dataset
"""
import argparse
import pandas as pd
from src import visualize as viz
from src.config import OUTPUTS, LABEL_COL
from src.data_loader import load_dataset
from src.detect import generate_alerts, save_alerts
from src.evaluate import cm, compare_models, report
from src.threshold import sweep
from src.simulate import save_simulated
from src.train import train_all


def run(source, path):
    print("[1/6] Loading data ...")
    df = load_dataset(source, path)
    print(df.head(), "\n")
    print(df[LABEL_COL].value_counts(), "\n")

    print("[2/6] Exploratory plots ...")
    if source == "simulated":
        viz.class_distribution(df)
        viz.feature_boxplots(df)

    print("[3/6] Training models (this takes ~30s) ...")
    models, iforest, (X_test, yb_test, ym_test) = train_all(df)

    print("[4/6] Evaluating ...")
    results = compare_models(models, iforest, X_test, yb_test)
    print(results, "\n")
    rf = models["Random Forest"]
    rf_pred = rf.predict(X_test)
    print(report(yb_test, rf_pred))
    (OUTPUTS / "reports").mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS / "reports" / "model_comparison.csv")
    viz.model_comparison(results)
    viz.confusion_matrix_plot(cm(yb_test, rf_pred))
    viz.feature_importance(rf)
    print("Threshold comparison (Random Forest):")
    sw = sweep(rf, X_test, yb_test)
    print(sw, "\n")
    sw.to_csv(OUTPUTS / "reports" / "threshold_sweep.csv", index=False)

    print("[5/6] Generating alerts on the unseen test set ...")
    alerts = generate_alerts(rf, X_test, iforest=iforest)
    alerts["true_label"] = ym_test.loc[alerts.index]
    print(f"{len(alerts)} alerts raised. Top 5:")
    cols = [c for c in ["attack_probability", "severity", "true_label", "reason"] if c in alerts]
    print(alerts[cols].head(), "\n")
    print(save_alerts(alerts.reset_index(drop=True)))

    print("[6/6] Alert visualisations ...")
    normal_view = X_test[yb_test == 0]
    if "unique_dst_ports" in alerts:
        viz.anomaly_scatter(alerts, normal_view)
    viz.alert_severity(alerts)
    print("Done. Check images/, outputs/alerts/ and outputs/reports/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "simulate"])
    p.add_argument("--source", default="simulated", choices=["simulated", "nslkdd"])
    p.add_argument("--path", default=None)
    a = p.parse_args()
    if a.command == "simulate":
        save_simulated()
        print("Simulated dataset saved to data/raw/simulated_traffic.csv")
    else:
        run(a.source, a.path)
