import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import joblib


def compute_psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 1, bins + 1)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts / len(actual))
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def monitor_model(snapshot_date_str):
    # Skip if not in OOT period
    from datetime import datetime
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    if snapshot_date < datetime(2024, 7, 1):
        print(f"Skipping monitoring for {snapshot_date_str} - not in OOT period")
        return

    # Load all predictions saved so far
    predictions_directory = "datamart/gold/model_predictions/"
    files = [f for f in os.listdir(predictions_directory) if f.endswith(".parquet")]
    
    if not files:
        print("No predictions found")
        return

    df_all_preds = pd.concat([
        pd.read_parquet(predictions_directory + f) for f in files
    ])
    df_all_preds["snapshot_date"] = pd.to_datetime(df_all_preds["snapshot_date"])

    # Load training baseline predictions for PSI
    model = joblib.load("model/artefacts/best_model.pkl")
    
    # Compute metrics month by month
    monitoring_results = []

    for date in sorted(df_all_preds["snapshot_date"].unique()):
        df_month = df_all_preds[df_all_preds["snapshot_date"] == date]
        preds = df_month["predicted_prob"].values
        actuals = df_month["actual_label"].values

        # Load training preds for PSI baseline
        train_preds = pd.read_parquet("model/artefacts/train_preds.parquet")["predicted_prob"].values

        monitoring_results.append({
            "snapshot_date": date,
            "auc": roc_auc_score(actuals, preds),
            "mean_predicted_prob": preds.mean(),
            "predicted_default_rate": (preds >= 0.5).mean(),
            "actual_default_rate": actuals.mean(),
            "psi": compute_psi(train_preds, preds)
        })

    df_monitoring = pd.DataFrame(monitoring_results).sort_values("snapshot_date")
    print(df_monitoring)

    # Save monitoring gold table
    os.makedirs("datamart/gold/model_monitoring", exist_ok=True)
    df_monitoring.to_parquet("datamart/gold/model_monitoring/model_monitoring.parquet", index=False)
    print("Monitoring table saved!")

    # Plot dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Monitoring Dashboard", fontsize=16)

    axes[0, 0].plot(df_monitoring["snapshot_date"], df_monitoring["auc"], marker="o", color="blue")
    axes[0, 0].axhline(y=0.75, color="red", linestyle="--", label="Minimum threshold")
    axes[0, 0].set_title("AUC Over Time")
    axes[0, 0].set_ylabel("AUC")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    axes[0, 1].plot(df_monitoring["snapshot_date"], df_monitoring["predicted_default_rate"], marker="o", label="Predicted", color="blue")
    axes[0, 1].plot(df_monitoring["snapshot_date"], df_monitoring["actual_default_rate"], marker="o", label="Actual", color="orange")
    axes[0, 1].set_title("Predicted vs Actual Default Rate")
    axes[0, 1].set_ylabel("Default Rate")
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)

    axes[1, 0].plot(df_monitoring["snapshot_date"], df_monitoring["mean_predicted_prob"], marker="o", color="green")
    axes[1, 0].set_title("Mean Predicted Probability Over Time")
    axes[1, 0].set_ylabel("Mean Predicted Prob")
    axes[1, 0].tick_params(axis='x', rotation=45)

    axes[1, 1].plot(df_monitoring["snapshot_date"], df_monitoring["psi"], marker="o", color="purple")
    axes[1, 1].axhline(y=0.1, color="orange", linestyle="--", label="Moderate drift (0.1)")
    axes[1, 1].axhline(y=0.2, color="red", linestyle="--", label="Significant drift (0.2)")
    axes[1, 1].set_title("PSI Over Time")
    axes[1, 1].set_ylabel("PSI")
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("model/monitoring/monitoring_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Dashboard saved!")