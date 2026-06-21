"""
Compare Churn Models: BG/NBD vs Survival Analysis.

Loads predictions from both approaches and determines the best model
based on AUC-ROC, F1-Score, and concordance metrics.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR


def compare_churn_models():
    """Compare all trained churn models and select the best one."""
    print("=" * 60)
    print("CHURN MODEL COMPARISON")
    print("=" * 60)

    save_dir = MODELS_DIR / "churn"

    # Load metrics
    metrics_files = list(save_dir.glob("*_metrics.csv"))
    all_metrics = []
    for f in metrics_files:
        df = pd.read_csv(f)
        all_metrics.append(df)

    if not all_metrics:
        print("No metrics found! Train models first.")
        return

    comparison = pd.concat(all_metrics, ignore_index=True)
    comparison = comparison.sort_values("auc_roc", ascending=False)

    print("\n📊 Model Comparison:")
    print(comparison.to_string(index=False))

    # Best model
    best = comparison.iloc[0]
    print(f"\n🏆 Best Overall Model: {best['model']}")
    print(f"   AUC-ROC: {best['auc_roc']:.4f}")
    print(f"   F1-Score: {best['f1_score']:.4f}")

    # Load predictions for ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"BG/NBD": "#2196F3", "Cox_PH": "#4CAF50", "Weibull_AFT": "#FF9800"}

    # Try to load BG/NBD predictions
    bgnbd_pred_file = save_dir / "bgnbd_predictions.parquet"
    survival_pred_file = save_dir / "survival_predictions.parquet"

    if bgnbd_pred_file.exists():
        bgnbd_preds = pd.read_parquet(bgnbd_pred_file)
        if "churn" in bgnbd_preds.columns and "churn_prob_bgnbd" in bgnbd_preds.columns:
            fpr, tpr, _ = roc_curve(bgnbd_preds["churn"], bgnbd_preds["churn_prob_bgnbd"])
            auc = roc_auc_score(bgnbd_preds["churn"], bgnbd_preds["churn_prob_bgnbd"])
            axes[0].plot(fpr, tpr, label=f"BG/NBD (AUC={auc:.3f})", color=colors["BG/NBD"], linewidth=2)

    if survival_pred_file.exists():
        surv_preds = pd.read_parquet(survival_pred_file)
        for col, name in [("churn_prob_cox", "Cox_PH"), ("churn_prob_aft", "Weibull_AFT")]:
            if col in surv_preds.columns:
                fpr, tpr, _ = roc_curve(surv_preds["churn_actual"], surv_preds[col])
                auc = roc_auc_score(surv_preds["churn_actual"], surv_preds[col])
                axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name], linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Bar chart comparison
    models = comparison["model"].tolist()
    x = np.arange(len(models))
    width = 0.2

    axes[1].bar(x - width, comparison["auc_roc"], width, label="AUC-ROC", color="#2196F3")
    axes[1].bar(x, comparison["f1_score"], width, label="F1-Score", color="#F44336")
    axes[1].bar(x + width, comparison["precision"], width, label="Precision", color="#4CAF50")

    axes[1].set_title("Metrics Comparison", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "churn_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save comparison
    comparison.to_csv(save_dir / "model_comparison.csv", index=False)

    # Save best model name
    with open(save_dir / "best_model.txt", "w") as f:
        f.write(best["model"])

    print(f"\n✅ Comparison saved to: {save_dir / 'churn_model_comparison.png'}")
    return comparison


if __name__ == "__main__":
    compare_churn_models()
