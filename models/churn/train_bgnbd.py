"""
BG/NBD + Gamma-Gamma Churn Model.

The BG/NBD (Beta-Geometric/Negative Binomial Distribution) model predicts
customer purchase behavior (alive/dead probability) based on:
- frequency: number of repeat purchases
- recency: time between first and last purchase
- T: customer age (time since first purchase)

Combined with Gamma-Gamma model for monetary value prediction.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BGNBD_PENALIZER, DATA_PROCESSED_DIR, MODELS_DIR


def train_bgnbd():
    """Train BG/NBD + Gamma-Gamma model and generate churn predictions."""
    print("=" * 60)
    print("BG/NBD + GAMMA-GAMMA CHURN MODEL")
    print("=" * 60)

    # Load BG/NBD summary data
    summary = pd.read_parquet(DATA_PROCESSED_DIR / "bgnbd_summary.parquet")
    print(f"\nLoaded {len(summary):,} customers")
    print(f"Churn rate: {summary['churn'].mean():.1%}")

    # ---- BG/NBD Model ----
    print("\n[1/4] Training BG/NBD model...")
    bgf = BetaGeoFitter(penalizer_coef=BGNBD_PENALIZER)
    bgf.fit(
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )
    print(f"  Model parameters: {bgf.summary}")

    # Predict probability of being alive
    summary["p_alive"] = bgf.conditional_probability_alive(
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )

    # Predict expected purchases in next 90 days
    summary["predicted_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90,
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )

    # Churn probability = 1 - p_alive
    summary["churn_prob_bgnbd"] = 1 - summary["p_alive"]

    # ---- Gamma-Gamma Model ----
    print("\n[2/4] Training Gamma-Gamma model...")
    # Filter customers with frequency > 0 for monetary value modeling
    returning_customers = summary[summary["frequency"] > 0].copy()

    if len(returning_customers) > 0 and "monetary_value" in returning_customers.columns:
        # Filter out zero/negative monetary values
        returning_customers = returning_customers[returning_customers["monetary_value"] > 0]

        ggf = GammaGammaFitter(penalizer_coef=0.001)
        ggf.fit(
            returning_customers["frequency"],
            returning_customers["monetary_value"],
        )

        # Predicted CLV
        returning_customers["predicted_clv"] = ggf.customer_lifetime_value(
            bgf,
            returning_customers["frequency"],
            returning_customers["recency"],
            returning_customers["T"],
            returning_customers["monetary_value"],
            time=3,  # 3 months
            discount_rate=0.01,
        )

        # Merge CLV back
        summary = summary.merge(
            returning_customers[["predicted_clv"]],
            left_index=True,
            right_index=True,
            how="left",
        )
        summary["predicted_clv"] = summary["predicted_clv"].fillna(0)
    else:
        print("  Skipping Gamma-Gamma (no returning customers with monetary data)")
        summary["predicted_clv"] = 0

    # ---- Evaluation ----
    print("\n[3/4] Evaluating BG/NBD churn predictions...")

    # Use optimal threshold based on F1
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        pred = (summary["churn_prob_bgnbd"] >= t).astype(int)
        f1 = f1_score(summary["churn"], pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    summary["churn_pred_bgnbd"] = (summary["churn_prob_bgnbd"] >= best_threshold).astype(int)

    auc = roc_auc_score(summary["churn"], summary["churn_prob_bgnbd"])
    precision = precision_score(summary["churn"], summary["churn_pred_bgnbd"])
    recall = recall_score(summary["churn"], summary["churn_pred_bgnbd"])
    f1 = f1_score(summary["churn"], summary["churn_pred_bgnbd"])
    accuracy = accuracy_score(summary["churn"], summary["churn_pred_bgnbd"])

    metrics = {
        "model": "BG/NBD",
        "auc_roc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "best_threshold": best_threshold,
    }

    print(f"\n  Results (threshold={best_threshold:.2f}):")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"\n{classification_report(summary['churn'], summary['churn_pred_bgnbd'])}")

    # ---- Save ----
    print("\n[4/4] Saving model and predictions...")
    save_dir = MODELS_DIR / "churn"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use lifetimes built-in save_model which handles pickling better
    bgf.save_model(save_dir / "bgnbd_model.pkl")

    # Check if ggf was defined in the local scope
    if "ggf" in locals():
        ggf.save_model(save_dir / "gamma_gamma_model.pkl")

    # Save predictions
    pred_cols = ["churn", "p_alive", "churn_prob_bgnbd", "churn_pred_bgnbd", "predicted_purchases_90d", "predicted_clv"]
    pred_cols = [c for c in pred_cols if c in summary.columns]
    summary[pred_cols].to_parquet(save_dir / "bgnbd_predictions.parquet")

    # Save metrics
    pd.DataFrame([metrics]).to_csv(save_dir / "bgnbd_metrics.csv", index=False)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # P(alive) distribution
    axes[0].hist(summary["p_alive"], bins=50, edgecolor="black", alpha=0.7, color="#2196F3")
    axes[0].set_title("Distribution of P(alive)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("P(alive)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(x=1 - best_threshold, color="red", linestyle="--", label=f"Threshold={best_threshold:.2f}")
    axes[0].legend()

    # Churn probability vs actual
    churned = summary[summary["churn"] == 1]["churn_prob_bgnbd"]
    retained = summary[summary["churn"] == 0]["churn_prob_bgnbd"]
    axes[1].hist(retained, bins=30, alpha=0.6, label="Retained", color="#4CAF50", edgecolor="black")
    axes[1].hist(churned, bins=30, alpha=0.6, label="Churned", color="#F44336", edgecolor="black")
    axes[1].set_title("Churn Probability Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Churn Probability")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # Predicted purchases vs churn
    axes[2].scatter(
        summary["predicted_purchases_90d"],
        summary["churn_prob_bgnbd"],
        c=summary["churn"],
        cmap="RdYlGn_r",
        alpha=0.3,
        s=5,
    )
    axes[2].set_title("Predicted Purchases vs Churn Prob", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Predicted Purchases (90d)")
    axes[2].set_ylabel("Churn Probability")

    plt.tight_layout()
    plt.savefig(save_dir / "bgnbd_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] BG/NBD model saved to: {save_dir}")
    return summary, metrics


if __name__ == "__main__":
    train_bgnbd()
