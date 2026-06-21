"""
Survival Analysis Churn Model.

Uses Cox Proportional Hazards and Accelerated Failure Time (AFT) models
to model the time-to-churn event for each customer.

Survival analysis is well-suited for churn because:
- It naturally handles censored data (customers who haven't churned yet)
- It provides time-varying churn hazard rates
- It gives interpretable feature effects
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ALL_FEATURES,
    COX_L1_RATIO,
    COX_PENALIZER,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
)


def prepare_survival_data():
    """
    Prepare data for survival analysis.

    For survival analysis we need:
    - duration: time from first purchase to churn event (or censoring)
    - event: 1 if churned (event observed), 0 if censored (still active)
    - features: covariates for the model
    """
    customer_features = pd.read_parquet(DATA_PROCESSED_DIR / "customer_features.parquet")

    # Duration = days_as_customer + recency (approximate time-to-event)
    # For churned customers: recency is the time since last purchase
    # For active customers: they're censored
    customer_features["duration"] = customer_features["days_as_customer"].clip(lower=1)

    # Event indicator
    customer_features["event"] = customer_features["churn"]

    # Select features
    feature_cols = [c for c in ALL_FEATURES if c in customer_features.columns]

    # Clean up features
    survival_df = customer_features[["customer_id", "duration", "event"] + feature_cols].copy()

    # Handle missing/infinite values
    survival_df = survival_df.replace([np.inf, -np.inf], np.nan)
    survival_df = survival_df.fillna(0)

    # Ensure duration > 0
    survival_df["duration"] = survival_df["duration"].clip(lower=1)

    print(f"Survival dataset: {len(survival_df):,} customers")
    print(f"Event rate (churned): {survival_df['event'].mean():.1%}")
    print(f"Median duration: {survival_df['duration'].median():.0f} days")

    return survival_df, feature_cols


def _drop_low_variance_cols(df, feature_cols, variance_threshold=1e-6):
    """Drop feature columns with near-zero variance to prevent convergence issues.

    Lifelines normalizes covariates by (X - mean) / std internally.
    When std ≈ 0, this produces NaN and kills the optimizer.
    """
    variances = df[feature_cols].var()
    low_var = variances[variances < variance_threshold].index.tolist()
    if low_var:
        print(f"  [!] Dropping low-variance columns: {low_var}")
    kept = [c for c in feature_cols if c not in low_var]
    return kept


def train_cox_ph(survival_df, feature_cols):
    """Train Cox Proportional Hazards model."""
    print("\n" + "-" * 40)
    print("Cox Proportional Hazards Model")
    print("-" * 40)

    # Drop near-zero-variance features
    safe_cols = _drop_low_variance_cols(survival_df, feature_cols)

    # Prepare data for CoxPH
    cox_df = survival_df[["duration", "event"] + safe_cols].copy()

    # Fit model
    cph = CoxPHFitter(penalizer=COX_PENALIZER, l1_ratio=COX_L1_RATIO)
    cph.fit(cox_df, duration_col="duration", event_col="event")

    # Print summary
    print("\nModel Summary:")
    cph.print_summary()

    # Concordance index
    c_index = cph.concordance_index_
    print(f"\nConcordance Index: {c_index:.4f}")

    return cph, safe_cols


def train_weibull_aft(survival_df, feature_cols):
    """Train Weibull Accelerated Failure Time model."""
    print("\n" + "-" * 40)
    print("Weibull AFT Model")
    print("-" * 40)

    # Drop near-zero-variance features
    safe_cols = _drop_low_variance_cols(survival_df, feature_cols)

    # Prepare data
    aft_df = survival_df[["duration", "event"] + safe_cols].copy()

    # Fit model
    aft = WeibullAFTFitter(penalizer=COX_PENALIZER)
    aft.fit(aft_df, duration_col="duration", event_col="event")

    # Print summary
    print("\nModel Summary:")
    aft.print_summary()

    # Concordance index
    c_index = concordance_index(
        aft_df["duration"],
        -aft.predict_median(aft_df),  # Negative because higher median = lower risk
        aft_df["event"],
    )
    print(f"\nConcordance Index: {c_index:.4f}")

    return aft, safe_cols


def evaluate_survival_model(model, survival_df, feature_cols, model_name):
    """Evaluate survival model's churn predictions."""
    print(f"\n  Evaluating {model_name}...")

    # Predict survival probability at CHURN_WINDOW_DAYS
    cox_df = survival_df[["duration", "event"] + feature_cols].copy()

    if isinstance(model, CoxPHFitter):
        # Cox model: predict partial hazard → convert to churn probability
        partial_hazard = model.predict_partial_hazard(cox_df)
        # Higher hazard = higher churn risk
        # Normalize to [0, 1] range
        churn_prob = partial_hazard / partial_hazard.max()
        churn_prob = churn_prob.clip(0, 1)
    else:
        # AFT model: predict survival function
        # Lower predicted median survival = higher churn risk
        median_survival = model.predict_median(cox_df)
        # Convert to probability: shorter survival → higher churn probability
        churn_prob = 1 - (median_survival / (median_survival.max() + 1))
        churn_prob = churn_prob.clip(0, 1)

    # Find best threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        pred = (churn_prob >= t).astype(int)
        f1 = f1_score(survival_df["event"], pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    churn_pred = (churn_prob >= best_threshold).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(survival_df["event"], churn_prob)
    except ValueError:
        auc = 0.5

    precision = precision_score(survival_df["event"], churn_pred, zero_division=0)
    recall = recall_score(survival_df["event"], churn_pred, zero_division=0)
    f1 = f1_score(survival_df["event"], churn_pred, zero_division=0)
    accuracy = accuracy_score(survival_df["event"], churn_pred)

    metrics = {
        "model": model_name,
        "auc_roc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "best_threshold": best_threshold,
    }

    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"\n{classification_report(survival_df['event'], churn_pred)}")

    return churn_prob, churn_pred, metrics


def train_survival():
    """Train and compare survival analysis models."""
    print("=" * 60)
    print("SURVIVAL ANALYSIS CHURN MODEL")
    print("=" * 60)

    # Prepare data
    survival_df, feature_cols = prepare_survival_data()

    # Train models
    print("\n[1/4] Training Cox PH model...")
    cph, cox_feature_cols = train_cox_ph(survival_df, feature_cols)

    print("\n[2/4] Training Weibull AFT model...")
    aft, aft_feature_cols = train_weibull_aft(survival_df, feature_cols)

    # Evaluate both (use the feature subset each model was trained on)
    print("\n[3/4] Evaluating models...")
    cox_prob, cox_pred, cox_metrics = evaluate_survival_model(cph, survival_df, cox_feature_cols, "Cox_PH")
    aft_prob, aft_pred, aft_metrics = evaluate_survival_model(aft, survival_df, aft_feature_cols, "Weibull_AFT")

    # Select best model
    best_model_name = "Cox_PH" if cox_metrics["auc_roc"] >= aft_metrics["auc_roc"] else "Weibull_AFT"
    best_model = cph if best_model_name == "Cox_PH" else aft
    best_prob = cox_prob if best_model_name == "Cox_PH" else aft_prob
    best_pred = cox_pred if best_model_name == "Cox_PH" else aft_pred
    best_metrics = cox_metrics if best_model_name == "Cox_PH" else aft_metrics

    print(f"\n  [*] Best survival model: {best_model_name}")

    # ---- Save ----
    print("\n[4/4] Saving models and predictions...")
    save_dir = MODELS_DIR / "churn"
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(cph, save_dir / "cox_ph_model.pkl")
    joblib.dump(aft, save_dir / "weibull_aft_model.pkl")
    joblib.dump(best_model, save_dir / "survival_best_model.pkl")

    # Save predictions
    predictions = survival_df[["customer_id"]].copy()
    predictions["churn_prob_cox"] = cox_prob.values
    predictions["churn_prob_aft"] = aft_prob.values
    predictions["churn_prob_survival"] = best_prob.values
    predictions["churn_pred_survival"] = best_pred.values
    predictions["churn_actual"] = survival_df["event"].values
    predictions.to_parquet(save_dir / "survival_predictions.parquet", index=False)

    # Save metrics
    all_metrics = pd.DataFrame([cox_metrics, aft_metrics])
    all_metrics.to_csv(save_dir / "survival_metrics.csv", index=False)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Kaplan-Meier curve
    kmf = KaplanMeierFitter()
    kmf.fit(survival_df["duration"], event_observed=survival_df["event"])
    kmf.plot_survival_function(ax=axes[0, 0])
    axes[0, 0].set_title("Kaplan-Meier Survival Curve", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Days")
    axes[0, 0].set_ylabel("Survival Probability")

    # Cox coefficients
    coefs = cph.summary[["coef"]].sort_values("coef")
    axes[0, 1].barh(coefs.index, coefs["coef"], color="#2196F3", edgecolor="black")
    axes[0, 1].set_title("Cox PH Coefficients", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Coefficient")

    # Churn probability distributions
    axes[1, 0].hist(cox_prob, bins=50, alpha=0.6, label="Cox PH", color="#4CAF50", edgecolor="black")
    axes[1, 0].hist(aft_prob, bins=50, alpha=0.6, label="Weibull AFT", color="#FF9800", edgecolor="black")
    axes[1, 0].set_title("Churn Probability Distributions", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Churn Probability")
    axes[1, 0].legend()

    # Model comparison
    models = [cox_metrics["model"], aft_metrics["model"]]
    auc_vals = [cox_metrics["auc_roc"], aft_metrics["auc_roc"]]
    f1_vals = [cox_metrics["f1_score"], aft_metrics["f1_score"]]

    x = np.arange(len(models))
    width = 0.35
    axes[1, 1].bar(x - width / 2, auc_vals, width, label="AUC-ROC", color="#2196F3")
    axes[1, 1].bar(x + width / 2, f1_vals, width, label="F1-Score", color="#F44336")
    axes[1, 1].set_title("Model Comparison", fontsize=13, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_dir / "survival_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Survival models saved to: {save_dir}")
    return predictions, best_metrics


if __name__ == "__main__":
    train_survival()
