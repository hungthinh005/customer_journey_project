"""
Churn Model Evaluation: AUC-ROC, Precision, Recall, F1.
Evaluates all trained churn models on the test set.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR


def evaluate_churn():
    print("=" * 60)
    print("CHURN MODEL EVALUATION")
    print("=" * 60)

    churn_dir = MODELS_DIR / "churn"
    results = []

    # BG/NBD
    bgnbd_file = churn_dir / "bgnbd_predictions.parquet"
    if bgnbd_file.exists():
        preds = pd.read_parquet(bgnbd_file)
        if "churn" in preds.columns and "churn_prob_bgnbd" in preds.columns:
            auc = roc_auc_score(preds["churn"], preds["churn_prob_bgnbd"])
            pred_labels = (
                preds["churn_pred_bgnbd"]
                if "churn_pred_bgnbd" in preds.columns
                else (preds["churn_prob_bgnbd"] > 0.5).astype(int)
            )
            results.append(
                {
                    "Model": "BG/NBD",
                    "AUC-ROC": auc,
                    "Precision": precision_score(preds["churn"], pred_labels),
                    "Recall": recall_score(preds["churn"], pred_labels),
                    "F1": f1_score(preds["churn"], pred_labels),
                    "Accuracy": accuracy_score(preds["churn"], pred_labels),
                }
            )
            print(f"\nBG/NBD: AUC={auc:.4f}")
            print(classification_report(preds["churn"], pred_labels))

    # Survival
    surv_file = churn_dir / "survival_predictions.parquet"
    if surv_file.exists():
        preds = pd.read_parquet(surv_file)
        for col, name in [("churn_prob_cox", "Cox PH"), ("churn_prob_aft", "Weibull AFT")]:
            if col in preds.columns and "churn_actual" in preds.columns:
                auc = roc_auc_score(preds["churn_actual"], preds[col])
                pred_labels = (preds[col] > 0.5).astype(int)
                results.append(
                    {
                        "Model": name,
                        "AUC-ROC": auc,
                        "Precision": precision_score(preds["churn_actual"], pred_labels),
                        "Recall": recall_score(preds["churn_actual"], pred_labels),
                        "F1": f1_score(preds["churn_actual"], pred_labels),
                        "Accuracy": accuracy_score(preds["churn_actual"], pred_labels),
                    }
                )
                print(f"\n{name}: AUC={auc:.4f}")

    if results:
        df = pd.DataFrame(results)
        print("\n📊 Summary:")
        print(df.to_string(index=False))
        df.to_csv(churn_dir / "churn_evaluation.csv", index=False)

    print(f"\n✅ Evaluation saved to: {churn_dir}")


if __name__ == "__main__":
    evaluate_churn()
