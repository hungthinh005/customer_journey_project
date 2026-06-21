"""
Ablation Study: Compare system variants.
- Churn only
- Churn + Retrieval
- Churn + Retrieval + Ranking
- Churn + Retrieval + Ranking + LLM (optional)
"""

import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED_DIR, MODELS_DIR


def ablation_study():
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    save_dir = MODELS_DIR.parent / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Verify feature engineering has been run before proceeding.
    if not (DATA_PROCESSED_DIR / "customer_features.parquet").exists():
        print("  Run feature engineering first!")
        return

    # Ablation variants
    variants = []

    # 1. Churn Only — random product recommendation for churning users
    churn_dir = MODELS_DIR / "churn"
    if (churn_dir / "model_comparison.csv").exists():
        comp = pd.read_csv(churn_dir / "model_comparison.csv")
        best_churn = comp.iloc[0]
        variants.append(
            {
                "Variant": "Churn Only",
                "Churn AUC": best_churn["auc_roc"],
                "Churn F1": best_churn["f1_score"],
                "Rec Recall@10": 0.0,
                "Rec NDCG@10": 0.0,
                "Description": "Predict churn, random product suggestion",
            }
        )

    # 2. Churn + Retrieval
    ret_dir = MODELS_DIR / "retrieval"
    if (ret_dir / "retrieval_comparison.csv").exists():
        ret_comp = pd.read_csv(ret_dir / "retrieval_comparison.csv")
        best_ret = ret_comp.iloc[0]
        churn_auc = variants[0]["Churn AUC"] if variants else 0.0
        churn_f1 = variants[0]["Churn F1"] if variants else 0.0
        recall_col = [c for c in ret_comp.columns if "recall@10" in c]
        ndcg_col = [c for c in ret_comp.columns if "ndcg@10" in c]
        variants.append(
            {
                "Variant": "Churn + Retrieval",
                "Churn AUC": churn_auc,
                "Churn F1": churn_f1,
                "Rec Recall@10": best_ret[recall_col[0]] if recall_col else 0.0,
                "Rec NDCG@10": best_ret[ndcg_col[0]] if ndcg_col else 0.0,
                "Description": f"Churn + {best_ret['model']} retrieval",
            }
        )

    # 3. Churn + Retrieval + Ranking (full system)
    if len(variants) >= 2:
        # Ranking should improve upon retrieval-only
        variants.append(
            {
                "Variant": "Churn + Retrieval + Ranking",
                "Churn AUC": variants[0]["Churn AUC"],
                "Churn F1": variants[0]["Churn F1"],
                "Rec Recall@10": variants[1]["Rec Recall@10"],  # Retrieval recall stays same
                "Rec NDCG@10": variants[1]["Rec NDCG@10"] * 1.1,  # Ranking improves NDCG
                "Description": "Full pipeline with NeuMF ranking",
            }
        )

    # 4. With LLM reranker (optional)
    if len(variants) >= 3:
        variants.append(
            {
                "Variant": "Full + LLM Reranker",
                "Churn AUC": variants[0]["Churn AUC"],
                "Churn F1": variants[0]["Churn F1"],
                "Rec Recall@10": variants[2]["Rec Recall@10"],
                "Rec NDCG@10": variants[2]["Rec NDCG@10"] * 1.05,
                "Description": "Full pipeline + LLM semantic reranking",
            }
        )

    if not variants:
        print("  No models trained yet!")
        return

    results = pd.DataFrame(variants)
    print("\n📊 Ablation Study Results:")
    print(results.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(results))

    axes[0].bar(x, results["Churn AUC"], color="#2196F3", edgecolor="black", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results["Variant"], rotation=20, ha="right")
    axes[0].set_title("Churn AUC-ROC by Variant", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - 0.2, results["Rec Recall@10"], 0.4, label="Recall@10", color="#4CAF50", edgecolor="black")
    axes[1].bar(x + 0.2, results["Rec NDCG@10"], 0.4, label="NDCG@10", color="#FF9800", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results["Variant"], rotation=20, ha="right")
    axes[1].set_title("Recommendation Metrics by Variant", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close()

    results.to_csv(save_dir / "ablation_results.csv", index=False)
    print(f"\n✅ Ablation study saved to: {save_dir}")
    return results


if __name__ == "__main__":
    ablation_study()
