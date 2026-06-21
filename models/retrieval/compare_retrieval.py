"""
Compare Retrieval Models: ALS vs Item2Vec vs Two-Tower.
Evaluates using Recall@K and NDCG@K on held-out interactions.
"""

import sys
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED_DIR, EVAL_K_VALUES, MODELS_DIR


def compute_recall_at_k(actual, predicted, k):
    if len(actual) == 0:
        return 0.0
    predicted_k = predicted[:k]
    return len(set(actual) & set(predicted_k)) / min(len(actual), k)


def compute_ndcg_at_k(actual, predicted, k):
    if len(actual) == 0:
        return 0.0
    predicted_k = predicted[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(predicted_k) if item in actual)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_model(user_embeddings, item_embeddings, mappings, interactions, model_name):
    print(f"\n  Evaluating {model_name}...")
    user_to_idx = mappings["user_to_idx"]
    idx_to_item = mappings["idx_to_item"]

    # Build ground truth from interactions
    ground_truth = interactions.groupby("customer_id")["stock_code"].apply(set).to_dict()

    results = {k: {"recall": [], "ndcg": []} for k in EVAL_K_VALUES}
    n_evaluated = 0

    for user_id, actual_items in ground_truth.items():
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        if user_idx >= len(user_embeddings):
            continue

        user_emb = user_embeddings[user_idx].reshape(1, -1)
        scores = cosine_similarity(user_emb, item_embeddings)[0]
        top_indices = np.argsort(scores)[::-1]

        predicted_items = [idx_to_item.get(idx, "") for idx in top_indices if idx in idx_to_item]

        for k in EVAL_K_VALUES:
            results[k]["recall"].append(compute_recall_at_k(actual_items, predicted_items, k))
            results[k]["ndcg"].append(compute_ndcg_at_k(actual_items, predicted_items, k))

        n_evaluated += 1
        if n_evaluated >= 1000:
            break

    metrics = {"model": model_name}
    for k in EVAL_K_VALUES:
        metrics[f"recall@{k}"] = np.mean(results[k]["recall"])
        metrics[f"ndcg@{k}"] = np.mean(results[k]["ndcg"])

    print(f"    Evaluated {n_evaluated} users")
    for k in EVAL_K_VALUES:
        print(f"    Recall@{k}: {metrics[f'recall@{k}']:.4f}  |  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")

    return metrics


def compare_retrieval_models():
    print("=" * 60)
    print("RETRIEVAL MODEL COMPARISON")
    print("=" * 60)

    interactions = pd.read_parquet(DATA_PROCESSED_DIR / "interactions.parquet")
    save_dir = MODELS_DIR / "retrieval"
    embed_dir = save_dir / "embeddings"
    all_metrics = []

    model_configs = [
        ("ALS", "als_user_embeddings.npy", "als_item_embeddings.npy", "als_mappings.pkl"),
        ("Item2Vec", "item2vec_user_embeddings.npy", "item2vec_item_embeddings.npy", "item2vec_mappings.pkl"),
        ("Two-Tower", "two_tower_user_embeddings.npy", "two_tower_item_embeddings.npy", "two_tower_mappings.pkl"),
    ]

    for name, user_file, item_file, map_file in model_configs:
        user_path = embed_dir / user_file
        item_path = embed_dir / item_file
        map_path = save_dir / map_file

        if not all(p.exists() for p in [user_path, item_path, map_path]):
            print(f"\n  ⚠️ Skipping {name} (files not found)")
            continue

        user_emb = np.load(user_path)
        item_emb = np.load(item_path)
        mappings = joblib.load(map_path)

        metrics = evaluate_retrieval_model(user_emb, item_emb, mappings, interactions, name)
        all_metrics.append(metrics)

    if not all_metrics:
        print("No models found!")
        return

    comparison = pd.DataFrame(all_metrics)
    print("\n📊 Retrieval Model Comparison:")
    print(comparison.to_string(index=False))

    # Determine best model
    best_idx = comparison[f"ndcg@{EVAL_K_VALUES[-1]}"].idxmax()
    best_model = comparison.loc[best_idx, "model"]
    print(f"\n🏆 Best Retrieval Model: {best_model}")

    # Save comparison
    comparison.to_csv(save_dir / "retrieval_comparison.csv", index=False)
    with open(save_dir / "best_model.txt", "w") as f:
        f.write(best_model)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(comparison))
    width = 0.25
    for i, k in enumerate(EVAL_K_VALUES):
        axes[0].bar(x + i * width, comparison[f"recall@{k}"], width, label=f"Recall@{k}")
        axes[1].bar(x + i * width, comparison[f"ndcg@{k}"], width, label=f"NDCG@{k}")

    for ax, title in zip(axes, ["Recall@K", "NDCG@K"]):
        ax.set_xticks(x + width)
        ax.set_xticklabels(comparison["model"])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "retrieval_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Comparison saved to: {save_dir}")
    return comparison


if __name__ == "__main__":
    compare_retrieval_models()
