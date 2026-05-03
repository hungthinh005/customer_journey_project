"""
ALS (Alternating Least Squares) Retrieval Model.

Uses the implicit library to learn user and item embeddings
from implicit feedback (purchase interactions).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ALS_FACTORS,
    ALS_ITERATIONS,
    ALS_REGULARIZATION,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)


def build_interaction_matrix(interactions):
    """Build sparse user-item interaction matrix."""
    # Create mappings
    user_ids = interactions["customer_id"].unique()
    item_ids = interactions["stock_code"].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

    # Build sparse matrix
    rows = interactions["customer_id"].map(user_to_idx).values
    cols = interactions["stock_code"].map(item_to_idx).values
    values = interactions["rating"].values

    interaction_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )

    print(f"  Interaction matrix: {interaction_matrix.shape}")
    print(f"  Non-zero entries: {interaction_matrix.nnz:,}")
    print(f"  Sparsity: {1 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4%}")

    mappings = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
    }

    return interaction_matrix, mappings


def train_als():
    """Train ALS model and extract embeddings."""
    print("=" * 60)
    print("ALS RETRIEVAL MODEL")
    print("=" * 60)

    # Load interactions
    interactions = pd.read_parquet(DATA_PROCESSED_DIR / "interactions.parquet")
    print(f"\nLoaded {len(interactions):,} interactions")

    # Build interaction matrix
    print("\n[1/3] Building interaction matrix...")
    interaction_matrix, mappings = build_interaction_matrix(interactions)

    # Train ALS
    print("\n[2/3] Training ALS model...")
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION,
        random_state=RANDOM_SEED,
    )
    model.fit(interaction_matrix)

    # Extract embeddings
    user_embeddings = model.user_factors
    item_embeddings = model.item_factors

    print(f"  User embeddings shape: {user_embeddings.shape}")
    print(f"  Item embeddings shape: {item_embeddings.shape}")

    # Save
    print("\n[3/3] Saving model and embeddings...")
    save_dir = MODELS_DIR / "retrieval"
    embed_dir = save_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_dir / "als_model.pkl")
    np.save(embed_dir / "als_user_embeddings.npy", user_embeddings)
    np.save(embed_dir / "als_item_embeddings.npy", item_embeddings)
    joblib.dump(mappings, save_dir / "als_mappings.pkl")
    joblib.dump(interaction_matrix, save_dir / "interaction_matrix.pkl")

    print(f"\n✅ ALS model saved to: {save_dir}")
    return model, user_embeddings, item_embeddings, mappings


if __name__ == "__main__":
    train_als()
