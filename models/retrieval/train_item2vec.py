"""
Item2Vec Retrieval Model.

Applies Word2Vec on purchase sequences to learn item embeddings.
Each customer's purchase history (ordered by time) is treated as a "sentence",
and each product is a "word". The model learns semantic item representations.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    EMBEDDING_DIM,
    ITEM2VEC_EPOCHS,
    ITEM2VEC_MIN_COUNT,
    ITEM2VEC_WINDOW,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)


def build_purchase_sequences(transactions):
    """
    Build purchase sequences from transaction data.
    Each customer's purchases are ordered chronologically to form a "sentence".
    """
    # Sort by customer and date
    transactions = transactions.sort_values(["customer_id", "invoice_date"])

    # Group by customer, get list of stock codes in order
    sequences = transactions.groupby("customer_id")["stock_code"].apply(list).reset_index()
    sequences.columns = ["customer_id", "sequence"]

    # Filter: only keep sequences with at least 2 items
    sequences = sequences[sequences["sequence"].apply(len) >= 2]

    print(f"  Purchase sequences: {len(sequences):,}")
    print(f"  Avg sequence length: {sequences['sequence'].apply(len).mean():.1f}")
    print(f"  Max sequence length: {sequences['sequence'].apply(len).max()}")

    return sequences


def train_item2vec():
    """Train Item2Vec model using Word2Vec on purchase sequences."""
    print("=" * 60)
    print("ITEM2VEC RETRIEVAL MODEL")
    print("=" * 60)

    # Load transactions
    transactions = pd.read_parquet(DATA_PROCESSED_DIR / "transactions_clean.parquet")
    print(f"\nLoaded {len(transactions):,} transactions")

    # Build purchase sequences
    print("\n[1/3] Building purchase sequences...")
    sequences = build_purchase_sequences(transactions)

    # Train Word2Vec (Item2Vec)
    print("\n[2/3] Training Item2Vec model...")
    sentence_list = sequences["sequence"].tolist()

    model = Word2Vec(
        sentences=sentence_list,
        vector_size=EMBEDDING_DIM,
        window=ITEM2VEC_WINDOW,
        min_count=ITEM2VEC_MIN_COUNT,
        epochs=ITEM2VEC_EPOCHS,
        sg=1,  # Skip-gram
        workers=4,
        seed=RANDOM_SEED,
    )

    print(f"  Vocabulary size: {len(model.wv):,}")
    print(f"  Embedding dim: {model.wv.vector_size}")

    # Extract item embeddings
    item_ids = list(model.wv.index_to_key)
    item_embeddings = np.array([model.wv[item] for item in item_ids])

    # Create item ID to index mapping
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}

    # Build user embeddings by averaging item embeddings in their purchase history
    print("\n  Building user embeddings from item averages...")
    user_embeddings_dict = {}
    for _, row in sequences.iterrows():
        user_id = row["customer_id"]
        items = [item for item in row["sequence"] if item in item_to_idx]
        if items:
            user_vec = np.mean([model.wv[item] for item in items], axis=0)
            user_embeddings_dict[user_id] = user_vec

    user_ids = list(user_embeddings_dict.keys())
    user_embeddings = np.array([user_embeddings_dict[uid] for uid in user_ids])
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    print(f"  User embeddings: {user_embeddings.shape}")
    print(f"  Item embeddings: {item_embeddings.shape}")

    # Save
    print("\n[3/3] Saving model and embeddings...")
    save_dir = MODELS_DIR / "retrieval"
    embed_dir = save_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(save_dir / "item2vec_model.model"))
    np.save(embed_dir / "item2vec_user_embeddings.npy", user_embeddings)
    np.save(embed_dir / "item2vec_item_embeddings.npy", item_embeddings)

    mappings = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": {v: k for k, v in user_to_idx.items()},
        "idx_to_item": {v: k for k, v in item_to_idx.items()},
    }
    joblib.dump(mappings, save_dir / "item2vec_mappings.pkl")

    print(f"\n✅ Item2Vec model saved to: {save_dir}")
    return model, user_embeddings, item_embeddings, mappings


if __name__ == "__main__":
    train_item2vec()
