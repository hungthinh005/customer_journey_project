"""
Build FAISS index from item embeddings for approximate nearest neighbor search.
Uses the best retrieval model's embeddings.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAISS_DIR, FAISS_TOP_N, MODELS_DIR


def get_best_retrieval_model():
    """Read which retrieval model was selected as best."""
    best_file = MODELS_DIR / "retrieval" / "best_model.txt"
    if best_file.exists():
        return best_file.read_text().strip()
    return "ALS"  # Default fallback


def load_embeddings(model_name):
    """Load embeddings for the specified model."""
    embed_dir = MODELS_DIR / "retrieval" / "embeddings"
    prefix_map = {"ALS": "als", "Item2Vec": "item2vec", "Two-Tower": "two_tower"}
    prefix = prefix_map.get(model_name, "als")

    item_emb = np.load(embed_dir / f"{prefix}_item_embeddings.npy")
    user_emb = np.load(embed_dir / f"{prefix}_user_embeddings.npy")

    mappings_file = MODELS_DIR / "retrieval" / f"{prefix}_mappings.pkl"
    mappings = joblib.load(mappings_file)

    return user_emb, item_emb, mappings


def build_faiss_index(item_embeddings, use_ivf=False, nlist=100):
    """
    Build FAISS index from item embeddings.

    Args:
        item_embeddings: numpy array of shape (n_items, embedding_dim)
        use_ivf: whether to use IVF index for faster search (for large datasets)
        nlist: number of clusters for IVF index

    Returns:
        FAISS index
    """
    dim = item_embeddings.shape[1]

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(item_embeddings)

    if use_ivf and len(item_embeddings) > 10000:
        # IVF index for large datasets
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, len(item_embeddings) // 10))
        index.train(item_embeddings)
        index.add(item_embeddings)
        index.nprobe = 10
        print(f"  Built IVF index: {index.ntotal} vectors, {nlist} clusters")
    else:
        # Flat index (exact search, good for moderate datasets)
        index = faiss.IndexFlatIP(dim)
        index.add(item_embeddings)
        print(f"  Built Flat index: {index.ntotal} vectors")

    return index


def search_similar_items(index, query_embedding, top_n=None):
    """Search for similar items given a query embedding."""
    if top_n is None:
        top_n = FAISS_TOP_N

    query = query_embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)

    distances, indices = index.search(query, top_n)
    return distances[0], indices[0]


def build_index():
    """Main function to build and save FAISS index."""
    print("=" * 60)
    print("FAISS INDEX BUILDING")
    print("=" * 60)

    # Get best retrieval model
    best_model = get_best_retrieval_model()
    print(f"\nUsing embeddings from: {best_model}")

    # Load embeddings
    user_embeddings, item_embeddings, mappings = load_embeddings(best_model)
    item_embeddings = item_embeddings.astype(np.float32).copy()
    print(f"  Item embeddings shape: {item_embeddings.shape}")

    # Build index
    print("\nBuilding FAISS index...")
    index = build_faiss_index(item_embeddings)

    # Test search
    print("\nTesting search with first user embedding...")
    user_emb = user_embeddings[0].astype(np.float32)
    distances, indices = search_similar_items(index, user_emb, top_n=5)
    idx_to_item = mappings.get("idx_to_item", {})
    print(f"  Top-5 items: {[idx_to_item.get(i, i) for i in indices]}")
    print(f"  Scores: {distances}")

    # Save
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_DIR / "faiss_index.bin"))
    joblib.dump(mappings, FAISS_DIR / "faiss_mappings.pkl")

    # Save metadata
    metadata = {
        "retrieval_model": best_model,
        "n_items": index.ntotal,
        "embedding_dim": item_embeddings.shape[1],
    }
    joblib.dump(metadata, FAISS_DIR / "faiss_metadata.pkl")

    print(f"\n✅ FAISS index saved to: {FAISS_DIR}")
    return index, mappings


if __name__ == "__main__":
    build_index()
