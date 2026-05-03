"""
Inference orchestrator for the Churn Prevention System.
Loads all models/artifacts at startup and orchestrates the full pipeline:
churn prediction → FAISS retrieval → NeuMF ranking → optional LLM reranking.
"""

import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED_DIR, EMBEDDING_DIM, FAISS_DIR, MODELS_DIR, RANKING_MLP_DIMS
from models.ranking.train_ranking import NeuMF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceEngine:
    """Orchestrates the full inference pipeline."""

    def __init__(self):
        self.loaded = {
            "churn_model": False, "retrieval": False,
            "faiss_index": False, "ranking_model": False,
        }
        self._load_models()

    def _load_models(self):
        """Load all model artifacts."""
        print("Loading models...")

        # 1. Load churn model (BG/NBD)
        try:
            churn_dir = MODELS_DIR / "churn"
            bgnbd_path = churn_dir / "bgnbd_model.pkl"
            if bgnbd_path.exists():
                self.churn_model = joblib.load(bgnbd_path)
                self.churn_type = "bgnbd"
            else:
                surv_path = churn_dir / "survival_best_model.pkl"
                self.churn_model = joblib.load(surv_path)
                self.churn_type = "survival"
            # Load precomputed churn predictions
            for f in ["bgnbd_predictions.parquet", "survival_predictions.parquet"]:
                fp = churn_dir / f
                if fp.exists():
                    self.churn_predictions = pd.read_parquet(fp)
                    break
            self.loaded["churn_model"] = True
            print("  [OK] Churn model loaded")
        except Exception as e:
            print(f"  [ERR] Churn model: {e}")

        # 2. Load retrieval embeddings
        try:
            best_file = MODELS_DIR / "retrieval" / "best_model.txt"
            prefix = "als"
            if best_file.exists():
                model_name = best_file.read_text().strip()
                prefix = {"ALS": "als", "Item2Vec": "item2vec", "Two-Tower": "two_tower"}.get(model_name, "als")

            embed_dir = MODELS_DIR / "retrieval" / "embeddings"
            self.user_embeddings = np.load(embed_dir / f"{prefix}_user_embeddings.npy")
            self.item_embeddings = np.load(embed_dir / f"{prefix}_item_embeddings.npy")
            self.retrieval_mappings = joblib.load(MODELS_DIR / "retrieval" / f"{prefix}_mappings.pkl")
            self.loaded["retrieval"] = True
            print(f"  [OK] Retrieval embeddings loaded ({prefix})")
        except Exception as e:
            print(f"  [ERR] Retrieval: {e}")

        # 3. Load FAISS index
        try:
            self.faiss_index = faiss.read_index(str(FAISS_DIR / "faiss_index.bin"))
            self.loaded["faiss_index"] = True
            print(f"  [OK] FAISS index loaded ({self.faiss_index.ntotal} items)")
        except Exception as e:
            print(f"  [ERR] FAISS: {e}")

        # 4. Load ranking model
        try:
            self.ranking_model = NeuMF(EMBEDDING_DIM, RANKING_MLP_DIMS).to(DEVICE)
            state = torch.load(MODELS_DIR / "ranking" / "ranking_model.pt", map_location=DEVICE)
            self.ranking_model.load_state_dict(state)
            self.ranking_model.eval()
            self.loaded["ranking_model"] = True
            print("  [OK] Ranking model loaded")
        except Exception as e:
            print(f"  [ERR] Ranking: {e}")

        # 5. Load item metadata
        try:
            self.item_metadata = pd.read_parquet(DATA_PROCESSED_DIR / "item_metadata.parquet")
        except Exception:
            self.item_metadata = pd.DataFrame()

        # 6. Load customer features (for BG/NBD summary data)
        try:
            self.bgnbd_summary = pd.read_parquet(DATA_PROCESSED_DIR / "bgnbd_summary.parquet")
        except Exception:
            self.bgnbd_summary = None

    def get_churn_prediction(self, customer_id):
        """Get churn probability for a customer."""
        result = {"churn_probability": 0.5, "p_alive": None, "predicted_clv": None}

        if not self.loaded["churn_model"]:
            return result

        if hasattr(self, "churn_predictions"):
            preds = self.churn_predictions
            if preds.index.name == "customer_id" or "customer_id" not in preds.columns:
                if customer_id in preds.index:
                    row = preds.loc[customer_id]
                    prob_col = [c for c in preds.columns if "churn_prob" in c]
                    if prob_col:
                        result["churn_probability"] = float(row[prob_col[0]])
                    if "p_alive" in preds.columns:
                        result["p_alive"] = float(row["p_alive"])
                    if "predicted_clv" in preds.columns:
                        result["predicted_clv"] = float(row["predicted_clv"])
            else:
                match = preds[preds["customer_id"] == customer_id]
                if len(match) > 0:
                    row = match.iloc[0]
                    prob_col = [c for c in preds.columns if "churn_prob" in c]
                    if prob_col:
                        result["churn_probability"] = float(row[prob_col[0]])
                    if "p_alive" in row:
                        result["p_alive"] = float(row.get("p_alive", 0))

        return result

    def get_candidates(self, customer_id, top_n=100):
        """Retrieve candidate items using FAISS."""
        if not (self.loaded["retrieval"] and self.loaded["faiss_index"]):
            return [], []

        user_to_idx = self.retrieval_mappings["user_to_idx"]
        if customer_id not in user_to_idx:
            return [], []

        user_idx = user_to_idx[customer_id]
        if user_idx >= len(self.user_embeddings):
            return [], []

        user_emb = self.user_embeddings[user_idx].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(user_emb)

        distances, indices = self.faiss_index.search(user_emb, top_n)

        idx_to_item = self.retrieval_mappings.get("idx_to_item", {})
        items = [idx_to_item.get(int(i), str(i)) for i in indices[0] if i >= 0]
        scores = distances[0][:len(items)].tolist()

        return items, scores

    def rank_candidates(self, customer_id, candidate_items, churn_prob, top_k=10):
        """Rank candidates using NeuMF."""
        if not self.loaded["ranking_model"] or not candidate_items:
            return candidate_items[:top_k], list(range(len(candidate_items[:top_k])))

        user_to_idx = self.retrieval_mappings["user_to_idx"]
        item_to_idx = self.retrieval_mappings["item_to_idx"]

        if customer_id not in user_to_idx:
            return candidate_items[:top_k], [0.0] * min(top_k, len(candidate_items))

        user_idx = user_to_idx[customer_id]
        user_emb = self.user_embeddings[user_idx]

        scores = []
        valid_items = []
        for item_id in candidate_items:
            if item_id in item_to_idx:
                i_idx = item_to_idx[item_id]
                if i_idx < len(self.item_embeddings):
                    valid_items.append(item_id)
                    item_emb = self.item_embeddings[i_idx]

                    with torch.no_grad():
                        u = torch.FloatTensor(user_emb).unsqueeze(0).to(DEVICE)
                        i = torch.FloatTensor(item_emb).unsqueeze(0).to(DEVICE)
                        c = torch.FloatTensor([churn_prob]).to(DEVICE)
                        score = self.ranking_model(u, i, c).item()
                    scores.append(score)

        if not scores:
            return candidate_items[:top_k], [0.0] * min(top_k, len(candidate_items))

        ranked = sorted(zip(valid_items, scores), key=lambda x: x[1], reverse=True)
        ranked_items = [x[0] for x in ranked[:top_k]]
        ranked_scores = [x[1] for x in ranked[:top_k]]

        return ranked_items, ranked_scores

    def get_item_description(self, stock_code):
        """Get item description from metadata."""
        if self.item_metadata.empty:
            return None
        match = self.item_metadata[self.item_metadata["stock_code"] == stock_code]
        return match.iloc[0]["description"] if len(match) > 0 else None

    def predict(self, customer_id, top_k=10, use_llm=False):
        """Full inference pipeline."""
        # 1. Churn prediction
        churn_result = self.get_churn_prediction(customer_id)
        churn_prob = churn_result["churn_probability"]

        # 2. Retrieve candidates
        candidates, retrieval_scores = self.get_candidates(customer_id)

        # 3. Rank candidates
        ranked_items, ranked_scores = self.rank_candidates(customer_id, candidates, churn_prob, top_k)

        # 4. Optional LLM reranking
        if use_llm:
            try:
                from models.reranker.llm_reranker import rerank_with_llm
                customer_features = pd.read_parquet(DATA_PROCESSED_DIR / "customer_features.parquet")
                interactions = pd.read_parquet(DATA_PROCESSED_DIR / "interactions.parquet")
                ranked_items, ranked_scores, _ = rerank_with_llm(
                    customer_id, ranked_items, ranked_scores,
                    customer_features, self.item_metadata, interactions, churn_prob,
                )
            except Exception as e:
                print(f"LLM reranking failed: {e}")

        # 5. Build response
        risk_level = "HIGH" if churn_prob > 0.7 else "MEDIUM" if churn_prob > 0.4 else "LOW"

        if risk_level == "HIGH":
            action = "Send personalized discount + top product recommendations"
        elif risk_level == "MEDIUM":
            action = "Send product recommendations + engagement email"
        else:
            action = "Standard product recommendations"

        recommendations = []
        for rank, (item, score) in enumerate(zip(ranked_items, ranked_scores), 1):
            recommendations.append({
                "stock_code": str(item),
                "description": self.get_item_description(str(item)),
                "score": float(score),
                "rank": rank,
            })

        return {
            "customer_id": customer_id,
            "churn_probability": churn_prob,
            "churn_risk_level": risk_level,
            "p_alive": churn_result.get("p_alive"),
            "predicted_clv": churn_result.get("predicted_clv"),
            "recommendations": recommendations,
            "retention_action": action,
            "model_info": self.loaded,
        }
