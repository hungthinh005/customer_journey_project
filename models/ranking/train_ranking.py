"""
NeuMF (Neural Matrix Factorization) Ranking Model.

Combines GMF (Generalized Matrix Factorization) and MLP paths.
Churn-aware: incorporates churn probability as an additional feature
to prioritize retention-focused recommendations.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    DATA_PROCESSED_DIR,
    EMBEDDING_DIM,
    MODELS_DIR,
    RANDOM_SEED,
    RANKING_BATCH_SIZE,
    RANKING_EPOCHS,
    RANKING_GMF_DIM,
    RANKING_LR,
    RANKING_MLP_DIMS,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RankingDataset(Dataset):
    """Dataset with user embedding, item embedding, churn prob, and label."""

    def __init__(self, user_embs, item_embs, churn_probs, labels):
        self.user_embs = torch.FloatTensor(user_embs)
        self.item_embs = torch.FloatTensor(item_embs)
        self.churn_probs = torch.FloatTensor(churn_probs)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_embs[idx], self.item_embs[idx], self.churn_probs[idx], self.labels[idx]


class NeuMF(nn.Module):
    """Neural Matrix Factorization with churn awareness."""

    def __init__(self, embedding_dim, mlp_dims, churn_feature_dim=1):
        super().__init__()
        self.embedding_dim = embedding_dim

        # GMF path
        self.gmf_user = nn.Linear(embedding_dim, RANKING_GMF_DIM)
        self.gmf_item = nn.Linear(embedding_dim, RANKING_GMF_DIM)

        # MLP path (concatenate user + item + churn)
        mlp_input = embedding_dim * 2 + churn_feature_dim
        layers = []
        prev_dim = mlp_input
        for dim in mlp_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = dim
        self.mlp = nn.Sequential(*layers)

        # Final prediction
        self.output = nn.Sequential(
            nn.Linear(RANKING_GMF_DIM + mlp_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_emb, item_emb, churn_prob):
        # GMF
        gmf_user = self.gmf_user(user_emb)
        gmf_item = self.gmf_item(item_emb)
        gmf_out = gmf_user * gmf_item

        # MLP
        mlp_input = torch.cat([user_emb, item_emb, churn_prob.unsqueeze(1)], dim=1)
        mlp_out = self.mlp(mlp_input)

        # Combine
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output(combined).squeeze(1)


def prepare_ranking_data():
    """Prepare training data for ranking model."""
    print("  Loading embeddings and churn predictions...")
    embed_dir = MODELS_DIR / "retrieval" / "embeddings"
    churn_dir = MODELS_DIR / "churn"

    # Determine best retrieval model
    best_file = MODELS_DIR / "retrieval" / "best_model.txt"
    prefix = "als"
    if best_file.exists():
        model_name = best_file.read_text().strip()
        prefix = {"ALS": "als", "Item2Vec": "item2vec", "Two-Tower": "two_tower"}.get(model_name, "als")

    user_embs = np.load(embed_dir / f"{prefix}_user_embeddings.npy")
    item_embs = np.load(embed_dir / f"{prefix}_item_embeddings.npy")
    mappings = joblib.load(MODELS_DIR / "retrieval" / f"{prefix}_mappings.pkl")

    # Load churn probabilities
    churn_preds = None
    for f in ["bgnbd_predictions.parquet", "survival_predictions.parquet"]:
        fp = churn_dir / f
        if fp.exists():
            churn_preds = pd.read_parquet(fp)
            break

    # Load interactions
    interactions = pd.read_parquet(DATA_PROCESSED_DIR / "interactions.parquet")

    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]

    # Build training data with positive and negative samples
    train_user_embs, train_item_embs, train_churn, train_labels = [], [], [], []

    for _, row in interactions.iterrows():
        uid = row["customer_id"]
        iid = row["stock_code"]
        if uid not in user_to_idx or iid not in item_to_idx:
            continue
        u_idx = user_to_idx[uid]
        i_idx = item_to_idx[iid]
        if u_idx >= len(user_embs) or i_idx >= len(item_embs):
            continue

        # Get churn probability
        churn_prob = 0.5
        if churn_preds is not None:
            churn_col = [c for c in churn_preds.columns if "churn_prob" in c]
            if churn_col:
                if uid in churn_preds.index:
                    churn_prob = churn_preds.loc[uid, churn_col[0]]
                elif "customer_id" in churn_preds.columns:
                    match = churn_preds[churn_preds["customer_id"] == uid]
                    if len(match) > 0:
                        churn_prob = match[churn_col[0]].values[0]

        # Positive sample
        train_user_embs.append(user_embs[u_idx])
        train_item_embs.append(item_embs[i_idx])
        train_churn.append(float(churn_prob))
        train_labels.append(1.0)

        # Negative sample
        neg_idx = np.random.randint(0, len(item_embs))
        train_user_embs.append(user_embs[u_idx])
        train_item_embs.append(item_embs[neg_idx])
        train_churn.append(float(churn_prob))
        train_labels.append(0.0)

    return (np.array(train_user_embs), np.array(train_item_embs), np.array(train_churn), np.array(train_labels))


def train_ranking():
    print("=" * 60)
    print("NeuMF RANKING MODEL")
    print("=" * 60)

    print("\n[1/3] Preparing ranking data...")
    user_embs, item_embs, churn_probs, labels = prepare_ranking_data()
    print(f"  Training samples: {len(labels):,}")
    print(f"  Positive ratio: {labels.mean():.2%}")

    dataset = RankingDataset(user_embs, item_embs, churn_probs, labels)
    dataloader = DataLoader(dataset, batch_size=RANKING_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    print("\n[2/3] Training NeuMF model...")
    model = NeuMF(EMBEDDING_DIM, RANKING_MLP_DIMS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=RANKING_LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(RANKING_EPOCHS):
        model.train()
        total_loss, n_batches = 0, 0
        for u, i, c, label in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{RANKING_EPOCHS}", leave=False):
            u, i, c, label = u.to(DEVICE), i.to(DEVICE), c.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(u, i, c), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}: Loss = {total_loss / max(n_batches, 1):.4f}")

    print("\n[3/3] Saving model...")
    save_dir = MODELS_DIR / "ranking"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "ranking_model.pt")
    joblib.dump({"embedding_dim": EMBEDDING_DIM, "mlp_dims": RANKING_MLP_DIMS}, save_dir / "ranking_config.pkl")
    print(f"\n✅ Ranking model saved to: {save_dir}")
    return model


if __name__ == "__main__":
    train_ranking()
