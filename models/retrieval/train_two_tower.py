"""
Two-Tower Retrieval Model (PyTorch).

Separate user and item towers learning embeddings in shared space.
Similarity computed via dot product.
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
    EMBEDDING_DIM, TWO_TOWER_BATCH_SIZE, TWO_TOWER_EPOCHS,
    TWO_TOWER_HIDDEN_DIM, TWO_TOWER_LR, DATA_PROCESSED_DIR,
    MODELS_DIR, RANDOM_SEED,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InteractionDataset(Dataset):
    def __init__(self, interactions, n_users, n_items, n_negatives=4):
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.user_indices = interactions["user_idx"].values
        self.item_indices = interactions["item_idx"].values
        self.positive_pairs = set(zip(self.user_indices, self.item_indices))

    def __len__(self):
        return len(self.user_indices) * (1 + self.n_negatives)

    def __getitem__(self, idx):
        n_pos = len(self.user_indices)
        if idx < n_pos:
            return (torch.tensor(self.user_indices[idx], dtype=torch.long),
                    torch.tensor(self.item_indices[idx], dtype=torch.long),
                    torch.tensor(1.0, dtype=torch.float))
        else:
            pos_idx = idx % n_pos
            user = self.user_indices[pos_idx]
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if (user, neg_item) not in self.positive_pairs:
                    break
            return (torch.tensor(user, dtype=torch.long),
                    torch.tensor(neg_item, dtype=torch.long),
                    torch.tensor(0.0, dtype=torch.float))


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, hidden_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim), nn.LayerNorm(embedding_dim),
        )
        self.item_embedding = nn.Embedding(n_items, hidden_dim)
        self.item_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim), nn.LayerNorm(embedding_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_user_embedding(self, user_idx):
        return self.user_tower(self.user_embedding(user_idx))

    def get_item_embedding(self, item_idx):
        return self.item_tower(self.item_embedding(item_idx))

    def forward(self, user_idx, item_idx):
        user_emb = self.get_user_embedding(user_idx)
        item_emb = self.get_item_embedding(item_idx)
        return (user_emb * item_emb).sum(dim=1)


def train_two_tower():
    print("=" * 60)
    print("TWO-TOWER RETRIEVAL MODEL")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    interactions = pd.read_parquet(DATA_PROCESSED_DIR / "interactions.parquet")
    user_ids = interactions["customer_id"].unique()
    item_ids = interactions["stock_code"].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    n_users, n_items = len(user_ids), len(item_ids)
    interactions["user_idx"] = interactions["customer_id"].map(user_to_idx)
    interactions["item_idx"] = interactions["stock_code"].map(item_to_idx)
    print(f"  Users: {n_users:,}, Items: {n_items:,}")

    dataset = InteractionDataset(interactions, n_users, n_items, n_negatives=4)
    dataloader = DataLoader(dataset, batch_size=TWO_TOWER_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    model = TwoTowerModel(n_users, n_items, EMBEDDING_DIM, TWO_TOWER_HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=TWO_TOWER_LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(TWO_TOWER_EPOCHS):
        model.train()
        total_loss, n_batches = 0, 0
        for user_idx, item_idx, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{TWO_TOWER_EPOCHS}", leave=False):
            user_idx, item_idx, label = user_idx.to(DEVICE), item_idx.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(user_idx, item_idx), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/max(n_batches,1):.4f}")

    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = [], []
        for i in range(0, n_users, 4096):
            batch = torch.arange(i, min(i+4096, n_users), device=DEVICE)
            user_embeddings.append(model.get_user_embedding(batch).cpu().numpy())
        for i in range(0, n_items, 4096):
            batch = torch.arange(i, min(i+4096, n_items), device=DEVICE)
            item_embeddings.append(model.get_item_embedding(batch).cpu().numpy())
        user_embeddings = np.concatenate(user_embeddings)
        item_embeddings = np.concatenate(item_embeddings)

    save_dir = MODELS_DIR / "retrieval"
    embed_dir = save_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "two_tower_model.pt")
    np.save(embed_dir / "two_tower_user_embeddings.npy", user_embeddings)
    np.save(embed_dir / "two_tower_item_embeddings.npy", item_embeddings)
    mappings = {"user_to_idx": user_to_idx, "item_to_idx": item_to_idx,
                "idx_to_user": {v: k for k, v in user_to_idx.items()},
                "idx_to_item": {v: k for k, v in item_to_idx.items()},
                "n_users": n_users, "n_items": n_items}
    joblib.dump(mappings, save_dir / "two_tower_mappings.pkl")
    print(f"\n✅ Two-Tower model saved to: {save_dir}")
    return model, user_embeddings, item_embeddings, mappings


if __name__ == "__main__":
    train_two_tower()
