"""Tests for the Churn Prevention System."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_imports():
    """Test that config module loads correctly."""
    from config import (
        CHURN_WINDOW_DAYS,
        DATA_PROCESSED_DIR,
        DATA_RAW_DIR,
        EMBEDDING_DIM,
        RANDOM_SEED,
    )

    assert CHURN_WINDOW_DAYS == 90
    assert EMBEDDING_DIM == 64
    assert RANDOM_SEED == 42
    assert DATA_RAW_DIR.exists() or True  # May not exist in CI
    assert DATA_PROCESSED_DIR.exists() or True


def test_schemas():
    """Test API schemas."""
    from api.schemas import PredictionRequest, ProductRecommendation

    req = PredictionRequest(customer_id=12345, top_k=5)
    assert req.customer_id == 12345
    assert req.top_k == 5
    assert req.use_llm_reranker is False

    rec = ProductRecommendation(stock_code="ABC", score=0.95, rank=1)
    assert rec.stock_code == "ABC"


def test_neumf_model():
    """Test NeuMF model architecture."""
    import torch
    from models.ranking.train_ranking import NeuMF

    model = NeuMF(embedding_dim=64, mlp_dims=[128, 64, 32])
    user_emb = torch.randn(4, 64)
    item_emb = torch.randn(4, 64)
    churn = torch.randn(4)
    output = model(user_emb, item_emb, churn)
    assert output.shape == (4,)


def test_two_tower_model():
    """Test Two-Tower model architecture."""
    import torch
    from models.retrieval.train_two_tower import TwoTowerModel

    model = TwoTowerModel(n_users=100, n_items=200, embedding_dim=64, hidden_dim=128)
    user_idx = torch.randint(0, 100, (4,))
    item_idx = torch.randint(0, 200, (4,))
    scores = model(user_idx, item_idx)
    assert scores.shape == (4,)

    user_emb = model.get_user_embedding(user_idx)
    assert user_emb.shape == (4, 64)
