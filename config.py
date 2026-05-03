"""
Centralized configuration for the Churn Prevention System.
All paths, hyperparameters, and settings are defined here.
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FAISS_DIR = PROJECT_ROOT / "faiss_index"

# Ensure directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FAISS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset
# ============================================================
DATASET_NAME = "mashlyn/online-retail-ii-uci"
RAW_DATA_FILE = DATA_RAW_DIR / "online_retail_II.xlsx"

# ============================================================
# Churn Definition
# ============================================================
CHURN_WINDOW_DAYS = 90  # Customer is churned if no purchase in this window
OBSERVATION_END = None  # Will be set dynamically from data (max InvoiceDate)

# Time-based split ratios
# Training: everything before SPLIT_TRAIN_END
# Validation: SPLIT_TRAIN_END to SPLIT_VAL_END
# Test label window: SPLIT_VAL_END + CHURN_WINDOW_DAYS
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ============================================================
# Feature Engineering
# ============================================================
RFM_FEATURES = ["recency", "frequency", "monetary"]
BEHAVIORAL_FEATURES = [
    "avg_basket_size",
    "avg_purchase_interval",
    "product_diversity",
    "avg_quantity_per_txn",
    "return_rate",
    "days_as_customer",
]
ALL_FEATURES = RFM_FEATURES + BEHAVIORAL_FEATURES

# ============================================================
# Churn Model Hyperparameters
# ============================================================
# BG/NBD + Gamma-Gamma
BGNBD_PENALIZER = 0.001

# Survival Analysis (Cox Proportional Hazards)
COX_PENALIZER = 0.01
COX_L1_RATIO = 0.0

# ============================================================
# Retrieval Model Hyperparameters
# ============================================================
EMBEDDING_DIM = 64

# ALS
ALS_FACTORS = EMBEDDING_DIM
ALS_ITERATIONS = 50
ALS_REGULARIZATION = 0.01

# Item2Vec
ITEM2VEC_WINDOW = 5
ITEM2VEC_MIN_COUNT = 5
ITEM2VEC_EPOCHS = 30

# Two-Tower
TWO_TOWER_LR = 1e-3
TWO_TOWER_EPOCHS = 30
TWO_TOWER_BATCH_SIZE = 1024
TWO_TOWER_HIDDEN_DIM = 128

# ============================================================
# FAISS
# ============================================================
FAISS_TOP_N = 100  # Number of candidates to retrieve
FAISS_NPROBE = 10

# ============================================================
# Ranking Model (NeuMF)
# ============================================================
RANKING_LR = 1e-3
RANKING_EPOCHS = 30
RANKING_BATCH_SIZE = 512
RANKING_MLP_DIMS = [128, 64, 32]
RANKING_GMF_DIM = EMBEDDING_DIM
RANKING_TOP_K = 10

# ============================================================
# LLM Reranker
# ============================================================
LLM_MODEL = "gpt-4o-mini"  # Can be changed to any OpenAI-compatible model
LLM_TEMPERATURE = 0.3
LLM_RERANK_TOP_K = 10
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================
# API
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000

# ============================================================
# Evaluation
# ============================================================
EVAL_K_VALUES = [5, 10, 20]

# ============================================================
# Random Seed
# ============================================================
RANDOM_SEED = 42
