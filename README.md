# Churn Prevention System

An end-to-end ML system combining **Churn Prediction** and **Recommendation System** for customer retention.

## 🎯 Project Objective

Build a production-ready system that:
1. **Predicts churn** using BG/NBD and Survival Analysis models
2. **Recommends products** using ALS, Item2Vec, and Two-Tower retrieval models
3. **Ranks recommendations** with a churn-aware NeuMF model
4. **Optionally reranks** using LLM for semantic refinement
5. **Serves predictions** via a FastAPI REST API

## 📊 Dataset

**Online Retail II (UCI)** — Real-world e-commerce transaction data from a UK-based retailer.
- Source: [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

## 🏗️ Architecture

```
┌─────────────────── OFFLINE PIPELINE ───────────────────┐
│                                                         │
│  Raw Data → Cleaning → Feature Engineering              │
│                 │                                       │
│     ┌───────────┼───────────┐                          │
│     ▼           ▼           ▼                          │
│  BG/NBD    Survival     Retrieval Models               │
│  Model      Analysis    (ALS/Item2Vec/Two-Tower)       │
│     │           │           │                          │
│     ▼           ▼           ▼                          │
│  Churn     Model        FAISS Index                    │
│  Scores    Comparison   + Embeddings                   │
│     │                       │                          │
│     └───────┬───────────────┘                          │
│             ▼                                          │
│      NeuMF Ranking Model                               │
└─────────────────────────────────────────────────────────┘

┌─────────────────── ONLINE PIPELINE ────────────────────┐
│                                                         │
│  API Request (customer_id)                              │
│       │                                                │
│       ▼                                                │
│  Churn Scoring ──► FAISS Retrieval ──► NeuMF Ranking   │
│       │                                    │           │
│       │                             LLM Reranker       │
│       │                                    │           │
│       └────────────┬───────────────────────┘           │
│                    ▼                                   │
│            Decision Layer                              │
│     (Retention Action + Top-K Products)                │
│                    │                                   │
│                    ▼                                   │
│             API Response                               │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Download Dataset
```bash
python download_data.py
```

### 2. Run Full Pipeline
You can run the entire training and evaluation pipeline using the provided script:
```bash
# Local execution
bash models/train_all.sh
```

Or

### 2. Docker (Recommended)
The easiest way to run the system is using Docker Compose.

**Train all models**
```bash
docker-compose run --rm churn-prevention-api bash models/train_all.sh
```

### 3. Start the API
```bash
docker-compose up
```

## 📡 API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict churn + get recommendations
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 13085, "top_k": 10}'

#or
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict" -Headers @{"Content-Type"="application/json"} -Body '{"customer_id": 13085, "top_k": 10}'
```

## 📈 Evaluation Metrics

### Churn Model
- AUC-ROC, Precision, Recall, F1-Score

### Recommendation
- Recall@K, NDCG@K

### Ablation Study
- Churn only vs Churn+Retrieval vs Full pipeline vs LLM

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Churn Models | lifetimes (BG/NBD), lifelines (Survival) |
| Retrieval | implicit (ALS), gensim (Item2Vec), PyTorch (Two-Tower) |
| FAISS | faiss-cpu |
| Ranking | PyTorch (NeuMF) |
| LLM Reranker | OpenAI API |
| API | FastAPI + Uvicorn |
| Deployment | Docker |
| CI/CD | GitHub Actions |

## 📁 Project Structure

```
├── config.py                 # Central configuration
├── download_data.py          # Dataset download
├── features/                 # Feature engineering
├── models/
│   ├── churn/               # BG/NBD + Survival Analysis
│   ├── retrieval/           # ALS, Item2Vec, Two-Tower
│   ├── ranking/             # NeuMF
│   └── reranker/            # LLM reranker
├── faiss_index/             # FAISS ANN index
├── api/                     # FastAPI serving
├── evaluation/              # Metrics & ablation study
├── tests/                   # Unit tests
├── .github/workflows/       # CI/CD pipeline
├── Dockerfile               # Docker image
└── docker-compose.yml       # Docker Compose
```
