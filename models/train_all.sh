#!/bin/bash
echo "Step 1: Feature Engineering..."
python features/feature_engineering.py

echo "Step 2: Training Churn Model 1 (BG/NBD)..."
python models/churn/train_bgnbd.py

echo "Training Model 2..."
python models/churn/train_survival.py

echo "Training Model 3..."
python models/churn/compare_churn.py

echo "Training Model 4..."
python models/retrieval/train_als.py

echo "Training Model 5..."
python models/retrieval/train_item2vec.py

echo "Training Model 6  ..."
python models/retrieval/train_two_tower.py

echo "Training Model 7..."
python models/retrieval/compare_retrieval.py

echo "Training Model 8..."
python faiss_index/build_index.py

echo "Training Model 9..."
python models/ranking/train_ranking.py

echo "Evaluating the Models..."
python evaluation/evaluate_churn.py
python evaluation/evaluate_rec.py
python evaluation/ablation_study.py

echo "All models trained successfully and ready to serve "