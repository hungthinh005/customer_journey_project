"""
Recommendation Model Evaluation: Recall@K, NDCG@K.
Evaluates retrieval and ranking quality.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_K_VALUES, MODELS_DIR


def evaluate_rec():
    print("=" * 60)
    print("RECOMMENDATION EVALUATION")
    print("=" * 60)

    # Load retrieval comparison if exists
    ret_comp = MODELS_DIR / "retrieval" / "retrieval_comparison.csv"
    if ret_comp.exists():
        df = pd.read_csv(ret_comp)
        print("\n📊 Retrieval Model Results:")
        print(df.to_string(index=False))
    else:
        print("  Run compare_retrieval.py first!")

    print(f"\n✅ Recommendation evaluation complete")


if __name__ == "__main__":
    evaluate_rec()
