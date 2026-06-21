"""
Load pipeline parquet outputs into the Postgres serving store.

Run after the training/feature pipeline so the API and agent read fresh data
from the database (single source of truth) instead of from parquet files.

    python db/load_to_db.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED_DIR, MODELS_DIR
from db.init_db import init_db
from db.models import ChurnPrediction, CustomerFeature
from db.session import session_scope


def _risk_level(prob: float) -> str:
    return "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"


def _synth_email(customer_id: int) -> str:
    """Synthesize a deterministic contact email (caught by MailHog locally)."""
    return f"customer{int(customer_id)}@example.com"


def _upsert(session, model, rows, index_elements):
    """Bulk upsert rows (list of dicts) using Postgres ON CONFLICT."""
    if not rows:
        return 0
    table = model.__table__
    batch_size = 1000
    total = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        stmt = insert(table).values(batch)
        update_cols = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in index_elements}
        stmt = stmt.on_conflict_do_update(index_elements=index_elements, set_=update_cols)
        session.execute(stmt)
        total += len(batch)
    return total


def load_customer_features():
    path = DATA_PROCESSED_DIR / "customer_features.parquet"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return 0

    df = pd.read_parquet(path)
    cols = [
        "recency",
        "frequency",
        "monetary",
        "avg_basket_size",
        "avg_purchase_interval",
        "product_diversity",
        "avg_quantity_per_txn",
        "return_rate",
        "days_as_customer",
    ]
    rows = []
    for _, r in df.iterrows():
        cid = int(r["customer_id"])
        row = {"customer_id": cid, "email": _synth_email(cid), "updated_at": datetime.utcnow()}
        for c in cols:
            if c in df.columns and pd.notna(r[c]):
                row[c] = float(r[c])
        rows.append(row)

    with session_scope() as session:
        n = _upsert(session, CustomerFeature, rows, ["customer_id"])
    print(f"  [OK] customer_features: {n:,} rows")
    return n


def load_churn_predictions(model_version="latest"):
    # Prefer BG/NBD, fall back to survival predictions.
    candidates = [
        ("bgnbd", MODELS_DIR / "churn" / "bgnbd_predictions.parquet"),
        ("survival", MODELS_DIR / "churn" / "survival_predictions.parquet"),
    ]
    path = next((p for _, p in candidates if p.exists()), None)
    if path is None:
        print("  [SKIP] no churn predictions parquet found")
        return 0

    df = pd.read_parquet(path)
    if df.index.name != "customer_id" and "customer_id" not in df.columns:
        df.index.name = "customer_id"
    df = df.reset_index()

    prob_col = next((c for c in df.columns if "churn_prob" in c), None)
    rows = []
    for _, r in df.iterrows():
        cid = int(r["customer_id"])
        prob = float(r[prob_col]) if prob_col and pd.notna(r[prob_col]) else 0.5
        rows.append(
            {
                "customer_id": cid,
                "churn_probability": prob,
                "p_alive": float(r["p_alive"]) if "p_alive" in df.columns and pd.notna(r.get("p_alive")) else None,
                "predicted_clv": float(r["predicted_clv"])
                if "predicted_clv" in df.columns and pd.notna(r.get("predicted_clv"))
                else None,
                "churn_risk_level": _risk_level(prob),
                "model_version": model_version,
                "scored_at": datetime.utcnow(),
            }
        )

    with session_scope() as session:
        n = _upsert(session, ChurnPrediction, rows, ["customer_id"])
    print(f"  [OK] churn_predictions: {n:,} rows")
    return n


def main():
    print("=" * 60)
    print("LOADING PARQUET -> POSTGRES")
    print("=" * 60)
    init_db()
    load_customer_features()
    load_churn_predictions()
    print("\nDone.")


if __name__ == "__main__":
    main()
