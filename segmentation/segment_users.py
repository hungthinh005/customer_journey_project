"""
Customer Segmentation.

Combines the churn model output (risk level + predicted CLV) with RFM /
behavioral features to assign every customer to an *actionable* segment and a
retention "playbook". The AI agent branches on the segment + playbook to decide
how aggressively to intervene and what tone/offer to use.

    python segmentation/segment_users.py

Reads from and writes to the Postgres serving store.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.init_db import init_db
from db.models import ChurnPrediction, CustomerFeature, Segment
from db.session import session_scope

# segment -> (playbook, base_priority, human description)
SEGMENT_PLAYBOOKS = {
    "high_value_at_risk": ("win_back_premium", 100, "High spend, elevated churn risk - protect aggressively"),
    "loyal_active": ("reward_loyalty", 40, "Healthy, engaged, high value - reward and upsell"),
    "new_promising": ("nurture_onboarding", 60, "Recently acquired with momentum - nurture the habit"),
    "price_sensitive_at_risk": ("value_offers", 70, "At risk and price-driven - lead with value/bundles"),
    "dormant_low_value": ("light_touch_winback", 30, "Low value and likely gone - cheap, low-effort win-back"),
    "standard": ("standard_recommendations", 10, "Stable mid-tier - standard personalized recs"),
}


def _load_frame() -> pd.DataFrame:
    """Join churn predictions with customer features into one frame."""
    with session_scope() as session:
        rows = (
            session.query(
                ChurnPrediction.customer_id,
                ChurnPrediction.churn_probability,
                ChurnPrediction.churn_risk_level,
                ChurnPrediction.predicted_clv,
                CustomerFeature.monetary,
                CustomerFeature.frequency,
                CustomerFeature.recency,
                CustomerFeature.days_as_customer,
                CustomerFeature.return_rate,
                CustomerFeature.product_diversity,
            )
            .outerjoin(CustomerFeature, CustomerFeature.customer_id == ChurnPrediction.customer_id)
            .all()
        )
    cols = [
        "customer_id", "churn_probability", "churn_risk_level", "predicted_clv",
        "monetary", "frequency", "recency", "days_as_customer",
        "return_rate", "product_diversity",
    ]
    return pd.DataFrame(rows, columns=cols)


def _assign_segment(row, value_threshold, new_customer_days, price_sensitive_return_rate):
    risk = (row["churn_risk_level"] or "LOW").upper()
    # Use predicted CLV when available, else fall back to historical monetary value.
    value = row["predicted_clv"] if pd.notna(row["predicted_clv"]) and row["predicted_clv"] > 0 else row["monetary"]
    value = value or 0.0
    is_high_value = value >= value_threshold
    days = row["days_as_customer"] or 0
    freq = row["frequency"] or 0
    return_rate = row["return_rate"] or 0.0

    at_risk = risk in ("HIGH", "MEDIUM")

    # New customers with some traction get nurtured first.
    if days <= new_customer_days and freq >= 2:
        return "new_promising"
    if at_risk and is_high_value:
        return "high_value_at_risk"
    if at_risk and return_rate >= price_sensitive_return_rate:
        return "price_sensitive_at_risk"
    if risk == "HIGH" and not is_high_value:
        return "dormant_low_value"
    if risk == "LOW" and is_high_value:
        return "loyal_active"
    return "standard"


def run_segmentation(value_percentile: float = 0.70, new_customer_days: int = 60):
    print("=" * 60)
    print("CUSTOMER SEGMENTATION")
    print("=" * 60)
    init_db()

    df = _load_frame()
    if df.empty:
        print("  [WARN] No churn predictions in DB. Run the pipeline + db/load_to_db.py first.")
        return {}

    value_series = df["predicted_clv"].where(df["predicted_clv"].fillna(0) > 0, df["monetary"]).fillna(0)
    value_threshold = float(value_series.quantile(value_percentile))
    # A small positive return rate already signals price/quality sensitivity.
    price_sensitive_return_rate = max(0.05, float(df["return_rate"].fillna(0).quantile(0.75)))

    print(f"  Customers: {len(df):,}")
    print(f"  Value threshold (p{int(value_percentile * 100)}): {value_threshold:,.2f}")

    rows = []
    for _, r in df.iterrows():
        seg = _assign_segment(r, value_threshold, new_customer_days, price_sensitive_return_rate)
        playbook, priority, desc = SEGMENT_PLAYBOOKS[seg]
        # Bump priority for higher churn probability so the agent acts on the
        # most urgent customers first when batch capacity is limited.
        churn_boost = int((r["churn_probability"] or 0) * 20)
        rows.append({
            "customer_id": int(r["customer_id"]),
            "segment": seg,
            "playbook": playbook,
            "priority": priority + churn_boost,
            "rationale": f"{desc}. risk={r['churn_risk_level']}, churn_p={r['churn_probability']:.2f}.",
            "updated_at": datetime.utcnow(),
        })

    counts = pd.Series([x["segment"] for x in rows]).value_counts().to_dict()

    with session_scope() as session:
        table = Segment.__table__
        for start in range(0, len(rows), 1000):
            batch = rows[start : start + 1000]
            stmt = insert(table).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["customer_id"],
                set_={
                    "segment": stmt.excluded.segment,
                    "playbook": stmt.excluded.playbook,
                    "priority": stmt.excluded.priority,
                    "rationale": stmt.excluded.rationale,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            session.execute(stmt)

    print("\n  Segment distribution:")
    for seg, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {seg:<26} {n:>8,}")
    print("\n[OK] Segments written to DB.")
    return counts


if __name__ == "__main__":
    run_segmentation()
