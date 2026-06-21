"""Read helpers over the serving store, shared by the API and the AI agent."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.models import ChurnPrediction, CustomerFeature, Notification, Segment
from db.session import session_scope


def get_customer_features(customer_id: int):
    with session_scope() as session:
        row = session.get(CustomerFeature, customer_id)
        if row is None:
            return None
        return {
            "customer_id": row.customer_id,
            "email": row.email,
            "recency": row.recency,
            "frequency": row.frequency,
            "monetary": row.monetary,
            "avg_basket_size": row.avg_basket_size,
            "avg_purchase_interval": row.avg_purchase_interval,
            "product_diversity": row.product_diversity,
            "avg_quantity_per_txn": row.avg_quantity_per_txn,
            "return_rate": row.return_rate,
            "days_as_customer": row.days_as_customer,
        }


def get_churn_prediction(customer_id: int):
    with session_scope() as session:
        row = session.get(ChurnPrediction, customer_id)
        if row is None:
            return None
        return {
            "churn_probability": row.churn_probability,
            "p_alive": row.p_alive,
            "predicted_clv": row.predicted_clv,
            "churn_risk_level": row.churn_risk_level,
            "model_version": row.model_version,
        }


def get_segment(customer_id: int):
    with session_scope() as session:
        row = session.get(Segment, customer_id)
        if row is None:
            return None
        return {
            "segment": row.segment,
            "playbook": row.playbook,
            "priority": row.priority,
            "rationale": row.rationale,
        }


def list_customers_by_segment(segment: str = None, limit: int = 1000):
    with session_scope() as session:
        q = session.query(Segment.customer_id, Segment.segment, Segment.priority)
        if segment:
            q = q.filter(Segment.segment == segment)
        q = q.order_by(Segment.priority.desc()).limit(limit)
        return [{"customer_id": c, "segment": s, "priority": p} for c, s, p in q.all()]


def was_recently_contacted(customer_id: int, within_days: int) -> bool:
    """Frequency-capping check: was a notification sent recently?"""
    cutoff = datetime.utcnow() - timedelta(days=within_days)
    with session_scope() as session:
        n = (
            session.query(Notification)
            .filter(
                Notification.customer_id == customer_id,
                Notification.status == "sent",
                Notification.sent_at >= cutoff,
            )
            .count()
        )
        return n > 0
