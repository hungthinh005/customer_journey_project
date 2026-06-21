"""Tests for the production upgrade: settings, segmentation, agent, DAGs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_settings_defaults():
    from settings import settings
    assert settings.agent_max_discount <= 1.0
    assert settings.smtp_port > 0
    assert "postgresql" in settings.database_url


def test_segment_assignment_high_value_at_risk():
    from segmentation.segment_users import _assign_segment

    row = {
        "churn_risk_level": "HIGH", "predicted_clv": 5000.0, "monetary": 4000.0,
        "days_as_customer": 400, "frequency": 20, "return_rate": 0.0,
    }
    seg = _assign_segment(row, value_threshold=1000.0, new_customer_days=60,
                          price_sensitive_return_rate=0.1)
    assert seg == "high_value_at_risk"


def test_segment_assignment_new_promising():
    from segmentation.segment_users import _assign_segment

    row = {
        "churn_risk_level": "MEDIUM", "predicted_clv": 50.0, "monetary": 50.0,
        "days_as_customer": 20, "frequency": 3, "return_rate": 0.0,
    }
    seg = _assign_segment(row, value_threshold=1000.0, new_customer_days=60,
                          price_sensitive_return_rate=0.1)
    assert seg == "new_promising"


def test_segment_assignment_loyal_active():
    from segmentation.segment_users import _assign_segment

    row = {
        "churn_risk_level": "LOW", "predicted_clv": 9000.0, "monetary": 9000.0,
        "days_as_customer": 500, "frequency": 30, "return_rate": 0.0,
    }
    seg = _assign_segment(row, value_threshold=1000.0, new_customer_days=60,
                          price_sensitive_return_rate=0.1)
    assert seg == "loyal_active"


def test_agent_guardrails_clamp_discount():
    from agent.run_agent import _apply_guardrails

    plan = {
        "offer": {"type": "discount", "discount_pct": 90, "description": "huge"},
        "recommended_items": ["AAA", "ZZZ"],
    }
    out = _apply_guardrails(plan, allowed_items={"AAA", "BBB"})
    # Discount clamped to the configured maximum (default 20%).
    assert out["offer"]["discount_pct"] <= 20
    # Hallucinated item ZZZ removed; only retrieved items kept.
    assert out["recommended_items"] == ["AAA"]


def test_agent_fallback_plan_structure():
    from agent.run_agent import _fallback_plan

    ctx = {
        "recommendations": [{"stock_code": "85123A", "description": "WHITE HANGING HEART"}],
        "history": [],
        "churn": {"churn_risk_level": "HIGH"},
    }
    plan = _fallback_plan(123, "high_value_at_risk", "win_back_premium", ctx)
    for key in ("strategy", "subject", "body", "offer", "recommended_items", "reasoning"):
        assert key in plan
    assert plan["offer"]["type"] == "discount"
    assert "85123A" in plan["recommended_items"]


def test_agent_json_extraction():
    from agent.graph import _extract_json

    raw = 'Here is the plan:\n```json\n{"subject": "Hi", "offer": {"type": "none"}}\n```'
    parsed = _extract_json(raw)
    assert parsed["subject"] == "Hi"


def test_dags_import():
    """DAG files should parse without errors (requires Airflow)."""
    pytest.importorskip("airflow")
    from airflow.models import DagBag

    dags_dir = Path(__file__).parent.parent / "dags"
    dag_bag = DagBag(dag_folder=str(dags_dir), include_examples=False)
    assert dag_bag.import_errors == {}, f"DAG import errors: {dag_bag.import_errors}"
    assert "training_pipeline" in dag_bag.dags
    assert "engagement_pipeline" in dag_bag.dags
