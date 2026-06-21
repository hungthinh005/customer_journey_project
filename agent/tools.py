"""
Tools the retention agent can call.

Each tool is also exposed as a plain function (``_fn`` suffix-free name in the
``PLAIN_TOOLS`` map) so the deterministic fallback path can gather the same
context without invoking the LLM.
"""

import functools
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED_DIR
from db import repository
from settings import settings


@functools.lru_cache(maxsize=1)
def _interactions():
    import pandas as pd

    path = DATA_PROCESSED_DIR / "interactions.parquet"
    return pd.read_parquet(path) if path.exists() else None


@functools.lru_cache(maxsize=1)
def _item_metadata():
    import pandas as pd

    path = DATA_PROCESSED_DIR / "item_metadata.parquet"
    return pd.read_parquet(path) if path.exists() else None


# ---- Plain implementations (used by both tools and the fallback) ----


def profile(customer_id: int) -> dict:
    return repository.get_customer_features(int(customer_id)) or {"customer_id": int(customer_id)}


def churn_and_clv(customer_id: int) -> dict:
    return repository.get_churn_prediction(int(customer_id)) or {
        "churn_probability": 0.5,
        "churn_risk_level": "MEDIUM",
        "p_alive": None,
        "predicted_clv": None,
    }


def segment(customer_id: int) -> dict:
    return repository.get_segment(int(customer_id)) or {"segment": "standard", "playbook": "standard_recommendations"}


def purchase_history(customer_id: int, top_n: int = 10) -> list:
    df = _interactions()
    if df is None:
        return []
    user = df[df["customer_id"] == int(customer_id)]
    if user.empty:
        return []
    top = user.nlargest(top_n, "total_quantity")
    meta = _item_metadata()
    out = []
    for _, r in top.iterrows():
        desc = None
        if meta is not None:
            m = meta[meta["stock_code"] == r["stock_code"]]
            if not m.empty:
                desc = m.iloc[0]["description"]
        out.append(
            {
                "stock_code": str(r["stock_code"]),
                "description": desc,
                "n_purchases": int(r["n_purchases"]),
                "total_quantity": int(r["total_quantity"]),
            }
        )
    return out


def recommendations(customer_id: int, top_k: int = 10) -> list:
    """Fetch model recommendations from the API service."""
    try:
        resp = requests.post(
            f"{settings.api_base_url}/predict",
            json={"customer_id": int(customer_id), "top_k": int(top_k)},
            headers={"x-api-key": settings.api_key},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("recommendations", [])
    except Exception:
        return []


PLAIN_TOOLS = {
    "get_customer_profile": profile,
    "get_churn_and_clv": churn_and_clv,
    "get_segment": segment,
    "get_purchase_history": purchase_history,
    "get_recommendations": recommendations,
}


def build_langchain_tools():
    """Construct LangChain tool objects (imported lazily to avoid hard dep)."""
    from langchain_core.tools import tool

    @tool
    def get_customer_profile(customer_id: int) -> str:
        """Get a customer's RFM and behavioral features and their contact email."""
        return json.dumps(profile(customer_id), default=str)

    @tool
    def get_churn_and_clv(customer_id: int) -> str:
        """Get the customer's churn probability, risk level, and predicted lifetime value."""
        return json.dumps(churn_and_clv(customer_id), default=str)

    @tool
    def get_segment(customer_id: int) -> str:
        """Get the customer's segment and recommended retention playbook."""
        return json.dumps(segment(customer_id), default=str)

    @tool
    def get_purchase_history(customer_id: int, top_n: int = 10) -> str:
        """Get the customer's most-purchased products with descriptions."""
        return json.dumps(purchase_history(customer_id, top_n), default=str)

    @tool
    def get_recommendations(customer_id: int, top_k: int = 10) -> str:
        """Get model-ranked product recommendations to feature in outreach."""
        return json.dumps(recommendations(customer_id, top_k), default=str)

    return [
        get_customer_profile,
        get_churn_and_clv,
        get_segment,
        get_purchase_history,
        get_recommendations,
    ]
