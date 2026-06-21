"""
LLM-based Reranker for Recommendation Refinement.

Takes the Top-K ranked recommendations and uses an LLM to semantically
rerank them based on user behavior summary and product metadata.
Falls back to original ranking if LLM is unavailable.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED_DIR, LLM_MODEL, LLM_RERANK_TOP_K, LLM_TEMPERATURE, OPENAI_API_KEY


def build_user_summary(customer_id, customer_features, interactions):
    """Build a text summary of user behavior for the LLM."""
    user = customer_features[customer_features["customer_id"] == customer_id]
    if len(user) == 0:
        return "Unknown customer."

    user = user.iloc[0]
    user_items = interactions[interactions["customer_id"] == customer_id]

    summary = f"""Customer #{customer_id}:
- Total spending: ${user.get("monetary", 0):.2f}
- Number of purchases: {user.get("frequency", 0)}
- Days since last purchase: {user.get("recency", "N/A")}
- Average basket size: {user.get("avg_basket_size", "N/A"):.1f} items
- Product diversity: {user.get("product_diversity", "N/A"):.2f}
- Purchase interval: {user.get("avg_purchase_interval", "N/A"):.1f} days
- Days as customer: {user.get("days_as_customer", "N/A")}
- Most purchased products: {", ".join(user_items.nlargest(5, "total_quantity")["stock_code"].tolist()) if len(user_items) > 0 else "N/A"}"""

    return summary


def build_product_info(item_ids, item_metadata):
    """Build product metadata text for the LLM."""
    products = []
    for item_id in item_ids:
        meta = item_metadata[item_metadata["stock_code"] == item_id]
        if len(meta) > 0:
            meta = meta.iloc[0]
            products.append(
                f"- {item_id}: {meta.get('description', 'N/A')} "
                f"(avg price: ${meta.get('avg_price', 0):.2f}, "
                f"sold to {meta.get('n_customers', 0)} customers)"
            )
        else:
            products.append(f"- {item_id}: No metadata available")
    return "\n".join(products)


def rerank_with_llm(
    customer_id,
    candidate_items,
    candidate_scores,
    customer_features,
    item_metadata,
    interactions,
    churn_probability=None,
):
    """
    Use LLM to rerank candidate items based on user context.

    Returns reranked item list and reasoning.
    """
    if not OPENAI_API_KEY:
        print("  ⚠️ No OpenAI API key set. Using original ranking.")
        return candidate_items, candidate_scores, "LLM unavailable - using model ranking"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        print("  ⚠️ openai package not installed. Using original ranking.")
        return candidate_items, candidate_scores, "OpenAI package not available"

    # Build context
    user_summary = build_user_summary(customer_id, customer_features, interactions)
    product_info = build_product_info(candidate_items[:LLM_RERANK_TOP_K], item_metadata)

    churn_context = ""
    if churn_probability is not None:
        risk_level = "HIGH" if churn_probability > 0.7 else "MEDIUM" if churn_probability > 0.4 else "LOW"
        churn_context = f"\nChurn Risk: {risk_level} (probability: {churn_probability:.2f})"

    prompt = f"""You are a recommendation system optimizer for an e-commerce retailer.
Your goal is to rerank product recommendations to maximize customer retention.

{user_summary}
{churn_context}

Current recommended products (ranked by model score):
{product_info}

Please rerank these products to maximize the chance of customer retention.
Consider:
1. Products similar to their purchase history
2. If churn risk is HIGH, prioritize products with proven retention impact
3. Product diversity to re-engage the customer
4. Complementary products to their favorites

Return your response as JSON with:
- "ranked_items": list of stock codes in your recommended order
- "reasoning": brief explanation of your reranking strategy"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        reranked = result.get("ranked_items", candidate_items)
        reasoning = result.get("reasoning", "No reasoning provided")

        # Ensure all items are included
        remaining = [i for i in candidate_items if i not in reranked]
        reranked = reranked + remaining

        return reranked, candidate_scores, reasoning

    except Exception as e:
        print(f"  ⚠️ LLM reranking failed: {e}")
        return candidate_items, candidate_scores, f"Error: {str(e)}"


def rerank_batch(customer_ids, candidates_dict, scores_dict, churn_probs=None):
    """Rerank recommendations for a batch of customers."""
    print("=" * 60)
    print("LLM RERANKER")
    print("=" * 60)

    # Load metadata
    customer_features = pd.read_parquet(DATA_PROCESSED_DIR / "customer_features.parquet")
    item_metadata = pd.read_parquet(DATA_PROCESSED_DIR / "item_metadata.parquet")
    interactions = pd.read_parquet(DATA_PROCESSED_DIR / "interactions.parquet")

    results = {}
    for cid in customer_ids:
        items = candidates_dict.get(cid, [])
        scores = scores_dict.get(cid, [])
        churn_prob = churn_probs.get(cid, 0.5) if churn_probs else 0.5

        reranked_items, reranked_scores, reasoning = rerank_with_llm(
            cid, items, scores, customer_features, item_metadata, interactions, churn_prob
        )

        results[cid] = {
            "original_items": items,
            "reranked_items": reranked_items,
            "reasoning": reasoning,
            "churn_probability": churn_prob,
        }
        print(f"  Customer {cid}: {reasoning[:80]}...")

    print(f"\n✅ Reranked {len(results)} customers")
    return results


if __name__ == "__main__":
    print("LLM Reranker module loaded. Use rerank_with_llm() or rerank_batch().")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")
