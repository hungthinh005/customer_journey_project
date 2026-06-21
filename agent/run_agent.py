"""
Agent orchestrator.

For a customer: enforce frequency capping, run the LangGraph agent (or a
deterministic fallback), apply offer guardrails, persist the full trace
(agent_runs + recommendations + notifications), and send the email.
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import agent.notifier as notifier
import agent.tools as tools
from agent.graph import compose_with_agent, llm_available
from db.models import AgentRun, Notification, Recommendation
from db.repository import list_customers_by_segment, was_recently_contacted
from db.session import session_scope
from segmentation.segment_users import SEGMENT_PLAYBOOKS
from settings import settings


def _new_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8]


def _fallback_plan(customer_id, segment, playbook, ctx):
    """Deterministic, no-LLM outreach plan (also the safety net on LLM failure)."""
    recs = ctx["recommendations"][:5] or ctx["history"][:5]
    item_lines = [f"- {r.get('description') or r['stock_code']}" for r in recs] or ["- A selection of our best-sellers"]

    discount_pct = 0
    offer_type = "none"
    offer_desc = "Personalized picks for you"
    if playbook in ("win_back_premium", "value_offers"):
        discount_pct = int(settings.agent_max_discount * 100)
        offer_type = "discount"
        offer_desc = f"{discount_pct}% off your next order"

    risk = ctx["churn"].get("churn_risk_level", "MEDIUM")
    subject = "We picked something for you" if risk == "LOW" else "We'd love to see you back"
    body = (
        "Hi there,\n\n"
        "Thanks for being our customer. Based on what you've enjoyed before, "
        "we thought you might like:\n"
        + "\n".join(item_lines)
        + (f"\n\nAs a thank-you, enjoy {offer_desc.lower()}." if offer_type != "none" else "")
        + "\n\nHappy shopping,\nThe Customer Journey Team"
    )
    return {
        "strategy": f"{playbook} (deterministic fallback)",
        "subject": subject,
        "body": body,
        "offer": {"type": offer_type, "discount_pct": discount_pct, "description": offer_desc},
        "recommended_items": [r["stock_code"] for r in recs],
        "reasoning": "Fallback template (LLM unavailable or failed).",
        "_tools_used": ["fallback"],
    }


def _apply_guardrails(plan, allowed_items):
    """Clamp discounts and restrict recommended items to retrieved ones."""
    notes = []
    max_pct = int(settings.agent_max_discount * 100)
    offer = plan.get("offer") or {}
    try:
        pct = float(offer.get("discount_pct", 0) or 0)
    except (TypeError, ValueError):
        pct = 0
    if pct > max_pct:
        notes.append(f"discount clamped {pct:.0f}->{max_pct}")
        offer["discount_pct"] = max_pct
        pct = max_pct
    if pct <= 0 and offer.get("type") == "discount":
        offer["type"] = "none"
    plan["offer"] = offer

    items = [str(i) for i in (plan.get("recommended_items") or []) if str(i) in allowed_items]
    if not items:
        items = list(allowed_items)[:5]
        notes.append("recommended_items replaced with retrieved items")
    plan["recommended_items"] = items
    plan["_guardrail_notes"] = notes
    return plan


def run_for_customer(customer_id: int, run_id: str = None) -> dict:
    run_id = run_id or _new_run_id()
    customer_id = int(customer_id)

    seg = tools.segment(customer_id)
    segment_name = seg.get("segment", "standard")
    playbook = seg.get("playbook") or SEGMENT_PLAYBOOKS.get(segment_name, ("standard_recommendations",))[0]

    # ---- Guardrail: frequency capping ----
    if was_recently_contacted(customer_id, settings.agent_frequency_cap_days):
        with session_scope() as session:
            session.add(
                AgentRun(
                    run_id=run_id,
                    customer_id=customer_id,
                    segment=segment_name,
                    status="skipped_frequency_cap",
                    strategy="skipped",
                    reasoning=f"Contacted within {settings.agent_frequency_cap_days} days.",
                )
            )
        return {"customer_id": customer_id, "status": "skipped_frequency_cap", "run_id": run_id}

    # ---- Gather context (also feeds the fallback) ----
    ctx = {
        "profile": tools.profile(customer_id),
        "churn": tools.churn_and_clv(customer_id),
        "history": tools.purchase_history(customer_id),
        "recommendations": tools.recommendations(customer_id, top_k=10),
    }
    allowed_items = {r["stock_code"] for r in ctx["recommendations"]} | {r["stock_code"] for r in ctx["history"]}

    # ---- Compose (agent or fallback) ----
    used_agent = False
    if llm_available():
        try:
            plan = compose_with_agent(customer_id, segment_name, playbook, settings.agent_max_discount * 100)
            used_agent = True
        except Exception as e:
            plan = _fallback_plan(customer_id, segment_name, playbook, ctx)
            plan["reasoning"] += f" (agent error: {e})"
    else:
        plan = _fallback_plan(customer_id, segment_name, playbook, ctx)

    plan = _apply_guardrails(plan, allowed_items)

    # ---- Persist trace + recommendations ----
    to_address = ctx["profile"].get("email") or f"customer{customer_id}@example.com"
    with session_scope() as session:
        session.add(
            AgentRun(
                run_id=run_id,
                customer_id=customer_id,
                segment=segment_name,
                churn_probability=ctx["churn"].get("churn_probability"),
                strategy=plan.get("strategy"),
                reasoning=plan.get("reasoning"),
                tools_used=plan.get("_tools_used"),
                recommended_items=plan.get("recommended_items"),
                offer=plan.get("offer"),
                status="agent" if used_agent else "fallback",
            )
        )
        for rank, code in enumerate(plan.get("recommended_items", []), 1):
            desc = next((r.get("description") for r in ctx["recommendations"] if r["stock_code"] == code), None)
            session.add(
                Recommendation(
                    run_id=run_id,
                    customer_id=customer_id,
                    stock_code=code,
                    description=desc,
                    rank=rank,
                    score=None,
                )
            )

    # ---- Send notification ----
    send_result = notifier.send_email(to_address, plan["subject"], plan["body"])
    with session_scope() as session:
        notif = Notification(
            run_id=run_id,
            customer_id=customer_id,
            channel="email",
            to_address=to_address,
            subject=plan["subject"],
            body=plan["body"],
            offer=plan.get("offer"),
            status=send_result["status"],
            error=send_result.get("error"),
            sent_at=datetime.utcnow() if send_result["status"] == "sent" else None,
        )
        session.add(notif)

    return {
        "customer_id": customer_id,
        "run_id": run_id,
        "segment": segment_name,
        "used_agent": used_agent,
        "status": send_result["status"],
        "subject": plan["subject"],
        "offer": plan.get("offer"),
    }


def run_for_segment(segment_name: str = None, limit: int = 50, run_id: str = None) -> list:
    """Run the agent for a batch of customers in a segment (priority order)."""
    run_id = run_id or _new_run_id()
    customers = list_customers_by_segment(segment_name, limit=limit)
    results = []
    for c in customers:
        results.append(run_for_customer(c["customer_id"], run_id=run_id))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the retention agent.")
    parser.add_argument("--customer-id", type=int, help="Single customer to process")
    parser.add_argument("--segment", type=str, default=None, help="Segment to batch-process")
    parser.add_argument("--limit", type=int, default=20, help="Max customers for segment batch")
    args = parser.parse_args()

    if args.customer_id:
        print(run_for_customer(args.customer_id))
    else:
        out = run_for_segment(args.segment, limit=args.limit)
        print(f"Processed {len(out)} customers.")
        for r in out:
            print(f"  {r['customer_id']}: {r['status']} ({r.get('segment')})")
