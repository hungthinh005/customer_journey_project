"""
Engagement pipeline DAG (daily).

Refresh scores into the serving store -> segment customers -> for each segment
run the AI agent over the highest-priority customers (dynamic task mapping) ->
the agent retrieves history + recommendations, composes a personalized message,
and sends an email -> summarize the run.

Params (trigger with config):
  per_segment_limit : max customers to contact per segment (default 25)
  drift_check       : run the Evidently drift report task (default True)
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_DIR = "/opt/airflow/project"

default_args = {
    "owner": "growth",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="engagement_pipeline",
    description="Daily segmentation + per-customer AI retention outreach",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["engagement", "agent"],
    params={
        "per_segment_limit": Param(25, type="integer"),
        "drift_check": Param(True, type="boolean"),
    },
)
def engagement_pipeline():

    @task
    def load_scores() -> int:
        """Load latest features + churn predictions into the serving DB."""
        import sys

        sys.path.insert(0, PROJECT_DIR)
        from db.load_to_db import main as load_main

        load_main()
        return 1

    @task
    def segment_customers(_loaded: int) -> list:
        """Assign segments and return the list of segments that have customers."""
        import sys

        sys.path.insert(0, PROJECT_DIR)
        from segmentation.segment_users import run_segmentation

        counts = run_segmentation()
        # Order segments by total priority interest (high-value/at-risk first).
        ordered = [
            "high_value_at_risk",
            "price_sensitive_at_risk",
            "new_promising",
            "dormant_low_value",
            "loyal_active",
            "standard",
        ]
        return [s for s in ordered if counts.get(s, 0) > 0]

    @task
    def engage(segment_name: str) -> dict:
        """Run the retention agent across one segment's top-priority customers."""
        import sys

        sys.path.insert(0, PROJECT_DIR)
        from airflow.operators.python import get_current_context
        from agent.run_agent import run_for_segment

        context = get_current_context()
        limit = int(context["params"]["per_segment_limit"])
        run_id = context["run_id"]
        results = run_for_segment(segment_name, limit=limit, run_id=run_id)
        sent = sum(1 for r in results if r.get("status") == "sent")
        return {"segment": segment_name, "processed": len(results), "sent": sent}

    @task
    def summarize(results: list) -> dict:
        total_processed = sum(r["processed"] for r in results)
        total_sent = sum(r["sent"] for r in results)
        print("=" * 50)
        print("ENGAGEMENT RUN SUMMARY")
        for r in results:
            print(f"  {r['segment']:<26} processed={r['processed']:>4} sent={r['sent']:>4}")
        print(f"  TOTAL processed={total_processed} sent={total_sent}")
        return {"processed": total_processed, "sent": total_sent}

    @task.branch
    def should_check_drift() -> str:
        from airflow.operators.python import get_current_context

        context = get_current_context()
        return "drift_report" if context["params"]["drift_check"] else "skip_drift"

    @task
    def drift_report() -> str:
        import sys

        sys.path.insert(0, PROJECT_DIR)
        from monitoring.drift_report import run_drift_report

        return run_drift_report()

    @task
    def skip_drift() -> str:
        return "drift check skipped"

    loaded = load_scores()
    segments = segment_customers(loaded)
    engaged = engage.expand(segment_name=segments)
    summary = summarize(engaged)

    branch = should_check_drift()
    summary >> branch
    branch >> [drift_report(), skip_drift()]


engagement_pipeline()
