"""
Data / prediction drift monitoring with Evidently.

Compares the current customer-feature distribution (live, from the serving DB)
against the training reference (customer_features.parquet) and writes an HTML
report. In production this would feed an alert + retraining trigger.

    python monitoring/drift_report.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ALL_FEATURES, DATA_PROCESSED_DIR, PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "monitoring" / "reports"


def _load_reference():
    import pandas as pd
    path = DATA_PROCESSED_DIR / "customer_features.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _load_current():
    import pandas as pd
    from db.session import engine
    try:
        return pd.read_sql("SELECT * FROM customer_features", engine)
    except Exception as e:
        print(f"  [WARN] could not read current features from DB: {e}")
        return None


def run_drift_report() -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference = _load_reference()
    current = _load_current()

    if reference is None or current is None or current.empty:
        msg = "drift report skipped (missing reference or current data)"
        print(f"  [SKIP] {msg}")
        return msg

    features = [c for c in ALL_FEATURES if c in reference.columns and c in current.columns]
    reference = reference[features].fillna(0)
    current = current[features].fillna(0)

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        out = REPORT_DIR / f"drift_{datetime.utcnow():%Y%m%d_%H%M%S}.html"
        report.save_html(str(out))

        result = report.as_dict()
        drift = result["metrics"][0]["result"]
        share = drift.get("share_of_drifted_columns", 0)
        n_drifted = drift.get("number_of_drifted_columns", 0)
        msg = f"drift report saved: {out.name} | drifted columns={n_drifted} share={share:.2f}"
        print(f"  [OK] {msg}")
        return msg
    except Exception as e:  # noqa: BLE001
        msg = f"evidently report failed: {e}"
        print(f"  [WARN] {msg}")
        return msg


if __name__ == "__main__":
    run_drift_report()
