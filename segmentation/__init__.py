"""Customer segmentation: turn churn risk + CLV + RFM into actionable segments."""

from segmentation.segment_users import SEGMENT_PLAYBOOKS, run_segmentation

__all__ = ["SEGMENT_PLAYBOOKS", "run_segmentation"]
