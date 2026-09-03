"""
Gating layer. The model NEVER autonomously blocks a merchant or a payment
outright - it only ever recommends bounded actions, each of which has an
explicit ceiling. This is what makes the agent "explainable, bounded and
gated" rather than a black box that silently declines revenue.

Actions (increasing severity):
  MONITOR        - log only, no friction added
  STEP_UP_AUTH   - request additional verification (OTP/3DS) on next txn
  SOFT_HOLD      - hold settlement for the window's txns pending review (capped ceiling)
  ESCALATE_HUMAN - page a human risk analyst; auto-actions stop here

Hard bounds (never overridden by the model):
  - SOFT_HOLD can only be auto-applied if the window's total exposure <= MAX_AUTO_HOLD
  - Anything above that ALWAYS requires human escalation, regardless of score
  - No action is ever "auto-block a merchant" - that remains human-only
"""

from dataclasses import dataclass

MAX_AUTO_HOLD_INR = 50_000  # bound: model can never auto-hold more than this
REVIEW_COST_INR = 45        # avg analyst cost to review one flagged window
AVG_FRAUD_LOSS_INR = 6_500  # avg loss if a true fraud window is missed


@dataclass
class Decision:
    action: str
    reason: str
    requires_human: bool
    bounded_exposure: float


def decide(row) -> Decision:
    score = row["risk_score"]
    exposure = row["total_amount"]

    if score < 0.45:
        return Decision("MONITOR", "risk score below alert threshold", False, 0)

    if score < 0.65:
        return Decision("STEP_UP_AUTH", "moderate anomaly - add verification friction", False, exposure)

    if score < 0.82 and exposure <= MAX_AUTO_HOLD_INR:
        return Decision("SOFT_HOLD", "high anomaly within auto-hold bound - settlement paused pending review", True, exposure)

    # either very high score OR exposure exceeds the hard auto-bound -> human only
    return Decision("ESCALATE_HUMAN", "score/exposure exceeds auto-action bound - human analyst required", True, exposure)


def cost_summary(labelled_df) -> dict:
    """Business cost accounting - the honesty check the track explicitly asks for."""
    fp = ((labelled_df["predicted"] == 1) & (labelled_df["is_fraud"] == 0)).sum()
    fn = ((labelled_df["predicted"] == 0) & (labelled_df["is_fraud"] == 1)).sum()
    tp = ((labelled_df["predicted"] == 1) & (labelled_df["is_fraud"] == 1)).sum()
    tn = ((labelled_df["predicted"] == 0) & (labelled_df["is_fraud"] == 0)).sum()

    fp_cost = fp * REVIEW_COST_INR
    fn_cost = fn * AVG_FRAUD_LOSS_INR
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positive_review_cost_inr": int(fp_cost),
        "false_negative_loss_cost_inr": int(fn_cost),
        "total_cost_inr": int(fp_cost + fn_cost),
    }
