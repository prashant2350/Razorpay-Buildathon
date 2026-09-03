"""
Live demo server for the Fraud-Spike Sentinel.

This is what turns the pipeline from "we ran a report once" into "here's a
live product": trains the detector once on startup (same synthetic data and
same code path as main.py), then exposes a /score endpoint that takes ONE
new transaction window's stats and returns a real-time risk score, bounded
action, and Claude-narrated explanation - the exact same logic main.py runs
in batch, just callable one request at a time.

Run:
    pip install fastapi uvicorn
    python3 server.py

Then open http://localhost:8000/demo for the click-to-fire UI, or POST
directly to http://localhost:8000/score - see the Pydantic model below for
the request shape.
"""

import sys
import os
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from data.synthetic_generator import generate, generate_multi_merchant, GenConfig
from detector.features import build_windows, FEATURE_COLUMNS
from detector.model import FraudSpikeDetector
from detector.rules_engine import decide
from detector.adaptive_threshold import AdaptiveThreshold
from evaluation.metrics import choose_threshold
from llm.explainer import explain, answer_question
from audit.audit_log import AuditLog
from webhooks.signature_verification import verify_webhook_signature, SignatureVerificationError

app = FastAPI(title="Fraud-Spike Sentinel — Live Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model_state = {}
_scored_windows = {}  # window_id -> record, so /feedback can look up what was predicted
audit = AuditLog(path=str(Path(__file__).resolve().parent / "outputs" / "live_audit_trail.jsonl"))


@app.on_event("startup")
def train_on_startup():
    """Trains once, on the same multi-merchant synthetic data main.py uses,
    so the live demo's decisions are consistent with the reported held-out
    metrics (see main.py's time_split - split per-merchant, not globally)."""
    print("Training detector for live demo (same data/code path as main.py)...")
    txns = generate_multi_merchant()
    feats = build_windows(txns)

    trains = []
    for merchant_id, group in feats.groupby("merchant_id"):
        group = group.sort_values("ts").reset_index(drop=True)
        n = len(group)
        trains.append(group.iloc[: int(n * 0.6)])
    train_df = pd.concat(trains, ignore_index=True)

    demo_merchant = feats[feats["merchant_id"] == "acc_merchant001"].sort_values("ts").reset_index(drop=True)
    n = len(demo_merchant)
    val_df = demo_merchant.iloc[int(n * 0.6): int(n * 0.8)]

    model = FraudSpikeDetector().fit(train_df)
    val_scored = model.score(val_df)
    threshold = choose_threshold(val_scored)

    _model_state["model"] = model
    _model_state["threshold"] = threshold
    _model_state["adaptive"] = AdaptiveThreshold(
        base_threshold=threshold,
        log_path=str(Path(__file__).resolve().parent / "outputs" / "threshold_adjustments.jsonl"),
    )
    _model_state["rolling_baseline"] = demo_merchant[["txn_count", "avg_amount"]].tail(48).agg(["mean", "std"]).to_dict()
    print(f"Ready. Threshold={threshold:.3f}")


class TransactionWindow(BaseModel):
    """One 15-min window's worth of stats - what a merchant's monitoring
    would aggregate in real time. Mirrors detector/features.py's FEATURE_COLUMNS."""
    merchant_id: str = "acc_merchant001"
    txn_count: int = Field(..., description="Number of transactions in this window")
    total_amount: float = Field(..., description="Total ₹ amount in this window")
    avg_amount: float = Field(..., description="Average ₹ ticket size")
    std_amount: float = Field(0.0, description="Std dev of ticket size")
    distinct_bins: int = Field(..., description="Distinct card IINs/BINs seen")
    distinct_countries: int = Field(1, description="Distinct countries seen")
    fail_rate: float = Field(0.0, description="Fraction of transactions that failed (0-1)")
    repeat_email_count: int = Field(0, description="How many prior high-value orders this same customer email placed in the last 30 days (refund abuse signal)")


@app.post("/score")
def score_transaction(window: TransactionWindow):
    if "model" not in _model_state:
        return {"error": "Model still training, try again in a few seconds."}

    if window.txn_count == 0:
        # An empty window (no activity) is not an anomaly - it's the
        # absence of a signal. Scoring it through the model would compute
        # nonsensical z-scores (dividing by a zero-transaction window's
        # own stats) and could produce a misleadingly high score. Nothing
        # happened, so nothing gets flagged - this is a MONITOR by
        # definition, not a model decision.
        return {
            "merchant_id": window.merchant_id,
            "window_start": str(pd.Timestamp.now()),
            "txn_count": 0,
            "total_amount_inr": 0.0,
            "risk_score": 0.0,
            "predicted_fraud": False,
            "deterministic_reasons": ["no transactions in this window - nothing to evaluate"],
            "action": "MONITOR",
            "requires_human": False,
            "bounded_exposure_inr": 0,
            "explanation": "Window had zero transactions - no activity to assess.",
            "check_first": None,
        }

    model = _model_state["model"]
    threshold = _model_state["adaptive"].get_threshold()
    baseline = _model_state["rolling_baseline"]

    bin_diversity = window.distinct_bins / max(window.txn_count, 1)
    txn_mean = baseline["txn_count"]["mean"]
    txn_std = baseline["txn_count"]["std"] or 1
    amt_mean = baseline["avg_amount"]["mean"]
    amt_std = baseline["avg_amount"]["std"] or 1

    row = pd.Series({
        "merchant_id": window.merchant_id,
        "ts": pd.Timestamp.now(),
        "txn_count": window.txn_count,
        "total_amount": window.total_amount,
        "avg_amount": window.avg_amount,
        "std_amount": window.std_amount,
        "distinct_bins": window.distinct_bins,
        "distinct_countries": window.distinct_countries,
        "fail_rate": window.fail_rate,
        "bin_diversity": bin_diversity,
        "txn_count_zscore": max(min((window.txn_count - txn_mean) / txn_std, 8), -8),
        "amount_zscore": max(min((window.avg_amount - amt_mean) / amt_std, 8), -8),
        "max_repeat_email_count": window.repeat_email_count,
    })

    scored_df = model.score(pd.DataFrame([row]))
    scored_row = scored_df.iloc[0]

    decision = decide(scored_row)
    reasons = model.explain_row(scored_row)
    narration = explain(scored_row, reasons, decision)
    if "llm_error" in narration:
        print(f"[WARN] /score: LLM call failed, using offline fallback. Reason: {narration['llm_error']}")

    window_id = uuid.uuid4().hex[:12]
    predicted_fraud = bool(scored_row["risk_score"] >= threshold)

    record = {
        "window_id": window_id,
        "merchant_id": window.merchant_id,
        "window_start": str(row["ts"]),
        "txn_count": window.txn_count,
        "total_amount_inr": round(window.total_amount, 2),
        "risk_score": round(float(scored_row["risk_score"]), 3),
        "predicted_fraud": predicted_fraud,
        "deterministic_reasons": reasons,
        "action": decision.action,
        "requires_human": decision.requires_human,
        "bounded_exposure_inr": decision.bounded_exposure,
        "explanation": narration.get("explanation"),
        "check_first": narration.get("check_first"),
        "feedback": None,  # filled in later via /feedback, None means "awaiting review"
    }
    audit.write(record)
    _scored_windows[window_id] = record
    return record


@app.post("/webhook")
async def receive_webhook(request: Request):
    """Real Razorpay webhook receiver. This is the endpoint you'd actually
    register in Dashboard > Settings > Webhooks for payment.captured and
    payment.failed events in a production deployment.

    Every incoming event is signature-verified BEFORE any of its data is
    trusted - see webhooks/signature_verification.py for why this matters
    for a fraud-detection pipeline specifically (an unverified endpoint is
    itself an attack surface: someone could inject fabricated events to
    poison what the detector sees as "normal"). Verification failure is
    logged to the audit trail and the request is rejected with 401 - it is
    never silently dropped, and it never falls through to processing.

    Note on scope: this endpoint verifies and logs the raw event faithfully.
    Feeding a single webhook event into the 15-min WINDOW-level detector
    (detector/model.py) would require buffering events into windows first,
    which is a real production concern (see README's roadmap section) -
    demonstrating that honestly here rather than papering over it with a
    fake instant per-event risk score."""
    raw_body = await request.body()
    raw_body_str = raw_body.decode("utf-8")
    signature = request.headers.get("x-razorpay-signature", "")

    try:
        is_valid = verify_webhook_signature(raw_body_str, signature)
    except SignatureVerificationError as e:
        audit.write({"type": "webhook_rejected", "reason": "not_configured", "detail": str(e)})
        return JSONResponse(status_code=500, content={"error": str(e)})

    if not is_valid:
        audit.write({
            "type": "webhook_rejected",
            "reason": "signature_mismatch",
            "signature_received": signature[:16] + "..." if signature else "(missing)",
        })
        return JSONResponse(status_code=401, content={"error": "Invalid webhook signature - request rejected, not processed."})

    import json as _json
    try:
        event = _json.loads(raw_body_str)
    except _json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Signature valid but body is not valid JSON."})

    event_type = event.get("event", "unknown")
    payment_entity = event.get("payload", {}).get("payment", {}).get("entity", {})

    audit.write({
        "type": "webhook_received",
        "event": event_type,
        "payment_id": payment_entity.get("id"),
        "amount": payment_entity.get("amount"),
        "status": payment_entity.get("status"),
        "signature_verified": True,
    })

    return {
        "received": True,
        "signature_verified": True,
        "event": event_type,
        "payment_id": payment_entity.get("id"),
        "note": "Event verified and logged. Window-level aggregation into the detector is a queued/async step in production - see README roadmap.",
    }


@app.get("/audit")
def get_live_audit():
    return {"records": audit.read_all()}


class FeedbackInput(BaseModel):
    """A human reviewer's verdict on a previously-scored window - was the
    model's prediction correct? This is the ONLY way the threshold moves;
    there is no automatic self-adjustment from unlabeled data. Consistent
    with the rules engine, a human stays in the loop for anything that
    changes future decisions."""
    window_id: str = Field(..., description="The window_id returned by /score")
    confirmed_fraud: bool = Field(..., description="Did a human reviewer confirm this was actually fraud?")


@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    if "adaptive" not in _model_state:
        return {"error": "Model still training, try again in a few seconds."}

    record = _scored_windows.get(feedback.window_id)
    if record is None:
        return {"error": f"Unknown window_id '{feedback.window_id}' - it may have been scored before this server started, or the ID is wrong."}

    predicted_fraud = record["predicted_fraud"]
    adaptive = _model_state["adaptive"]
    old_threshold = adaptive.get_threshold()

    adaptive.record_feedback(predicted_fraud=predicted_fraud, confirmed_fraud=feedback.confirmed_fraud)
    new_threshold = adaptive.get_threshold()

    record["feedback"] = {
        "confirmed_fraud": feedback.confirmed_fraud,
        "was_correct": predicted_fraud == feedback.confirmed_fraud,
    }
    audit.write({
        "type": "feedback",
        "window_id": feedback.window_id,
        "predicted_fraud": predicted_fraud,
        "confirmed_fraud": feedback.confirmed_fraud,
        "was_correct": predicted_fraud == feedback.confirmed_fraud,
        "threshold_before": round(old_threshold, 4),
        "threshold_after": round(new_threshold, 4),
    })

    return {
        "window_id": feedback.window_id,
        "was_correct": predicted_fraud == feedback.confirmed_fraud,
        "threshold_before": round(old_threshold, 4),
        "threshold_after": round(new_threshold, 4),
        "threshold_changed": abs(new_threshold - old_threshold) > 1e-9,
        "feedback_count_so_far": len(adaptive.feedback_log),
    }


class AskInput(BaseModel):
    """A free-form follow-up question about an already-scored window - the
    interactive layer judges can use in the live demo. Claude only answers
    from the window's own aggregated stats; it never invents identifying
    details (card numbers, names) that weren't in the payload, and it never
    revisits the bounded action - that stays the rules engine's decision."""
    window_id: str = Field(..., description="The window_id returned by /score")
    question: str = Field(..., min_length=1, max_length=500)


@app.post("/ask")
def ask_about_window(ask: AskInput):
    record = _scored_windows.get(ask.window_id)
    if record is None:
        return {"error": f"Unknown window_id '{ask.window_id}' - it may have been scored before this server started, or the ID is wrong."}

    result = answer_question(
        question=ask.question,
        row=record,
        reasons=record.get("deterministic_reasons", []),
        decision_action=record.get("action"),
        bounded_exposure=record.get("bounded_exposure_inr", 0),
        prior_explanation=record.get("explanation"),
    )
    if "llm_error" in result:
        print(f"[WARN] /ask: LLM call failed, using offline fallback. Reason: {result['llm_error']}")

    audit.write({
        "type": "qa",
        "window_id": ask.window_id,
        "question": ask.question,
        "answer": result.get("answer"),
    })

    return {
        "window_id": ask.window_id,
        "question": ask.question,
        "answer": result.get("answer"),
        "offline_mode": "llm_error" in result,
    }


@app.get("/threshold_status")
def threshold_status():
    """Shows the adaptive threshold's current state - useful for a demo
    panel proving the bound is real, not just claimed."""
    if "adaptive" not in _model_state:
        return {"error": "Model still training."}
    adaptive = _model_state["adaptive"]
    return {
        "base_threshold": round(adaptive.base_threshold, 4),
        "current_threshold": round(adaptive.current_threshold, 4),
        "allowed_band": [round(adaptive.base_threshold - adaptive.band, 4),
                          round(adaptive.base_threshold + adaptive.band, 4)],
        "feedback_received": len(adaptive.feedback_log),
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return (Path(__file__).resolve().parent / "dashboard" / "live_demo.html").read_text()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
