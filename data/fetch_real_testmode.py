"""
Fetches REAL payments from your Razorpay test-mode account via the Fetch
All Payments API and converts them into the same DataFrame shape
data/synthetic_generator.py produces - so features.py, model.py, and
rules_engine.py work completely unchanged on real API output.

Because your test account has no real fraud (it's a sandbox), records
fetched here are labelled is_fraud=0 across the board - this is HONEST:
we are not pretending to have ground truth we don't have. Real-fetched
traffic is meant to validate that the feature engineering + detector run
correctly against Razorpay's real response shape, NOT to replace the
labelled synthetic evaluation set used for precision/recall in main.py.

Usage:
    export RAZORPAY_KEY_ID=rzp_test_xxxx
    export RAZORPAY_KEY_SECRET=xxxx
    python3 data/fetch_real_testmode.py
"""

import pandas as pd
from datetime import datetime, timezone
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.razorpay_client import fetch_all_payments, RazorpayNotConfigured

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def _entity_to_row(p: dict, merchant_id: str = "acc_merchant001") -> dict:
    """Same target shape as synthetic_generator._payment_entity(), built
    from a REAL API response instead of a synthetic one."""
    card = p.get("card") or {}
    return {
        "id": p.get("id"),
        "entity": p.get("entity"),
        "amount": p.get("amount"),
        "currency": p.get("currency"),
        "status": p.get("status"),
        "order_id": p.get("order_id"),
        "invoice_id": p.get("invoice_id"),
        "international": p.get("international"),
        "method": p.get("method"),
        "amount_refunded": p.get("amount_refunded"),
        "refund_status": p.get("refund_status"),
        "captured": p.get("captured"),
        "card_id": p.get("card_id"),
        "card_iin": card.get("iin"),
        "card_network": card.get("network"),
        "card_type": card.get("type"),
        "bank": p.get("bank"),
        "vpa": p.get("vpa"),
        "email": p.get("email"),
        "contact": p.get("contact"),
        "notes": p.get("notes"),
        "fee": p.get("fee"),
        "tax": p.get("tax"),
        "error_code": p.get("error_code"),
        "error_reason": p.get("error_reason"),
        "error_description": p.get("error_description"),
        "error_source": p.get("error_source"),
        "error_step": p.get("error_step"),
        "acquirer_data": p.get("acquirer_data"),
        "created_at": p.get("created_at"),
        # --- evaluation-only metadata ---
        "merchant_id": merchant_id,
        "ts": datetime.fromtimestamp(p["created_at"], tz=timezone.utc).replace(tzinfo=None)
              if p.get("created_at") else None,
        "country": "IN" if not p.get("international") else "XX",
        "is_fraud": 0,        # HONEST: sandbox has no real fraud ground truth
        "fraud_type": None,
    }


def fetch_real_dataframe(max_records: int = 500) -> pd.DataFrame:
    all_rows = []
    skip = 0
    page = 100
    while len(all_rows) < max_records:
        resp = fetch_all_payments(count=min(page, max_records - len(all_rows)), skip=skip)
        items = resp.get("items", [])
        if not items:
            break
        all_rows.extend(_entity_to_row(p) for p in items)
        skip += len(items)
        if len(items) < page:
            break
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    try:
        df = fetch_real_dataframe()
    except RazorpayNotConfigured as e:
        print(f"NOT CONFIGURED: {e}")
        raise SystemExit(1)

    if df.empty:
        print("No payments found on your test account yet. Run "
              "data/seed_razorpay_testmode.py first, or create a few test "
              "payments through Razorpay's Checkout test UI.")
    else:
        print(f"Fetched {len(df)} real test-mode payments.")
        print(df[["id", "amount", "status", "method", "created_at"]].head(10).to_string())
        df.to_csv(OUTPUTS / "real_testmode_payments.csv", index=False)
        print(f"\nSaved to {OUTPUTS / 'real_testmode_payments.csv'}")
