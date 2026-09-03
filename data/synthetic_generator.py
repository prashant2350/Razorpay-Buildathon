"""
Synthetic transaction generator for the Razorpay AI Buildathon
Track 02 - AI Risk Manager (Fraud-Spike Detector)

IMPORTANT: transactions are generated in Razorpay's real `payment.entity`
shape (the object returned by the Fetch Payment API and delivered in
payment.* webhooks), NOT an invented schema. Field names, nesting, the
amount-in-paise convention, status values, and error codes are all taken
directly from Razorpay's public API docs:
  https://razorpay.com/docs/api/payments/fetch-with-id/
  https://razorpay.com/docs/webhooks/payloads/payments/
  https://razorpay.com/docs/payments/payments/test-card-details/

We inject fraud/hard-negative *patterns* synthetically (Razorpay's sandbox
has no real fraud to observe), but every record produced - fraudulent or
not - is shaped exactly like what a merchant's webhook handler or
`payments.fetch()` call would actually receive. That means detector/features.py
and detector/model.py can be pointed at a real payments.all() dump with
zero changes to their field references beyond the flattening step below.

Ground truth labels (`is_fraud`, `fraud_type`) are metadata WE attach for
evaluation only - they are not part of the real Razorpay schema and are
stripped before anything is treated as "the API response".

Four canonical fraud/abuse patterns:
  1. Card-testing burst   -> many small txns, high fail-then-success rate, many distinct card IINs
  2. High-value takeover  -> sudden large-ticket txns, new geography/bank
  3. Geo-velocity spike   -> same card token, multiple countries, short window
  4. Refund abuse ring    -> same customer email, repeated high-value orders +
     refunds in a tight cycle across MULTIPLE windows/days - a slower,
     behavioral pattern rather than a burst, deliberately included because
     it can't be caught by the same volume/velocity features as 1-3.
     Grounded in Ravelin's 2025 Fraud Trends Report (54% of merchants saw a
     rise in refund abuse) and Razorpay's own chargebacks blog, which flags
     friendly fraud/refund abuse as a distinct, high-cost loss category
     separate from card-present fraud - https://razorpay.com/blog/international-payment-chargebacks-for-indian-businesses-how-to-win-prevent-and-handle-them
Plus a hard-negative pattern (legit flash-sale burst) so precision isn't trivial.
"""

import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

RNG = np.random.default_rng(42)

# --- Real-shaped reference data (not invented) -----------------------------
# IINs (issuer identification numbers = the real name for what card networks
# call a "BIN") taken from Razorpay's published test card numbers.
TEST_CARD_IINS = {
    "411111": ("Visa", "credit"),      # 4111 1111 1111 1111 - domestic test card
    "555555": ("MasterCard", "debit"), # 5555 5555 5555 4444
    "510405": ("MasterCard", "credit"),# 5104 0155 5555 5558
    "401288": ("Visa", "credit"),      # 4012 8888 8888 1881 - international
    "510510": ("MasterCard", "credit"),# 5105 1051 0510 5100 - international
    "524181": ("MasterCard", "debit"), # 5241 8100 0000 0000
}
NORMAL_IINS = list(TEST_CARD_IINS.keys())

# Razorpay's real error taxonomy (error_code / error_reason), from Capture
# Payment + Test Card Details docs - not invented strings.
ERROR_TIMED_OUT = dict(error_code="GATEWAY_ERROR", error_reason="payment_timed_out",
                        error_description="Your payment could not be completed due to a temporary issue.",
                        error_source="gateway", error_step="payment_authorization")
ERROR_INSUFFICIENT_FUNDS = dict(error_code="BAD_REQUEST_ERROR", error_reason="insufficient_fund",
                                 error_description="Your payment could not be completed due to insufficient funds.",
                                 error_source="issuer", error_step="payment_authorization")
NO_ERROR = dict(error_code=None, error_reason=None, error_description=None,
                 error_source=None, error_step=None)

BANKS = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK"]
METHODS = ["card", "upi", "netbanking"]
COUNTRIES = ["IN", "IN", "IN", "IN", "IN", "AE", "SG", "US", "NG", "RU"]  # IN-heavy baseline


@dataclass
class GenConfig:
    n_days: int = 45
    merchant_id: str = "acc_merchant001"
    base_txns_per_hour: int = 40
    n_fraud_spikes: int = 34
    n_hard_negative_spikes: int = 14   # legit bursts that LOOK anomalous but aren't fraud
    n_refund_abuse_rings: int = 5      # slow, multi-day behavioral pattern (see _inject_refund_abuse_ring)


def _rid(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:14]}"


def _payment_entity(ts, merchant_id, amount_inr, method, status, iin, network, card_type,
                     country, error=NO_ERROR, is_fraud=0, fraud_type=None):
    """Builds one record in Razorpay's real payment.entity shape.
    amount is stored in paise (subunits) per Razorpay convention."""
    captured = status == "captured"
    return {
        "id": _rid("pay"),
        "entity": "payment",
        "amount": int(round(amount_inr * 100)),   # paise, per Razorpay docs
        "currency": "INR",
        "status": status,                          # created|authorized|captured|failed
        "order_id": _rid("order"),
        "invoice_id": None,
        "international": country != "IN",
        "method": method,
        "amount_refunded": 0,
        "refund_status": None,
        "captured": captured,
        "card_id": _rid("card") if method == "card" else None,
        "card_iin": iin if method == "card" else None,           # = card BIN
        "card_network": network if method == "card" else None,
        "card_type": card_type if method == "card" else None,
        "bank": RNG.choice(BANKS) if method in ("netbanking", "card") else None,
        "vpa": f"user{RNG.integers(1000,9999)}@okhdfcbank" if method == "upi" else None,
        "email": f"user{RNG.integers(1000,9999)}@example.com",
        "contact": f"+91{RNG.integers(6000000000,9999999999)}",
        "notes": [],
        "fee": int(round(amount_inr * 100 * 0.02)),
        "tax": int(round(amount_inr * 100 * 0.0036)),
        **error,
        "acquirer_data": {"rrn": f"{RNG.integers(10**11,10**12)}"},
        "created_at": int(ts.replace(tzinfo=timezone.utc).timestamp()),
        # --- evaluation-only metadata, NOT part of the real Razorpay schema ---
        "merchant_id": merchant_id,
        "ts": ts,
        "country": country,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
    }


def _normal_hour_volume(hour_of_day: int, base: int) -> int:
    curve = 1.0 + 0.6 * np.exp(-((hour_of_day - 13) ** 2) / 8) + 0.5 * np.exp(-((hour_of_day - 20) ** 2) / 6)
    lam = max(base * curve, 2)
    return RNG.poisson(lam)


def _gen_normal_txn(ts, merchant_id):
    method = RNG.choice(METHODS, p=[0.55, 0.35, 0.10])
    iin = RNG.choice(NORMAL_IINS)
    network, card_type = TEST_CARD_IINS[iin]
    failed = RNG.random() < 0.05
    if failed:
        error = ERROR_INSUFFICIENT_FUNDS if RNG.random() < 0.6 else ERROR_TIMED_OUT
        status = "failed"
    else:
        error, status = NO_ERROR, "captured"
    return _payment_entity(ts, merchant_id, max(50, RNG.normal(650, 220)), method, status,
                            iin, network, card_type, "IN", error)


def _inject_card_testing_burst(start_ts, merchant_id, n=25):
    rows = []
    # card-testing rings cycle through many IINs quickly, most outside the
    # "known good" set - modelled as synthetic IINs in the same 6-digit format
    burst_iins = [f"{RNG.integers(400000,599999)}" for _ in range(n // 2)]
    for i in range(n):
        ts = start_ts + timedelta(seconds=int(RNG.integers(0, 90)))
        failed = RNG.random() < 0.75  # card-testing = mostly declines
        iin = RNG.choice(burst_iins)
        network = "Visa" if iin.startswith("4") else "MasterCard"
        status = "failed" if failed else "captured"
        error = ERROR_INSUFFICIENT_FUNDS if failed else NO_ERROR
        rows.append(_payment_entity(ts, merchant_id, max(1, RNG.normal(15, 8)), "card", status,
                                     iin, network, "credit", RNG.choice(COUNTRIES), error,
                                     is_fraud=1, fraud_type="card_testing"))
    return rows


def _inject_high_value_takeover(start_ts, merchant_id, n=6):
    rows = []
    for i in range(n):
        ts = start_ts + timedelta(seconds=int(RNG.integers(0, 300)))
        iin = RNG.choice(NORMAL_IINS)
        network, card_type = TEST_CARD_IINS[iin]
        rows.append(_payment_entity(ts, merchant_id, max(5000, RNG.normal(18000, 4000)), "card",
                                     "captured", iin, network, card_type,
                                     RNG.choice(["US", "NG", "RU"]), NO_ERROR,
                                     is_fraud=1, fraud_type="high_value_takeover"))
    return rows


def _inject_geo_velocity_spike(start_ts, merchant_id, n=8):
    rows = []
    iin = RNG.choice(NORMAL_IINS)
    network, card_type = TEST_CARD_IINS[iin]
    countries = RNG.choice(["US", "SG", "AE", "NG", "RU"], size=n, replace=True)
    for i in range(n):
        ts = start_ts + timedelta(seconds=int(RNG.integers(0, 120)))
        rows.append(_payment_entity(ts, merchant_id, max(200, RNG.normal(2200, 500)), "card",
                                     "captured", iin, network, card_type, countries[i], NO_ERROR,
                                     is_fraud=1, fraud_type="geo_velocity"))
    return rows


def _inject_refund_abuse_ring(start_ts, merchant_id, n_cycles=4):
    """Refund abuse ring: the SAME customer email places a high-value order,
    it's captured, then - across the following days - the same email places
    another high-value order in a tight repeating cycle. Unlike patterns 1-3
    this is NOT a burst inside one 15-min window: it's a slow, multi-day
    behavioral signature (same identity, repeated order-then-reorder cycles)
    that only shows up when you track a customer's history ACROSS windows,
    not within one. Grounded in the real industry pattern described in
    Ravelin's 2025 Fraud Trends Report and Razorpay's own chargebacks blog
    (see module docstring) - the merchant ships repeatedly to the same
    customer who disputes or requests refunds on a recurring cycle.

    Returns rows spread across several days, all sharing one email/contact -
    that shared identity is the signal, not transaction volume or amount."""
    rows = []
    abuser_email = f"repeat.abuser{RNG.integers(1000,9999)}@example.com"
    abuser_contact = f"+91{RNG.integers(6000000000,9999999999)}"
    iin = RNG.choice(NORMAL_IINS)
    network, card_type = TEST_CARD_IINS[iin]

    for cycle in range(n_cycles):
        cycle_ts = start_ts + timedelta(days=cycle * 3, hours=int(RNG.integers(0, 20)))
        amount = max(3000, RNG.normal(9000, 2000))
        row = _payment_entity(cycle_ts, merchant_id, amount, "card", "captured",
                               iin, network, card_type, "IN", NO_ERROR,
                               is_fraud=1, fraud_type="refund_abuse_ring")
        row["email"] = abuser_email
        row["contact"] = abuser_contact
        rows.append(row)
    return rows


def _inject_flash_sale_burst(start_ts, merchant_id, n=45):
    """Hard negative: a legitimate flash-sale traffic spike. High volume like a
    fraud burst, but normal fail-rate, normal IIN diversity, IN-only geo."""
    rows = []
    for i in range(n):
        ts = start_ts + timedelta(seconds=int(RNG.integers(0, 900)))
        rows.append(_gen_normal_txn(ts, merchant_id))
    return rows


def generate(cfg: GenConfig = GenConfig()) -> pd.DataFrame:
    """Generates transactions for a SINGLE merchant (kept for backwards
    compatibility - main.py and server.py's single-merchant demo still call
    this directly)."""
    start = datetime(2026, 8, 1)
    rows = []
    for day in range(cfg.n_days):
        for hour in range(24):
            hour_start = start + timedelta(days=day, hours=hour)
            n_txn = _normal_hour_volume(hour, cfg.base_txns_per_hour)
            for _ in range(n_txn):
                ts = hour_start + timedelta(seconds=int(RNG.integers(0, 3600)))
                rows.append(_gen_normal_txn(ts, cfg.merchant_id))

    total_hours = cfg.n_days * 24
    spike_hours = RNG.choice(range(6, total_hours - 6), size=cfg.n_fraud_spikes, replace=False)
    spike_types = RNG.choice(
        ["card_testing", "high_value_takeover", "geo_velocity"],
        size=cfg.n_fraud_spikes, p=[0.5, 0.25, 0.25]
    )
    for h, stype in zip(spike_hours, spike_types):
        spike_ts = start + timedelta(hours=int(h), minutes=int(RNG.integers(0, 50)))
        if stype == "card_testing":
            rows.extend(_inject_card_testing_burst(spike_ts, cfg.merchant_id))
        elif stype == "high_value_takeover":
            rows.extend(_inject_high_value_takeover(spike_ts, cfg.merchant_id))
        else:
            rows.extend(_inject_geo_velocity_spike(spike_ts, cfg.merchant_id))

    hn_hours = RNG.choice(
        [h for h in range(6, total_hours - 6) if h not in set(spike_hours)],
        size=cfg.n_hard_negative_spikes, replace=False
    )
    for h in hn_hours:
        hn_ts = start + timedelta(hours=int(h), minutes=int(RNG.integers(0, 50)))
        rows.extend(_inject_flash_sale_burst(hn_ts, cfg.merchant_id))

    # Refund abuse rings: multi-day pattern (4 cycles x 3 days apart = ~12
    # days of runway needed), so only start these within the first
    # (n_days - 15) days to guarantee every ring completes inside the window.
    if cfg.n_days > 20:
        max_start_day = cfg.n_days - 15
        ring_start_days = RNG.choice(range(1, max_start_day), size=min(cfg.n_refund_abuse_rings, max_start_day - 1), replace=False)
        for d in ring_start_days:
            ring_ts = start + timedelta(days=int(d), hours=int(RNG.integers(8, 20)))
            rows.extend(_inject_refund_abuse_ring(ring_ts, cfg.merchant_id))

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


# Three merchants with genuinely different baseline traffic - not just a
# renamed copy of the same config. This is what lets us honestly claim the
# detector generalizes across merchants rather than overfitting to one
# specific volume/spike pattern, since each merchant's rolling z-score
# baseline (see detector/features.py) is computed independently per merchant.
MULTI_MERCHANT_CONFIGS = [
    GenConfig(merchant_id="acc_merchant001", base_txns_per_hour=40,
              n_fraud_spikes=34, n_hard_negative_spikes=14),   # mid-size retailer
    GenConfig(merchant_id="acc_merchant002", base_txns_per_hour=12,
              n_fraud_spikes=10, n_hard_negative_spikes=4),    # small/low-volume merchant
    GenConfig(merchant_id="acc_merchant003", base_txns_per_hour=110,
              n_fraud_spikes=48, n_hard_negative_spikes=22),   # high-volume merchant
]


def generate_multi_merchant(configs=None) -> pd.DataFrame:
    """Generates and combines transactions for several merchants with
    distinct baseline volumes. Each merchant's rolling baseline in
    features.py is computed per-merchant (groupby('merchant_id')), so this
    is what actually exercises that per-merchant logic instead of always
    running it against a single group."""
    configs = configs or MULTI_MERCHANT_CONFIGS
    frames = [generate(cfg) for cfg in configs]
    return pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    df = generate_multi_merchant()
    print(df.shape, df["is_fraud"].sum(), "fraud txns out of", len(df))
    print(df.groupby("merchant_id").size())
    print(df[["id", "amount", "currency", "status", "method", "card_iin", "error_reason"]].head(5).to_string())
