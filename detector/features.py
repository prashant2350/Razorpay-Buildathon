"""
Turns raw Razorpay payment.entity records into 15-minute per-merchant windows
with engineered features. A window is labelled fraud (1) if ANY transaction
inside it is fraudulent (ground truth from the generator, evaluation-only
metadata) - this simulates how a real detector only ever sees aggregated
signals, not the ground-truth label.

`from_razorpay_entities()` is the adapter boundary: it's the ONE place that
knows about Razorpay's real field names (amount in paise, card_iin, status
values). Point it at a real `payments.all()`/webhook dump instead of the
synthetic generator and nothing downstream (features, model, rules engine)
needs to change.
"""

import pandas as pd
import numpy as np

WINDOW = "15min"


def from_razorpay_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Adapter: real Razorpay payment.entity fields -> flat working columns.
    amount: paise -> INR. card_iin -> card_bin (BIN, same concept).
    status: Razorpay's captured/authorized/failed collapsed to success/failed
    for fail-rate purposes (authorized-but-not-yet-captured counts as success
    here since it wasn't declined)."""
    out = df.copy()
    out["amount"] = out["amount"] / 100.0
    out["card_bin"] = out["card_iin"]
    out["status_norm"] = np.where(out["status"] == "failed", "failed", "success")
    return out


def build_windows(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = from_razorpay_entities(raw_df)
    df["ts"] = pd.to_datetime(df["ts"])

    # --- Refund-abuse signal: per-customer repeat frequency across days ---
    # Computed BEFORE windowing, on the full (unwindowed) transaction stream,
    # because this pattern is invisible inside any single 15-min window - it
    # only shows up when you count how many times the SAME email has placed
    # a high-value order in the trailing 30 days. This is deliberately a
    # different kind of feature than the burst/velocity ones below: it's a
    # per-identity behavioral count, not a per-window aggregate.
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    df_sorted["_is_high_value"] = df_sorted["amount"] > 3000  # amount already in ₹ here (post from_razorpay_entities)
    repeat_counts = []
    email_history = {}  # email -> list of high-value txn timestamps (rolling 30d)
    for _, r in df_sorted.iterrows():
        email = r["email"]
        ts = r["ts"]
        hist = email_history.get(email, [])
        hist = [t for t in hist if (ts - t).days <= 30]
        repeat_counts.append(len(hist))
        if r["_is_high_value"]:
            hist.append(ts)
        email_history[email] = hist
    df_sorted["repeat_email_count_30d"] = repeat_counts

    df = df_sorted.set_index("ts")
    g = df.groupby([pd.Grouper(freq=WINDOW), "merchant_id"])

    feats = g.agg(
        txn_count=("amount", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        distinct_bins=("card_bin", "nunique"),
        distinct_countries=("country", "nunique"),
        fail_count=("status_norm", lambda s: (s == "failed").sum()),
        max_repeat_email_count=("repeat_email_count_30d", "max"),
        is_fraud=("is_fraud", "max"),
    ).reset_index()

    feats["std_amount"] = feats["std_amount"].fillna(0)
    feats["fail_rate"] = (feats["fail_count"] / feats["txn_count"]).fillna(0)
    feats["bin_diversity"] = (feats["distinct_bins"] / feats["txn_count"]).fillna(0)
    feats["max_repeat_email_count"] = feats["max_repeat_email_count"].fillna(0)

    # rolling per-merchant baselines (statistical signal, computed causally -
    # only uses PAST windows so there's no leakage from future/held-out data)
    feats = feats.sort_values(["merchant_id", "ts"])
    roll = feats.groupby("merchant_id")[["txn_count", "avg_amount"]].transform(
        lambda s: s.shift(1).rolling(48, min_periods=8)
    )
    feats["txn_count_roll_mean"] = feats.groupby("merchant_id")["txn_count"].transform(
        lambda s: s.shift(1).rolling(48, min_periods=8).mean())
    feats["txn_count_roll_std"] = feats.groupby("merchant_id")["txn_count"].transform(
        lambda s: s.shift(1).rolling(48, min_periods=8).std())
    feats["amount_roll_mean"] = feats.groupby("merchant_id")["avg_amount"].transform(
        lambda s: s.shift(1).rolling(48, min_periods=8).mean())
    feats["amount_roll_std"] = feats.groupby("merchant_id")["avg_amount"].transform(
        lambda s: s.shift(1).rolling(48, min_periods=8).std())

    feats["txn_count_zscore"] = (
        (feats["txn_count"] - feats["txn_count_roll_mean"]) / feats["txn_count_roll_std"].replace(0, np.nan)
    ).fillna(0).clip(-8, 8)
    feats["amount_zscore"] = (
        (feats["avg_amount"] - feats["amount_roll_mean"]) / feats["amount_roll_std"].replace(0, np.nan)
    ).fillna(0).clip(-8, 8)

    # drop the warm-up period where rolling stats aren't populated yet
    feats = feats[feats["txn_count_roll_mean"].notna()].reset_index(drop=True)
    return feats


FEATURE_COLUMNS = [
    "txn_count", "total_amount", "avg_amount", "std_amount",
    "distinct_bins", "distinct_countries", "fail_rate", "bin_diversity",
    "txn_count_zscore", "amount_zscore", "max_repeat_email_count",
]
