"""
Fraud-spike detector: an ensemble of
  (a) a transparent statistical rule (rolling z-score on volume & ticket size)
  (b) an unsupervised Isolation Forest over the full feature vector

The two are blended into a single 0-1 risk score. Blending a transparent
statistical signal with a learned one keeps the model both EXPLAINABLE
(every score can be decomposed into "which stat looked abnormal") and
sensitive to multivariate patterns a single rule would miss (e.g. card
testing, where volume looks normal but bin-diversity + fail-rate spike).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from detector.features import FEATURE_COLUMNS


class FraudSpikeDetector:
    def __init__(self, contamination=0.09, random_state=42):
        self.scaler = StandardScaler()
        self.iso = IsolationForest(
            n_estimators=300, contamination=contamination,
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, train_df: pd.DataFrame):
        X = train_df[FEATURE_COLUMNS].fillna(0).values
        Xs = self.scaler.fit_transform(X)
        self.iso.fit(Xs)

        # Save fixed normalization ranges from the TRAINING distribution, not
        # the batch being scored. Using batch min/max (as an earlier version
        # of this code did) breaks completely for single-row live scoring:
        # with one row, min == max, so every score collapses to the same
        # constant regardless of how anomalous the row actually is. Fixed
        # training-time ranges keep scores meaningful and comparable whether
        # you score 1 row live or 10,000 rows in a batch.
        iso_raw_train = -self.iso.score_samples(Xs)
        self._iso_raw_min = float(iso_raw_train.min())
        self._iso_raw_max = float(iso_raw_train.max())

        stat_raw_train = self._stat_score_raw(train_df)
        self._stat_raw_max = float(stat_raw_train.max()) or 1.0

        self.fitted = True
        return self

    def _stat_score_raw(self, df: pd.DataFrame) -> np.ndarray:
        z = df[["txn_count_zscore", "amount_zscore"]].abs().max(axis=1)
        card_testing_signal = ((df["fail_rate"] > 0.5) & (df["bin_diversity"] > 0.3)).astype(float) * 4
        return (z.values + card_testing_signal.values)

    def _stat_score(self, df: pd.DataFrame) -> np.ndarray:
        # transparent rule-based score: combination of volume & amount deviation
        # plus a hard signal for the card-testing signature (many bins, high fail rate)
        # Normalized against the FIXED training-time max (see fit()), so a
        # single extreme row scores near 1.0 instead of always normalizing to 1.0.
        raw = self._stat_score_raw(df)
        return np.clip(raw / (self._stat_raw_max + 1e-9), 0, 1)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "call .fit() first"
        out = df.copy()
        X = out[FEATURE_COLUMNS].fillna(0).values
        Xs = self.scaler.transform(X)

        # isolation forest: more negative = more anomalous -> flip & normalise to 0..1
        # against the FIXED training-time range, not this batch's own min/max
        # (see fit() docstring - batch normalization breaks single-row scoring)
        iso_raw = -self.iso.score_samples(Xs)
        span = (self._iso_raw_max - self._iso_raw_min) or 1e-9
        iso_norm = np.clip((iso_raw - self._iso_raw_min) / span, 0, 1)

        stat_norm = self._stat_score(out)

        out["iso_score"] = iso_norm
        out["stat_score"] = stat_norm
        out["risk_score"] = 0.55 * iso_norm + 0.45 * stat_norm
        return out

    def explain_row(self, row) -> list:
        """Return the top contributing factors for a scored row (rule-based, deterministic)."""
        reasons = []
        if abs(row["txn_count_zscore"]) > 2.5:
            reasons.append(f"transaction volume {row['txn_count_zscore']:.1f} std-dev from this merchant's rolling baseline")
        if abs(row["amount_zscore"]) > 2.5:
            reasons.append(f"average ticket size {row['amount_zscore']:.1f} std-dev from baseline")
        if row["fail_rate"] > 0.5:
            reasons.append(f"unusually high decline rate ({row['fail_rate']:.0%})")
        if row["bin_diversity"] > 0.3 and row["fail_rate"] > 0.3:
            # card-testing needs BOTH signals together - high BIN diversity
            # alone is common in small, low-volume windows (few transactions,
            # naturally few repeated cards) and isn't suspicious on its own.
            reasons.append(f"high card-BIN diversity relative to volume ({row['bin_diversity']:.0%}) with a high decline rate ({row['fail_rate']:.0%}) — card-testing signature")
        if row.get("max_repeat_email_count", 0) >= 1:
            reasons.append(f"same customer email placed {int(row['max_repeat_email_count'])} prior high-value orders in the last 30 days — refund abuse ring signature")
        if row["distinct_countries"] >= 3:
            reasons.append(f"transactions spanning {int(row['distinct_countries'])} countries in one window")
        if not reasons:
            reasons.append("multivariate pattern flagged by the anomaly model (no single feature dominant)")
        return reasons
