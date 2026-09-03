"""
Adaptive threshold: the risk-score cutoff (see evaluation/metrics.py's
choose_threshold) is picked once on a validation split in main.py/server.py's
startup. That's correct for the reported held-out metrics, but a merchant's
traffic pattern genuinely drifts over time - a festival sale can triple
normal volume for a week, permanently changing what "normal" looks like.
A threshold picked once in August and never revisited will misfire in
November.

This module implements a bounded, auditable form of adaptation: as human
reviewers confirm or reject flagged windows (was this actually fraud or a
false positive?), the threshold nudges up or down within a hard band around
its original validation-set value. It never free-floats - the band keeps
this from being an uncontrolled feedback loop, and every adjustment is
logged with the reason, satisfying the same "bounded and explainable"
principle as detector/rules_engine.py's action layer.

This is NOT a second model - it's a simple, fully transparent PID-style
nudge on ONE number, on purpose. A opaque "the threshold retrains itself"
story would violate the same explainability bar the rest of this project
holds itself to.
"""

import json
from pathlib import Path
from datetime import datetime, timezone


class AdaptiveThreshold:
    def __init__(self, base_threshold: float, band: float = 0.15,
                 nudge_step: float = 0.02, min_feedback_for_nudge: int = 5,
                 log_path: str = None):
        """
        base_threshold: the validation-set-selected threshold (ground truth anchor)
        band: max allowed drift from base_threshold in either direction (bounded!)
        nudge_step: how much one recalibration cycle can move the threshold
        min_feedback_for_nudge: don't adjust on fewer than this many labeled reviews
        """
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.band = band
        self.nudge_step = nudge_step
        self.min_feedback_for_nudge = min_feedback_for_nudge
        self.feedback_log = []  # list of (predicted_fraud: bool, confirmed_fraud: bool)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_feedback(self, predicted_fraud: bool, confirmed_fraud: bool):
        """A human reviewer confirms or overturns a flagged window. Called
        from the audit/review workflow, not automatically - this keeps a
        human in the loop for every data point that moves the threshold,
        consistent with the rules engine never acting fully autonomously."""
        self.feedback_log.append({"predicted": predicted_fraud, "confirmed": confirmed_fraud})
        if len(self.feedback_log) >= self.min_feedback_for_nudge:
            self._recalibrate()

    def _recalibrate(self):
        recent = self.feedback_log[-self.min_feedback_for_nudge:]
        false_positives = sum(1 for r in recent if r["predicted"] and not r["confirmed"])
        false_negatives = sum(1 for r in recent if not r["predicted"] and r["confirmed"])

        old_threshold = self.current_threshold
        reason = None

        # Too many false positives -> raise the bar (require higher score to flag)
        if false_positives >= 3:
            proposed = self.current_threshold + self.nudge_step
            reason = f"{false_positives}/{len(recent)} recent flags were false positives - raising threshold"
        # Too many false negatives (fraud slipping through) -> lower the bar
        elif false_negatives >= 2:
            proposed = self.current_threshold - self.nudge_step
            reason = f"{false_negatives}/{len(recent)} recent fraud cases were missed - lowering threshold"
        else:
            return  # feedback looks healthy, no adjustment needed

        # HARD BOUND: never drift more than `band` from the original
        # validation-set threshold, in either direction. This is what keeps
        # this "adaptive" instead of "unbounded self-tuning that could drift
        # to always-flag or never-flag over time.
        lower_bound = self.base_threshold - self.band
        upper_bound = self.base_threshold + self.band
        new_threshold = max(lower_bound, min(upper_bound, proposed))

        self.current_threshold = new_threshold
        self._log_adjustment(old_threshold, new_threshold, reason)

    def _log_adjustment(self, old_threshold, new_threshold, reason):
        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "old_threshold": round(old_threshold, 4),
            "new_threshold": round(new_threshold, 4),
            "base_threshold": round(self.base_threshold, 4),
            "bound": [round(self.base_threshold - self.band, 4), round(self.base_threshold + self.band, 4)],
            "reason": reason,
        }
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def get_threshold(self) -> float:
        return self.current_threshold
