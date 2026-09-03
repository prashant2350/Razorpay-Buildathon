import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, auc


def choose_threshold(val_df: pd.DataFrame) -> float:
    """Pick the score threshold that maximises F1 on a validation split."""
    prec, rec, thr = precision_recall_curve(val_df["is_fraud"], val_df["risk_score"])
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-9), 0)
    best_idx = np.argmax(f1[:-1]) if len(thr) else 0
    return float(thr[best_idx]) if len(thr) else 0.5


def evaluate(test_df: pd.DataFrame, threshold: float) -> dict:
    y_true = test_df["is_fraud"].values
    y_pred = (test_df["risk_score"].values >= threshold).astype(int)

    prec, rec, thr = precision_recall_curve(y_true, test_df["risk_score"].values)
    pr_auc = auc(rec, prec)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": pr_auc,
        "n_test_windows": len(test_df),
        "n_positive_windows": int(y_true.sum()),
        "n_flagged": int(y_pred.sum()),
        "y_pred": y_pred,
    }
