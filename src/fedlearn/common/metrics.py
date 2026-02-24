from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_binary_metrics(model, X, y) -> dict[str, float]:
    """
    Compute binary model metrics (accuracy, log loss, ROC-AUC) with failure flags.
    """
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))

    log_loss_failed = 0.0
    try:
        y_proba = model.predict_proba(X)
        loss = float(log_loss(y, y_proba, labels=model.classes_))
    except ValueError:
        loss = float("nan")
        log_loss_failed = 1.0

    roc_auc, roc_auc_failed = compute_roc_auc(y, model, X)

    return {
        "accuracy": acc,
        "loss": loss,
        "roc_auc": roc_auc,
        "log-loss-failed": log_loss_failed,
        "roc-auc-failed": roc_auc_failed,
    }


def compute_roc_auc(y_true, model, X) -> tuple[float, float]:
    """
    Compute ROC-AUC safely for binary classification.

    Returns:
        (roc_auc, failed_flag). If the score cannot be computed, a fallback value of (0.5, 1.0) is returned.
    """
    # AUC requires both classes
    if len(np.unique(y_true)) < 2:
        return 0.5, 1.0

    # try probability scores first
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X)[:, 1]
            return float(roc_auc_score(y_true, y_score)), 0.0
        except ValueError:
            pass

    # fallback to decision scores
    if hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X)
            return float(roc_auc_score(y_true, y_score)), 0.0
        except ValueError:
            pass

    return 0.5, 1.0
