"""Experiment result dataclass and metric computation utilities.

ExperimentResult collects all evaluation metrics for a single trained model
into one structured object.  compute_metrics() builds one from raw arrays.
compare_models() returns a pandas DataFrame for multi-model comparison.

Example::

    result = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=CLASS_NAMES,
        model_name="LightGBM",
        cv_scores=[0.87, 0.88, 0.86, 0.89, 0.87],
        n_features=156,
        training_time=12.4,
        n_samples=len(y_test),
    )
    print(result.accuracy, result.macro_f1)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────
# ExperimentResult dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """All evaluation metrics for a single model training/evaluation run.

    Attributes:
        model_name:          short identifier (e.g. "LightGBM", "SVM_RBF")
        accuracy:            overall accuracy (0–1)
        macro_f1:            macro-averaged F1 score
        weighted_f1:         weighted-averaged F1 score
        per_class_accuracy:  dict mapping class_name → per-class accuracy
        confusion_matrix:    (N, N) raw count confusion matrix
        cv_scores:           list of cross-validation fold accuracies
        cv_mean:             mean of cv_scores (computed automatically)
        cv_std:              std  of cv_scores (computed automatically)
        n_features:          number of input features used
        training_time:       training time in seconds
        n_samples:           total number of evaluation samples
    """
    model_name:         str
    accuracy:           float
    macro_f1:           float
    weighted_f1:        float
    per_class_accuracy: Dict[str, float]     = field(default_factory=dict)
    confusion_matrix:   np.ndarray           = field(default_factory=lambda: np.array([]))
    cv_scores:          List[float]          = field(default_factory=list)
    cv_mean:            float                = 0.0
    cv_std:             float                = 0.0
    n_features:         int                  = 0
    training_time:      float                = 0.0
    n_samples:          int                  = 0

    def __post_init__(self) -> None:
        """Auto-compute cv_mean and cv_std from cv_scores if not set."""
        if self.cv_scores and self.cv_mean == 0.0:
            self.cv_mean = float(np.mean(self.cv_scores))
        if self.cv_scores and self.cv_std == 0.0:
            self.cv_std = float(np.std(self.cv_scores))

    def to_dict(self) -> Dict[str, object]:
        """Flat dict representation for DataFrame construction."""
        return {
            "model_name":    self.model_name,
            "accuracy":      round(self.accuracy, 4),
            "macro_f1":      round(self.macro_f1, 4),
            "weighted_f1":   round(self.weighted_f1, 4),
            "cv_mean":       round(self.cv_mean, 4),
            "cv_std":        round(self.cv_std, 4),
            "n_features":    self.n_features,
            "training_time": round(self.training_time, 2),
            "n_samples":     self.n_samples,
        }

    def __repr__(self) -> str:
        return (
            f"ExperimentResult("
            f"model='{self.model_name}', "
            f"acc={self.accuracy:.4f}, "
            f"macro_f1={self.macro_f1:.4f}, "
            f"cv={self.cv_mean:.4f}±{self.cv_std:.4f})"
        )


# ─────────────────────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    model_name: str = "model",
    cv_scores: Optional[List[float]] = None,
    n_features: int = 0,
    training_time: float = 0.0,
    n_samples: Optional[int] = None,
) -> ExperimentResult:
    """Build an ExperimentResult from prediction arrays.

    Args:
        y_true:        (N,) ground-truth integer or string labels.
        y_pred:        (N,) predicted integer or string labels.
        y_proba:       (N, C) predicted probabilities (optional, reserved
                       for future ROC-AUC computation).
        class_names:   list of class name strings. If None, inferred from
                       unique values in y_true.
        model_name:    identifier string for this result.
        cv_scores:     list of per-fold CV accuracies (optional).
        n_features:    number of features used for this run.
        training_time: training wall time in seconds.
        n_samples:     number of evaluation samples.  If None, len(y_true).

    Returns:
        ExperimentResult

    Raises:
        ImportError: if scikit-learn is not installed.
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for compute_metrics: pip install scikit-learn"
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        labels_unique = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        class_names = [str(l) for l in labels_unique]

    acc        = float(accuracy_score(y_true, y_pred))
    macro_f1   = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy (diagonal / row sum)
    per_class: Dict[str, float] = {}
    for i, cls_name in enumerate(class_names):
        if i < len(cm):
            row_sum = int(cm[i].sum())
            per_class[cls_name] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0

    cv_list = list(cv_scores) if cv_scores is not None else []
    cv_mean = float(np.mean(cv_list)) if cv_list else acc
    cv_std  = float(np.std(cv_list))  if cv_list else 0.0

    return ExperimentResult(
        model_name=model_name,
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_accuracy=per_class,
        confusion_matrix=cm,
        cv_scores=cv_list,
        cv_mean=cv_mean,
        cv_std=cv_std,
        n_features=n_features,
        training_time=training_time,
        n_samples=int(n_samples) if n_samples is not None else len(y_true),
    )


# ─────────────────────────────────────────────────────────────
# compare_models
# ─────────────────────────────────────────────────────────────

def compare_models(results: List[ExperimentResult]) -> "pd.DataFrame":
    """Build a summary DataFrame sorted by accuracy (descending).

    Args:
        results: list of ExperimentResult objects.

    Returns:
        pandas DataFrame with one row per model, columns:
        model_name, accuracy, macro_f1, weighted_f1,
        cv_mean, cv_std, n_features, training_time, n_samples.

    Raises:
        ImportError: if pandas is not installed.
    """
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for compare_models: pip install pandas"
        )

    rows = [r.to_dict() for r in results]
    df   = pd.DataFrame(rows)
    df   = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    # Add rank column
    df.insert(0, "rank", range(1, len(df) + 1))

    return df
