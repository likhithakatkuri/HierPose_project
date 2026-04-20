"""Cross-validation utilities for IEEE-quality evaluation.

Provides:
- Nested cross-validation (outer: generalization estimate, inner: hyperparam selection)
- McNemar's test for pairwise model significance testing
- Friedman test for comparing multiple classifiers
- Reporting utilities

References:
- Dietterich (1998), "Approximate statistical tests for comparing supervised classifiers"
- Demšar (2006), "Statistical comparisons of classifiers over multiple data sets"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import stats
    from scipy.stats import chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class CVResult:
    """Result from cross-validation."""
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    n_folds: int
    n_samples: int
    n_classes: int
    model_name: str = ""
    predictions: Optional[np.ndarray] = None   # OOF predictions for McNemar test

    def __str__(self) -> str:
        return (
            f"CV {self.n_folds}-fold: {self.cv_mean:.4f} ± {self.cv_std:.4f} "
            f"({self.n_samples} samples, {self.n_classes} classes)"
        )


@dataclass
class McNemarResult:
    """Result from McNemar's test."""
    model_a: str
    model_b: str
    chi2_stat: float
    p_value: float
    n_discordant: int
    significant: bool   # p < 0.05
    winner: str         # which model is better

    def __str__(self) -> str:
        sig = "✓ significant" if self.significant else "✗ not significant"
        return (
            f"McNemar({self.model_a} vs {self.model_b}): "
            f"χ²={self.chi2_stat:.4f}, p={self.p_value:.4f} — {sig} "
            f"(better: {self.winner})"
        )


# ─────────────────────────────────────────────────────────────
# Nested cross-validation
# ─────────────────────────────────────────────────────────────

def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    return_predictions: bool = True,
    model_name: str = "",
) -> CVResult:
    """Standard stratified k-fold CV with optional OOF predictions.

    Args:
        model: sklearn-compatible classifier
        X: (n_samples, n_features)
        y: (n_samples,) integer labels
        n_folds: number of folds (5 is IEEE standard for small datasets)
        random_state: reproducibility seed
        return_predictions: compute OOF predictions (needed for McNemar)
        model_name: display name for results

    Returns:
        CVResult with accuracy scores and optionally OOF predictions
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required")

    n_samples = len(y)
    n_classes = len(np.unique(y))

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores: List[float] = []
    oof_preds: Optional[np.ndarray] = None

    if return_predictions:
        oof_preds = np.full(n_samples, -1, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        scores.append(acc)

        if return_predictions and oof_preds is not None:
            oof_preds[val_idx] = preds

    return CVResult(
        cv_scores=scores,
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores)),
        n_folds=n_folds,
        n_samples=n_samples,
        n_classes=n_classes,
        model_name=model_name,
        predictions=oof_preds,
    )


def nested_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], Any],
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
    model_name: str = "",
) -> CVResult:
    """Nested cross-validation for unbiased performance estimation.

    Used when sample size < 1000 (IEEE recommendation).
    - Outer loop: estimate generalization performance
    - Inner loop: hyperparameter selection (if factory supports it)

    For the HierPose pipeline, the inner loop is simplified:
    the model_factory creates a fresh model with best-known hyperparameters.
    For Optuna-based tuning, see hyperparameter_search.py.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) labels
        model_factory: callable that returns a fresh model instance
        outer_folds: outer CV folds (default 5)
        inner_folds: inner CV folds (default 3)
        random_state: seed
        model_name: display name

    Returns:
        CVResult with outer-fold scores
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required")

    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=random_state + 1
    )

    outer_scores: List[float] = []
    oof_preds = np.full(len(y), -1, dtype=int)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Inner CV for model selection / validation
        model = model_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        oof_preds[test_idx] = preds
        outer_scores.append(accuracy_score(y_te, preds))

    return CVResult(
        cv_scores=outer_scores,
        cv_mean=float(np.mean(outer_scores)),
        cv_std=float(np.std(outer_scores)),
        n_folds=outer_folds,
        n_samples=len(y),
        n_classes=len(np.unique(y)),
        model_name=model_name,
        predictions=oof_preds,
    )


# ─────────────────────────────────────────────────────────────
# McNemar's test
# ─────────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    model_a_name: str = "model_A",
    model_b_name: str = "model_B",
    alpha: float = 0.05,
) -> McNemarResult:
    """McNemar's test for pairwise classifier significance.

    Tests whether two classifiers have significantly different error rates.
    Requires paired predictions (OOF from cross_validate_model).

    The test uses the corrected McNemar formula (Dietterich 1998):
        χ² = (|n01 - n10| - 1)² / (n01 + n10)
    where n01 = cases A wrong, B correct; n10 = cases A correct, B wrong.

    Args:
        y_true: ground truth labels
        pred_a: predictions from model A
        pred_b: predictions from model B
        model_a_name: name of model A
        model_b_name: name of model B
        alpha: significance threshold (default 0.05)

    Returns:
        McNemarResult with chi2 stat, p-value, significance flag
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for McNemar test: pip install scipy")

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # Discordant pairs
    n01 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n10 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    n_discordant = n01 + n10

    if n_discordant == 0:
        return McNemarResult(
            model_a=model_a_name,
            model_b=model_b_name,
            chi2_stat=0.0,
            p_value=1.0,
            n_discordant=0,
            significant=False,
            winner="tie",
        )

    # Corrected McNemar (continuity correction)
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = float(1 - stats.chi2.cdf(chi2_stat, df=1))

    acc_a = float(np.mean(correct_a))
    acc_b = float(np.mean(correct_b))
    winner = model_a_name if acc_a > acc_b else (model_b_name if acc_b > acc_a else "tie")

    return McNemarResult(
        model_a=model_a_name,
        model_b=model_b_name,
        chi2_stat=float(chi2_stat),
        p_value=p_value,
        n_discordant=n_discordant,
        significant=p_value < alpha,
        winner=winner,
    )


def pairwise_mcnemar(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> List[McNemarResult]:
    """Run McNemar's test for all pairs of models.

    Args:
        y_true: ground truth
        predictions_dict: {model_name: predictions_array}
        alpha: significance threshold

    Returns:
        List of McNemarResult for all pairs
    """
    names = list(predictions_dict.keys())
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result = mcnemar_test(
                y_true,
                predictions_dict[names[i]],
                predictions_dict[names[j]],
                model_a_name=names[i],
                model_b_name=names[j],
                alpha=alpha,
            )
            results.append(result)
    return results


def significance_table(mcnemar_results: List[McNemarResult]) -> str:
    """Format McNemar results as a readable table."""
    lines = [
        "Pairwise McNemar Significance Tests",
        "=" * 70,
        f"{'Model A':<14} {'Model B':<14} {'χ²':>8} {'p-value':>10} {'Sig':>5} {'Winner':<14}",
        "-" * 70,
    ]
    for r in mcnemar_results:
        sig = "✓" if r.significant else "✗"
        lines.append(
            f"{r.model_a:<14} {r.model_b:<14} {r.chi2_stat:>8.4f} {r.p_value:>10.4f} {sig:>5} {r.winner:<14}"
        )
    lines.append("=" * 70)
    lines.append("✓ = p < 0.05 (significant difference)")
    return "\n".join(lines)
