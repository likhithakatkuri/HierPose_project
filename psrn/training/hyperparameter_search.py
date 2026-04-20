"""Optuna-based hyperparameter search for all HierPose models.

Uses Optuna's Tree-structured Parzen Estimator (TPE) sampler for
efficient Bayesian optimization. Each model has a tailored search
space based on the JHMDB dataset characteristics (~672 train samples,
21 classes).

Usage:
    from psrn.training.hyperparameter_search import tune_lightgbm
    best_params = tune_lightgbm(X_train, y_train, n_trials=50)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _cv_score(model: Any, X: np.ndarray, y: np.ndarray, n_folds: int = 3) -> float:
    """Quick 3-fold CV score for Optuna objectives."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=2)
    return float(scores.mean())


# ─────────────────────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────────────────────

def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    n_jobs: int = 4,
    random_state: int = 42,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Optuna hyperparameter search for LightGBM on JHMDB 21-class.

    Args:
        X_train: (n_samples, n_features)
        y_train: (n_samples,) integer labels
        n_trials: number of Optuna trials (50 = good balance of quality/speed)
        n_jobs: parallel jobs for CV within each trial
        random_state: seed
        timeout: optional time limit in seconds

    Returns:
        dict of best hyperparameters
    """
    if not HAS_OPTUNA:
        print("Optuna not installed. pip install optuna. Using default params.")
        return _default_lgbm_params(len(np.unique(y_train)))

    if not HAS_LGBM:
        raise ImportError("lightgbm required: pip install lightgbm")

    n_classes = len(np.unique(y_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "class_weight": "balanced",
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbosity": -1,
        }
        if n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes

        model = lgb.LGBMClassifier(**params)
        return _cv_score(model, X_train, y_train)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params.copy()
    best["class_weight"] = "balanced"
    best["n_jobs"] = n_jobs
    best["random_state"] = random_state
    best["verbosity"] = -1
    if n_classes > 2:
        best["objective"] = "multiclass"
        best["num_class"] = n_classes

    print(f"LightGBM best CV: {study.best_value:.4f} (trial {study.best_trial.number})")
    return best


def _default_lgbm_params(n_classes: int) -> Dict[str, Any]:
    params = {
        "n_estimators": 500,
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "class_weight": "balanced",
        "n_jobs": 4,
        "random_state": 42,
        "verbosity": -1,
    }
    if n_classes > 2:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
    return params


# ─────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    n_jobs: int = 4,
    random_state: int = 42,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Optuna hyperparameter search for XGBoost."""
    if not HAS_OPTUNA:
        print("Optuna not installed. Using default params.")
        return _default_xgb_params(len(np.unique(y_train)))

    if not HAS_XGB:
        raise ImportError("xgboost required: pip install xgboost")

    n_classes = len(np.unique(y_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbosity": 0,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }
        if n_classes > 2:
            params["objective"] = "multi:softmax"
            params["num_class"] = n_classes

        model = xgb.XGBClassifier(**params)
        return _cv_score(model, X_train, y_train)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params.copy()
    best.update({
        "tree_method": "hist",
        "n_jobs": n_jobs,
        "random_state": random_state,
        "verbosity": 0,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    })
    if n_classes > 2:
        best["objective"] = "multi:softmax"
        best["num_class"] = n_classes

    print(f"XGBoost best CV: {study.best_value:.4f}")
    return best


def _default_xgb_params(n_classes: int) -> Dict[str, Any]:
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": 42,
        "verbosity": 0,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    }
    if n_classes > 2:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
    return params


# ─────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────

def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    n_jobs: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Optuna hyperparameter search for Random Forest."""
    if not HAS_OPTUNA:
        return _default_rf_params()

    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required")

    from sklearn.ensemble import RandomForestClassifier

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 15, 20]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": "balanced",
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        model = RandomForestClassifier(**params)
        return _cv_score(model, X_train, y_train)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    best.update({"class_weight": "balanced", "n_jobs": n_jobs, "random_state": random_state})
    print(f"Random Forest best CV: {study.best_value:.4f}")
    return best


def _default_rf_params() -> Dict[str, Any]:
    return {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 3,
        "class_weight": "balanced",
        "oob_score": True,
        "n_jobs": 4,
        "random_state": 42,
    }
