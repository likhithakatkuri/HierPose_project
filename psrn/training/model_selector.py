"""Model selector: train all classifiers, compare, and build best ensemble.

Models evaluated:
- LightGBM (lgbm)
- XGBoost (xgb)
- Random Forest (rf)
- SVM with RBF kernel (svm)
- Linear Discriminant Analysis (lda)
- Soft-voting Ensemble of top-3 models (ensemble)

Auto-selection logic:
1. Train each model with 5-fold stratified CV
2. Rank by CV accuracy
3. Build soft-voting ensemble from top-3
4. Return ModelSelectionReport with full comparison table
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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


# ─────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class SingleModelResult:
    model_name: str
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    train_time_s: float
    model_object: Any = field(repr=False)

    def __str__(self) -> str:
        return (
            f"{self.model_name:12s} | CV: {self.cv_mean:.4f}±{self.cv_std:.4f} "
            f"| Time: {self.train_time_s:.1f}s"
        )


@dataclass
class ModelSelectionReport:
    results: List[SingleModelResult]
    best_model_name: str
    ensemble: Any = field(repr=False, default=None)
    ensemble_cv_mean: float = 0.0
    ensemble_cv_std: float = 0.0
    n_classes: int = 0
    n_train: int = 0
    n_features: int = 0

    def summary_table(self) -> str:
        """Return a printable summary table."""
        lines = [
            "=" * 65,
            f"{'Model':<14} {'CV Acc':>8} {'±Std':>7} {'Time(s)':>9}",
            "-" * 65,
        ]
        sorted_results = sorted(self.results, key=lambda r: -r.cv_mean)
        for r in sorted_results:
            lines.append(
                f"{r.model_name:<14} {r.cv_mean:>8.4f} {r.cv_std:>7.4f} {r.train_time_s:>9.1f}"
            )
        if self.ensemble is not None:
            lines.append(
                f"{'ensemble':<14} {self.ensemble_cv_mean:>8.4f} {self.ensemble_cv_std:>7.4f} {'N/A':>9}"
            )
        lines.append("=" * 65)
        lines.append(f"Best single model: {self.best_model_name}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────

def _build_lgbm(n_classes: int, n_jobs: int = 4, random_state: int = 42) -> Any:
    if not HAS_LGBM:
        return None
    params = dict(
        n_estimators=500,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=10,
        min_child_weight=1e-3,
        lambda_l1=0.1,
        lambda_l2=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=-1,
    )
    if n_classes > 2:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
        params["metric"] = "multi_logloss"
    return lgb.LGBMClassifier(**params)


def _build_xgb(n_classes: int, n_jobs: int = 4, random_state: int = 42) -> Any:
    if not HAS_XGB:
        return None
    params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    if n_classes > 2:
        params["objective"] = "multi:softmax"
        params["num_class"] = n_classes
    else:
        params["objective"] = "binary:logistic"
    return xgb.XGBClassifier(**params)


def _build_rf(n_jobs: int = 4, random_state: int = 42) -> Any:
    if not HAS_SKLEARN:
        return None
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _build_svm(C: float = 100.0, gamma: float = 0.001, random_state: int = 42) -> Any:
    if not HAS_SKLEARN:
        return None
    return SVC(
        C=C,
        kernel="rbf",
        gamma=gamma,
        class_weight="balanced",
        probability=True,
        random_state=random_state,
        cache_size=4000,
    )


def _build_lda() -> Any:
    if not HAS_SKLEARN:
        return None
    return LinearDiscriminantAnalysis(solver="svd")


def _tune_svm_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_tune_samples: int = 1000,
) -> Any:
    """Fast grid search for SVM on a small stratified subsample (9 combos, 3-fold)."""
    if not HAS_SKLEARN:
        return _build_svm(random_state=random_state)

    from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

    # Subsample — SVM is O(n²), keep it small for speed
    if len(y_train) > max_tune_samples:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_tune_samples,
                                     random_state=random_state)
        idx, _ = next(sss.split(X_train, y_train))
        X_t, y_t = X_train[idx], y_train[idx]
    else:
        X_t, y_t = X_train, y_train

    param_grid = {
        "C":     [10, 50, 100, 500],
        "gamma": [0.0005, 0.001, 0.005],
    }
    base = SVC(kernel="rbf", class_weight="balanced", probability=False,
               random_state=random_state, cache_size=2000)
    gs = GridSearchCV(base, param_grid, cv=3, scoring="accuracy",
                      n_jobs=-1, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_t, y_t)

    best = gs.best_params_
    print(f"    SVM grid best: C={best['C']}, gamma={best['gamma']} "
          f"(CV={gs.best_score_:.4f}, subsample={len(y_t)})")
    return _build_svm(C=best["C"], gamma=best["gamma"], random_state=random_state)


def _tune_lgbm_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    n_jobs: int = 4,
    random_state: int = 42,
) -> Any:
    """Fast grid search for LightGBM (12 combos, 3-fold CV)."""
    if not HAS_LGBM:
        return _build_lgbm(n_classes, n_jobs, random_state)

    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "num_leaves":    [31, 63],
        "learning_rate": [0.05, 0.1],
    }
    base_params = dict(
        n_estimators=300,
        class_weight="balanced", n_jobs=n_jobs,
        random_state=random_state, verbosity=-1,
    )
    if n_classes > 2:
        base_params["objective"] = "multiclass"
        base_params["num_class"] = n_classes
    base = lgb.LGBMClassifier(**base_params)
    gs = GridSearchCV(base, param_grid, cv=3, scoring="accuracy",
                      n_jobs=n_jobs, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_train, y_train)

    best = gs.best_params_
    print(f"    LGBM grid best: {best} (CV={gs.best_score_:.4f})")
    final_params = {**base_params, **best}
    return lgb.LGBMClassifier(**final_params)


def _tune_lgbm_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    n_trials: int = 20,
    cv: int = 3,
    n_jobs: int = 4,
    random_state: int = 42,
) -> Any:
    """Use Optuna to find best LightGBM hyperparameters."""
    if not (HAS_OPTUNA and HAS_LGBM):
        return _build_lgbm(n_classes, n_jobs, random_state)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1000),
            num_leaves=trial.suggest_int("num_leaves", 20, 80),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 1.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 1.0, log=True),
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=-1,
        )
        if n_classes > 2:
            params["objective"] = "multiclass"
            params["num_class"] = n_classes
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=skf,
                                  scoring="accuracy", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"    LGBM best CV={study.best_value:.4f} params={best}")
    params = {**best, "class_weight": "balanced", "n_jobs": n_jobs,
              "random_state": random_state, "verbosity": -1}
    if n_classes > 2:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
    return lgb.LGBMClassifier(**params)


# ─────────────────────────────────────────────────────────────
# Model Selector
# ─────────────────────────────────────────────────────────────

class ModelSelector:
    """Train all classifiers, compare, and build best ensemble.

    Args:
        n_cv_folds: number of cross-validation folds (default 5)
        n_jobs: parallel jobs for tree-based models
        random_state: for reproducibility
        models_to_train: list of model names to train;
                         None = train all available
        verbose: print progress

    Example:
        selector = ModelSelector(n_cv_folds=5)
        report = selector.fit_all(X_train, y_train)
        print(report.summary_table())
        best_model = report.results[0].model_object  # top CV model
        ensemble = report.ensemble
    """

    AVAILABLE_MODELS = ["lgbm", "xgb", "rf", "svm", "lda"]

    def __init__(
        self,
        n_cv_folds: int = 5,
        n_jobs: int = 4,
        random_state: int = 42,
        models_to_train: Optional[List[str]] = None,
        verbose: bool = True,
        tune_hyperparams: bool = True,
        n_optuna_trials: int = 40,
    ) -> None:
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required. pip install scikit-learn")
        self.n_cv_folds = n_cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models_to_train = models_to_train or self.AVAILABLE_MODELS
        self.verbose = verbose
        self.tune_hyperparams = tune_hyperparams
        self.n_optuna_trials = n_optuna_trials

    def _build_models(self, n_classes: int, X_train=None, y_train=None) -> Dict[str, Any]:
        if self.tune_hyperparams and X_train is not None:
            if self.verbose:
                print("  Tuning SVM (fast grid search)...", flush=True)
            tuned_svm = _tune_svm_grid(
                X_train, y_train,
                random_state=self.random_state,
            )
            # LightGBM: use fixed high-quality params (grid search is too slow on 2640 samples)
            tuned_lgbm = _build_lgbm(n_classes, self.n_jobs, self.random_state)
        else:
            tuned_svm = _build_svm(random_state=self.random_state)
            tuned_lgbm = _build_lgbm(n_classes, self.n_jobs, self.random_state)

        builders = {
            "lgbm": lambda: tuned_lgbm,
            "xgb":  lambda: _build_xgb(n_classes, self.n_jobs, self.random_state),
            "rf":   lambda: _build_rf(self.n_jobs, self.random_state),
            "svm":  lambda: tuned_svm,
            "lda":  lambda: _build_lda(),
        }
        result = {}
        for name in self.models_to_train:
            if name in builders:
                model = builders[name]()
                if model is not None:
                    result[name] = model
                elif self.verbose:
                    print(f"  Skipping {name} (package not installed)")
        return result

    def fit_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ModelSelectionReport:
        """Train all models with CV and return comparison report.

        Args:
            X_train: (n_samples, n_features) training features
            y_train: (n_samples,) integer class labels

        Returns:
            ModelSelectionReport with all model results + ensemble
        """
        n_classes = len(np.unique(y_train))
        n_samples, n_features = X_train.shape

        if self.verbose:
            print(f"\nTraining {len(self.models_to_train)} models on "
                  f"{n_samples} samples, {n_features} features, {n_classes} classes")

        models = self._build_models(n_classes, X_train, y_train)
        cv = StratifiedKFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state
        )

        results: List[SingleModelResult] = []
        for name, model in models.items():
            if self.verbose:
                print(f"  Training {name}...", end=" ", flush=True)

            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring="accuracy", n_jobs=self.n_jobs,
                )
            train_time = time.time() - t0

            if self.verbose:
                print(f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f} ({train_time:.1f}s)")

            results.append(SingleModelResult(
                model_name=name,
                cv_scores=cv_scores.tolist(),
                cv_mean=float(cv_scores.mean()),
                cv_std=float(cv_scores.std()),
                train_time_s=train_time,
                model_object=model,
            ))

        # Rank by CV mean accuracy
        results.sort(key=lambda r: -r.cv_mean)
        best_name = results[0].model_name if results else ""

        # Soft-voting ensemble from top-3 (excluding LDA)
        # NOTE: StackingClassifier overfits on small datasets — soft voting is safer
        ensemble = None
        ensemble_cv_mean = 0.0
        ensemble_cv_std = 0.0

        vote_candidates = [r for r in results if r.model_name != "lda"]
        if len(vote_candidates) >= 2:
            top_k = min(3, len(vote_candidates))
            top_models = [(r.model_name, r.model_object) for r in vote_candidates[:top_k]]
            ensemble = VotingClassifier(estimators=top_models, voting="soft", n_jobs=self.n_jobs)

            if self.verbose:
                print(f"  Training soft-voting ensemble (top-{top_k}: "
                      f"{[m[0] for m in top_models]})...", end=" ", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ens_cv = cross_val_score(
                        ensemble, X_train, y_train,
                        cv=cv, scoring="accuracy", n_jobs=self.n_jobs,
                    )
                ensemble_cv_mean = float(ens_cv.mean())
                ensemble_cv_std = float(ens_cv.std())
                if self.verbose:
                    print(f"CV={ensemble_cv_mean:.4f}±{ensemble_cv_std:.4f}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed ({e}), falling back to best single model")
                ensemble = None

        report = ModelSelectionReport(
            results=results,
            best_model_name=best_name,
            ensemble=ensemble,
            ensemble_cv_mean=ensemble_cv_mean,
            ensemble_cv_std=ensemble_cv_std,
            n_classes=n_classes,
            n_train=n_samples,
            n_features=n_features,
        )

        if self.verbose:
            print()
            print(report.summary_table())

        return report

    def fit_best(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "ensemble",
        X_train_full: Optional[np.ndarray] = None,
        y_train_full: Optional[np.ndarray] = None,
    ) -> Tuple[Any, ModelSelectionReport]:
        """Train all models with CV on X_train, fit final model on X_train_full.

        X_train       — original (non-augmented) data, used for honest CV
        X_train_full  — augmented data, used for final model fit only
                        (if None, falls back to X_train)
        """
        report = self.fit_all(X_train, y_train)

        if model_type in ("ensemble", "auto"):
            if report.ensemble is not None:
                chosen = report.ensemble
            else:
                chosen = report.results[0].model_object
        else:
            matching = [r for r in report.results if r.model_name == model_type]
            if not matching:
                raise ValueError(
                    f"Model '{model_type}' not found in trained models. "
                    f"Available: {[r.model_name for r in report.results]}"
                )
            chosen = matching[0].model_object

        X_fit = X_train_full if X_train_full is not None else X_train
        y_fit = y_train_full if y_train_full is not None else y_train

        if self.verbose:
            print(f"\nFitting final model on {len(y_fit)} samples "
                  f"({'augmented' if X_train_full is not None else 'original'})...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chosen.fit(X_fit, y_fit)

        return chosen, report
