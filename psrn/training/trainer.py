"""HierPose main ML experiment runner.

Orchestrates the complete pipeline:
1. Load JHMDB data (real labels from official split files)
2. Extract hierarchical features (with augmentation on train)
3. Preprocess with StandardScaler (no PCA — preserves SHAP interpretability)
4. Train all models + build ensemble
5. Evaluate on held-out test set (per-class accuracy, F1, confusion matrix)
6. SHAP analysis
7. Save all artifacts

Critical fix over original train_ml.py:
- Labels come from JHMDBSplitLoader (real 0-20 indices), NOT hash(video_name) % 10
- No PCA before SHAP (PCA destroys interpretability)
- Proper train/test split (official JHMDB splits, not random)

Usage:
    config = HierPoseConfig(data_root=Path("data/JHMDB"), split_num=1)
    trainer = HierPoseTrainer(config)
    result = trainer.run()
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from psrn.configs import HierPoseConfig
from psrn.data.jhmdb_loader import JHMDB_CLASSES, JHMDBSplitLoader
from psrn.data.augmentation import AugConfig, KeypointAugmenter
from psrn.features.extractor import FeatureConfig, HierarchicalFeatureExtractor
from psrn.features.registry import ALL_GROUPS, feature_names_to_group

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Full result from one HierPose experiment."""
    model_name: str
    accuracy: float
    accuracy_per_class: Dict[str, float]
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    class_names: List[str]
    cv_mean: float
    cv_std: float
    cv_scores: List[float]
    train_accuracy: float
    n_train: int
    n_test: int
    n_features: int
    feature_names: List[str]
    experiment_name: str
    output_dir: str

    def summary(self) -> str:
        lines = [
            f"=== {self.experiment_name} ===",
            f"Model:        {self.model_name}",
            f"Test Acc:     {self.accuracy:.4f}",
            f"Train Acc:    {self.train_accuracy:.4f}",
            f"Macro F1:     {self.macro_f1:.4f}",
            f"Weighted F1:  {self.weighted_f1:.4f}",
            f"CV (5-fold):  {self.cv_mean:.4f} ± {self.cv_std:.4f}",
            f"Samples:      {self.n_train} train / {self.n_test} test",
            f"Features:     {self.n_features}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "cv_scores": self.cv_scores,
            "train_accuracy": self.train_accuracy,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "experiment_name": self.experiment_name,
            "accuracy_per_class": self.accuracy_per_class,
        }


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class HierPoseTrainer:
    """Main HierPose ML experiment orchestrator.

    Args:
        config: HierPoseConfig
        verbose: print progress
    """

    def __init__(self, config: HierPoseConfig, verbose: bool = True) -> None:
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        self.config = config
        self.verbose = verbose
        self._exp_dir = self._create_experiment_dir()

        # Set random seed
        np.random.seed(config.random_seed)

    def _create_experiment_dir(self) -> Path:
        exp_dir = Path(self.config.output_dir) / self.config.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Data Loading ────────────────────────────────────────

    def _load_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load JHMDB data with real labels and extract features.

        Returns:
            X_train, y_train, X_test, y_test, feature_names
        """
        self._log(f"\nLoading JHMDB split {self.config.split_num} from {self.config.data_root}")

        loader = JHMDBSplitLoader(
            data_root=self.config.data_root,
            split_num=self.config.split_num,
            require_mat=True,
        )

        test_samples = loader.get_test_samples()

        if getattr(self.config, "use_all_splits", False):
            # Combine training samples from all 3 splits (deduplicated by video name)
            seen = set()
            train_samples = []
            for s in [1, 2, 3]:
                l = JHMDBSplitLoader(
                    data_root=self.config.data_root,
                    split_num=s, require_mat=True,
                )
                for samp in l.get_train_samples():
                    if samp.video_name not in seen:
                        seen.add(samp.video_name)
                        train_samples.append(samp)
            self._log(f"  Using all 3 splits: {len(train_samples)} unique train, {len(test_samples)} test samples")
        else:
            train_samples = loader.get_train_samples()
            self._log(f"  Found {len(train_samples)} train, {len(test_samples)} test samples")

        # Feature extractor
        feat_config = FeatureConfig.from_hierpose_config(self.config)
        extractor = HierarchicalFeatureExtractor(feat_config)

        # Augmenter for training only
        augmenter = None
        if self.config.augment_train:
            aug_config = AugConfig(
                flip_prob=self.config.aug_flip_prob,
                rotate_prob=self.config.aug_rotate_prob,
                rotate_max_deg=self.config.aug_rotate_max_deg,
                scale_prob=self.config.aug_scale_prob,
                noise_prob=self.config.aug_noise_prob,
            )
            augmenter = KeypointAugmenter(aug_config)

        # Original samples only — used for honest CV (no leakage from augmented copies)
        self._log("  Extracting train features (original, for CV)...")
        X_train_orig, y_train_orig, feature_names = extractor.extract_batch(
            train_samples,
            cache_dir=Path(self.config.feature_cache_dir),
            augmenter=None,
            n_augment_copies=0,
            show_progress=self.verbose,
        )

        # Augmented copies — used only for final model fitting (after CV)
        X_train_aug, y_train_aug = X_train_orig, y_train_orig  # default: no aug
        if augmenter is not None:
            self._log("  Extracting augmented train features (orig + 3 copies, for final fit)...")
            X_train_aug, y_train_aug, _ = extractor.extract_batch(
                train_samples,
                cache_dir=None,  # Don't cache augmented versions
                augmenter=augmenter,
                n_augment_copies=3,
                show_progress=self.verbose,
            )

        self._log("  Extracting test features...")
        X_test, y_test, _ = extractor.extract_batch(
            test_samples,
            cache_dir=Path(self.config.feature_cache_dir),
            augmenter=None,
            show_progress=self.verbose,
        )

        self._log(
            f"  Features: {X_train_orig.shape[1]} dims | "
            f"Train (CV): {X_train_orig.shape[0]} | Train (fit): {X_train_aug.shape[0]} | Test: {X_test.shape[0]}"
        )

        # Pack augmented data so _train_model can use it for final fitting
        self._X_train_aug = X_train_aug
        self._y_train_aug = y_train_aug

        # Return original data — CV will be run on this (no leakage)
        X_train, y_train = X_train_orig, y_train_orig

        return X_train, y_train, X_test, y_test, feature_names

    # ── Preprocessing ────────────────────────────────────────

    def _preprocess(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Build a scaler+selector preprocessing pipeline fitted on original train only.

        Returns raw (unscaled) arrays — the preprocessing pipeline is stored in
        self._preproc_pipeline and applied by _train_model when needed.
        All transforms happen inside the pipeline so feature counts always match.
        """
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.pipeline import Pipeline

        k = min(200, X_train.shape[1])
        self._log(f"  Building preprocessing pipeline: StandardScaler → SelectKBest(k={k})")

        preproc = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=k)),
        ])

        X_train_pp = preproc.fit_transform(X_train, y_train).astype(np.float32)
        X_test_pp  = preproc.transform(X_test).astype(np.float32)

        self._preproc_pipeline = preproc
        self._feature_selector = preproc.named_steps["selector"]

        # Update feature names
        if hasattr(self, "_all_feature_names") and self._all_feature_names:
            mask = preproc.named_steps["selector"].get_support()
            self._all_feature_names = [
                n for n, m in zip(self._all_feature_names, mask) if m
            ]

        # Transform augmented data through the same pipeline
        if hasattr(self, "_X_train_aug") and self._X_train_aug is not None:
            self._X_train_aug = preproc.transform(self._X_train_aug).astype(np.float32)

        self._log(f"  After selection: {X_train_pp.shape[1]} features | "
                  f"Train: {X_train_pp.shape[0]} | Test: {X_test_pp.shape[0]}")

        return X_train_pp, X_test_pp, preproc

    def _extract_feature_subset(
        self,
        X_all: np.ndarray,
        X_test: np.ndarray,
        enabled_groups: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a column subset of features for ablation studies.

        This is only used by AblationStudy — in normal training, all features are used.
        """
        if not hasattr(self, "_all_feature_names"):
            return X_all, X_test

        group_map = feature_names_to_group(self._all_feature_names)
        mask = np.array([
            group_map.get(name, "unknown") in enabled_groups
            for name in self._all_feature_names
        ])

        if not mask.any():
            return np.zeros((X_all.shape[0], 0)), np.zeros((X_test.shape[0], 0))

        return X_all[:, mask], X_test[:, mask]

    # ── Training ─────────────────────────────────────────────

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[Any, "ModelSelectionReport"]:  # noqa: F821
        """Train models and select best.

        If config.tune_hyperparams, runs Optuna before training.

        Returns:
            (fitted_model, report)
        """
        from psrn.training.model_selector import ModelSelector

        selector = ModelSelector(
            n_cv_folds=self.config.n_cv_folds,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_seed,
            models_to_train=self._get_models_to_train(),
            verbose=self.verbose,
            tune_hyperparams=self.config.tune_hyperparams,
            n_optuna_trials=self.config.n_optuna_trials,
        )

        model_type = self.config.model_type
        # CV on original data; final fit on augmented data (avoids leakage)
        X_fit = getattr(self, "_X_train_aug", X_train)
        y_fit = getattr(self, "_y_train_aug", y_train)
        fitted_model, report = selector.fit_best(
            X_train, y_train, model_type, X_train_full=X_fit, y_train_full=y_fit
        )
        return fitted_model, report

    def _get_models_to_train(self) -> List[str]:
        """Return list of model names to train based on config."""
        if self.config.model_type == "ensemble":
            return ["lgbm", "xgb", "rf", "svm", "lda"]
        elif self.config.model_type == "all":
            return ["lgbm", "xgb", "rf", "svm", "lda"]
        else:
            # Single model + a few for comparison
            return [self.config.model_type, "lgbm"] if self.config.model_type != "lgbm" else ["lgbm", "xgb", "rf"]

    # ── Evaluation ───────────────────────────────────────────

    def _evaluate(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cv_mean: float,
        cv_std: float,
        cv_scores: List[float],
        model_name: str,
    ) -> ExperimentResult:
        """Compute full evaluation metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)

        test_acc = float(accuracy_score(y_test, y_pred))
        train_acc = float(accuracy_score(y_train, y_pred_train))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(JHMDB_CLASSES))))

        # Per-class accuracy
        per_class: Dict[str, float] = {}
        for c_idx, c_name in enumerate(JHMDB_CLASSES):
            mask = y_test == c_idx
            if mask.sum() > 0:
                per_class[c_name] = float(accuracy_score(y_test[mask], y_pred[mask]))

        self._log("\n" + "=" * 50)
        self._log(f"Test Accuracy:  {test_acc:.4f}")
        self._log(f"Macro F1:       {macro_f1:.4f}")
        self._log(f"Weighted F1:    {weighted_f1:.4f}")
        self._log("=" * 50)

        return ExperimentResult(
            model_name=model_name,
            accuracy=test_acc,
            accuracy_per_class=per_class,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            confusion_matrix=cm,
            class_names=list(JHMDB_CLASSES),
            cv_mean=cv_mean,
            cv_std=cv_std,
            cv_scores=cv_scores,
            train_accuracy=train_acc,
            n_train=len(y_train),
            n_test=len(y_test),
            n_features=X_test.shape[1],
            feature_names=feature_names,
            experiment_name=self.config.experiment_name,
            output_dir=str(self._exp_dir),
        )

    # ── Artifacts ────────────────────────────────────────────

    def _save_artifacts(
        self,
        model: Any,
        scaler: StandardScaler,
        result: ExperimentResult,
        report: Any,
    ) -> None:
        """Save all experiment artifacts to output directory."""
        if not HAS_JOBLIB:
            self._log("Warning: joblib not installed — models not saved")
            return

        # Models
        joblib.dump(model, self._exp_dir / "model.pkl")
        joblib.dump(scaler, self._exp_dir / "scaler.pkl")
        if hasattr(self, "_feature_selector") and self._feature_selector is not None:
            joblib.dump(self._feature_selector, self._exp_dir / "feature_selector.pkl")

        # Results JSON
        result_dict = result.to_dict()
        result_dict["confusion_matrix"] = result.confusion_matrix.tolist()
        with open(self._exp_dir / "results.json", "w") as f:
            json.dump(result_dict, f, indent=2)

        # Model selection report
        if report is not None:
            report_data = [
                {
                    "model": r.model_name,
                    "cv_mean": r.cv_mean,
                    "cv_std": r.cv_std,
                    "train_time_s": r.train_time_s,
                }
                for r in report.results
            ]
            if report.ensemble is not None:
                report_data.append({
                    "model": "ensemble",
                    "cv_mean": report.ensemble_cv_mean,
                    "cv_std": report.ensemble_cv_std,
                    "train_time_s": -1,
                })
                joblib.dump(report.ensemble, self._exp_dir / "ensemble.pkl")

            with open(self._exp_dir / "model_comparison.json", "w") as f:
                json.dump(report_data, f, indent=2)

        self._log(f"\nArtifacts saved to: {self._exp_dir}")

    # ── Main Run ─────────────────────────────────────────────

    def run(self) -> ExperimentResult:
        """Execute the full training pipeline.

        Returns:
            ExperimentResult with all metrics
        """
        t_start = time.time()
        self._log(f"\n{'='*60}")
        self._log(f"HierPose Experiment: {self.config.experiment_name}")
        self._log(f"{'='*60}")

        # 1. Load data
        X_train, y_train, X_test, y_test, feature_names = self._load_data()
        self._all_feature_names = feature_names

        # 2. Preprocess — pipeline(scaler + selector) fitted on original data only
        X_train_sc, X_test_sc, preproc = self._preprocess(X_train, X_test, y_train)
        scaler = preproc  # kept for artifact saving compatibility

        # 3. Train
        model, report = self._train_model(X_train_sc, y_train)

        # Get CV scores from report
        cv_mean = report.results[0].cv_mean if report.results else 0.0
        cv_std = report.results[0].cv_std if report.results else 0.0
        cv_scores = report.results[0].cv_scores if report.results else []
        model_name = self.config.model_type

        if self.config.model_type in ("ensemble", "auto") and report.ensemble is not None:
            model_name = "ensemble"
            cv_mean = report.ensemble_cv_mean
            cv_std = report.ensemble_cv_std

        # 4. Evaluate
        result = self._evaluate(
            model, X_train_sc, y_train, X_test_sc, y_test,
            feature_names, cv_mean, cv_std, cv_scores, model_name,
        )

        # 5. Save artifacts
        self._save_artifacts(model, scaler, result, report)

        # 6. SHAP analysis (optional)
        if self.config.save_shap:
            try:
                from psrn.explainability.shap_analysis import SHAPAnalyzer
                self._log("\nRunning SHAP analysis...")
                analyzer = SHAPAnalyzer(
                    model=model,
                    X=X_test_sc,
                    y=y_test,
                    feature_names=feature_names,
                    class_names=JHMDB_CLASSES,
                )
                analyzer.compute(max_samples=min(200, len(X_test_sc)))
                analyzer.plot_bar_summary(self._exp_dir / "shap_importance.png")
                analyzer.feature_importance_table().to_csv(
                    self._exp_dir / "shap_feature_importance.csv", index=False
                )
                group_imp = analyzer.map_to_anatomical_groups()
                with open(self._exp_dir / "shap_group_importance.json", "w") as f:
                    json.dump(group_imp, f, indent=2)
                self._log("  SHAP analysis complete")
            except Exception as e:
                self._log(f"  SHAP analysis failed: {e}")

        # 7. Plots
        if self.config.save_confusion_matrix:
            try:
                from psrn.visualization.plots import plot_confusion_matrix
                plot_confusion_matrix(
                    result.confusion_matrix,
                    result.class_names,
                    self._exp_dir / "confusion_matrix.png",
                )
                self._log("  Confusion matrix saved")
            except Exception as e:
                self._log(f"  Confusion matrix failed: {e}")

        if self.config.save_tsne:
            try:
                from psrn.visualization.plots import plot_tsne_feature_space
                plot_tsne_feature_space(
                    X_test_sc, y_test, JHMDB_CLASSES,
                    self._exp_dir / "tsne_feature_space.png",
                )
                self._log("  t-SNE plot saved")
            except Exception as e:
                self._log(f"  t-SNE failed: {e}")

        elapsed = time.time() - t_start
        self._log(f"\nTotal time: {elapsed:.1f}s")
        self._log(result.summary())

        return result
