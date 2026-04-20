"""Ablation study engine for HierPose feature groups.

Runs systematic experiments to quantify each feature group's contribution.
Two modes:
1. leave_one_out: train with all groups except one, measure accuracy drop
2. incremental: add groups one-by-one, measure accuracy gain at each step

Results map directly to IEEE Table 3 (ablation study).

Usage:
    study = AblationStudy(base_config, trainer_class=HierPoseTrainer)
    df = study.run_leave_one_out()
    print(df.to_string())
    df.to_csv("output/ablation/leave_one_out.csv")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from psrn.features.registry import (
    ALL_GROUPS, STATIC_GROUPS, TEMPORAL_PER_FRAME_GROUPS,
    TEMPORAL_CLIP_GROUPS, get_ablation_subsets,
)


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class AblationRun:
    """Result of a single ablation experiment (one group subset)."""
    run_name: str
    enabled_groups: List[str]
    disabled_group: Optional[str]       # for leave-one-out mode
    cv_mean: float
    cv_std: float
    test_accuracy: float
    n_features: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "disabled_group": self.disabled_group,
            "enabled_groups": ",".join(self.enabled_groups),
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "test_accuracy": self.test_accuracy,
            "n_features": self.n_features,
        }


# ─────────────────────────────────────────────────────────────
# Ablation Study
# ─────────────────────────────────────────────────────────────

class AblationStudy:
    """Systematic feature group ablation.

    The study re-runs the full HierPoseTrainer for each feature subset.
    This ensures fair comparison (same CV folds, same preprocessing).

    Args:
        base_config: HierPoseConfig with full feature set
        trainer_class: HierPoseTrainer class (for dependency injection)
        output_dir: directory to save results CSV and JSON
        verbose: print progress
    """

    def __init__(
        self,
        base_config,              # HierPoseConfig
        trainer_class=None,       # HierPoseTrainer (avoid circular import)
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.base_config = base_config
        self._trainer_class = trainer_class
        self.output_dir = Path(output_dir or base_config.output_dir) / "ablation"
        self.verbose = verbose

    def _get_trainer_class(self):
        if self._trainer_class is not None:
            return self._trainer_class
        from psrn.training.trainer import HierPoseTrainer
        return HierPoseTrainer

    def _run_experiment(
        self,
        run_name: str,
        enabled_groups: List[str],
        disabled_group: Optional[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> AblationRun:
        """Run a single experiment with the given feature group subset."""
        if self.verbose:
            status = f"all groups except '{disabled_group}'" if disabled_group else f"groups: {enabled_groups[:3]}..."
            print(f"  [{run_name}] {status}", end=" ", flush=True)

        # Find indices of features belonging to enabled groups
        from psrn.features.registry import feature_names_to_group

        # Use the trainer with modified config
        import copy
        config = copy.deepcopy(self.base_config)
        config.enabled_feature_groups = enabled_groups

        TrainerClass = self._get_trainer_class()
        trainer = TrainerClass(config, verbose=False)

        # Extract features for the given group subset
        X_tr_sub, X_te_sub = trainer._extract_feature_subset(
            X_train, X_test, enabled_groups
        )

        if X_tr_sub.shape[1] == 0:
            if self.verbose:
                print("  [SKIP] no features in subset")
            return AblationRun(
                run_name=run_name,
                enabled_groups=enabled_groups,
                disabled_group=disabled_group,
                cv_mean=0.0,
                cv_std=0.0,
                test_accuracy=0.0,
                n_features=0,
            )

        # Quick 5-fold CV on train set
        from psrn.training.model_selector import _build_lgbm
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        import warnings

        n_classes = len(np.unique(y_train))
        model = _build_lgbm(n_classes, n_jobs=2, random_state=self.base_config.random_seed)

        cv = StratifiedKFold(
            n_splits=self.base_config.n_cv_folds,
            shuffle=True,
            random_state=self.base_config.random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X_tr_sub, y_train, cv=cv, scoring="accuracy")
            model.fit(X_tr_sub, y_train)
            test_acc = float(np.mean(model.predict(X_te_sub) == y_test))

        if self.verbose:
            print(f"CV={scores.mean():.4f} test={test_acc:.4f}")

        return AblationRun(
            run_name=run_name,
            enabled_groups=enabled_groups,
            disabled_group=disabled_group,
            cv_mean=float(scores.mean()),
            cv_std=float(scores.std()),
            test_accuracy=test_acc,
            n_features=X_tr_sub.shape[1],
        )

    def run_leave_one_out(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> "pd.DataFrame":
        """Run leave-one-out ablation.

        For each group G, trains with all groups except G.
        Records accuracy drop (baseline - ablated) to show each group's importance.

        Args:
            X_train: (n_train, n_all_features)
            y_train: training labels
            X_test: (n_test, n_all_features)
            y_test: test labels
            feature_names: list of all feature names (same order as X columns)

        Returns:
            DataFrame with columns: group_name, cv_accuracy, test_accuracy,
                                    n_features, delta_cv_from_baseline, delta_test_from_baseline
        """
        if self.verbose:
            print("\n=== Leave-One-Out Ablation Study ===")

        # First: baseline (all groups)
        baseline = self._run_experiment(
            "baseline_all", ALL_GROUPS, None,
            X_train, y_train, X_test, y_test,
        )

        results: List[Dict] = []
        results.append({
            "group_disabled": "none (baseline)",
            "cv_accuracy": baseline.cv_mean,
            "cv_std": baseline.cv_std,
            "test_accuracy": baseline.test_accuracy,
            "n_features": baseline.n_features,
            "delta_cv": 0.0,
            "delta_test": 0.0,
        })

        for group in ALL_GROUPS:
            subset = [g for g in ALL_GROUPS if g != group]
            run = self._run_experiment(
                f"no_{group}", subset, group,
                X_train, y_train, X_test, y_test,
            )
            results.append({
                "group_disabled": group,
                "cv_accuracy": run.cv_mean,
                "cv_std": run.cv_std,
                "test_accuracy": run.test_accuracy,
                "n_features": run.n_features,
                "delta_cv": run.cv_mean - baseline.cv_mean,
                "delta_test": run.test_accuracy - baseline.test_accuracy,
            })

        # Save and return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_PANDAS:
            import pandas as pd
            df = pd.DataFrame(results)
            df = df.sort_values("delta_cv")  # most important group first (largest drop)
            df.to_csv(self.output_dir / "leave_one_out.csv", index=False)
            if self.verbose:
                print(f"\nResults saved to {self.output_dir / 'leave_one_out.csv'}")
            return df
        else:
            with open(self.output_dir / "leave_one_out.json", "w") as f:
                json.dump(results, f, indent=2)
            return results

    def run_incremental(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> "pd.DataFrame":
        """Run incremental ablation.

        Adds feature groups one-by-one in anatomical order.
        Shows accuracy gain curve — perfect for a research figure.

        Args:
            Same as run_leave_one_out

        Returns:
            DataFrame showing cumulative accuracy as groups are added
        """
        if self.verbose:
            print("\n=== Incremental Ablation Study ===")

        # Order: static groups first (anatomically ordered), then temporal
        ordered_groups = (
            ["angles", "distances", "ratios", "centroids",
             "symmetry", "orientation", "extent", "cross_body"]
            + ["temporal_vel", "temporal_acc", "temporal_ma", "motion_energy"]
            + ["range_of_motion", "temporal_variance", "peak_velocity", "direction_reversals"]
        )
        # Filter to only groups that exist
        ordered_groups = [g for g in ordered_groups if g in ALL_GROUPS]

        results: List[Dict] = []
        for i, group in enumerate(ordered_groups):
            subset = ordered_groups[:i + 1]
            run = self._run_experiment(
                f"step{i+1:02d}_{group}", subset, None,
                X_train, y_train, X_test, y_test,
            )
            results.append({
                "step": i + 1,
                "group_added": group,
                "groups_included": ",".join(subset),
                "cv_accuracy": run.cv_mean,
                "cv_std": run.cv_std,
                "test_accuracy": run.test_accuracy,
                "n_features": run.n_features,
            })

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_PANDAS:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir / "incremental.csv", index=False)
            return df
        else:
            with open(self.output_dir / "incremental.json", "w") as f:
                json.dump(results, f, indent=2)
            return results

    def run_full(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Run both leave-one-out and incremental ablation studies."""
        loo = self.run_leave_one_out(X_train, y_train, X_test, y_test, feature_names)
        inc = self.run_incremental(X_train, y_train, X_test, y_test, feature_names)
        return {"leave_one_out": loo, "incremental": inc}
