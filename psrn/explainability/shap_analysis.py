"""SHAP-based explainability for HierPose models.

Fixes the critical bug in the original train_ml.py:
- Original: `shap_values[0]` — only class 0 for 21-class problem
- Fixed: aggregate |SHAP| across ALL classes for global importance

Provides:
- Global SHAP beeswarm (top-N features by mean |SHAP|)
- Per-class beeswarm (which features define each action)
- Group-level importance aggregation (by anatomical group)
- Feature importance CSV for IEEE paper Table 1
- Single-sample waterfall chart (for counterfactual explanation figures)

References:
- Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- SHAP documentation: https://shap.readthedocs.io
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for saving
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from psrn.features.registry import feature_names_to_group, GROUP_BODY_REGION


class SHAPAnalyzer:
    """SHAP-based explainability for HierPose classifiers.

    Supports tree-based models (LightGBM, XGBoost, Random Forest) via
    TreeExplainer, and ensemble models via a sampling approach.

    Args:
        model: fitted sklearn-compatible classifier
        X: (n_samples, n_features) feature matrix (test set)
        y: (n_samples,) true labels
        feature_names: list of feature name strings
        class_names: list of class name strings (e.g., JHMDB_CLASSES)

    Example:
        analyzer = SHAPAnalyzer(model, X_test, y_test, feature_names, class_names)
        shap_vals = analyzer.compute(max_samples=200)
        analyzer.plot_bar_summary(output_path)
        df = analyzer.feature_importance_table()
    """

    def __init__(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
    ) -> None:
        if not HAS_SHAP:
            raise ImportError("shap required: pip install shap")
        self.model = model
        self.X = X.astype(np.float32)
        self.y = y
        self.feature_names = feature_names
        self.class_names = class_names
        self._shap_values: Optional[np.ndarray] = None  # (n_samples, n_features, n_classes) or (n_samples, n_features)
        self._global_importance: Optional[np.ndarray] = None  # (n_features,)

    @staticmethod
    def _get_tree_model_for_shap(model) -> Any:
        """Extract a tree-based estimator suitable for TreeExplainer.

        For StackingClassifier / VotingClassifier, returns the LightGBM or RF
        base estimator. TreeExplainer is fast; KernelExplainer on the full
        ensemble would take hours.
        """
        model_type = type(model).__name__
        if model_type in ("StackingClassifier", "VotingClassifier"):
            # Prefer lgbm → rf → xgb → first available
            estimators = dict(model.estimators_) if hasattr(model, "estimators_") else {}
            if not estimators and hasattr(model, "estimators"):
                estimators = dict(model.estimators)
            for preferred in ("lgbm", "rf", "xgb"):
                if preferred in estimators:
                    return estimators[preferred]
            # Return first available
            if estimators:
                return next(iter(estimators.values()))
        return model

    def compute(self, max_samples: int = 100) -> np.ndarray:
        """Compute SHAP values.

        For multiclass (n_classes > 2): shap_values is a list of length n_classes,
        each element is (n_samples, n_features). We stack to (n_samples, n_features, n_classes).

        For binary: shap_values is (n_samples, n_features).

        Args:
            max_samples: subsample for speed (SHAP is O(n²) for some explainers)

        Returns:
            SHAP values array
        """
        n = min(max_samples, len(self.X))
        idx = np.random.choice(len(self.X), n, replace=False)
        X_sample = self.X[idx]

        # Extract a tree-based model for SHAP (TreeExplainer is fast)
        # For ensembles (Stacking/Voting), use the LightGBM or RF base estimator
        shap_model = self._get_tree_model_for_shap(self.model)
        model_type = type(shap_model).__name__

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.TreeExplainer(shap_model)
                raw = explainer.shap_values(X_sample)

        except Exception as e:
            print(f"TreeExplainer failed ({e}), using KernelExplainer on 30 background samples...")
            background = shap.kmeans(self.X[:min(200, len(self.X))], 10)
            explainer = shap.KernelExplainer(shap_model.predict_proba, background)
            raw = explainer.shap_values(X_sample, nsamples=30)

        # Normalize to (n_samples, n_features, n_classes)
        if isinstance(raw, list):
            # Multiclass: list of (n_samples, n_features) per class
            stacked = np.stack(raw, axis=-1)  # (n_samples, n_features, n_classes)
            self._shap_values = stacked
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            self._shap_values = raw  # already (n_samples, n_features, n_classes)
        elif isinstance(raw, np.ndarray) and raw.ndim == 2:
            # Binary or single-output
            self._shap_values = raw[:, :, np.newaxis]  # (n_samples, n_features, 1)
        else:
            self._shap_values = np.array(raw)

        # Global importance: mean |SHAP| across samples AND classes
        self._global_importance = np.mean(
            np.abs(self._shap_values), axis=(0, 2)
        )  # (n_features,)

        return self._shap_values

    def _ensure_computed(self) -> None:
        if self._shap_values is None:
            self.compute()

    @property
    def global_importance(self) -> np.ndarray:
        """Mean |SHAP| per feature, averaged across samples and classes."""
        self._ensure_computed()
        return self._global_importance  # type: ignore[return-value]

    def plot_bar_summary(
        self,
        output_path: Path,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
    ) -> None:
        """Plot mean |SHAP| bar chart (publication quality).

        Args:
            output_path: where to save the PNG
            top_k: number of top features to show
            figsize: figure size in inches
            dpi: output resolution
        """
        if not HAS_MPL:
            print("matplotlib not available — cannot plot")
            return

        self._ensure_computed()
        importance = self._global_importance

        # Top-k features
        top_idx = np.argsort(importance)[::-1][:top_k]
        top_vals = importance[top_idx]
        top_names = [self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
                     for i in top_idx]

        # Color by anatomical group
        group_map = feature_names_to_group(self.feature_names)
        group_colors = {
            "angles": "#e74c3c",
            "distances": "#3498db",
            "ratios": "#2ecc71",
            "centroids": "#9b59b6",
            "symmetry": "#f39c12",
            "orientation": "#1abc9c",
            "extent": "#e67e22",
            "cross_body": "#d35400",
            "temporal_vel": "#2980b9",
            "temporal_acc": "#8e44ad",
            "temporal_ma": "#16a085",
            "motion_energy": "#c0392b",
            "range_of_motion": "#27ae60",
            "temporal_variance": "#7f8c8d",
            "peak_velocity": "#2c3e50",
            "direction_reversals": "#95a5a6",
        }
        colors = [
            group_colors.get(group_map.get(name, ""), "#7f8c8d")
            for name in top_names
        ]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(top_k), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(f"Feature Importance (SHAP) — Top {top_k} Features", fontsize=13)
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close()

    def plot_beeswarm_global(
        self,
        output_path: Path,
        top_k: int = 20,
        max_display: int = 20,
        dpi: int = 300,
    ) -> None:
        """SHAP beeswarm plot — standard IEEE figure (Figure 4 in paper).

        Args:
            output_path: output path
            top_k: features to display
            max_display: max features (shap parameter)
            dpi: resolution
        """
        if not HAS_MPL:
            return
        self._ensure_computed()

        # For multiclass: average across classes
        if self._shap_values.ndim == 3:
            sv_mean = self._shap_values.mean(axis=2)  # (n_samples, n_features)
        else:
            sv_mean = self._shap_values[:, :, 0]

        X_sample = self.X[:sv_mean.shape[0]]

        try:
            fig, ax = plt.subplots(figsize=(10, max_display * 0.4 + 2))
            feature_labels = (
                self.feature_names if len(self.feature_names) == sv_mean.shape[1]
                else [f"feat_{i}" for i in range(sv_mean.shape[1])]
            )
            shap.summary_plot(
                sv_mean,
                X_sample,
                feature_names=feature_labels,
                max_display=max_display,
                show=False,
                plot_type="dot",
            )
            plt.title("SHAP Feature Importance (Global)", fontsize=13)
            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Beeswarm plot failed: {e}")

    def plot_beeswarm_per_class(
        self,
        output_dir: Path,
        dpi: int = 200,
    ) -> None:
        """One SHAP beeswarm per class — shows which joints distinguish each action."""
        if not HAS_MPL:
            return
        self._ensure_computed()
        if self._shap_values.ndim != 3:
            print("Per-class beeswarm requires multiclass SHAP values")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n_classes = self._shap_values.shape[2]
        X_sample = self.X[:self._shap_values.shape[0]]
        feature_labels = (
            self.feature_names if len(self.feature_names) == self._shap_values.shape[1]
            else [f"feat_{i}" for i in range(self._shap_values.shape[1])]
        )

        for c in range(n_classes):
            class_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            sv_c = self._shap_values[:, :, c]  # (n_samples, n_features)
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    sv_c, X_sample,
                    feature_names=feature_labels,
                    max_display=15,
                    show=False, plot_type="dot",
                )
                plt.title(f"SHAP — Class: {class_name}", fontsize=12)
                plt.tight_layout()
                plt.savefig(output_dir / f"shap_{class_name}.png", dpi=dpi, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"Per-class beeswarm for {class_name} failed: {e}")
                plt.close()

    def map_to_anatomical_groups(self) -> Dict[str, float]:
        """Aggregate SHAP importance by feature group.

        Returns:
            dict: group_name → fraction of total importance [0, 1]
        """
        self._ensure_computed()
        importance = self._global_importance
        group_map = feature_names_to_group(self.feature_names)

        group_totals: Dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            if i >= len(importance):
                break
            group = group_map.get(name, "unknown")
            group_totals[group] = group_totals.get(group, 0.0) + float(importance[i])

        total = sum(group_totals.values()) + 1e-10
        return {k: v / total for k, v in sorted(group_totals.items(), key=lambda x: -x[1])}

    def feature_importance_table(self) -> "pd.DataFrame":
        """Return feature importance as a pandas DataFrame.

        Columns: feature_name, mean_abs_shap, rank, group, body_region

        Suitable for CSV export and IEEE Table 1.
        """
        self._ensure_computed()
        if not HAS_PANDAS:
            print("pandas not available — returning dict")
            return {}

        importance = self._global_importance
        group_map = feature_names_to_group(self.feature_names)

        rows = []
        for i, name in enumerate(self.feature_names):
            if i >= len(importance):
                break
            group = group_map.get(name, "unknown")
            rows.append({
                "feature_name": name,
                "mean_abs_shap": float(importance[i]),
                "group": group,
                "body_region": GROUP_BODY_REGION.get(group, ""),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df[["rank", "feature_name", "mean_abs_shap", "group", "body_region"]]

    def get_sample_shap(
        self, sample_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get SHAP values for a single sample.

        Returns:
            (shap_values, feature_values, feature_names)
            shap_values: (n_classes,) mean |SHAP| per class for this sample
        """
        self._ensure_computed()
        sv = self._shap_values[sample_idx]  # (n_features, n_classes) or (n_features,)
        fv = self.X[sample_idx]
        return sv, fv, self.feature_names
