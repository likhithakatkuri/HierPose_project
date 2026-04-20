"""Counterfactual Pose Guidance (CPG) engine.

The CPG engine answers: "What minimal changes to the current pose
would make the model predict a different (target) class?"

This is the novel XAI contribution of HierPose. For each input pose,
CPG generates:
1. A list of specific joint angle/distance corrections ranked by importance
2. Human-readable instructions in domain-specific language
3. The corrected feature vector (for visualization)

Method:
- Uses scipy.optimize.minimize to find minimal L2 perturbation in
  feature space that flips the model prediction to the target class
- Corrections are mapped back to anatomical feature descriptions
- Domain-specific feedback templates format the output for each use case

Applications:
- Medical: "For accurate PA chest X-ray, rotate left shoulder 12° forward"
- Sports: "Improve squat: bend knees 18° more (112° → 90°)"
- Ergonomics: "HIGH RISK: reduce neck flexion by 22° (45° → 23°)"
- Action recognition: "To match 'wave': increase wrist velocity by 34%"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize, Bounds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class PoseCorrection:
    """A single corrective instruction for one feature."""
    feature_name: str
    current_value: float
    target_value: float
    delta: float                    # target - current (signed)
    delta_abs: float                # |delta|
    direction: str                  # "increase" | "decrease" | "change"
    body_part: str                  # anatomical body part label
    feature_group: str              # e.g. "angles", "temporal_vel"
    importance_rank: int            # 1 = most critical
    is_angle: bool = False          # True if units are degrees

    def to_text(self, domain: str = "general") -> str:
        """Format as human-readable instruction."""
        unit = "°" if self.is_angle else ""
        direction_verb = {
            "increase": "increase",
            "decrease": "decrease",
            "change": "adjust",
        }.get(self.direction, "adjust")

        part = self.body_part.replace("_", " ")
        return (
            f"{direction_verb.capitalize()} {part} by "
            f"{abs(self.delta):.1f}{unit} "
            f"(currently {self.current_value:.1f}{unit}, "
            f"target {self.target_value:.1f}{unit})"
        )


@dataclass
class CounterfactualResult:
    """Full counterfactual explanation for one sample."""
    original_features: np.ndarray          # (n_features,) original
    counterfactual_features: np.ndarray    # (n_features,) minimal perturbation
    predicted_original: str
    predicted_target: str
    corrections: List[PoseCorrection]      # ranked by importance (most → least)
    l2_distance: float                     # feature-space distance of perturbation
    optimization_success: bool
    n_corrections: int                     # len(corrections)

    def summary(self, domain: str = "general", max_corrections: int = 5) -> str:
        """Human-readable correction summary."""
        lines = [
            f"Current pose: {self.predicted_original}",
            f"Target pose:  {self.predicted_target}",
            f"Change score: {self.l2_distance:.4f}",
            "",
            f"Required corrections ({min(self.n_corrections, max_corrections)}/{self.n_corrections} shown):",
        ]
        for i, c in enumerate(self.corrections[:max_corrections]):
            lines.append(f"  {i+1}. {c.to_text(domain)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Feedback templates per domain
# ─────────────────────────────────────────────────────────────

DOMAIN_TEMPLATES: Dict[str, Dict[str, str]] = {
    "medical": {
        "intro": "Patient positioning adjustment needed for {target}:",
        "angle_increase": "Extend {part} by {delta:.1f}° (currently {current:.1f}°, target {target_val:.1f}°)",
        "angle_decrease": "Reduce {part} angle by {delta:.1f}° (currently {current:.1f}°, target {target_val:.1f}°)",
        "distance_change": "Adjust {part} spacing by {delta:.2f} units",
        "footer": "Estimated repositioning required: {n_corrections} adjustments",
    },
    "sports": {
        "intro": "Form correction needed to achieve {target}:",
        "angle_increase": "Bend {part} further: {current:.1f}° → {target_val:.1f}° (increase {delta:.1f}°)",
        "angle_decrease": "Straighten {part}: {current:.1f}° → {target_val:.1f}° (decrease {delta:.1f}°)",
        "distance_change": "Move {part} {delta:.2f} units",
        "footer": "Form score will improve with {n_corrections} adjustments",
    },
    "ergonomics": {
        "intro": "⚠ POSTURE RISK: Corrections needed for safe {target}:",
        "angle_increase": "RAISE {part}: needs {delta:.1f}° more (current {current:.1f}°)",
        "angle_decrease": "LOWER {part}: reduce by {delta:.1f}° (current {current:.1f}°)",
        "distance_change": "Reposition {part} by {delta:.2f} normalized units",
        "footer": "Apply {n_corrections} corrections to reduce injury risk",
    },
    "general": {
        "intro": "To classify as '{target}', adjust:",
        "angle_increase": "Increase {part} angle: {current:.1f}° → {target_val:.1f}°",
        "angle_decrease": "Decrease {part} angle: {current:.1f}° → {target_val:.1f}°",
        "distance_change": "Adjust {part}: Δ={delta:.3f}",
        "footer": "{n_corrections} correction(s) needed",
    },
}


# ─────────────────────────────────────────────────────────────
# CPG Engine
# ─────────────────────────────────────────────────────────────

class CounterfactualPoseGuide:
    """Generate minimal pose corrections to achieve a target prediction.

    Args:
        model: fitted sklearn-compatible classifier (with predict_proba)
        scaler: fitted StandardScaler (used to invert feature normalization)
        feature_names: list of feature name strings
        class_names: list of class name strings
        domain: domain name for feedback templates ("medical"/"sports"/"ergonomics"/"general")
        min_delta_threshold: minimum |delta| to include in corrections list
        max_corrections: maximum number of corrections to return

    Example:
        cpg = CounterfactualPoseGuide(model, scaler, feature_names, class_names)
        result = cpg.generate(current_features_scaled, target_class="wave")
        print(result.summary(domain="sports"))
    """

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_names: List[str],
        class_names: List[str],
        domain: str = "general",
        min_delta_threshold: float = 0.01,
        max_corrections: int = 10,
    ) -> None:
        if not HAS_SCIPY:
            raise ImportError("scipy required for counterfactuals: pip install scipy")
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.class_names = class_names
        self.domain = domain
        self.min_delta_threshold = min_delta_threshold
        self.max_corrections = max_corrections
        self._class_idx = {name: i for i, name in enumerate(class_names)}

    def generate(
        self,
        features_scaled: np.ndarray,
        target_class: str,
        max_iter: int = 500,
        lambda_l2: float = 1.0,
    ) -> CounterfactualResult:
        """Find minimal feature perturbation to predict target_class.

        Optimization objective:
            minimize  ||x' - x||_2 + λ * penalty
            subject to: model.predict(x') == target_class_idx

        Uses L-BFGS-B optimizer with a differentiable loss that pushes
        the model's probability for target_class above 0.5.

        Args:
            features_scaled: (n_features,) scaled feature vector
            target_class: desired prediction class name
            max_iter: optimization iterations
            lambda_l2: weight for L2 distance term

        Returns:
            CounterfactualResult
        """
        x0 = features_scaled.flatten().astype(np.float64)
        target_idx = self._class_idx.get(target_class, 0)
        n_classes = len(self.class_names)

        # Get current prediction
        proba = self.model.predict_proba(x0.reshape(1, -1))[0]
        predicted_original = self.class_names[int(np.argmax(proba))]

        def objective(x: np.ndarray) -> float:
            """Minimize: L2 distance + classification loss for target."""
            dist = float(np.sum((x - x0) ** 2))
            p = self.model.predict_proba(x.reshape(1, -1))[0]
            target_prob = p[target_idx] if target_idx < len(p) else 0.0
            # Classification loss: push target_prob toward 1
            clf_loss = -float(np.log(target_prob + 1e-10))
            return lambda_l2 * dist + clf_loss

        def gradient_approx(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
            """Numerical gradient (finite differences)."""
            grad = np.zeros_like(x)
            fx = objective(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                grad[i] = (objective(x_plus) - fx) / eps
            return grad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = minimize(
                    objective,
                    x0,
                    jac=gradient_approx,
                    method="L-BFGS-B",
                    options={"maxiter": max_iter, "ftol": 1e-8},
                )
                x_cf = res.x
                success = res.success
            except Exception:
                x_cf = x0.copy()
                success = False

        # Check prediction on counterfactual
        pred_cf = self.class_names[
            int(np.argmax(self.model.predict_proba(x_cf.reshape(1, -1))[0]))
        ]

        # Build correction list from feature differences
        l2_dist = float(np.sqrt(np.sum((x_cf - x0) ** 2)))
        corrections = self._build_corrections(x0, x_cf)

        return CounterfactualResult(
            original_features=x0,
            counterfactual_features=x_cf,
            predicted_original=predicted_original,
            predicted_target=pred_cf,
            corrections=corrections,
            l2_distance=l2_dist,
            optimization_success=success,
            n_corrections=len(corrections),
        )

    def _build_corrections(
        self,
        x_orig: np.ndarray,
        x_cf: np.ndarray,
    ) -> List[PoseCorrection]:
        """Map feature differences to anatomical corrections.

        Inverts StandardScaler to get original feature values,
        then computes delta in interpretable units (degrees, normalized dist, etc.)
        """
        from psrn.features.registry import feature_names_to_group

        # Invert scaling to get human-readable values
        if self.scaler is not None and HAS_SKLEARN:
            try:
                x_orig_raw = self.scaler.inverse_transform(x_orig.reshape(1, -1))[0]
                x_cf_raw = self.scaler.inverse_transform(x_cf.reshape(1, -1))[0]
            except Exception:
                x_orig_raw = x_orig
                x_cf_raw = x_cf
        else:
            x_orig_raw = x_orig
            x_cf_raw = x_cf

        group_map = feature_names_to_group(self.feature_names)
        deltas = x_cf_raw - x_orig_raw

        corrections: List[PoseCorrection] = []
        for i, (name, delta) in enumerate(zip(self.feature_names, deltas)):
            if abs(delta) < self.min_delta_threshold:
                continue

            group = group_map.get(name, "unknown")
            is_angle = "_deg" in name or "_angle" in name
            body_part = self._extract_body_part(name)

            corrections.append(PoseCorrection(
                feature_name=name,
                current_value=float(x_orig_raw[i]),
                target_value=float(x_cf_raw[i]),
                delta=float(delta),
                delta_abs=float(abs(delta)),
                direction="increase" if delta > 0 else "decrease",
                body_part=body_part,
                feature_group=group,
                importance_rank=0,  # set after sort
                is_angle=is_angle,
            ))

        # Sort by |delta| descending and assign ranks
        corrections.sort(key=lambda c: -c.delta_abs)
        for rank, c in enumerate(corrections):
            c.importance_rank = rank + 1

        return corrections[:self.max_corrections]

    def _extract_body_part(self, feature_name: str) -> str:
        """Extract body part label from a feature name."""
        # Remove prefixes like "angle_", "dist_", "vel_", etc.
        for prefix in ("angle_", "dist_", "ratio_", "centroid_", "sym_", "orient_",
                       "extent_", "cross_", "vel_ma_", "vel_", "acc_", "motion_",
                       "rom_", "var_", "peak_timing_", "reversals_"):
            if feature_name.startswith(prefix):
                part = feature_name[len(prefix):]
                # Remove suffix like "_cos", "_deg", "_mean", "_std"
                for suffix in ("_cos", "_deg", "_mean", "_std", "_q25", "_q75"):
                    if part.endswith(suffix):
                        part = part[:-len(suffix)]
                        break
                return part
        return feature_name

    def explain_as_text(
        self,
        result: CounterfactualResult,
        max_corrections: int = 5,
    ) -> str:
        """Format CounterfactualResult as human-readable domain-specific text.

        Args:
            result: CounterfactualResult from generate()
            max_corrections: max items to include

        Returns:
            Multi-line string with correction instructions
        """
        template = DOMAIN_TEMPLATES.get(self.domain, DOMAIN_TEMPLATES["general"])

        lines = [
            template["intro"].format(target=result.predicted_target),
            "",
        ]

        for i, c in enumerate(result.corrections[:max_corrections]):
            part = c.body_part.replace("_", " ")
            delta = abs(c.delta)
            if c.is_angle:
                key = "angle_increase" if c.direction == "increase" else "angle_decrease"
                line = template[key].format(
                    part=part,
                    delta=delta,
                    current=c.current_value,
                    target_val=c.target_value,
                )
            else:
                line = template["distance_change"].format(
                    part=part,
                    delta=c.delta,
                )
            lines.append(f"  {i+1}. {line}")

        lines.append("")
        lines.append(
            template["footer"].format(n_corrections=result.n_corrections)
        )

        if not result.optimization_success:
            lines.append("⚠ Note: Optimization may not have converged — corrections are approximate")

        return "\n".join(lines)

    def generate_for_misclassification(
        self,
        features_scaled: np.ndarray,
        true_label: str,
    ) -> Tuple[CounterfactualResult, str]:
        """Explain why a sample was misclassified and how to correct it.

        Args:
            features_scaled: scaled feature vector
            true_label: the true class name

        Returns:
            (CounterfactualResult, explanation text)
        """
        proba = self.model.predict_proba(features_scaled.reshape(1, -1))[0]
        predicted = self.class_names[int(np.argmax(proba))]

        result = self.generate(features_scaled, target_class=true_label)

        explanation = (
            f"Misclassification Analysis\n"
            f"{'=' * 40}\n"
            f"Predicted:  {predicted} ({proba.max():.1%} confidence)\n"
            f"True label: {true_label}\n\n"
        ) + self.explain_as_text(result)

        return result, explanation
