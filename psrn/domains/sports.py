"""Sports biomechanics domain modules.

SquatFormDomain:
    Analyzes squat form quality and flags common errors such as
    insufficient depth, knee cave, and excessive forward lean.
    Provides corrections like "Squat depth insufficient. Knee angle 112°, target ≤90°".

GenericSportsDomain:
    A flexible domain for any sport where joint angle deviations
    from reference poses determine risk level.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from psrn.domains.base import BaseDomain, DomainPoseClass, PoseScore

# ─────────────────────────────────────────────────────────────
# JHMDB joint constants (re-declared to avoid circular import)
# ─────────────────────────────────────────────────────────────
NECK = 0
BELLY = 1
FACE = 2
R_SHOULDER = 3
L_SHOULDER = 4
R_HIP = 5
L_HIP = 6
R_ELBOW = 7
L_ELBOW = 8
R_KNEE = 9
L_KNEE = 10
R_WRIST = 11
L_WRIST = 12
R_ANKLE = 13
L_ANKLE = 14


# ─────────────────────────────────────────────────────────────
# Squat Form Domain
# ─────────────────────────────────────────────────────────────

class SquatFormDomain(BaseDomain):
    """Biomechanical squat form assessment.

    Classes:
        correct_depth   — knee ≤90°, back upright, knees over ankles
        too_shallow     — knee angle > 90° (insufficient depth)
        knee_cave       — knees collapsing inward (valgus)
        forward_lean    — excessive trunk inclination > 45°
        good_form       — composite score ≥ 80 across all criteria

    Scoring criteria:
        - Knee angle contribution  : target ≤ 90° (flexion depth)
        - Back angle contribution  : target ≤ 45° from vertical
        - Knee alignment           : L/R knee within 15° of ankle plumb line
        - Depth score              : derived from hip height relative to knees
    """

    @property
    def domain_name(self) -> str:
        return "squat_form"

    @property
    def display_name(self) -> str:
        return "Squat Form Analyser"

    @property
    def description(self) -> str:
        return "Assess and correct squat biomechanics — depth, back angle, and knee tracking"

    @property
    def pose_classes(self) -> List[DomainPoseClass]:
        return [
            DomainPoseClass(
                name="correct_depth",
                display_name="Correct Depth Squat",
                description="Hip crease below parallel, knees tracking over ankles, back neutral",
                acceptable_ranges={
                    "angle_r_knee_deg": (70.0, 90.0),
                    "angle_l_knee_deg": (70.0, 90.0),
                    "angle_trunk_inclination_deg": (0.0, 45.0),
                },
            ),
            DomainPoseClass(
                name="too_shallow",
                display_name="Insufficient Depth",
                description="Knee angle > 90° — hip crease above parallel",
                acceptable_ranges={
                    "angle_r_knee_deg": (91.0, 150.0),
                    "angle_l_knee_deg": (91.0, 150.0),
                },
            ),
            DomainPoseClass(
                name="knee_cave",
                display_name="Knee Valgus (Cave)",
                description="Knees collapsing inward — L/R knee more than 15° off ankle plumb",
                acceptable_ranges={},
            ),
            DomainPoseClass(
                name="forward_lean",
                display_name="Excessive Forward Lean",
                description="Trunk angle > 45° from vertical — potential lower-back stress",
                acceptable_ranges={
                    "angle_trunk_inclination_deg": (46.0, 90.0),
                },
            ),
            DomainPoseClass(
                name="good_form",
                display_name="Good Overall Form",
                description="Composite score ≥ 80 across depth, back angle, and knee alignment",
                acceptable_ranges={
                    "angle_r_knee_deg": (60.0, 90.0),
                    "angle_l_knee_deg": (60.0, 90.0),
                    "angle_trunk_inclination_deg": (0.0, 45.0),
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feature(
        self,
        feat_dict: Dict[str, float],
        *candidates: str,
        default: float = 0.0,
    ) -> float:
        """Return the first matching feature name, else default."""
        for name in candidates:
            if name in feat_dict:
                return float(feat_dict[name])
        return default

    def _knee_angle_score(self, knee_angle_deg: float) -> float:
        """Score knee flexion depth. Target ≤ 90°; 100 at 90°, 0 at 180°."""
        if knee_angle_deg <= 90.0:
            return 100.0
        # Linear penalty: 90° → 100, 180° → 0
        return max(0.0, 100.0 * (180.0 - knee_angle_deg) / 90.0)

    def _back_angle_score(self, trunk_angle_deg: float) -> float:
        """Score trunk inclination. Target ≤ 45°; 100 at 0°, 0 at 90°."""
        if trunk_angle_deg <= 45.0:
            return 100.0
        return max(0.0, 100.0 * (90.0 - trunk_angle_deg) / 45.0)

    def _knee_alignment_score(self, knee_ankle_angle_diff: float) -> float:
        """Score lateral knee alignment. Target diff ≤ 15°."""
        if knee_ankle_angle_diff <= 15.0:
            return 100.0
        return max(0.0, 100.0 * (1.0 - (knee_ankle_angle_diff - 15.0) / 45.0))

    # ------------------------------------------------------------------
    # BaseDomain interface
    # ------------------------------------------------------------------

    def compute_pose_score(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predicted_class: str,
    ) -> PoseScore:
        """Composite squat form score 0–100.

        Components:
            - Knee depth score  (avg of L+R knee angles vs ≤90° target)
            - Back angle score  (trunk inclination vs ≤45° target)
            - Knee alignment    (knee–ankle lateral deviation vs ≤15° target)
        """
        feat_dict: Dict[str, float] = dict(zip(feature_names, features.tolist()))

        # --- Extract relevant features ---
        r_knee = self._get_feature(feat_dict, "angle_r_knee_deg", "angle_r_knee", default=120.0)
        l_knee = self._get_feature(feat_dict, "angle_l_knee_deg", "angle_l_knee", default=120.0)
        trunk  = self._get_feature(feat_dict, "angle_trunk_inclination_deg",
                                   "angle_trunk_inclination", default=30.0)
        # Knee–ankle alignment proxy: bilateral symmetry of knee-ankle vector
        knee_align = self._get_feature(feat_dict, "sym_knee_ankle_angle_diff",
                                       "cross_knee_ankle_diff", default=10.0)

        # --- Component scores ---
        knee_score  = 0.5 * (self._knee_angle_score(r_knee) + self._knee_angle_score(l_knee))
        back_score  = self._back_angle_score(trunk)
        align_score = self._knee_alignment_score(abs(knee_align))
        depth_score = knee_score  # alias for clarity

        # Weighted composite: depth 40%, back 35%, alignment 25%
        composite = 0.40 * depth_score + 0.35 * back_score + 0.25 * align_score

        # Risk level
        if composite >= 75.0:
            risk_level = "good"
        elif composite >= 50.0:
            risk_level = "caution"
        else:
            risk_level = "poor"

        details = {
            "knee_depth_score":  round(depth_score, 1),
            "back_angle_score":  round(back_score, 1),
            "alignment_score":   round(align_score, 1),
            "r_knee_angle_deg":  round(r_knee, 1),
            "l_knee_angle_deg":  round(l_knee, 1),
            "trunk_angle_deg":   round(trunk, 1),
            "knee_align_diff":   round(abs(knee_align), 1),
            "predicted_class":   predicted_class,
        }

        feedback = self._build_score_feedback(
            composite, r_knee, l_knee, trunk, abs(knee_align), predicted_class
        )

        return PoseScore(
            score=round(composite, 1),
            risk_level=risk_level,
            feedback=feedback,
            details=details,
        )

    def _build_score_feedback(
        self,
        score: float,
        r_knee: float,
        l_knee: float,
        trunk: float,
        knee_align: float,
        predicted_class: str,
    ) -> str:
        """Short feedback summary for score display."""
        avg_knee = 0.5 * (r_knee + l_knee)
        issues: List[str] = []
        if avg_knee > 90.0:
            issues.append(f"knee depth {avg_knee:.0f}° (target ≤90°)")
        if trunk > 45.0:
            issues.append(f"forward lean {trunk:.0f}° (target ≤45°)")
        if knee_align > 15.0:
            issues.append(f"knee cave {knee_align:.0f}° (target ≤15°)")

        if not issues:
            return f"Squat form score {score:.0f}/100 — good form overall"
        return f"Squat form score {score:.0f}/100 — issues: {'; '.join(issues)}"

    def generate_feedback(
        self,
        corrections: List[Any],
        target_class: str,
    ) -> str:
        """Generate squat-specific correction instructions.

        Prioritises the most clinically significant corrections first.
        Each correction references the current and target angle values.
        """
        cls = self.get_class(target_class)
        cls_name = cls.display_name if cls else target_class

        lines: List[str] = [f"Squat Form Corrections — target: {cls_name}", ""]

        if not corrections:
            lines.append("No corrections required — form is within acceptable ranges.")
            return "\n".join(lines)

        lines.append("Required adjustments (ranked by importance):")
        for i, c in enumerate(corrections[:6]):
            part = c.body_part.replace("_", " ").title()
            if c.is_angle:
                if "_knee" in c.feature_name.lower():
                    if c.direction == "increase":
                        lines.append(
                            f"  {i+1}. Squat depth insufficient. "
                            f"Knee angle {c.current_value:.0f}°, target ≤90° "
                            f"— bend knees {abs(c.delta):.0f}° more."
                        )
                    else:
                        lines.append(
                            f"  {i+1}. Knee over-flexed. "
                            f"Knee angle {c.current_value:.0f}°, ease back to ≥90°."
                        )
                elif "trunk" in c.feature_name.lower() or "back" in c.feature_name.lower():
                    if c.direction == "decrease":
                        lines.append(
                            f"  {i+1}. Excessive forward lean. "
                            f"Trunk angle {c.current_value:.0f}°, target ≤45° "
                            f"— brace core and raise chest."
                        )
                    else:
                        lines.append(
                            f"  {i+1}. Adjust trunk angle: {c.current_value:.0f}° → {c.target_value:.0f}°."
                        )
                else:
                    direction = "Increase" if c.direction == "increase" else "Decrease"
                    lines.append(
                        f"  {i+1}. {direction} {part}: "
                        f"{c.current_value:.0f}° → {c.target_value:.0f}° "
                        f"(Δ {abs(c.delta):.0f}°)"
                    )
            else:
                lines.append(
                    f"  {i+1}. Adjust {part} position (Δ = {c.delta:.3f})"
                )

        lines.append("")
        lines.append(f"Total: {len(corrections)} adjustment(s) needed.")
        return "\n".join(lines)

    def get_feedback_template(self) -> Dict[str, str]:
        return {
            "intro": "Squat form corrections to achieve '{target}':",
            "angle_increase": "Bend {part} further: {current:.0f}° → {target_val:.0f}° (add {delta:.0f}°)",
            "angle_decrease": "Ease {part}: {current:.0f}° → {target_val:.0f}° (reduce {delta:.0f}°)",
            "distance_change": "Adjust {part} positioning by {delta:.3f}",
            "footer": "{n_corrections} form correction(s) needed",
        }


# ─────────────────────────────────────────────────────────────
# Generic Sports Domain
# ─────────────────────────────────────────────────────────────

class GenericSportsDomain(BaseDomain):
    """Generic sports biomechanics domain.

    Risk classification driven by joint angle deviations from
    user-supplied (or default) reference poses.

    Risk classes:
        optimal      — deviation < 10° across all joints
        suboptimal   — deviation 10–25°
        at_risk      — deviation 25–45°
        injury_risk  — deviation > 45°

    Usage:
        domain = GenericSportsDomain(sport_name="tennis_serve")
        score  = domain.compute_pose_score(features, feature_names, "at_risk")
    """

    def __init__(
        self,
        sport_name: str = "generic",
        reference_angles: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            sport_name: human label shown in feedback (e.g. "tennis_serve")
            reference_angles: dict mapping feature names to ideal angles (degrees).
                              If None, all joint angles default to 160° (near-straight).
        """
        self._sport_name = sport_name
        self.reference_angles: Dict[str, float] = reference_angles or {}

    # ---- BaseDomain properties ----

    @property
    def domain_name(self) -> str:
        return "sports_generic"

    @property
    def display_name(self) -> str:
        return f"Sports — {self._sport_name.replace('_', ' ').title()}"

    @property
    def description(self) -> str:
        return (
            f"Generic sports biomechanics risk assessment for {self._sport_name}. "
            "Classifies poses into optimal / suboptimal / at_risk / injury_risk."
        )

    @property
    def pose_classes(self) -> List[DomainPoseClass]:
        return [
            DomainPoseClass(
                name="optimal",
                display_name="Optimal Form",
                description="All joint angles within 10° of reference — minimal injury risk",
                acceptable_ranges={},
            ),
            DomainPoseClass(
                name="suboptimal",
                display_name="Suboptimal Form",
                description="One or more joints 10–25° from reference — monitor closely",
                acceptable_ranges={},
            ),
            DomainPoseClass(
                name="at_risk",
                display_name="At Risk",
                description="Joints 25–45° from reference — technique intervention recommended",
                acceptable_ranges={},
            ),
            DomainPoseClass(
                name="injury_risk",
                display_name="Injury Risk",
                description="Joints > 45° from reference — immediate technique correction required",
                acceptable_ranges={},
            ),
        ]

    # ---- Internal helpers ----

    def _angle_features(
        self,
        feat_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """Extract all angle features from the feature dict."""
        return {
            name: val
            for name, val in feat_dict.items()
            if "_deg" in name or "_angle" in name
        }

    def _compute_deviation(self, feat_dict: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Compute mean absolute deviation from reference angles.

        Returns (mean_deviation_deg, per_joint_deviations).
        """
        angle_feats = self._angle_features(feat_dict)
        if not angle_feats:
            return 0.0, {}

        deviations: Dict[str, float] = {}
        for name, val in angle_feats.items():
            ref = self.reference_angles.get(name, 160.0)
            deviations[name] = abs(val - ref)

        mean_dev = float(np.mean(list(deviations.values())))
        return mean_dev, deviations

    # ---- BaseDomain interface ----

    def compute_pose_score(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predicted_class: str,
    ) -> PoseScore:
        """Score based on joint angle deviation from reference poses.

        Score 100 = perfect match to reference.
        Score falls linearly with mean deviation up to 60°.
        """
        feat_dict: Dict[str, float] = dict(zip(feature_names, features.tolist()))
        mean_dev, per_joint = self._compute_deviation(feat_dict)

        # Clamp and invert: 0 deviation → 100, 60° deviation → 0
        score = max(0.0, 100.0 * (1.0 - mean_dev / 60.0))

        if score >= 80.0:
            risk_level = "optimal"
        elif score >= 60.0:
            risk_level = "suboptimal"
        elif score >= 40.0:
            risk_level = "at_risk"
        else:
            risk_level = "injury_risk"

        # Top 3 worst joints for feedback
        worst = sorted(per_joint.items(), key=lambda x: -x[1])[:3]
        worst_strs = [f"{k.replace('_', ' ')} ({v:.0f}°)" for k, v in worst]

        feedback = (
            f"{self.display_name} score {score:.0f}/100 "
            f"(mean deviation {mean_dev:.1f}°)"
        )
        if worst_strs:
            feedback += f" — worst joints: {', '.join(worst_strs)}"

        return PoseScore(
            score=round(score, 1),
            risk_level=risk_level,
            feedback=feedback,
            details={
                "mean_deviation_deg": round(mean_dev, 2),
                "per_joint_deviations": {k: round(v, 1) for k, v in per_joint.items()},
                "predicted_class": predicted_class,
                "sport": self._sport_name,
            },
        )

    def generate_feedback(
        self,
        corrections: List[Any],
        target_class: str,
    ) -> str:
        """Generate sports-specific correction instructions."""
        cls = self.get_class(target_class)
        cls_name = cls.display_name if cls else target_class

        lines: List[str] = [
            f"Sports Biomechanics Feedback — {self.display_name}",
            f"Target form: {cls_name}",
            "",
        ]

        if not corrections:
            lines.append("Form is within acceptable parameters — no corrections needed.")
            return "\n".join(lines)

        lines.append("Form corrections (most impactful first):")
        for i, c in enumerate(corrections[:6]):
            part = c.body_part.replace("_", " ").title()
            if c.is_angle:
                direction = "Increase" if c.direction == "increase" else "Decrease"
                lines.append(
                    f"  {i+1}. {direction} {part} by {abs(c.delta):.1f}°"
                    f" ({c.current_value:.1f}° → {c.target_value:.1f}°)"
                )
            else:
                lines.append(
                    f"  {i+1}. Reposition {part} (Δ = {c.delta:.3f})"
                )

        lines.append("")
        lines.append(
            f"Addressing {len(corrections)} correction(s) will move pose toward '{cls_name}'."
        )
        return "\n".join(lines)

    def get_feedback_template(self) -> Dict[str, str]:
        return {
            "intro": f"Form correction for {self._sport_name} to achieve '{{target}}':",
            "angle_increase": "Bend {part}: {current:.1f}° → {target_val:.1f}° (add {delta:.1f}°)",
            "angle_decrease": "Extend {part}: {current:.1f}° → {target_val:.1f}° (reduce {delta:.1f}°)",
            "distance_change": "Adjust {part} by {delta:.3f} units",
            "footer": "{n_corrections} adjustment(s) needed for optimal form",
        }
