"""Workplace ergonomics domain module.

WorkplaceErgonomicsDomain:
    RULA-proxy scoring from body keypoints to classify workplace
    posture into four risk tiers (neutral, low_risk, medium_risk, high_risk).

    Feedback examples:
      "Neck is bent 38° forward. Raise monitor height by ~19cm or tilt screen up."
      "Upper arm raised 97°. Lower workstation surface or use an armrest."

    Reference:
        McAtamney & Corlett (1993) RULA: A survey method for the
        investigation of work-related upper limb disorders.
        Applied Ergonomics, 24(2), 91–99.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from psrn.domains.base import BaseDomain, DomainPoseClass, PoseScore


# ─────────────────────────────────────────────────────────────
# WorkplaceErgonomicsDomain
# ─────────────────────────────────────────────────────────────

class WorkplaceErgonomicsDomain(BaseDomain):
    """Workplace ergonomics risk assessment using a RULA proxy.

    Risk classes and score bands:
        neutral     — score 80–100  (posture is safe, no intervention needed)
        low_risk    — score 60–79   (some deviations, monitor and adjust)
        medium_risk — score 40–59   (investigate and implement changes soon)
        high_risk   — score  0–39   (immediate corrective action required)

    RULA proxy components scored from keypoints:
        1. upper_arm_angle   : shoulder–elbow deviation from neutral (0° = hanging)
        2. neck_angle        : head–neck flexion forward/back
        3. trunk_angle       : trunk inclination from vertical
        4. wrist_deviation   : wrist–forearm alignment proxy

    The composite score weights these four components and maps them
    to a 0–100 ergonomic risk scale (higher = better / safer).
    """

    # Component weight configuration
    _WEIGHTS: Dict[str, float] = {
        "upper_arm": 0.30,
        "neck":      0.30,
        "trunk":     0.25,
        "wrist":     0.15,
    }

    @property
    def domain_name(self) -> str:
        return "workplace_ergonomics"

    @property
    def display_name(self) -> str:
        return "Workplace Ergonomics Assessor"

    @property
    def description(self) -> str:
        return (
            "RULA-proxy ergonomic risk scoring: classifies seated/standing postures "
            "into neutral, low-risk, medium-risk, and high-risk tiers."
        )

    @property
    def pose_classes(self) -> List[DomainPoseClass]:
        return [
            DomainPoseClass(
                name="neutral",
                display_name="Neutral Posture",
                description=(
                    "Upper arm <20°, neck <20°, trunk <20° — "
                    "safe working posture, no intervention required"
                ),
                acceptable_ranges={
                    "ergo_upper_arm_angle": (0.0, 20.0),
                    "ergo_neck_angle":      (0.0, 20.0),
                    "ergo_trunk_angle":     (0.0, 20.0),
                },
            ),
            DomainPoseClass(
                name="low_risk",
                display_name="Low Risk",
                description=(
                    "Some joints outside neutral range — "
                    "monitor and consider minor workstation adjustments"
                ),
                acceptable_ranges={
                    "ergo_upper_arm_angle": (20.0, 45.0),
                    "ergo_neck_angle":      (20.0, 30.0),
                    "ergo_trunk_angle":     (20.0, 40.0),
                },
            ),
            DomainPoseClass(
                name="medium_risk",
                display_name="Medium Risk",
                description=(
                    "Multiple joints outside safe range — "
                    "investigate workstation and implement changes soon"
                ),
                acceptable_ranges={
                    "ergo_upper_arm_angle": (45.0, 90.0),
                    "ergo_neck_angle":      (30.0, 45.0),
                    "ergo_trunk_angle":     (40.0, 60.0),
                },
            ),
            DomainPoseClass(
                name="high_risk",
                display_name="High Risk",
                description=(
                    "Severe deviations present — "
                    "immediate corrective action required to prevent injury"
                ),
                acceptable_ranges={
                    "ergo_upper_arm_angle": (90.0, 180.0),
                    "ergo_neck_angle":      (45.0, 90.0),
                    "ergo_trunk_angle":     (60.0, 90.0),
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Internal angle extraction helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        feat_dict: Dict[str, float],
        *candidates: str,
        default: float = 0.0,
    ) -> float:
        for name in candidates:
            if name in feat_dict:
                return float(feat_dict[name])
        return default

    def _upper_arm_score(self, angle_deg: float) -> float:
        """RULA upper arm score proxy.

        0°  = arm hanging at side (neutral, score 100)
        45° = moderate raise     (score ~50)
        90° = horizontal         (score ~0)
        >90°= high risk          (score 0)
        """
        if angle_deg <= 20.0:
            return 100.0
        if angle_deg <= 45.0:
            return 100.0 - 50.0 * (angle_deg - 20.0) / 25.0
        if angle_deg <= 90.0:
            return 50.0 - 50.0 * (angle_deg - 45.0) / 45.0
        return 0.0

    def _neck_score(self, angle_deg: float) -> float:
        """RULA neck flexion score proxy.

        <20° = neutral  (score 100)
        20–45° = risk   (linear decay from 100 to 0)
        >45°  = high risk (score 0)
        """
        if angle_deg < 20.0:
            return 100.0
        if angle_deg <= 45.0:
            return 100.0 * (1.0 - (angle_deg - 20.0) / 25.0)
        return 0.0

    def _trunk_score(self, angle_deg: float) -> float:
        """RULA trunk inclination score proxy.

        <20° = neutral   (score 100)
        20–60° = risk    (linear decay)
        >60°  = high risk (score 0)
        """
        if angle_deg < 20.0:
            return 100.0
        if angle_deg <= 60.0:
            return 100.0 * (1.0 - (angle_deg - 20.0) / 40.0)
        return 0.0

    def _wrist_score(self, deviation: float) -> float:
        """Wrist deviation score proxy (0 = straight, 1 = fully deviated).

        0.0–0.2 = neutral  (score 100)
        0.2–1.0 = risk     (linear decay)
        """
        if deviation <= 0.2:
            return 100.0
        return max(0.0, 100.0 * (1.0 - (deviation - 0.2) / 0.8))

    # ------------------------------------------------------------------
    # BaseDomain interface
    # ------------------------------------------------------------------

    def compute_pose_score(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predicted_class: str,
    ) -> PoseScore:
        """Compute RULA-proxy ergonomic risk score 0–100.

        Pulls four angle proxies from the feature vector, scores each,
        then computes a weighted composite.  Falls back to generic
        angle feature names when domain-specific ones are absent.

        Returns:
            PoseScore with per-joint breakdown in details dict.
        """
        feat_dict: Dict[str, float] = dict(zip(feature_names, features.tolist()))

        # --- Extract component angles ---
        upper_arm = self._get(
            feat_dict,
            "ergo_upper_arm_angle",
            "angle_r_shoulder_deg",
            "angle_l_shoulder_deg",
            default=30.0,
        )
        neck = self._get(
            feat_dict,
            "ergo_neck_angle",
            "angle_neck_deg",
            "angle_head_neck_deg",
            default=15.0,
        )
        trunk = self._get(
            feat_dict,
            "ergo_trunk_angle",
            "angle_trunk_inclination_deg",
            "angle_trunk_inclination",
            default=15.0,
        )
        wrist = self._get(
            feat_dict,
            "ergo_wrist_deviation",
            "sym_wrist_deviation",
            "cross_wrist_diff",
            default=0.1,
        )
        # Normalise wrist: if value > 2 assume it is in degrees (0–90°), convert
        if wrist > 2.0:
            wrist = min(1.0, wrist / 90.0)

        # --- Component scores ---
        s_arm   = self._upper_arm_score(upper_arm)
        s_neck  = self._neck_score(neck)
        s_trunk = self._trunk_score(trunk)
        s_wrist = self._wrist_score(wrist)

        w = self._WEIGHTS
        composite = (
            w["upper_arm"] * s_arm
            + w["neck"]    * s_neck
            + w["trunk"]   * s_trunk
            + w["wrist"]   * s_wrist
        )

        # Map composite → risk class
        if composite >= 80.0:
            risk_level = "neutral"
        elif composite >= 60.0:
            risk_level = "low_risk"
        elif composite >= 40.0:
            risk_level = "medium_risk"
        else:
            risk_level = "high_risk"

        details: Dict[str, Any] = {
            "upper_arm_angle_deg": round(upper_arm, 1),
            "neck_angle_deg":      round(neck, 1),
            "trunk_angle_deg":     round(trunk, 1),
            "wrist_deviation":     round(wrist, 3),
            "upper_arm_score":     round(s_arm, 1),
            "neck_score":          round(s_neck, 1),
            "trunk_score":         round(s_trunk, 1),
            "wrist_score":         round(s_wrist, 1),
            "predicted_class":     predicted_class,
        }

        feedback = self._build_score_feedback(
            composite, upper_arm, neck, trunk, wrist
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
        upper_arm: float,
        neck: float,
        trunk: float,
        wrist: float,
    ) -> str:
        """Short one-line ergonomic risk feedback."""
        level_map = {
            "neutral":      "No intervention required",
            "low_risk":     "Monitor and consider minor adjustments",
            "medium_risk":  "Investigate workstation — changes recommended",
            "high_risk":    "IMMEDIATE correction required",
        }
        if score >= 80.0:
            level = "neutral"
        elif score >= 60.0:
            level = "low_risk"
        elif score >= 40.0:
            level = "medium_risk"
        else:
            level = "high_risk"

        worst: List[str] = []
        if upper_arm > 45.0:
            worst.append(f"upper arm {upper_arm:.0f}°")
        if neck > 20.0:
            worst.append(f"neck {neck:.0f}°")
        if trunk > 20.0:
            worst.append(f"trunk {trunk:.0f}°")

        base = f"Ergonomic score {score:.0f}/100 — {level_map[level]}"
        if worst:
            base += f". Risk factors: {', '.join(worst)}."
        return base

    def generate_feedback(
        self,
        corrections: List[Any],
        target_class: str,
    ) -> str:
        """Generate RULA-informed ergonomic correction instructions.

        Provides actionable workstation guidance for each correction,
        including estimated monitor height changes for neck deviations
        and armrest / surface-height guidance for upper-arm deviations.
        """
        cls = self.get_class(target_class)
        cls_name = cls.display_name if cls else target_class

        lines: List[str] = [
            f"Ergonomic Risk Assessment — Target: {cls_name}",
            "",
        ]

        if not corrections:
            lines.append(
                "Posture is within acceptable ergonomic ranges. No corrections needed."
            )
            return "\n".join(lines)

        lines.append("Recommended corrections (highest priority first):")

        for i, c in enumerate(corrections[:6]):
            feat  = c.feature_name.lower()
            part  = c.body_part.replace("_", " ").title()
            angle = abs(c.current_value)
            delta = abs(c.delta)

            # --- Neck-specific guidance ---
            if "neck" in feat or "head" in feat:
                cm_raise = round(delta * 2.5, 0)          # rough: 1° ≈ 2.5 cm at screen
                lines.append(
                    f"  {i+1}. Neck is bent {angle:.0f}° forward. "
                    f"Raise monitor height by ~{cm_raise:.0f} cm or tilt screen up."
                )

            # --- Upper arm / shoulder guidance ---
            elif "shoulder" in feat or "arm" in feat or "upper_arm" in feat:
                if angle > 90.0:
                    lines.append(
                        f"  {i+1}. Upper arm raised {angle:.0f}° (high risk). "
                        f"Lower workstation surface or use an adjustable armrest."
                    )
                elif angle > 45.0:
                    lines.append(
                        f"  {i+1}. Upper arm elevated {angle:.0f}°. "
                        f"Lower keyboard/mouse surface by ~{delta * 1.5:.0f} cm."
                    )
                else:
                    lines.append(
                        f"  {i+1}. Adjust {part}: {c.current_value:.0f}° → {c.target_value:.0f}°."
                    )

            # --- Trunk guidance ---
            elif "trunk" in feat or "back" in feat:
                if c.direction == "decrease":
                    lines.append(
                        f"  {i+1}. Trunk inclined {angle:.0f}° forward. "
                        f"Move screen/work closer or add lumbar support."
                    )
                else:
                    lines.append(
                        f"  {i+1}. Adjust trunk posture: "
                        f"{c.current_value:.0f}° → {c.target_value:.0f}°."
                    )

            # --- Wrist guidance ---
            elif "wrist" in feat:
                lines.append(
                    f"  {i+1}. Wrist deviated ({c.current_value:.2f}). "
                    f"Use a wrist rest or tilt keyboard to achieve neutral wrist posture."
                )

            # --- Generic fallback ---
            else:
                if c.is_angle:
                    direction = "Increase" if c.direction == "increase" else "Decrease"
                    lines.append(
                        f"  {i+1}. {direction} {part}: "
                        f"{c.current_value:.0f}° → {c.target_value:.0f}° "
                        f"(Δ {delta:.0f}°)"
                    )
                else:
                    lines.append(
                        f"  {i+1}. Reposition {part} (Δ = {c.delta:.3f})"
                    )

        lines.append("")
        lines.append(
            f"Implementing {len(corrections)} correction(s) will reduce ergonomic risk."
        )
        return "\n".join(lines)

    def get_feedback_template(self) -> Dict[str, str]:
        return {
            "intro": "⚠ Ergonomic risk: corrections needed to reach '{target}':",
            "angle_increase": "RAISE {part}: needs {delta:.1f}° more (current {current:.1f}°, target {target_val:.1f}°)",
            "angle_decrease": "LOWER {part}: reduce by {delta:.1f}° (current {current:.1f}°, target {target_val:.1f}°)",
            "distance_change": "Reposition {part} by {delta:.3f} normalised units",
            "footer": "Apply {n_corrections} correction(s) to reduce injury risk",
        }
