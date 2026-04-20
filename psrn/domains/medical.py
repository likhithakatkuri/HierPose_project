"""Medical imaging and rehabilitation domain modules.

XrayPositioningDomain:
    Detects whether a patient is in the correct position for common
    radiographic views (PA chest, lateral chest, AP knee, etc.).
    Provides corrections like "Rotate left shoulder 12° forward for
    accurate PA chest X-ray positioning."

RehabMonitoringDomain:
    Tracks exercise completion during physical therapy.
    Monitors joint angles against prescribed ranges
    (e.g., "Knee flexion at 67°, target 90° — bend knee further").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from psrn.domains.base import BaseDomain, DomainPoseClass, PoseScore


# ─────────────────────────────────────────────────────────────
# X-Ray Positioning Domain
# ─────────────────────────────────────────────────────────────

class XrayPositioningDomain(BaseDomain):
    """Radiographic patient positioning guidance.

    Pose classes correspond to standard radiographic views.
    Reference: Merrill's Atlas of Radiographic Positioning (standard protocols).

    Classes:
    - pa_chest: PA (posterior-anterior) chest X-ray
    - lateral_chest: Lateral chest X-ray
    - ap_knee: AP knee X-ray
    - lateral_knee: Lateral knee X-ray
    - ap_pelvis: AP pelvis / hip X-ray
    """

    @property
    def domain_name(self) -> str:
        return "xray_positioning"

    @property
    def display_name(self) -> str:
        return "X-Ray Positioning Assistant"

    @property
    def description(self) -> str:
        return "Guide patients into correct positions for accurate radiographic imaging"

    @property
    def pose_classes(self) -> List[DomainPoseClass]:
        return [
            DomainPoseClass(
                name="pa_chest",
                display_name="PA Chest X-Ray",
                description="Posterior-anterior chest view: chin up, shoulders forward, arms rotated",
                acceptable_ranges={
                    "angle_shoulder_span_deg": (140, 180),
                    "sym_shoulder_angle_diff": (0, 15),
                    "angle_trunk_inclination_deg": (80, 100),
                },
            ),
            DomainPoseClass(
                name="lateral_chest",
                display_name="Lateral Chest X-Ray",
                description="Lateral chest view: arms raised, body turned 90°",
                acceptable_ranges={
                    "angle_reach_angle_deg": (60, 120),
                },
            ),
            DomainPoseClass(
                name="ap_knee",
                display_name="AP Knee X-Ray",
                description="Anterior-posterior knee view: leg extended, toes pointing forward",
                acceptable_ranges={
                    "angle_l_knee_deg": (160, 180),
                    "angle_r_knee_deg": (160, 180),
                },
            ),
            DomainPoseClass(
                name="lateral_knee",
                display_name="Lateral Knee X-Ray",
                description="Lateral knee view: knee flexed 25-30°",
                acceptable_ranges={
                    "angle_l_knee_deg": (145, 160),
                    "angle_r_knee_deg": (145, 160),
                },
            ),
            DomainPoseClass(
                name="ap_pelvis",
                display_name="AP Pelvis/Hip X-Ray",
                description="AP pelvis view: feet internally rotated 15-20°",
                acceptable_ranges={
                    "cross_foot_spread_ratio": (0.8, 1.2),
                    "sym_lower_score": (0.7, 1.0),
                },
            ),
        ]

    def compute_pose_score(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predicted_class: str,
    ) -> PoseScore:
        """Score positioning quality against acceptable angle ranges."""
        cls = self.get_class(predicted_class)
        if cls is None or not cls.acceptable_ranges:
            return PoseScore(
                score=50.0,
                risk_level="medium",
                feedback=f"Class '{predicted_class}' not found in domain",
            )

        feat_dict = dict(zip(feature_names, features))
        violations = []
        checks = 0

        for feat_name, (low, high) in cls.acceptable_ranges.items():
            if feat_name in feat_dict:
                val = float(feat_dict[feat_name])
                checks += 1
                if not (low <= val <= high):
                    diff = min(abs(val - low), abs(val - high))
                    violations.append(f"{feat_name}: {val:.1f} (expected {low:.0f}–{high:.0f})")

        if checks == 0:
            score = 70.0
        else:
            score = 100.0 * max(0, checks - len(violations)) / checks

        risk_level = "low" if score >= 80 else "medium" if score >= 50 else "high"
        feedback = (
            f"Positioning quality: {score:.0f}%"
            if not violations else
            f"Positioning issues ({len(violations)} found): " + "; ".join(violations[:3])
        )

        return PoseScore(score=score, risk_level=risk_level, feedback=feedback,
                         details={"violations": violations})

    def generate_feedback(self, corrections: List[Any], target_class: str) -> str:
        cls = self.get_class(target_class)
        cls_name = cls.display_name if cls else target_class

        lines = [f"Patient Positioning Guidance — {cls_name}", ""]
        if cls:
            lines.append(f"Protocol: {cls.description}")
            lines.append("")

        lines.append("Required adjustments:")
        for i, c in enumerate(corrections[:5]):
            part = c.body_part.replace("_", " ")
            if c.is_angle:
                direction = "Increase" if c.direction == "increase" else "Decrease"
                lines.append(
                    f"  {i+1}. {direction} {part} angle by {abs(c.delta):.1f}°"
                    f" ({c.current_value:.1f}° → {c.target_value:.1f}°)"
                )
            else:
                lines.append(
                    f"  {i+1}. Adjust {part} position (Δ={c.delta:.3f})"
                )

        lines.append("")
        lines.append(f"({len(corrections)} total adjustment(s) needed)")
        return "\n".join(lines)

    def get_feedback_template(self) -> Dict[str, str]:
        return {
            "intro": "Patient positioning adjustment for {target}:",
            "angle_increase": "Extend {part} by {delta:.1f}° ({current:.1f}° → {target_val:.1f}°)",
            "angle_decrease": "Reduce {part} by {delta:.1f}° ({current:.1f}° → {target_val:.1f}°)",
            "distance_change": "Reposition {part} by {delta:.3f} units",
            "footer": "{n_corrections} positioning adjustments required",
        }


# ─────────────────────────────────────────────────────────────
# Rehabilitation Monitoring Domain
# ─────────────────────────────────────────────────────────────

class RehabMonitoringDomain(BaseDomain):
    """Physical therapy exercise monitoring.

    Tracks patient exercise completion against prescribed targets.
    Provides real-time feedback on exercise quality.

    Example: "Knee flexion at 67°, target 90° — bend knee 23° further"
    """

    @property
    def domain_name(self) -> str:
        return "rehab_monitoring"

    @property
    def display_name(self) -> str:
        return "Rehabilitation Monitoring"

    @property
    def description(self) -> str:
        return "Monitor physical therapy exercises and provide real-time completion feedback"

    @property
    def pose_classes(self) -> List[DomainPoseClass]:
        return [
            DomainPoseClass(
                name="knee_flexion_30",
                display_name="Knee Flexion 30°",
                description="Light knee bend — early post-op",
                acceptable_ranges={"angle_l_knee_deg": (145, 160), "angle_r_knee_deg": (145, 160)},
            ),
            DomainPoseClass(
                name="knee_flexion_60",
                display_name="Knee Flexion 60°",
                description="Moderate knee flexion",
                acceptable_ranges={"angle_l_knee_deg": (115, 130), "angle_r_knee_deg": (115, 130)},
            ),
            DomainPoseClass(
                name="knee_flexion_90",
                display_name="Knee Flexion 90°",
                description="Full weight-bearing flexion goal",
                acceptable_ranges={"angle_l_knee_deg": (85, 100), "angle_r_knee_deg": (85, 100)},
            ),
            DomainPoseClass(
                name="shoulder_abduction",
                display_name="Shoulder Abduction",
                description="Arm raise to 90° from side",
                acceptable_ranges={"angle_l_shoulder_deg": (60, 100), "angle_r_shoulder_deg": (60, 100)},
            ),
            DomainPoseClass(
                name="standing_straight",
                display_name="Standing Balance",
                description="Upright posture assessment",
                acceptable_ranges={
                    "angle_trunk_inclination_deg": (75, 95),
                    "sym_lower_score": (0.7, 1.0),
                },
            ),
        ]

    def compute_pose_score(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predicted_class: str,
    ) -> PoseScore:
        """Score exercise completion percentage."""
        cls = self.get_class(predicted_class)
        if cls is None:
            return PoseScore(score=50.0, risk_level="medium", feedback="Exercise class not found")

        feat_dict = dict(zip(feature_names, features))
        total_completion = 0.0
        n_checked = 0

        for feat_name, (target_low, target_high) in cls.acceptable_ranges.items():
            if feat_name in feat_dict:
                val = float(feat_dict[feat_name])
                target_mid = (target_low + target_high) / 2
                target_range = (target_high - target_low) / 2
                deviation = abs(val - target_mid)
                completion = max(0, 1 - deviation / (target_range + 1e-8))
                total_completion += completion
                n_checked += 1

        score = 100 * (total_completion / max(n_checked, 1))
        risk_level = "low" if score >= 80 else "medium" if score >= 50 else "high"

        feedback = f"Exercise completion: {score:.0f}%"
        if score < 60:
            feedback += " — more range of motion needed"
        elif score >= 90:
            feedback += " — excellent form!"

        return PoseScore(score=score, risk_level=risk_level, feedback=feedback)

    def generate_feedback(self, corrections: List[Any], target_class: str) -> str:
        cls = self.get_class(target_class)
        target_name = cls.display_name if cls else target_class

        lines = [f"Exercise Feedback — {target_name}", ""]
        if cls:
            lines.append(f"Goal: {cls.description}")
            lines.append("")

        lines.append("To reach target position:")
        for i, c in enumerate(corrections[:5]):
            part = c.body_part.replace("_", " ")
            if c.is_angle:
                direction = "Bend more" if c.direction == "increase" else "Extend more"
                lines.append(
                    f"  {i+1}. {direction}: {part} — "
                    f"{c.current_value:.1f}° → {c.target_value:.1f}° "
                    f"(need {abs(c.delta):.1f}° more)"
                )

        return "\n".join(lines)

    def get_feedback_template(self) -> Dict[str, str]:
        return {
            "intro": "Exercise corrections for {target}:",
            "angle_increase": "Bend {part} {delta:.1f}° more ({current:.1f}° → {target_val:.1f}°)",
            "angle_decrease": "Extend {part} {delta:.1f}° ({current:.1f}° → {target_val:.1f}°)",
            "distance_change": "Adjust {part} position",
            "footer": "Complete {n_corrections} adjustment(s) to reach target",
        }
