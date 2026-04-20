"""Abstract base class for HierPose application domains.

A domain encapsulates:
1. Pose class definitions (what classes exist in this domain)
2. Reference poses (ideal/target keypoint positions per class)
3. Scoring (domain-specific quality metric)
4. Feedback generation (correction language for the CPG engine)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DomainPoseClass:
    """Definition of one pose class within a domain."""
    name: str
    display_name: str
    description: str
    reference_keypoints: Optional[np.ndarray] = None   # (N_joints, 2) ideal pose
    acceptable_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # e.g., {"l_knee_angle_deg": (80, 100), "r_knee_angle_deg": (80, 100)}


@dataclass
class PoseScore:
    """Pose quality score from a domain-specific scoring function."""
    score: float                        # 0 (worst) to 100 (perfect)
    risk_level: str                     # "low" | "medium" | "high" | "critical"
    feedback: str                       # human-readable summary
    details: Dict[str, Any] = field(default_factory=dict)


class BaseDomain(ABC):
    """Abstract base for all HierPose application domains.

    Subclasses define:
    - What pose classes exist in this domain
    - What the reference/ideal poses look like
    - How to score an observed pose
    - How to generate feedback (feeds into CPG engine)
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Short machine name: "medical", "sports", "ergonomics", "action" """

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for UI display."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown in UI sidebar."""

    @property
    @abstractmethod
    def pose_classes(self) -> List[DomainPoseClass]:
        """All pose classes defined in this domain."""

    @property
    def class_names(self) -> List[str]:
        """Short list of class names."""
        return [c.name for c in self.pose_classes]

    def get_class(self, name: str) -> Optional[DomainPoseClass]:
        """Return DomainPoseClass by name, or None."""
        for c in self.pose_classes:
            if c.name == name:
                return c
        return None

    def get_reference_pose(self, class_name: str) -> Optional[np.ndarray]:
        """Return reference keypoints for a class, or None."""
        cls = self.get_class(class_name)
        return cls.reference_keypoints if cls else None

    @abstractmethod
    def compute_pose_score(
        self,
        features: np.ndarray,
        feature_names: List[str],
        predicted_class: str,
    ) -> PoseScore:
        """Compute domain-specific quality score for an observed pose.

        Args:
            features: (n_features,) extracted feature vector
            feature_names: corresponding feature name strings
            predicted_class: model's predicted class name

        Returns:
            PoseScore with score, risk_level, and feedback
        """

    @abstractmethod
    def generate_feedback(
        self,
        corrections: List[Any],  # List[PoseCorrection]
        target_class: str,
    ) -> str:
        """Generate domain-specific human-readable correction instructions.

        Args:
            corrections: list of PoseCorrection from CPG engine
            target_class: the desired target pose class

        Returns:
            Multi-line formatted correction string
        """

    def get_feedback_template(self) -> Dict[str, str]:
        """Return feedback templates for the CPG engine."""
        return {
            "intro": f"Corrections needed for {self.display_name}:",
            "angle_increase": "Increase {{part}} angle by {{delta:.1f}}° ({{current:.1f}}° → {{target_val:.1f}}°)",
            "angle_decrease": "Decrease {{part}} angle by {{delta:.1f}}° ({{current:.1f}}° → {{target_val:.1f}}°)",
            "distance_change": "Adjust {{part}} distance by {{delta:.3f}}",
            "footer": "{{n_corrections}} adjustment(s) required",
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(domain='{self.domain_name}', classes={len(self.pose_classes)})"
