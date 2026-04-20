"""Skeleton visualization with SHAP heatmaps and correction arrows.

Provides:
1. draw_skeleton_shap_heatmap — joints colored by SHAP importance
   Red = high importance, Blue = low importance
   Joint radius ∝ |SHAP value|
   → Figure 5 in IEEE paper

2. draw_correction_skeleton — current pose (gray) + target pose overlay
   Arrows from current → target positions
   Arrow color = correction urgency (red/yellow/green)
   → Demo UI's most visually impressive feature

3. draw_skeleton_plain — clean skeleton without any overlay
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from psrn.data.jhmdb_loader import JHMDB_BONES, JHMDB_KP_NAMES


# ─────────────────────────────────────────────────────────────
# Color utilities
# ─────────────────────────────────────────────────────────────

def _importance_to_color(importance: float, vmin: float, vmax: float) -> Tuple[float, float, float]:
    """Map importance value to RGB color (blue → red gradient)."""
    if vmax <= vmin:
        return (0.5, 0.5, 0.8)
    t = max(0.0, min(1.0, (importance - vmin) / (vmax - vmin)))
    # Blue (0,0,1) → White (1,1,1) → Red (1,0,0)
    if t < 0.5:
        r = t * 2
        g = t * 2
        b = 1.0
    else:
        r = 1.0
        g = 2.0 * (1.0 - t)
        b = 2.0 * (1.0 - t)
    return (r, g, b)


def _severity_to_color(delta_abs: float, max_delta: float) -> str:
    """Map correction severity to traffic-light color."""
    if max_delta <= 0:
        return "green"
    t = delta_abs / max_delta
    if t > 0.6:
        return "#e74c3c"   # red — urgent
    elif t > 0.3:
        return "#f39c12"   # orange — moderate
    else:
        return "#2ecc71"   # green — minor


# ─────────────────────────────────────────────────────────────
# Joint importance mapping
# ─────────────────────────────────────────────────────────────

def feature_importances_to_joints(
    feature_names: List[str],
    shap_importances: Dict[str, float],
    n_joints: int = 15,
) -> np.ndarray:
    """Map feature-level SHAP importances to joint-level values.

    Aggregates all features associated with each joint by summing their
    SHAP importances. Joint names are extracted from feature names.

    Args:
        feature_names: list of feature names
        shap_importances: dict of feature_name → mean |SHAP|
        n_joints: number of joints in the skeleton

    Returns:
        joint_importance: (n_joints,) array of importance values
    """
    from psrn.data.jhmdb_loader import JHMDB_KP_NAMES

    joint_importance = np.zeros(n_joints, dtype=np.float32)

    # Map feature name → joint indices (a feature may involve multiple joints)
    kp_name_map = {name.lower(): i for i, name in enumerate(JHMDB_KP_NAMES)}

    for feat_name, imp in shap_importances.items():
        # Find joint indices mentioned in this feature name
        for jname, jidx in kp_name_map.items():
            if jname in feat_name.lower():
                joint_importance[jidx] += float(imp)

    return joint_importance


# ─────────────────────────────────────────────────────────────
# Skeleton drawing functions
# ─────────────────────────────────────────────────────────────

def draw_skeleton_plain(
    frame: np.ndarray,
    ax: Optional[Any] = None,
    joint_color: str = "#2c3e50",
    bone_color: str = "#7f8c8d",
    joint_radius: float = 6.0,
    bone_lw: float = 2.0,
    title: str = "",
    flip_y: bool = True,
) -> Any:
    """Draw a plain skeleton without any overlay.

    Args:
        frame: (N_joints, 2) joint positions
        ax: matplotlib Axes (creates new figure if None)
        joint_color: color for joint circles
        bone_color: color for bone lines
        joint_radius: circle radius in points
        bone_lw: bone line width
        title: plot title
        flip_y: flip y-axis (image coordinates have y increasing downward)

    Returns:
        matplotlib Axes
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    xs = frame[:, 0]
    ys = frame[:, 1] if not flip_y else -frame[:, 1]

    for a, b in JHMDB_BONES:
        if a < len(frame) and b < len(frame):
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                    color=bone_color, linewidth=bone_lw, zorder=1)

    ax.scatter(xs, ys, s=joint_radius ** 2, c=joint_color, zorder=2)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def draw_skeleton_shap_heatmap(
    frame: np.ndarray,
    feature_names: List[str],
    shap_importances: Dict[str, float],
    image: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "SHAP Feature Importance",
    figsize: Tuple[int, int] = (6, 7),
    dpi: int = 200,
    flip_y: bool = True,
) -> Any:
    """Draw skeleton with joints colored by SHAP importance.

    Red = high SHAP importance (most discriminative for the predicted class)
    Blue = low SHAP importance
    Joint radius ∝ importance magnitude

    Args:
        frame: (N_joints, 2) joint positions
        feature_names: list of feature name strings
        shap_importances: dict feature_name → |SHAP value|
        image: optional (H, W, 3) background image
        output_path: if provided, save PNG here
        title: plot title
        figsize: figure size
        dpi: output resolution
        flip_y: flip y coordinates

    Returns:
        matplotlib Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    n_joints = len(frame)
    joint_imp = feature_importances_to_joints(feature_names, shap_importances, n_joints)

    vmin = joint_imp.min()
    vmax = joint_imp.max()

    fig, ax = plt.subplots(figsize=figsize)

    if image is not None:
        ax.imshow(image, alpha=0.3, aspect="auto")
        xs = frame[:, 0]
        ys = frame[:, 1]
    else:
        xs = frame[:, 0]
        ys = -frame[:, 1] if flip_y else frame[:, 1]

    # Draw bones
    for a, b in JHMDB_BONES:
        if a < n_joints and b < n_joints:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                    color="#95a5a6", linewidth=2, zorder=1, alpha=0.7)

    # Draw joints colored by importance
    max_radius = 200
    min_radius = 30
    for j in range(n_joints):
        imp = float(joint_imp[j])
        color = _importance_to_color(imp, vmin, vmax)
        radius = min_radius + (max_radius - min_radius) * (imp - vmin) / (vmax - vmin + 1e-8)
        ax.scatter(xs[j], ys[j], s=radius, c=[color], zorder=3, edgecolors="white", linewidths=0.5)

    # Colorbar
    cmap = plt.cm.RdBu_r
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="SHAP Importance")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")

    return fig


def draw_correction_skeleton(
    current_frame: np.ndarray,
    target_frame: Optional[np.ndarray],
    corrections: List[Any],  # List[PoseCorrection]
    output_path: Optional[Path] = None,
    title: str = "Pose Correction",
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 200,
    flip_y: bool = True,
) -> Any:
    """Draw current pose with correction arrows toward target.

    Current pose is shown in gray.
    Target pose is shown in green (if provided).
    Arrows show where joints need to move, colored by urgency.

    Args:
        current_frame: (N_joints, 2) current pose
        target_frame: (N_joints, 2) optional target pose (if None, derived from corrections)
        corrections: list of PoseCorrection objects
        output_path: save path
        title: plot title
        figsize: figure dimensions
        dpi: resolution
        flip_y: flip y-axis

    Returns:
        matplotlib Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    n_joints = len(current_frame)
    ys_sign = -1 if flip_y else 1

    fig, ax = plt.subplots(figsize=figsize)

    # Current pose in gray
    xs_c = current_frame[:, 0]
    ys_c = current_frame[:, 1] * ys_sign

    for a, b in JHMDB_BONES:
        if a < n_joints and b < n_joints:
            ax.plot([xs_c[a], xs_c[b]], [ys_c[a], ys_c[b]],
                    color="#bdc3c7", linewidth=2, zorder=1, alpha=0.8)
    ax.scatter(xs_c, ys_c, s=60, c="#95a5a6", zorder=2, label="Current pose")

    # Target pose in green (if provided)
    if target_frame is not None:
        xs_t = target_frame[:, 0]
        ys_t = target_frame[:, 1] * ys_sign
        for a, b in JHMDB_BONES:
            if a < n_joints and b < n_joints:
                ax.plot([xs_t[a], xs_t[b]], [ys_t[a], ys_t[b]],
                        color="#27ae60", linewidth=2, zorder=1, alpha=0.6, linestyle="--")
        ax.scatter(xs_t, ys_t, s=60, c="#2ecc71", zorder=2, alpha=0.7, label="Target pose")

    # Correction arrows for corrected joints
    max_delta = max((c.delta_abs for c in corrections), default=1.0)

    from psrn.data.jhmdb_loader import JHMDB_KP_NAMES
    kp_map = {name.lower(): i for i, name in enumerate(JHMDB_KP_NAMES)}

    annotated_joints = set()
    for corr in corrections[:8]:  # show top-8 corrections
        # Find which joint this correction relates to
        joint_idx = None
        for kname, kidx in kp_map.items():
            if kname in corr.body_part.lower():
                joint_idx = kidx
                break

        if joint_idx is None or joint_idx >= n_joints:
            continue
        if joint_idx in annotated_joints:
            continue
        annotated_joints.add(joint_idx)

        color = _severity_to_color(corr.delta_abs, max_delta)
        x0, y0 = xs_c[joint_idx], ys_c[joint_idx]

        if target_frame is not None:
            dx = xs_t[joint_idx] - x0
            dy = ys_t[joint_idx] - y0
        else:
            # Approximate arrow direction from delta sign
            dx = 0.02 if corr.direction == "increase" else -0.02
            dy = 0.0

        if abs(dx) > 1e-4 or abs(dy) > 1e-4:
            ax.annotate(
                "", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
                zorder=5,
            )

    # Legend
    legend_elements = [
        mpatches.Patch(color="#95a5a6", label="Current pose"),
        mpatches.Patch(color="#2ecc71", label="Target pose"),
        mpatches.Patch(color="#e74c3c", label="Urgent correction"),
        mpatches.Patch(color="#f39c12", label="Moderate correction"),
        mpatches.Patch(color="#2ecc71", label="Minor correction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")

    return fig


def draw_skeleton_comparison(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    shap_a: Dict[str, float],
    shap_b: Dict[str, float],
    feature_names: List[str],
    class_names: Tuple[str, str],
    output_path: Optional[Path] = None,
    dpi: int = 200,
) -> Any:
    """Side-by-side SHAP skeleton comparison for two classes.

    Shows which joints are important for distinguishing class A vs. class B.

    Args:
        frame_a, frame_b: pose frames for each class
        shap_a, shap_b: SHAP importance dicts for each class
        feature_names: feature names list
        class_names: (class_a_name, class_b_name)
        output_path: save path
        dpi: resolution

    Returns:
        matplotlib Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    for ax, frame, shap_imp, cls_name in zip(
        axes, [frame_a, frame_b], [shap_a, shap_b], class_names
    ):
        n_joints = len(frame)
        joint_imp = feature_importances_to_joints(feature_names, shap_imp, n_joints)
        vmin = joint_imp.min()
        vmax = joint_imp.max()
        xs = frame[:, 0]
        ys = -frame[:, 1]  # flip y

        for a, b in JHMDB_BONES:
            if a < n_joints and b < n_joints:
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#95a5a6", linewidth=2, zorder=1)

        for j in range(n_joints):
            imp = float(joint_imp[j])
            color = _importance_to_color(imp, vmin, vmax)
            r = 30 + 170 * (imp - vmin) / (vmax - vmin + 1e-8)
            ax.scatter(xs[j], ys[j], s=r, c=[color], zorder=3, edgecolors="white", linewidths=0.5)

        ax.set_title(cls_name, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("SHAP Skeleton Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")

    return fig
