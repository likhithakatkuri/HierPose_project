"""Feature group registry for ablation-ready feature selection.

The registry maps group names to the static/temporal feature functions.
Using this registry, any combination of feature groups can be enabled
or disabled with a single config change — critical for ablation studies.

Usage:
    from psrn.features.registry import FEATURE_GROUPS, get_feature_group_names

    # All available group names
    print(get_feature_group_names())

    # Custom subset for ablation (e.g., no temporal features)
    subset = [g for g in get_feature_group_names() if not g.startswith("temporal")]
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

# ─────────────────────────────────────────────────────────────
# Registry structure
# ─────────────────────────────────────────────────────────────

# Static groups (per-frame): these are included in frame_all_static_features()
STATIC_GROUPS: List[str] = [
    "angles",       # Joint angles (cosine + degrees) — 28 features
    "distances",    # Inter-joint distances (torso-normalized) — 14 features
    "ratios",       # Limb proportion ratios — 4 features
    "centroids",    # Anatomical group centroids — 12 features
    "symmetry",     # Left-right symmetry — 12 features
    "orientation",  # Limb direction unit vectors — 12 features
    "extent",       # Pose compactness / bounding box — 8 features
    "cross_body",   # Cross-body coordination distances — 10 features
]

# Per-frame temporal groups (included in sequence_all_temporal_per_frame())
TEMPORAL_PER_FRAME_GROUPS: List[str] = [
    "temporal_vel",    # Joint velocity magnitudes — N_VEL_JOINTS features/frame
    "temporal_acc",    # Joint acceleration — N_VEL_JOINTS features/frame
    "temporal_ma",     # Smoothed (moving-average) velocity — N_VEL_JOINTS features/frame
    "motion_energy",   # Global motion energy + 8-dir histogram — 9 features/frame
]

# Clip-level temporal groups (one feature vector per video clip)
TEMPORAL_CLIP_GROUPS: List[str] = [
    "range_of_motion",      # Max-min angle per joint — n_angles features
    "temporal_variance",    # Temporal variance of angles — n_angles features
    "peak_velocity",        # Normalized peak velocity timing — N_VEL_JOINTS features
    "direction_reversals",  # Oscillation count per joint — N_VEL_JOINTS features
]

ALL_GROUPS: List[str] = STATIC_GROUPS + TEMPORAL_PER_FRAME_GROUPS + TEMPORAL_CLIP_GROUPS

# Human-readable descriptions for UI and paper
GROUP_DESCRIPTIONS: Dict[str, str] = {
    "angles":              "Joint angles (cosine + degrees) for 14 anatomical triplets",
    "distances":           "Torso-normalized distances between 14 joint pairs",
    "ratios":              "Scale-invariant limb proportion ratios (4)",
    "centroids":           "Anatomical group centroid offsets from neck (12)",
    "symmetry":            "Left-right bilateral symmetry features (12)",
    "orientation":         "Normalized limb direction unit vectors (12)",
    "extent":              "Pose bounding box and compactness features (8)",
    "cross_body":          "Cross-body coordination distances (10)",
    "temporal_vel":        "Joint velocity magnitudes per frame",
    "temporal_acc":        "Joint acceleration per frame",
    "temporal_ma":         "Moving-average smoothed velocity per frame",
    "motion_energy":       "Global motion energy + 8-direction histogram per frame",
    "range_of_motion":     "Range of motion (max-min angle) over clip",
    "temporal_variance":   "Temporal variance of joint angles over clip",
    "peak_velocity":       "Normalized peak velocity timing per joint",
    "direction_reversals": "Oscillation count per joint (direction reversals)",
}

# Approximate feature count per group (useful for Table 1 in paper)
# Exact counts from get_static_feature_count() / actual extraction
GROUP_FEATURE_COUNTS: Dict[str, str] = {
    "angles":              "28 (2 per triplet × 14)",
    "distances":           "14",
    "ratios":              "4",
    "centroids":           "12 (dx,dy × 6 groups)",
    "symmetry":            "12",
    "orientation":         "12 (dx,dy × 6 segments)",
    "extent":              "8",
    "cross_body":          "10",
    "temporal_vel":        "10/frame (N_VEL_JOINTS)",
    "temporal_acc":        "10/frame",
    "temporal_ma":         "10/frame",
    "motion_energy":       "9/frame (1 energy + 8 direction bins)",
    "range_of_motion":     "14 (one per angle triplet)",
    "temporal_variance":   "14",
    "peak_velocity":       "10 (one per tracked joint)",
    "direction_reversals": "10",
}

# Anatomical grouping for SHAP analysis (maps group → body region)
GROUP_BODY_REGION: Dict[str, str] = {
    "angles":              "Joints (angle)",
    "distances":           "Limbs (distance)",
    "ratios":              "Proportions",
    "centroids":           "Body segments (position)",
    "symmetry":            "Symmetry",
    "orientation":         "Limb directions",
    "extent":              "Pose shape",
    "cross_body":          "Cross-body",
    "temporal_vel":        "Temporal dynamics",
    "temporal_acc":        "Temporal dynamics",
    "temporal_ma":         "Temporal dynamics",
    "motion_energy":       "Motion energy",
    "range_of_motion":     "Motion range",
    "temporal_variance":   "Temporal dynamics",
    "peak_velocity":       "Temporal dynamics",
    "direction_reversals": "Cyclic motion",
}

# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

# The actual callable registry is populated lazily to avoid circular imports.
# Users should import FEATURE_GROUPS after feature modules are loaded.
FEATURE_GROUPS: Dict[str, dict] = {
    group: {
        "description": GROUP_DESCRIPTIONS.get(group, ""),
        "feature_count": GROUP_FEATURE_COUNTS.get(group, "?"),
        "body_region": GROUP_BODY_REGION.get(group, ""),
        "type": (
            "static" if group in STATIC_GROUPS else
            "temporal_per_frame" if group in TEMPORAL_PER_FRAME_GROUPS else
            "temporal_clip"
        ),
    }
    for group in ALL_GROUPS
}


def get_feature_group_names(
    include_static: bool = True,
    include_temporal_per_frame: bool = True,
    include_temporal_clip: bool = True,
) -> List[str]:
    """Return list of available feature group names.

    Args:
        include_static: include per-frame static groups
        include_temporal_per_frame: include per-frame temporal groups
        include_temporal_clip: include clip-level temporal groups

    Returns:
        Filtered list of group names
    """
    result: List[str] = []
    if include_static:
        result.extend(STATIC_GROUPS)
    if include_temporal_per_frame:
        result.extend(TEMPORAL_PER_FRAME_GROUPS)
    if include_temporal_clip:
        result.extend(TEMPORAL_CLIP_GROUPS)
    return result


def validate_group_names(groups: List[str]) -> None:
    """Raise ValueError if any group name is not in the registry."""
    unknown = set(groups) - set(ALL_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown feature groups: {unknown}. "
            f"Valid groups: {ALL_GROUPS}"
        )


def get_ablation_subsets(mode: str = "leave_one_out") -> List[List[str]]:
    """Generate feature group subsets for ablation experiments.

    Args:
        mode: "leave_one_out" — each subset drops one group from ALL_GROUPS
              "incremental"   — each subset adds one group sequentially

    Returns:
        List of group name lists, one per ablation experiment
    """
    if mode == "leave_one_out":
        return [
            [g for g in ALL_GROUPS if g != leave_out]
            for leave_out in ALL_GROUPS
        ]
    elif mode == "incremental":
        return [ALL_GROUPS[:i + 1] for i in range(len(ALL_GROUPS))]
    else:
        raise ValueError(f"Unknown ablation mode: {mode}. Use 'leave_one_out' or 'incremental'")


def feature_names_to_group(feature_names: List[str]) -> Dict[str, str]:
    """Map each feature name to its group (for SHAP group-level analysis).

    Returns:
        dict: feature_name → group_name
    """
    mapping: Dict[str, str] = {}
    for name in feature_names:
        for prefix, group in [
            ("angle_", "angles"),
            ("dist_", "distances"),
            ("ratio_", "ratios"),
            ("centroid_", "centroids"),
            ("sym_", "symmetry"),
            ("orient_", "orientation"),
            ("extent_", "extent"),
            ("cross_", "cross_body"),
            ("vel_ma_", "temporal_ma"),
            ("vel_", "temporal_vel"),
            ("acc_", "temporal_acc"),
            ("motion_", "motion_energy"),
            ("rom_", "range_of_motion"),
            ("var_", "temporal_variance"),
            ("peak_timing_", "peak_velocity"),
            ("reversals_", "direction_reversals"),
        ]:
            if name.startswith(prefix):
                mapping[name] = group
                break
        else:
            mapping[name] = "unknown"
    return mapping
