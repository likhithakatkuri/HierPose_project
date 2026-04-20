"""Static (per-frame) hierarchical pose features.

Computes anatomically structured features from a single frame's joint positions.
All features are normalized for scale and translation invariance.

Feature groups:
    A. Joint angles (cosine + degrees) — 14 triplets → 28 features
    B. Inter-joint distances (torso-normalized) — 14 pairs → 14 features
    C. Limb ratios (scale-invariant proportions) — 4 features
    D. Anatomical centroids (6 groups × dx,dy) — 12 features
    E. Bilateral symmetry (L/R angle differences + symmetry scores) — 12 features
    F. Joint orientation vectors (limb direction unit vectors) — 12 features
    G. Spatial extent (bounding box, pose compactness) — 8 features
    H. Cross-body coordination (wrist-hip, elbow-knee proximity) — 10 features

Total static features per frame: ~100 features (exact count in get_static_feature_count())
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from psrn.data.jhmdb_loader import JHMDB_KP_NAMES

# ─────────────────────────────────────────────────────────────
# JHMDB joint index constants (for readability)
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
N_JOINTS = 15


# ─────────────────────────────────────────────────────────────
# Helper math
# ─────────────────────────────────────────────────────────────

def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector from joint a to joint b."""
    return b - a


def _length(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    d = b - a
    return float(np.sqrt(d[0] ** 2 + d[1] ** 2))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float]:
    """Angle between two 2D vectors.

    Returns:
        (cosine, angle_degrees)  — cosine is in [-1, 1]
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0, 0.0
    cos_val = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    angle_deg = float(math.degrees(math.acos(cos_val)))
    return cos_val, angle_deg


def _torso_scale(frame: np.ndarray) -> float:
    """Torso scale: distance from mid-shoulder to mid-hip.

    Used to normalize all distances for scale invariance.
    Returns 1.0 if degenerate.
    """
    mid_shoulder = (frame[R_SHOULDER] + frame[L_SHOULDER]) / 2.0
    mid_hip = (frame[R_HIP] + frame[L_HIP]) / 2.0
    scale = _length(mid_shoulder, mid_hip)
    return scale if scale > 1e-8 else 1.0


def _normalize_dist(dist: float, scale: float) -> float:
    return dist / scale if scale > 1e-8 else dist


def _joint_angle(frame: np.ndarray, a: int, b: int, c: int) -> Tuple[float, float]:
    """Angle at joint b in the triplet a-b-c.

    Returns:
        (cosine, degrees)
    """
    v1 = _vec(frame[b], frame[a])
    v2 = _vec(frame[b], frame[c])
    return _angle_between(v1, v2)


# ─────────────────────────────────────────────────────────────
# Group A: Joint Angles
# ─────────────────────────────────────────────────────────────

# (a, vertex_b, c, name)
ANGLE_TRIPLETS: List[Tuple[int, int, int, str]] = [
    # Original 8 angles
    (L_SHOULDER, L_ELBOW, L_WRIST,   "l_elbow"),
    (R_SHOULDER, R_ELBOW, R_WRIST,   "r_elbow"),
    (NECK,       L_SHOULDER, L_ELBOW, "l_shoulder"),
    (NECK,       R_SHOULDER, R_ELBOW, "r_shoulder"),
    (L_SHOULDER, L_HIP, L_KNEE,      "l_hip"),
    (R_SHOULDER, R_HIP, R_KNEE,      "r_hip"),
    (L_HIP,      L_KNEE, L_ANKLE,    "l_knee"),
    (R_HIP,      R_KNEE, R_ANKLE,    "r_knee"),
    # Extended 6 angles (Group A extension)
    (R_SHOULDER, NECK, L_SHOULDER,   "shoulder_span"),  # shoulder opening angle
    (R_HIP, BELLY, L_HIP,            "hip_span"),        # hip opening angle
    (NECK, BELLY, R_HIP,             "trunk_inclination"),# spine angle (lean)
    (L_KNEE, L_ANKLE, R_ANKLE,       "l_ankle_dorsi"),   # ankle/foot angle
    (R_KNEE, R_ANKLE, L_ANKLE,       "r_ankle_dorsi"),
    (L_WRIST, NECK, R_WRIST,         "reach_angle"),      # arm reach angle
]


def frame_angle_features(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute joint angles for all triplets.

    Returns:
        features: (28,) array — 2 values per triplet (cosine, degrees)
        names: list of 28 feature name strings
    """
    feats, names = [], []
    for a, b, c, label in ANGLE_TRIPLETS:
        cos_val, deg_val = _joint_angle(frame, a, b, c)
        feats.extend([cos_val, deg_val])
        names.extend([f"angle_{label}_cos", f"angle_{label}_deg"])
    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Group B: Inter-Joint Distances
# ─────────────────────────────────────────────────────────────

DISTANCE_PAIRS: List[Tuple[int, int, str]] = [
    # Limb segment distances
    (L_SHOULDER, L_ELBOW,    "l_upper_arm"),
    (R_SHOULDER, R_ELBOW,    "r_upper_arm"),
    (L_ELBOW, L_WRIST,       "l_forearm"),
    (R_ELBOW, R_WRIST,       "r_forearm"),
    (L_HIP, L_KNEE,          "l_thigh"),
    (R_HIP, R_KNEE,          "r_thigh"),
    (L_KNEE, L_ANKLE,        "l_shin"),
    (R_KNEE, R_ANKLE,        "r_shin"),
    # Span distances
    (L_SHOULDER, R_SHOULDER, "shoulder_width"),
    (L_HIP, R_HIP,           "hip_width"),
    (NECK, BELLY,            "spine_length"),
    (L_WRIST, R_WRIST,       "wrist_span"),
    (L_ANKLE, R_ANKLE,       "ankle_span"),
    # Functional distances
    (R_WRIST, FACE,          "r_hand_to_face"),
]


def frame_distance_features(
    frame: np.ndarray, scale: Optional[float] = None
) -> Tuple[np.ndarray, List[str]]:
    """Compute torso-normalized inter-joint distances.

    Returns:
        features: (14,) array of normalized distances
        names: list of 14 feature name strings
    """
    if scale is None:
        scale = _torso_scale(frame)
    feats, names = [], []
    for a, b, label in DISTANCE_PAIRS:
        dist = _normalize_dist(_length(frame[a], frame[b]), scale)
        feats.append(dist)
        names.append(f"dist_{label}")
    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Group C: Limb Ratios
# ─────────────────────────────────────────────────────────────

def frame_ratio_features(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute scale-invariant limb proportion ratios.

    Returns:
        features: (4,) array
        names: list of 4 feature name strings
    """
    eps = 1e-8
    l_upper = _length(frame[L_SHOULDER], frame[L_ELBOW]) + eps
    r_upper = _length(frame[R_SHOULDER], frame[R_ELBOW]) + eps
    l_lower = _length(frame[L_ELBOW], frame[L_WRIST])
    r_lower = _length(frame[R_ELBOW], frame[R_WRIST])
    l_thigh = _length(frame[L_HIP], frame[L_KNEE]) + eps
    r_thigh = _length(frame[R_HIP], frame[R_KNEE]) + eps
    l_shin = _length(frame[L_KNEE], frame[L_ANKLE])
    r_shin = _length(frame[R_KNEE], frame[R_ANKLE])

    feats = [
        l_lower / l_upper,   # left forearm/upper-arm ratio
        r_lower / r_upper,   # right forearm/upper-arm ratio
        l_shin / l_thigh,    # left shin/thigh ratio
        r_shin / r_thigh,    # right shin/thigh ratio
    ]
    names = [
        "ratio_l_forearm_upper", "ratio_r_forearm_upper",
        "ratio_l_shin_thigh", "ratio_r_shin_thigh",
    ]
    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Group D: Anatomical Centroids
# ─────────────────────────────────────────────────────────────

ANATOMICAL_GROUPS: List[Tuple[str, List[int]]] = [
    ("head",       [FACE, NECK]),
    ("torso",      [NECK, BELLY, R_SHOULDER, L_SHOULDER]),
    ("l_arm",      [L_SHOULDER, L_ELBOW, L_WRIST]),
    ("r_arm",      [R_SHOULDER, R_ELBOW, R_WRIST]),
    ("l_leg",      [L_HIP, L_KNEE, L_ANKLE]),
    ("r_leg",      [R_HIP, R_KNEE, R_ANKLE]),
]


def frame_centroid_features(
    frame: np.ndarray, scale: Optional[float] = None
) -> Tuple[np.ndarray, List[str]]:
    """Compute centroid position of each anatomical group.

    Returns offsets (dx, dy) from NECK, normalized by torso scale.

    Returns:
        features: (12,) array — 2 per group
        names: list of 12 feature name strings
    """
    if scale is None:
        scale = _torso_scale(frame)
    origin = frame[NECK]
    feats, names = [], []
    for group_name, joint_indices in ANATOMICAL_GROUPS:
        pts = frame[joint_indices]          # (k, 2)
        centroid = pts.mean(axis=0)         # (2,)
        offset = (centroid - origin) / scale
        feats.extend([float(offset[0]), float(offset[1])])
        names.extend([f"centroid_{group_name}_dx", f"centroid_{group_name}_dy"])
    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Group E: Bilateral Symmetry
# ─────────────────────────────────────────────────────────────

def frame_symmetry_features(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute left-right symmetry features.

    Features:
    - L/R angle differences for elbow, shoulder, hip, knee (4 abs differences)
    - L/R wrist height relative to shoulder (2 values)
    - L/R ankle height relative to hip (2 values)
    - Bilateral symmetry score for upper body (1 value, [0,1] where 1=perfectly symmetric)
    - Bilateral symmetry score for lower body (1 value)
    - Wrist height asymmetry (1 value)
    - Shoulder height difference (1 value)

    Returns:
        features: (12,) array
        names: list of 12 strings
    """
    feats, names = [], []

    # Angle differences (absolute)
    _, l_elbow_deg = _joint_angle(frame, L_SHOULDER, L_ELBOW, L_WRIST)
    _, r_elbow_deg = _joint_angle(frame, R_SHOULDER, R_ELBOW, R_WRIST)
    _, l_shoulder_deg = _joint_angle(frame, NECK, L_SHOULDER, L_ELBOW)
    _, r_shoulder_deg = _joint_angle(frame, NECK, R_SHOULDER, R_ELBOW)
    _, l_hip_deg = _joint_angle(frame, L_SHOULDER, L_HIP, L_KNEE)
    _, r_hip_deg = _joint_angle(frame, R_SHOULDER, R_HIP, R_KNEE)
    _, l_knee_deg = _joint_angle(frame, L_HIP, L_KNEE, L_ANKLE)
    _, r_knee_deg = _joint_angle(frame, R_HIP, R_KNEE, R_ANKLE)

    feats.extend([
        abs(l_elbow_deg - r_elbow_deg),
        abs(l_shoulder_deg - r_shoulder_deg),
        abs(l_hip_deg - r_hip_deg),
        abs(l_knee_deg - r_knee_deg),
    ])
    names.extend([
        "sym_elbow_angle_diff", "sym_shoulder_angle_diff",
        "sym_hip_angle_diff", "sym_knee_angle_diff",
    ])

    # Wrist height relative to shoulder (y-axis, larger y = lower in image coords)
    scale = _torso_scale(frame)
    l_wrist_rel = (frame[L_WRIST][1] - frame[L_SHOULDER][1]) / scale
    r_wrist_rel = (frame[R_WRIST][1] - frame[R_SHOULDER][1]) / scale
    feats.extend([l_wrist_rel, r_wrist_rel])
    names.extend(["sym_l_wrist_height", "sym_r_wrist_height"])

    # Ankle height relative to hip
    l_ankle_rel = (frame[L_ANKLE][1] - frame[L_HIP][1]) / scale
    r_ankle_rel = (frame[R_ANKLE][1] - frame[R_HIP][1]) / scale
    feats.extend([l_ankle_rel, r_ankle_rel])
    names.extend(["sym_l_ankle_height", "sym_r_ankle_height"])

    # Bilateral symmetry score: 1 - |diff| / (|left| + |right| + eps)
    def _sym_score(left_pts: np.ndarray, right_pts: np.ndarray) -> float:
        # Mirror right joints by flipping x around mid-line
        mid_x = (frame[L_SHOULDER][0] + frame[R_SHOULDER][0]) / 2
        mirrored_right = right_pts.copy()
        mirrored_right[:, 0] = 2 * mid_x - mirrored_right[:, 0]
        diff = np.mean(np.linalg.norm(left_pts - mirrored_right, axis=1))
        total = np.mean(np.linalg.norm(left_pts - frame[NECK], axis=1)) + \
                np.mean(np.linalg.norm(right_pts - frame[NECK], axis=1)) + 1e-8
        return float(1.0 - min(diff / total, 1.0))

    upper_l = frame[[L_SHOULDER, L_ELBOW, L_WRIST]]
    upper_r = frame[[R_SHOULDER, R_ELBOW, R_WRIST]]
    lower_l = frame[[L_HIP, L_KNEE, L_ANKLE]]
    lower_r = frame[[R_HIP, R_KNEE, R_ANKLE]]

    feats.append(_sym_score(upper_l, upper_r))
    feats.append(_sym_score(lower_l, lower_r))
    names.extend(["sym_upper_score", "sym_lower_score"])

    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Group F: Joint Orientation Vectors
# ─────────────────────────────────────────────────────────────

ORIENTATION_SEGMENTS: List[Tuple[int, int, str]] = [
    (L_SHOULDER, L_ELBOW,   "l_upper_arm"),
    (R_SHOULDER, R_ELBOW,   "r_upper_arm"),
    (L_ELBOW,    L_WRIST,   "l_forearm"),
    (R_ELBOW,    R_WRIST,   "r_forearm"),
    (L_HIP,      L_KNEE,    "l_thigh"),
    (R_HIP,      R_KNEE,    "r_thigh"),
]


def frame_orientation_features(frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute normalized direction vectors for key limb segments.

    Each limb gives a unit vector (dx, dy) in 2D — captures directionality
    lost in angle magnitude (e.g., arm pointing up vs. arm pointing down
    can have same angle but different direction).

    Returns:
        features: (12,) array — 2 per segment
        names: list of 12 strings
    """
    feats, names = [], []
    for a, b, label in ORIENTATION_SEGMENTS:
        vec = _vec(frame[a], frame[b])
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            unit = vec / norm
        else:
            unit = np.zeros(2)
        feats.extend([float(unit[0]), float(unit[1])])
        names.extend([f"orient_{label}_dx", f"orient_{label}_dy"])
    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Group G: Spatial Extent (Pose Compactness)
# ─────────────────────────────────────────────────────────────

def frame_extent_features(
    frame: np.ndarray, scale: Optional[float] = None
) -> Tuple[np.ndarray, List[str]]:
    """Compute bounding box and pose compactness features.

    Returns:
        features: (8,) array
        names: list of 8 strings
    """
    if scale is None:
        scale = _torso_scale(frame)

    xs = frame[:, 0]
    ys = frame[:, 1]

    bbox_w = (xs.max() - xs.min()) / scale
    bbox_h = (ys.max() - ys.min()) / scale
    bbox_ar = bbox_w / (bbox_h + 1e-8)          # aspect ratio
    bbox_area = bbox_w * bbox_h                  # normalized area

    # Compactness: ratio of (sum of dist to centroid) over bbox diagonal
    centroid = frame.mean(axis=0)
    dists = np.linalg.norm(frame - centroid, axis=1)
    avg_dist = dists.mean() / scale
    bbox_diag = math.sqrt(bbox_w ** 2 + bbox_h ** 2 + 1e-8)
    compactness = avg_dist / (bbox_diag + 1e-8)

    # Vertical extent: head (face) to lowest ankle, normalized
    v_extent = abs(frame[FACE][1] - max(frame[L_ANKLE][1], frame[R_ANKLE][1])) / scale

    # Horizontal span: max wrist-to-wrist
    h_span = abs(frame[L_WRIST][0] - frame[R_WRIST][0]) / scale

    # Y-position of center of mass (height of pose in frame)
    com_y = float(centroid[1]) / (ys.max() - ys.min() + 1e-8)

    feats = [bbox_w, bbox_h, bbox_ar, bbox_area, compactness, v_extent, h_span, com_y]
    names = [
        "extent_bbox_w", "extent_bbox_h", "extent_bbox_ar", "extent_bbox_area",
        "extent_compactness", "extent_vertical", "extent_wrist_span", "extent_com_y",
    ]
    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Group H: Cross-Body Coordination
# ─────────────────────────────────────────────────────────────

def frame_cross_body_features(
    frame: np.ndarray, scale: Optional[float] = None
) -> Tuple[np.ndarray, List[str]]:
    """Compute cross-body proximity features.

    These capture gestures like crossing arms, kicking, reaching across body.

    Returns:
        features: (10,) array
        names: list of 10 strings
    """
    if scale is None:
        scale = _torso_scale(frame)

    feats, names = [], []

    # Wrist to opposite hip (crossing gesture detection)
    l_wrist_r_hip = _normalize_dist(_length(frame[L_WRIST], frame[R_HIP]), scale)
    r_wrist_l_hip = _normalize_dist(_length(frame[R_WRIST], frame[L_HIP]), scale)
    feats.extend([l_wrist_r_hip, r_wrist_l_hip])
    names.extend(["cross_l_wrist_r_hip", "cross_r_wrist_l_hip"])

    # Elbow to opposite knee (kicking, martial arts)
    l_elbow_r_knee = _normalize_dist(_length(frame[L_ELBOW], frame[R_KNEE]), scale)
    r_elbow_l_knee = _normalize_dist(_length(frame[R_ELBOW], frame[L_KNEE]), scale)
    feats.extend([l_elbow_r_knee, r_elbow_l_knee])
    names.extend(["cross_l_elbow_r_knee", "cross_r_elbow_l_knee"])

    # Wrist to face (reaching to face/head gesture)
    l_wrist_face = _normalize_dist(_length(frame[L_WRIST], frame[FACE]), scale)
    r_wrist_face = _normalize_dist(_length(frame[R_WRIST], frame[FACE]), scale)
    feats.extend([l_wrist_face, r_wrist_face])
    names.extend(["cross_l_wrist_face", "cross_r_wrist_face"])

    # Foot spread (ankle-to-ankle relative to hip width)
    ankle_span = _length(frame[L_ANKLE], frame[R_ANKLE])
    hip_span = _length(frame[L_HIP], frame[R_HIP]) + 1e-8
    foot_spread = ankle_span / hip_span
    feats.append(foot_spread)
    names.append("cross_foot_spread_ratio")

    # Wrist-to-wrist (hands together vs. apart)
    wrist_dist = _normalize_dist(_length(frame[L_WRIST], frame[R_WRIST]), scale)
    feats.append(wrist_dist)
    names.append("cross_wrist_distance")

    # Center of hands relative to center of body
    hand_center = (frame[L_WRIST] + frame[R_WRIST]) / 2
    body_center = (frame[R_SHOULDER] + frame[L_SHOULDER] + frame[R_HIP] + frame[L_HIP]) / 4
    hand_body_dist = _normalize_dist(_length(hand_center, body_center), scale)
    feats.append(hand_body_dist)
    names.append("cross_hand_body_dist")

    return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────
# Combined per-frame static features
# ─────────────────────────────────────────────────────────────

def frame_all_static_features(
    frame: np.ndarray,
    enabled_groups: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute all (or a subset of) static features for a single frame.

    Args:
        frame: (N_joints, 2) joint position array
        enabled_groups: list of group names to include, or None for all.
            Valid: "angles", "distances", "ratios", "centroids",
                   "symmetry", "orientation", "extent", "cross_body"

    Returns:
        features: (n_features,) concatenated array
        names: list of n_features strings
    """
    if frame.shape[0] < N_JOINTS:
        # Pad with zeros if fewer joints
        pad = np.zeros((N_JOINTS - frame.shape[0], frame.shape[1]), dtype=np.float32)
        frame = np.vstack([frame, pad])

    scale = _torso_scale(frame)
    all_groups = enabled_groups  # None means include all

    all_feats: List[np.ndarray] = []
    all_names: List[str] = []

    def _include(group_name: str) -> bool:
        return all_groups is None or group_name in all_groups

    if _include("angles"):
        f, n = frame_angle_features(frame)
        all_feats.append(f); all_names.extend(n)

    if _include("distances"):
        f, n = frame_distance_features(frame, scale)
        all_feats.append(f); all_names.extend(n)

    if _include("ratios"):
        f, n = frame_ratio_features(frame)
        all_feats.append(f); all_names.extend(n)

    if _include("centroids"):
        f, n = frame_centroid_features(frame, scale)
        all_feats.append(f); all_names.extend(n)

    if _include("symmetry"):
        f, n = frame_symmetry_features(frame)
        all_feats.append(f); all_names.extend(n)

    if _include("orientation"):
        f, n = frame_orientation_features(frame)
        all_feats.append(f); all_names.extend(n)

    if _include("extent"):
        f, n = frame_extent_features(frame, scale)
        all_feats.append(f); all_names.extend(n)

    if _include("cross_body"):
        f, n = frame_cross_body_features(frame, scale)
        all_feats.append(f); all_names.extend(n)

    if not all_feats:
        return np.zeros(0, dtype=np.float32), []

    return np.concatenate(all_feats).astype(np.float32), all_names


def get_static_feature_count(enabled_groups: Optional[List[str]] = None) -> int:
    """Return the number of static features for the given group selection."""
    dummy = np.zeros((N_JOINTS, 2), dtype=np.float32)
    # Give a non-degenerate pose so scale != 0
    dummy[R_SHOULDER] = [0.3, 0.3]
    dummy[L_SHOULDER] = [0.7, 0.3]
    dummy[R_HIP] = [0.35, 0.6]
    dummy[L_HIP] = [0.65, 0.6]
    feats, _ = frame_all_static_features(dummy, enabled_groups)
    return len(feats)
