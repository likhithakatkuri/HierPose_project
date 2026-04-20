"""Temporal pose features computed across a video sequence.

Two types:
1. Per-frame temporal features (computed using neighboring frames):
   - Velocity: joint displacement magnitude frame-to-frame
   - Acceleration: change in velocity
   - Moving-average velocity (smoothed, 5-frame window)
   - Motion energy: global motion across all joints

2. Clip-level sequence features (one value per clip):
   - Range of motion: max - min for each angle over the clip
   - Temporal variance of angles
   - Peak velocity frame index (normalized to [0,1])
   - Directional histogram: dominant motion direction distribution
   - Number of direction reversals (oscillation proxy)

These are returned as separate arrays and combined in the extractor.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from psrn.data.jhmdb_loader import (
    JHMDB_KP_NAMES, N_JOINTS,
    L_WRIST, R_WRIST, L_ELBOW, R_ELBOW,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    L_SHOULDER, R_SHOULDER, NECK, BELLY,
)
from psrn.features.static import ANGLE_TRIPLETS, _joint_angle


# Key joints tracked for velocity/acceleration
VEL_JOINTS: List[int] = [
    L_WRIST, R_WRIST,
    L_ELBOW, R_ELBOW,
    L_KNEE,  R_KNEE,
    L_ANKLE, R_ANKLE,
    L_SHOULDER, R_SHOULDER,
]
VEL_JOINT_NAMES: List[str] = [
    "l_wrist", "r_wrist",
    "l_elbow", "r_elbow",
    "l_knee",  "r_knee",
    "l_ankle", "r_ankle",
    "l_shoulder", "r_shoulder",
]
N_VEL_JOINTS = len(VEL_JOINTS)


# ─────────────────────────────────────────────────────────────
# Per-frame temporal features
# ─────────────────────────────────────────────────────────────

def sequence_velocity_features(
    frames: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Frame-to-frame joint velocity (displacement magnitude).

    Args:
        frames: (T, N_joints, 2)

    Returns:
        features: (T, N_VEL_JOINTS) — velocity magnitude per joint per frame
        names: list of N_VEL_JOINTS strings
    """
    T = frames.shape[0]
    feats = np.zeros((T, N_VEL_JOINTS), dtype=np.float32)

    for t in range(1, T):
        for j_idx, joint in enumerate(VEL_JOINTS):
            diff = frames[t, joint] - frames[t - 1, joint]
            feats[t, j_idx] = float(np.sqrt(diff[0] ** 2 + diff[1] ** 2))

    names = [f"vel_{name}" for name in VEL_JOINT_NAMES]
    return feats, names


def sequence_acceleration_features(
    frames: np.ndarray,
    velocities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Frame-to-frame joint acceleration (change in velocity magnitude).

    Args:
        frames: (T, N_joints, 2)
        velocities: (T, N_VEL_JOINTS) — precomputed velocities for efficiency

    Returns:
        features: (T, N_VEL_JOINTS)
        names: list of N_VEL_JOINTS strings
    """
    if velocities is None:
        velocities, _ = sequence_velocity_features(frames)

    T = frames.shape[0]
    feats = np.zeros((T, N_VEL_JOINTS), dtype=np.float32)

    for t in range(1, T):
        feats[t] = np.abs(velocities[t] - velocities[t - 1])

    names = [f"acc_{name}" for name in VEL_JOINT_NAMES]
    return feats, names


def sequence_moving_average_velocity(
    frames: np.ndarray,
    window: int = 5,
    velocities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Smoothed velocity using a sliding-window moving average.

    Reduces high-frequency noise from detection jitter.

    Args:
        frames: (T, N_joints, 2)
        window: smoothing window size (default 5)
        velocities: precomputed (T, N_VEL_JOINTS) velocities

    Returns:
        features: (T, N_VEL_JOINTS) smoothed velocities
        names: list of N_VEL_JOINTS strings
    """
    if velocities is None:
        velocities, _ = sequence_velocity_features(frames)

    T, J = velocities.shape
    feats = np.zeros_like(velocities)
    half = window // 2

    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        feats[t] = velocities[start:end].mean(axis=0)

    names = [f"vel_ma_{name}" for name in VEL_JOINT_NAMES]
    return feats, names


def sequence_motion_energy_features(
    frames: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Global motion energy and directional histogram per frame.

    Captures overall amount and direction of motion — distinguishes
    active (large energy) vs. static actions.

    Args:
        frames: (T, N_joints, 2)

    Returns:
        features: (T, 9) — 1 energy value + 8 directional histogram bins
        names: list of 9 strings
    """
    T = frames.shape[0]
    N_BINS = 8
    feats = np.zeros((T, 1 + N_BINS), dtype=np.float32)

    bin_edges = np.linspace(-np.pi, np.pi, N_BINS + 1)

    for t in range(1, T):
        displacements = frames[t] - frames[t - 1]   # (N_joints, 2)
        magnitudes = np.linalg.norm(displacements, axis=1)  # (N_joints,)

        # Total motion energy
        feats[t, 0] = float(magnitudes.sum())

        # Directional histogram (weighted by magnitude)
        angles = np.arctan2(displacements[:, 1], displacements[:, 0])  # (N_joints,)
        hist, _ = np.histogram(angles, bins=bin_edges, weights=magnitudes)
        total = hist.sum() + 1e-8
        feats[t, 1:] = (hist / total).astype(np.float32)

    names = ["motion_energy"] + [f"motion_dir_bin{i}" for i in range(N_BINS)]
    return feats, names


# ─────────────────────────────────────────────────────────────
# Clip-level sequence features (one vector per clip)
# ─────────────────────────────────────────────────────────────

def sequence_range_of_motion(
    frames: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Range of motion: max - min angle value over the clip, for each angle.

    High ROM → dynamic action (jump, wave); low ROM → static (stand, sit).

    Args:
        frames: (T, N_joints, 2)

    Returns:
        features: (n_angles,) — one ROM value per angle triplet
        names: list of strings
    """
    T = frames.shape[0]
    n_angles = len(ANGLE_TRIPLETS)
    angle_series = np.zeros((T, n_angles), dtype=np.float32)

    for t in range(T):
        for i, (a, b, c, _) in enumerate(ANGLE_TRIPLETS):
            _, deg = _joint_angle(frames[t], a, b, c)
            angle_series[t, i] = deg

    rom = angle_series.max(axis=0) - angle_series.min(axis=0)  # (n_angles,)
    names = [f"rom_{label}" for _, _, _, label in ANGLE_TRIPLETS]
    return rom.astype(np.float32), names


def sequence_temporal_variance(
    frames: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Temporal variance of each joint angle over the clip.

    High variance → lots of movement; low → static.

    Returns:
        features: (n_angles,)
        names: list of strings
    """
    T = frames.shape[0]
    n_angles = len(ANGLE_TRIPLETS)
    angle_series = np.zeros((T, n_angles), dtype=np.float32)

    for t in range(T):
        for i, (a, b, c, _) in enumerate(ANGLE_TRIPLETS):
            _, deg = _joint_angle(frames[t], a, b, c)
            angle_series[t, i] = deg

    variance = angle_series.var(axis=0)  # (n_angles,)
    names = [f"var_{label}" for _, _, _, label in ANGLE_TRIPLETS]
    return variance.astype(np.float32), names


def sequence_peak_velocity_timing(
    frames: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Normalized timing of peak velocity for key joints.

    Captures where in the action the fastest motion occurs:
    - Early peak → sudden start
    - Late peak → build-up then explosion

    Returns:
        features: (N_VEL_JOINTS,) — peak timing as fraction of clip [0, 1]
        names: list of strings
    """
    velocities, _ = sequence_velocity_features(frames)
    T = velocities.shape[0]

    peak_timing = np.argmax(velocities, axis=0).astype(np.float32)
    peak_timing /= max(T - 1, 1)  # normalize to [0, 1]

    names = [f"peak_timing_{name}" for name in VEL_JOINT_NAMES]
    return peak_timing, names


def sequence_direction_reversals(
    frames: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Count direction reversals for each key joint (oscillation proxy).

    Useful for distinguishing cyclic actions (clap, wave) vs. one-shot (throw, kick).

    Returns:
        features: (N_VEL_JOINTS,) — number of velocity sign changes per joint
        names: list of strings
    """
    velocities, _ = sequence_velocity_features(frames)
    T = velocities.shape[0]

    reversals = np.zeros(N_VEL_JOINTS, dtype=np.float32)
    for j in range(N_VEL_JOINTS):
        v = velocities[:, j]
        signs = np.sign(v[1:] - v[:-1])
        reversals[j] = float(np.sum(np.abs(np.diff(signs)) > 0))

    # Normalize by sequence length
    reversals /= max(T - 1, 1)

    names = [f"reversals_{name}" for name in VEL_JOINT_NAMES]
    return reversals, names


# ─────────────────────────────────────────────────────────────
# Combined temporal features
# ─────────────────────────────────────────────────────────────

def sequence_all_temporal_per_frame(
    frames: np.ndarray,
    enabled_groups: Optional[List[str]] = None,
    window: int = 5,
) -> Tuple[np.ndarray, List[str]]:
    """Compute all per-frame temporal features.

    Returns:
        features: (T, n_temporal_features) array
        names: list of n_temporal_features strings
    """
    def _include(g: str) -> bool:
        return enabled_groups is None or g in enabled_groups

    all_feats: List[np.ndarray] = []
    all_names: List[str] = []

    velocities: Optional[np.ndarray] = None

    if _include("temporal_vel") or _include("temporal_acc") or _include("temporal_ma"):
        velocities, vel_names = sequence_velocity_features(frames)
        if _include("temporal_vel"):
            all_feats.append(velocities)
            all_names.extend(vel_names)

    if _include("temporal_acc") and velocities is not None:
        f, n = sequence_acceleration_features(frames, velocities)
        all_feats.append(f); all_names.extend(n)

    if _include("temporal_ma") and velocities is not None:
        f, n = sequence_moving_average_velocity(frames, window, velocities)
        all_feats.append(f); all_names.extend(n)

    if _include("motion_energy"):
        f, n = sequence_motion_energy_features(frames)
        all_feats.append(f); all_names.extend(n)

    if not all_feats:
        T = frames.shape[0]
        return np.zeros((T, 0), dtype=np.float32), []

    return np.concatenate(all_feats, axis=1).astype(np.float32), all_names


def sequence_all_clip_level(
    frames: np.ndarray,
    enabled_groups: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute all clip-level (sequence-level) features.

    Returns:
        features: (n_clip_features,) flat array
        names: list of strings
    """
    def _include(g: str) -> bool:
        return enabled_groups is None or g in enabled_groups

    all_feats: List[np.ndarray] = []
    all_names: List[str] = []

    if _include("range_of_motion"):
        f, n = sequence_range_of_motion(frames)
        all_feats.append(f); all_names.extend(n)

    if _include("temporal_variance"):
        f, n = sequence_temporal_variance(frames)
        all_feats.append(f); all_names.extend(n)

    if _include("peak_velocity"):
        f, n = sequence_peak_velocity_timing(frames)
        all_feats.append(f); all_names.extend(n)

    if _include("direction_reversals"):
        f, n = sequence_direction_reversals(frames)
        all_feats.append(f); all_names.extend(n)

    if not all_feats:
        return np.zeros(0, dtype=np.float32), []

    return np.concatenate(all_feats).astype(np.float32), all_names
