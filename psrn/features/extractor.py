"""Hierarchical Feature Extractor — main orchestrator.

Combines static per-frame features with temporal features and
provides the primary API for the HierPose ML pipeline.

Key design:
- FeatureConfig controls which groups are enabled (registry-based)
- Temporal aggregation uses [mean, std, Q1, Q3] — richer than mean-only
- Clip-level sequence features are concatenated at the end
- Caching to .npz files for fast repeated use

Usage:
    config = FeatureConfig(enabled_groups=["angles", "distances", "temporal_vel"])
    extractor = HierarchicalFeatureExtractor(config)

    # Extract from a sequence
    frames = np.random.randn(15, 15, 2).astype(np.float32)  # (T, 15, 2)
    features, names = extractor.extract_and_pool(frames)
    # features: (n_features,)  — ready for ML training

    # Or per-frame features
    feat_matrix, names = extractor.extract(frames)
    # feat_matrix: (T, n_per_frame_features)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from psrn.features.registry import (
    ALL_GROUPS,
    STATIC_GROUPS,
    TEMPORAL_CLIP_GROUPS,
    TEMPORAL_PER_FRAME_GROUPS,
    get_feature_group_names,
    validate_group_names,
)
from psrn.features.static import frame_all_static_features
from psrn.features.temporal import (
    sequence_all_clip_level,
    sequence_all_temporal_per_frame,
)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Joint schema: "jhmdb" (15 joints) or "mediapipe" (33 joints)
    joint_schema: str = "jhmdb"

    # Normalization
    normalize_by_torso: bool = True

    # Temporal moving average window
    temporal_window: int = 5

    # Feature groups to enable (None = all)
    enabled_groups: Optional[List[str]] = None

    # Temporal aggregation statistics for per-frame features
    # "mean", "std", "q25", "q75" — all four recommended for best performance
    temporal_agg_stats: List[str] = field(
        default_factory=lambda: ["mean", "std", "q25", "q75"]
    )

    def __post_init__(self) -> None:
        if self.enabled_groups is not None:
            validate_group_names(self.enabled_groups)

    def get_static_groups(self) -> List[str]:
        """Return enabled static group names."""
        if self.enabled_groups is None:
            return list(STATIC_GROUPS)
        return [g for g in self.enabled_groups if g in STATIC_GROUPS]

    def get_temporal_per_frame_groups(self) -> List[str]:
        """Return enabled per-frame temporal group names."""
        if self.enabled_groups is None:
            return list(TEMPORAL_PER_FRAME_GROUPS)
        return [g for g in self.enabled_groups if g in TEMPORAL_PER_FRAME_GROUPS]

    def get_temporal_clip_groups(self) -> List[str]:
        """Return enabled clip-level temporal group names."""
        if self.enabled_groups is None:
            return list(TEMPORAL_CLIP_GROUPS)
        return [g for g in self.enabled_groups if g in TEMPORAL_CLIP_GROUPS]

    @classmethod
    def from_hierpose_config(cls, hierpose_config) -> "FeatureConfig":
        """Create FeatureConfig from a HierPoseConfig."""
        return cls(
            joint_schema="jhmdb",
            normalize_by_torso=hierpose_config.normalize_by_torso,
            temporal_window=hierpose_config.temporal_window,
            enabled_groups=hierpose_config.enabled_feature_groups,
        )


# ─────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────

def save_features_npz(
    path: str,
    features: np.ndarray,
    feature_names: List[str],
) -> None:
    """Save features to compressed .npz file."""
    np.savez_compressed(
        path,
        feats=features,
        names=np.array(feature_names, dtype=object),
    )


def load_features_npz(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load features from .npz cache file.

    Returns:
        (features array, feature_names list)
    """
    data = np.load(path, allow_pickle=True)
    feats = data["feats"]
    names = list(data["names"])
    return feats, names


# ─────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────

class HierarchicalFeatureExtractor:
    """Extract hierarchical pose features from a joint sequence.

    The extractor is stateless — it can process any number of sequences.
    Feature names are consistent across all calls with the same config.

    Args:
        config: FeatureConfig controlling which groups are extracted

    Example:
        extractor = HierarchicalFeatureExtractor(FeatureConfig())
        features, names = extractor.extract_and_pool(frames)
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self._feature_names: Optional[List[str]] = None  # cached after first call

    def extract(self, frames: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract per-frame + clip-level features from a sequence.

        Note: This returns a (T, n_per_frame) matrix for the per-frame part.
        For ML training, use extract_and_pool() which returns a flat vector.

        Args:
            frames: (T, N_joints, 2) joint position array

        Returns:
            per_frame_features: (T, n_per_frame_features) array
            names: list of n_per_frame_features feature name strings
        """
        cfg = self.config
        static_groups = cfg.get_static_groups()
        temporal_pf_groups = cfg.get_temporal_per_frame_groups()

        T = frames.shape[0]
        all_feats: List[np.ndarray] = []
        all_names: List[str] = []

        # --- Static features (per frame) ---
        if static_groups:
            static_list: List[np.ndarray] = []
            static_names: Optional[List[str]] = None

            for t in range(T):
                f, n = frame_all_static_features(frames[t], static_groups)
                static_list.append(f)
                if static_names is None:
                    static_names = n

            static_matrix = np.vstack(static_list)  # (T, n_static)
            all_feats.append(static_matrix)
            all_names.extend(static_names or [])

        # --- Per-frame temporal features ---
        if temporal_pf_groups:
            temp_pf, temp_names = sequence_all_temporal_per_frame(
                frames, temporal_pf_groups, cfg.temporal_window
            )
            if temp_pf.shape[1] > 0:
                all_feats.append(temp_pf)
                all_names.extend(temp_names)

        if not all_feats:
            return np.zeros((T, 0), dtype=np.float32), []

        per_frame = np.concatenate(all_feats, axis=1).astype(np.float32)
        return per_frame, all_names

    def extract_and_pool(
        self,
        frames: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features and aggregate to a single vector per clip.

        Temporal aggregation: [mean, std, Q1, Q3] for per-frame features
        + clip-level features appended.

        Args:
            frames: (T, N_joints, 2)

        Returns:
            features: (n_features,) flat vector for ML training
            names: list of n_features strings
        """
        cfg = self.config

        # --- Per-frame features → aggregate ---
        per_frame, pf_names = self.extract(frames)
        aggregated_feats: List[np.ndarray] = []
        aggregated_names: List[str] = []

        if per_frame.shape[1] > 0:
            stats = cfg.temporal_agg_stats
            if "mean" in stats:
                m = per_frame.mean(axis=0)
                aggregated_feats.append(m)
                aggregated_names.extend([f"{n}_mean" for n in pf_names])
            if "std" in stats:
                s = per_frame.std(axis=0)
                aggregated_feats.append(s)
                aggregated_names.extend([f"{n}_std" for n in pf_names])
            if "q25" in stats:
                q25 = np.percentile(per_frame, 25, axis=0).astype(np.float32)
                aggregated_feats.append(q25)
                aggregated_names.extend([f"{n}_q25" for n in pf_names])
            if "q75" in stats:
                q75 = np.percentile(per_frame, 75, axis=0).astype(np.float32)
                aggregated_feats.append(q75)
                aggregated_names.extend([f"{n}_q75" for n in pf_names])

        # --- Clip-level sequence features ---
        clip_groups = cfg.get_temporal_clip_groups()
        if clip_groups:
            clip_feats, clip_names = sequence_all_clip_level(frames, clip_groups)
            if len(clip_feats) > 0:
                aggregated_feats.append(clip_feats)
                aggregated_names.extend(clip_names)

        if not aggregated_feats:
            return np.zeros(0, dtype=np.float32), []

        result = np.concatenate(aggregated_feats).astype(np.float32)
        self._feature_names = aggregated_names
        return result, aggregated_names

    def get_feature_names(self) -> List[str]:
        """Return feature names from the last extract_and_pool() call.

        Raises RuntimeError if extract_and_pool() hasn't been called yet.
        """
        if self._feature_names is None:
            # Run on dummy data to get names
            dummy = np.zeros((3, 15, 2), dtype=np.float32)
            dummy[0, 3] = [0.3, 0.3]  # r_shoulder
            dummy[0, 4] = [0.7, 0.3]  # l_shoulder
            dummy[0, 5] = [0.35, 0.6]  # r_hip
            dummy[0, 6] = [0.65, 0.6]  # l_hip
            for t in range(1, 3):
                dummy[t] = dummy[0] + np.random.randn(15, 2).astype(np.float32) * 0.01
            _, names = self.extract_and_pool(dummy)
            self._feature_names = names
        return list(self._feature_names)

    def extract_from_sample(
        self,
        sample,  # JHMDBSample
        cache_dir: Optional[Path] = None,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from a JHMDBSample, with optional caching.

        Args:
            sample: JHMDBSample instance
            cache_dir: directory to cache .npz files
            force_recompute: ignore cache and recompute

        Returns:
            features: (n_features,) flat vector
            names: list of feature name strings
        """
        from psrn.data.jhmdb_loader import load_mat_joints

        # Check cache
        if cache_dir is not None and not force_recompute:
            cache_path = Path(cache_dir) / f"{sample.video_name}.npz"
            if cache_path.exists():
                return load_features_npz(str(cache_path))

        # Load joints and extract
        joints = load_mat_joints(sample.mat_path)   # (T, 15, 2)
        features, names = self.extract_and_pool(joints)

        # Save to cache
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_path = Path(cache_dir) / f"{sample.video_name}.npz"
            save_features_npz(str(cache_path), features, names)

        return features, names

    def extract_batch(
        self,
        samples,   # List[JHMDBSample]
        cache_dir: Optional[Path] = None,
        augmenter=None,
        n_augment_copies: int = 0,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features from a batch of JHMDB samples.

        Args:
            samples: list of JHMDBSample instances
            cache_dir: optional cache directory
            augmenter: optional KeypointAugmenter (applied to joints before extraction)
            n_augment_copies: number of ADDITIONAL augmented copies per sample.
                              0 = original only (or one augmented pass if augmenter set).
                              3 = original + 3 augmented → 4x dataset size.
            show_progress: show tqdm progress bar

        Returns:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) label array
            names: list of feature name strings
        """
        from psrn.data.jhmdb_loader import load_mat_joints

        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="Extracting features") if show_progress else samples
        except ImportError:
            iterator = samples

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        names: Optional[List[str]] = None

        for sample in iterator:
            try:
                joints_orig = load_mat_joints(sample.mat_path)

                # ── Original sample (cached when no augmentation) ──────
                use_cache = cache_dir is not None and augmenter is None and n_augment_copies == 0
                if use_cache:
                    cache_path = Path(cache_dir) / f"{sample.video_name}.npz"
                    if cache_path.exists():
                        feats, feat_names = load_features_npz(str(cache_path))
                        X_list.append(feats)
                        y_list.append(sample.label_idx)
                        if names is None:
                            names = feat_names
                        continue

                # Compute original (or single-augmented) feature
                joints = augmenter(joints_orig) if (augmenter is not None and n_augment_copies == 0) else joints_orig
                feats, feat_names = self.extract_and_pool(joints)
                X_list.append(feats)
                y_list.append(sample.label_idx)
                if names is None:
                    names = feat_names

                if use_cache:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    save_features_npz(str(Path(cache_dir) / f"{sample.video_name}.npz"), feats, feat_names)

                # ── Additional augmented copies ─────────────────────────
                if augmenter is not None and n_augment_copies > 0:
                    for _ in range(n_augment_copies):
                        aug_joints = augmenter(joints_orig)
                        aug_feats, _ = self.extract_and_pool(aug_joints)
                        X_list.append(aug_feats)
                        y_list.append(sample.label_idx)

            except Exception as e:
                print(f"Warning: Failed to process {sample.video_name}: {e}")
                continue

        if not X_list:
            raise ValueError("No features could be extracted from the provided samples")

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y, names or []
