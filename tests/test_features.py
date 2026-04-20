"""Tests for feature extraction pipeline."""
import numpy as np
import pytest
from psrn.features.extractor import HierarchicalFeatureExtractor, FeatureConfig
from psrn.features.static import frame_all_static_features, frame_angle_features


def make_random_frames(T: int = 20, J: int = 15) -> np.ndarray:
    """Create plausible random skeleton frames (T, J, 2) in [0, 1]."""
    return np.random.rand(T, J, 2)


def test_angle_features_shape():
    frames = make_random_frames()
    feats = frame_angle_features(frames[0])
    assert feats.ndim == 1
    assert len(feats) == 28  # 14 triplets × 2


def test_static_features_shape():
    frames = make_random_frames()
    feats = frame_all_static_features(frames[0])
    assert feats.ndim == 1
    assert len(feats) > 0


def test_extractor_single():
    extractor = HierarchicalFeatureExtractor(FeatureConfig())
    frames = make_random_frames()
    vec = extractor.extract_and_pool(frames)
    assert vec.ndim == 1
    assert len(vec) > 0
    assert not np.any(np.isnan(vec))


def test_extractor_batch_shapes():
    extractor = HierarchicalFeatureExtractor(FeatureConfig())
    results = [extractor.extract_and_pool(make_random_frames()) for _ in range(5)]
    assert all(r.shape == results[0].shape for r in results)


def test_no_pca_by_default():
    cfg = FeatureConfig()
    assert not getattr(cfg, "use_pca", False), "PCA must be disabled by default"
