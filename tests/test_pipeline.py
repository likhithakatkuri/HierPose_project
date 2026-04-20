"""Integration test for training pipeline (uses mock data)."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_mock_samples(n: int = 30, T: int = 15, J: int = 15):
    from psrn.data.jhmdb_loader import JHMDBSample

    samples = []
    classes = ["walk", "run", "jump"]
    for i in range(n):
        cls = classes[i % 3]
        s = JHMDBSample(
            video_dir="",
            mat_path="fake.mat",
            class_name=cls,
            label_idx=i % 3,
            video_name=f"vid_{i:03d}",
            split=1,
        )
        samples.append(s)
    return samples


def test_feature_extraction_mock():
    from psrn.features.extractor import HierarchicalFeatureExtractor, FeatureConfig

    extractor = HierarchicalFeatureExtractor(FeatureConfig())
    frames = np.random.rand(15, 15, 2)
    vec = extractor.extract_and_pool(frames)
    assert vec.ndim == 1
    assert not np.any(np.isnan(vec))


def test_model_training_mock():
    """Test that model fits and predicts without errors."""
    from psrn.training.model_selector import ModelSelector

    X = np.random.rand(60, 50)
    y = np.array([i % 3 for i in range(60)])
    selector = ModelSelector(n_jobs=1, verbose=False)
    report = selector.fit_all(X, y)
    assert report is not None
    assert len(report.cv_scores) > 0
