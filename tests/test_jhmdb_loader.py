"""Tests for JHMDB loader (mocked — no real data needed)."""
import numpy as np
import pytest
from unittest.mock import patch
from psrn.data.jhmdb_loader import load_mat_joints, JHMDB_CLASSES, JHMDB_KP_NAMES


def test_jhmdb_classes_count():
    assert len(JHMDB_CLASSES) == 21


def test_jhmdb_kp_names_count():
    assert len(JHMDB_KP_NAMES) == 15


def test_load_mat_joints_shape():
    """Test mat loading with mocked scipy.io."""
    mock_mat = {"pos_img": np.random.rand(2, 10, 15)}
    with patch("scipy.io.loadmat", return_value=mock_mat):
        frames = load_mat_joints("fake_path.mat")
    assert frames.shape == (10, 15, 2), f"Expected (10,15,2), got {frames.shape}"


def test_classes_sorted():
    assert JHMDB_CLASSES == sorted(JHMDB_CLASSES)
