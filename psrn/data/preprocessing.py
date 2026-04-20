"""Pose and image preprocessing utilities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms


def load_pose_from_json(json_path: Path) -> dict[str, list]:
    """Load pose keypoints from JSON file.
    
    Expected format: {"people": [{"keypoints": [[x, y, conf], ...]}, ...]}
    
    Args:
        json_path: path to JSON file with pose annotations
    
    Returns:
        dict with 'people' key containing list of person keypoints
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_keypoints_to_parts(
    keypoints: list[list[float]], num_keypoints: int = 14
) -> dict[str, np.ndarray]:
    """Extract body parts from keypoints.
    
    From Sec. 3.2, Fig. 3b: Split 14 joints into 5 parts:
    - head (8-D): top-of-head, neck, and 6 face points
    - left arm (6-D): left shoulder, elbow, wrist
    - right arm (6-D): right shoulder, elbow, wrist
    - left leg (6-D): left hip, knee, ankle
    - right leg (6-D): right hip, knee, ankle
    
    Args:
        keypoints: list of [x, y, conf] for each keypoint
        num_keypoints: expected number of keypoints (14)
    
    Returns:
        dict with keys ['head', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        each containing (part_dim, 2) array of [x, y] coordinates
    """
    # Ensure we have the right number of keypoints
    while len(keypoints) < num_keypoints:
        keypoints.append([0.0, 0.0, 0.0])

    # Extract [x, y] coordinates (ignore confidence)
    coords = np.array([[k[0], k[1]] for k in keypoints[:num_keypoints]])

    # Map keypoints to parts (assuming COCO-18 style minus eyes/ears/nose)
    # Indices: 0=top_head, 1=neck, 2=L_shoulder, 3=L_elbow, 4=L_wrist,
    #          5=R_shoulder, 6=R_elbow, 7=R_wrist,
    #          8=L_hip, 9=L_knee, 10=L_ankle,
    #          11=R_hip, 12=R_knee, 13=R_ankle
    parts = {
        "head": coords[[0, 1]],  # top_head, neck (2 points, but we need 8)
        # Pad head to 8-D by repeating/reusing points
        "left_arm": coords[[2, 3, 4]],  # L_shoulder, L_elbow, L_wrist
        "right_arm": coords[[5, 6, 7]],  # R_shoulder, R_elbow, R_wrist
        "left_leg": coords[[8, 9, 10]],  # L_hip, L_knee, L_ankle
        "right_leg": coords[[11, 12, 13]],  # R_hip, R_knee, R_ankle
    }

    # Pad head to 8-D (repeat neck point to fill)
    head_coords = coords[[0, 1]]
    if head_coords.shape[0] < 4:
        # Repeat to get 4 points, then flatten
        padded_head = np.tile(head_coords, (4, 1))[:4]
        parts["head"] = padded_head

    # Flatten each part to (part_dim, 2) then to (part_dim * 2,) for MLP input
    # Actually, we need (part_dim, 2) format
    return parts


def normalize_poses(
    poses: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Normalize pose coordinates to [0, 1].
    
    From Sec. 3.2: Divide keypoints by frame width/height.
    
    Args:
        poses: (T, N, 14, 2) array of pose coordinates
        width: frame width
        height: frame height
    
    Returns:
        Normalized poses in [0, 1] range
    """
    normalized = poses.copy()
    normalized[:, :, :, 0] /= width
    normalized[:, :, :, 1] /= height
    return np.clip(normalized, 0.0, 1.0)


def pad_poses(
    poses_list: list[np.ndarray], fill_value: float = 0.0
) -> np.ndarray:
    """Pad poses to have same number of persons.
    
    From Sec. 3.2, Fig. 3a: Find N = max persons in any frame;
    for frames with < N, append virtual persons (0, 0).
    
    Args:
        poses_list: list of (N_i, 14, 2) arrays per frame
        fill_value: value to use for padding (0.0 for virtual persons)
    
    Returns:
        (T, N_max, 14, 2) padded pose array
    """
    max_persons = max(p.shape[0] for p in poses_list)
    T = len(poses_list)
    num_keypoints = poses_list[0].shape[1]
    num_dims = poses_list[0].shape[2]

    padded = np.full(
        (T, max_persons, num_keypoints, num_dims), fill_value, dtype=np.float32
    )

    for t, pose_frame in enumerate(poses_list):
        N = pose_frame.shape[0]
        padded[t, :N] = pose_frame

    return padded


def preprocess_image(image_path: Path | str, size: int = 224) -> Tensor:
    """Preprocess image for VGG16 object stream.
    
    Args:
        image_path: path to image file
        size: target image size (224 for VGG16)
    
    Returns:
        (3, 224, 224) normalized tensor
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image)

