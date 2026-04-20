"""JHMDB dataset adapter — wraps JHMDBSplitLoader."""
from psrn.data.jhmdb_loader import JHMDBSplitLoader, JHMDBSample, load_mat_joints, JHMDB_CLASSES
import numpy as np
from typing import List
from pathlib import Path


class JHMDBDataset:
    def __init__(self, data_root: str, split: int = 1):
        self.data_root = Path(data_root)
        self.split = split
        self.loader = JHMDBSplitLoader(data_root, split)

    def get_split(self, subset: str = "train") -> List[JHMDBSample]:
        if subset == "train":
            return self.loader.get_train_samples()
        elif subset == "test":
            return self.loader.get_test_samples()
        else:
            raise ValueError(f"subset must be 'train' or 'test', got {subset!r}")

    def load_frames(self, sample: JHMDBSample) -> np.ndarray:
        return load_mat_joints(sample.mat_path)

    @property
    def class_names(self) -> List[str]:
        return JHMDB_CLASSES

    @property
    def n_classes(self) -> int:
        return len(JHMDB_CLASSES)
