"""Reproducibility helpers."""
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: integer seed value (default 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
