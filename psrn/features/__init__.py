"""HierPose hierarchical feature extraction package.

Public API:
    HierarchicalFeatureExtractor  — main class for extracting features
    FeatureConfig                 — configuration dataclass
    FEATURE_GROUPS                — registry of all named feature groups
    get_feature_group_names       — list all available group names
"""

from psrn.features.extractor import FeatureConfig, HierarchicalFeatureExtractor
from psrn.features.registry import FEATURE_GROUPS, get_feature_group_names

__all__ = [
    "HierarchicalFeatureExtractor",
    "FeatureConfig",
    "FEATURE_GROUPS",
    "get_feature_group_names",
]
