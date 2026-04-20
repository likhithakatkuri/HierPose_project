"""HierPose training package.

Modules:
    trainer              — HierPoseTrainer: full ML experiment pipeline
    model_selector       — ModelSelector: train all models, build best ensemble
    cross_validation     — nested CV, McNemar statistical test
    hyperparameter_search — Optuna-based tuning for all models
    ablation             — AblationStudy: leave-one-out and incremental
"""

from psrn.training.trainer import HierPoseTrainer
from psrn.training.model_selector import ModelSelector

__all__ = ["HierPoseTrainer", "ModelSelector"]
