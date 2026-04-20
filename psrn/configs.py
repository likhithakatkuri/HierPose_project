"""Configuration dataclasses for the HierPose ML pipeline.

Covers:
- HierPoseConfig   — JHMDB data loading, feature extraction, model training
- DomainConfig     — per-application domain (medical, sports, ergonomics)
- AblationConfig   — feature group ablation study settings
- ExperimentConfig — top-level YAML-loadable config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HierPoseConfig:
    """Configuration for the HierPose interpretable ML pipeline.

    Governs: JHMDB data loading, feature extraction, model training,
    cross-validation, and output artifacts.

    Dataset structure expected at data_root:
        data/JHMDB/
        ├── joint_positions/   <class>/<video>/joint_positions.mat
        └── splits/            <class>_test_split<N>.txt
    """

    # ── Data ────────────────────────────────────────────────────
    data_root: Path = Path("data/JHMDB")
    split_num: int = 1                              # Official JHMDB split: 1, 2, or 3
    use_all_splits: bool = False                    # Combine train from splits 1+2+3
    feature_cache_dir: Path = Path("cache/features")

    # ── Feature Engineering ─────────────────────────────────────
    enabled_feature_groups: Optional[List[str]] = None  # None = all 16 groups
    normalize_by_torso: bool = True
    temporal_window: int = 5
    augment_train: bool = True
    aug_flip_prob: float = 0.5
    aug_rotate_prob: float = 0.5
    aug_rotate_max_deg: float = 15.0
    aug_scale_prob: float = 0.3
    aug_noise_prob: float = 0.5

    # ── Preprocessing ────────────────────────────────────────────
    use_pca: bool = False   # Keep False — PCA destroys SHAP interpretability

    # ── Model ────────────────────────────────────────────────────
    model_type: str = "ensemble"   # lgbm | xgb | rf | svm | lda | ensemble
    n_jobs: int = 4

    # ── Cross-Validation ─────────────────────────────────────────
    n_cv_folds: int = 5
    cv_random_state: int = 42

    # ── Hyperparameter Tuning ────────────────────────────────────
    tune_hyperparams: bool = True
    n_optuna_trials: int = 50

    # ── Output ───────────────────────────────────────────────────
    output_dir: Path = Path("outputs/experiments")
    experiment_name: str = "hierpose_jhmdb_split1"
    save_shap: bool = True
    save_confusion_matrix: bool = True
    save_tsne: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.feature_cache_dir = Path(self.feature_cache_dir)
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HierPoseConfig":
        """Create from a flat or nested dict (e.g., loaded from YAML)."""
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        valid = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in flat.items() if k in valid}
        return cls(**filtered)


@dataclass
class DomainConfig:
    """Per-application domain configuration (medical / sports / ergonomics / action)."""

    domain_name: str = "action"
    pose_classes: List[str] = field(default_factory=list)
    reference_poses: Dict[str, Any] = field(default_factory=dict)
    feedback_templates: Dict[str, str] = field(default_factory=dict)
    severity_thresholds: Dict[str, float] = field(default_factory=dict)
    display_name: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.domain_name.replace("_", " ").title()


@dataclass
class AblationConfig:
    """Configuration for feature group ablation studies (IEEE Table 3)."""

    ablation_mode: str = "leave_one_out"   # "leave_one_out" | "incremental"
    groups_to_test: Optional[List[str]] = None
    n_cv_folds: int = 5
    random_seed: int = 42
    output_dir: Path = Path("outputs/ablation")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass
class ExperimentConfig:
    """Top-level config — loadable from YAML via ExperimentConfig.from_yaml(path).

    YAML structure (configs/default.yaml):
        hierpose:
          data_root: data/JHMDB
          split_num: 1
          model_type: ensemble
          ...
        ablation:
          ablation_mode: leave_one_out
          ...
    """

    description: str = ""
    paper_table_id: str = ""
    hierpose: HierPoseConfig = field(default_factory=HierPoseConfig)
    ablation: Optional[AblationConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentConfig":
        import yaml
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        hp_raw = raw.get("hierpose", {})
        hierpose = HierPoseConfig.from_dict(hp_raw)

        ablation = None
        if "ablation" in raw:
            ab_raw = raw["ablation"]
            valid = {f for f in AblationConfig.__dataclass_fields__}
            filtered = {k: v for k, v in ab_raw.items() if k in valid}
            ablation = AblationConfig(**filtered)

        return cls(
            description=raw.get("description", ""),
            paper_table_id=raw.get("paper_table_id", ""),
            hierpose=hierpose,
            ablation=ablation,
        )
