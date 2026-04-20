#!/usr/bin/env python
"""CLI entry point for training. Delegates to HierPoseTrainer."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="HierPose training pipeline")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--output_dir", default="outputs/experiment")
    parser.add_argument(
        "--model_type",
        default="ensemble",
        choices=["lgbm", "xgb", "rf", "svm", "lda", "ensemble"],
    )
    parser.add_argument("--no_tune", action="store_true")
    parser.add_argument("--all_splits", action="store_true",
                        help="Combine train data from all 3 splits (~1980 samples)")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    # Import here to avoid slow startup
    from psrn.configs import HierPoseConfig
    from psrn.training.trainer import HierPoseTrainer

    if args.config:
        from psrn.configs import ExperimentConfig
        cfg = ExperimentConfig.from_yaml(args.config).hierpose
        cfg.model_type = args.model_type  # CLI overrides yaml
    else:
        cfg = HierPoseConfig(
            data_root=args.data_root,
            split_num=args.split,
            output_dir=args.output_dir,
            model_type=args.model_type,
            tune_hyperparams=not args.no_tune,
            use_all_splits=args.all_splits,
        )

    trainer = HierPoseTrainer(cfg)
    result = trainer.run()
    print("\n=== Results ===")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    print(f"Output:   {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
