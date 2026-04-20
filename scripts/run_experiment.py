#!/usr/bin/env python
"""Run full HierPose experiment: train, evaluate, SHAP, LaTeX."""
import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--output_dir", default="outputs/experiment")
    parser.add_argument("--model_type", default="ensemble")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no_tune", action="store_true")
    parser.add_argument("--no_shap", action="store_true")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    args = parser.parse_args()

    from psrn.configs import HierPoseConfig
    from psrn.training.trainer import HierPoseTrainer
    from psrn.utils.reproducibility import set_seed

    set_seed(42)

    cfg = HierPoseConfig(
        data_root=args.data_root,
        split_num=args.split,
        output_dir=args.output_dir,
        model_type=args.model_type,
        tune_hyperparams=not args.no_tune,
    )

    print(
        f"[experiment] data_root={args.data_root}, split={args.split}, "
        f"model={args.model_type}"
    )
    t0 = time.time()

    trainer = HierPoseTrainer(cfg)
    result = trainer.run()

    elapsed = time.time() - t0
    print(f"\n=== Experiment Complete ({elapsed:.1f}s) ===")
    print(f"  Accuracy:    {result.accuracy:.4f}")
    print(f"  Macro F1:    {result.macro_f1:.4f}")
    print(f"  Weighted F1: {result.weighted_f1:.4f}")
    print(f"  Output:      {cfg.output_dir}")

    if args.latex:
        from psrn.evaluation.reporting import generate_latex_tables

        latex_dir = Path(cfg.output_dir) / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)
        generate_latex_tables([result], None, str(latex_dir))
        print(f"  LaTeX:       {latex_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
