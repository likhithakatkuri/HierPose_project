#!/usr/bin/env python
"""Run feature group ablation study."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/ablation")
    parser.add_argument(
        "--mode",
        default="leave_one_out",
        choices=["leave_one_out", "incremental"],
    )
    parser.add_argument("--model_type", default="lgbm")
    args = parser.parse_args()

    from psrn.configs import HierPoseConfig, AblationConfig
    from psrn.training.ablation import AblationStudy
    from psrn.visualization.plots import plot_ablation_curve
    from psrn.utils.reproducibility import set_seed

    set_seed(42)

    cfg = HierPoseConfig(
        data_root=args.data_root,
        split_num=args.split,
        model_type=args.model_type,
    )
    ablation_cfg = AblationConfig(ablation_mode=args.mode, output_dir=args.output_dir)

    study = AblationStudy(cfg, ablation_cfg)

    if args.mode == "leave_one_out":
        df = study.run_leave_one_out()
    else:
        df = study.run_incremental()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_results.csv", index=False)

    plot_ablation_curve(df, save_path=str(out / "ablation_curve.png"))
    print(f"Ablation results saved to {out}")
    print(df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
