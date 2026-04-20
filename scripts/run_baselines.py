#!/usr/bin/env python
"""Run all baseline models and generate comparison table."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/baselines")
    args = parser.parse_args()

    from psrn.configs import HierPoseConfig
    from psrn.training.trainer import HierPoseTrainer
    from psrn.evaluation.metrics import compare_models
    from psrn.evaluation.reporting import LaTeXReporter, generate_latex_tables
    from psrn.visualization.plots import plot_model_comparison
    from psrn.utils.reproducibility import set_seed

    set_seed(42)

    models = ["lgbm", "xgb", "rf", "svm", "lda", "ensemble"]
    results = []

    for model_type in models:
        print(f"\n[baselines] Training {model_type}...")
        cfg = HierPoseConfig(
            data_root=args.data_root,
            split_num=args.split,
            model_type=model_type,
            tune_hyperparams=False,
            output_dir=f"{args.output_dir}/{model_type}",
        )
        trainer = HierPoseTrainer(cfg)
        result = trainer.run()
        results.append(result)
        print(f"  {model_type}: acc={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}")

    df = compare_models(results)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "model_comparison.csv", index=False)
    print(f"\n{df.to_string()}")

    plot_model_comparison(
        {
            r.model_name: {
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
            }
            for r in results
        },
        save_path=str(out / "model_comparison.png"),
    )

    reporter = LaTeXReporter()
    tex = reporter.model_comparison_table(results)
    (out / "model_comparison.tex").write_text(tex)
    print(f"Results saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
