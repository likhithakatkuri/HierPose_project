"""HierPose evaluation package.

Provides structured experiment result tracking, metric computation,
model comparison utilities, and LaTeX table generation for IEEE papers.

Components:
    ExperimentResult   — dataclass holding all metrics for one model run
    EvaluationReport   — (alias for ExperimentResult, for API clarity)
    compute_metrics    — build ExperimentResult from y_true / y_pred arrays
    generate_latex_tables — write all .tex tables to an output directory

Example::

    from psrn.evaluation import compute_metrics, generate_latex_tables, ExperimentResult

    result = compute_metrics(y_true, y_pred, class_names=class_names)
    print(result.accuracy, result.macro_f1)

    generate_latex_tables([result], ablation_df, output_dir="paper/tables")
"""

from psrn.evaluation.metrics import (
    ExperimentResult,
    compute_metrics,
    compare_models,
)
from psrn.evaluation.reporting import (
    LaTeXReporter,
    generate_latex_tables,
)

# EvaluationReport is an alias kept for API compatibility
EvaluationReport = ExperimentResult

__all__ = [
    "ExperimentResult",
    "EvaluationReport",
    "compute_metrics",
    "compare_models",
    "LaTeXReporter",
    "generate_latex_tables",
]
