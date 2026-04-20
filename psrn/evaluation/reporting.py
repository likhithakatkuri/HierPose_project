"""LaTeX table generation for IEEE-format papers.

LaTeXReporter generates publication-ready \begin{table}...\end{table}
environments directly from ExperimentResult objects and ablation DataFrames.

generate_latex_tables() is the convenience function that writes all
standard tables to an output directory in one call.

Example::

    from psrn.evaluation.reporting import LaTeXReporter, generate_latex_tables

    reporter = LaTeXReporter()
    tex = reporter.model_comparison_table(results_list)
    print(tex)

    generate_latex_tables(results_list, ablation_df, output_dir="paper/tables")
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sklearn.metrics import precision_recall_fscore_support
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for char, replacement in replacements:
        text = text.replace(char, replacement)
    return text


def _bold(val: float, best: float, threshold: float = 1e-4) -> str:
    """Return LaTeX-bold formatted value if it equals the best."""
    fmt = f"{val:.4f}"
    if abs(val - best) <= threshold:
        return rf"\textbf{{{fmt}}}"
    return fmt


# ─────────────────────────────────────────────────────────────
# LaTeXReporter
# ─────────────────────────────────────────────────────────────

class LaTeXReporter:
    """Generate IEEE-format LaTeX tables from HierPose experiment results.

    All tables follow IEEE Transactions style:
        - \\begin{table}[htbp]
        - \\caption{} placed above the tabular
        - \\label{tab:...}
        - \\toprule / \\midrule / \\bottomrule (booktabs)
        - Best results are \\textbf{bold}

    Methods:
        model_comparison_table  — multi-model accuracy/F1 comparison
        ablation_table          — feature group contribution table
        per_class_table         — per-class precision / recall / F1
    """

    def __init__(self, label_prefix: str = "tab") -> None:
        """
        Args:
            label_prefix: prefix for \\label{} identifiers (default "tab").
        """
        self.label_prefix = label_prefix

    # ------------------------------------------------------------------
    # 1. Model comparison table
    # ------------------------------------------------------------------

    def model_comparison_table(
        self,
        results_list: List["ExperimentResult"],
        caption: str = "Comparison of classification models on the JHMDB dataset.",
        label: str = "model_comparison",
        include_cv: bool = True,
        include_time: bool = True,
    ) -> str:
        """IEEE-format table comparing models on accuracy, macro F1, weighted F1.

        Args:
            results_list: list of ExperimentResult objects.
            caption:      LaTeX \\caption{} text.
            label:        \\label{} suffix (prefixed with label_prefix).
            include_cv:   include CV mean ± std columns.
            include_time: include training time column.

        Returns:
            Multi-line LaTeX string for the complete table.
        """
        # Sort by accuracy descending
        sorted_results = sorted(results_list, key=lambda r: -r.accuracy)

        best_acc  = max(r.accuracy    for r in sorted_results)
        best_mf1  = max(r.macro_f1   for r in sorted_results)
        best_wf1  = max(r.weighted_f1 for r in sorted_results)
        best_cv   = max(r.cv_mean     for r in sorted_results)

        # Build column spec
        cols = ["l", "c", "c", "c"]
        headers = ["Model", "Accuracy", "Macro F1", "Weighted F1"]

        if include_cv:
            cols.append("c")
            headers.append("CV Mean ± Std")

        if include_time:
            cols.append("r")
            headers.append("Train Time (s)")

        col_spec = " ".join(cols)

        lines: List[str] = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{_escape_latex(caption)}}}",
            rf"\label{{{self.label_prefix}:{label}}}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            " & ".join(rf"\textbf{{{h}}}" for h in headers) + r" \\",
            r"\midrule",
        ]

        for r in sorted_results:
            name = _escape_latex(r.model_name)
            row = [
                name,
                _bold(r.accuracy,    best_acc),
                _bold(r.macro_f1,    best_mf1),
                _bold(r.weighted_f1, best_wf1),
            ]

            if include_cv:
                cv_str = f"{r.cv_mean:.4f} \\pm {r.cv_std:.4f}"
                if abs(r.cv_mean - best_cv) <= 1e-4:
                    cv_str = rf"\textbf{{{r.cv_mean:.4f}}} $\pm$ {r.cv_std:.4f}"
                else:
                    cv_str = f"{r.cv_mean:.4f} $\\pm$ {r.cv_std:.4f}"
                row.append(cv_str)

            if include_time:
                row.append(f"{r.training_time:.1f}")

            lines.append(" & ".join(row) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 2. Ablation table
    # ------------------------------------------------------------------

    def ablation_table(
        self,
        df_ablation: "pd.DataFrame",
        metric_col: str = "cv_accuracy",
        group_col: str = "feature_group",
        caption: str = "Feature group ablation study showing contribution to classification accuracy.",
        label: str = "ablation",
    ) -> str:
        """IEEE-format feature group contribution table.

        Args:
            df_ablation: DataFrame with at minimum columns:
                             - group_col  : feature group label
                             - metric_col : metric value per group
                         Optional columns included if present:
                             - n_features : number of features in group
                             - delta      : accuracy delta vs baseline
            metric_col:  column name for the main metric.
            group_col:   column name for group labels.
            caption:     \\caption{} text.
            label:       \\label{} suffix.

        Returns:
            Multi-line LaTeX string.
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for ablation_table.")

        has_n_features = "n_features" in df_ablation.columns
        has_delta      = "delta"      in df_ablation.columns

        # Column spec
        cols    = ["l", "c"]
        headers = [
            group_col.replace("_", " ").title(),
            metric_col.replace("_", " ").title(),
        ]
        if has_n_features:
            cols.append("r")
            headers.append("\\# Features")
        if has_delta:
            cols.append("c")
            headers.append("$\\Delta$ Accuracy")

        col_spec  = " ".join(cols)
        best_val  = float(df_ablation[metric_col].max())

        lines: List[str] = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{_escape_latex(caption)}}}",
            rf"\label{{{self.label_prefix}:{label}}}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            " & ".join(rf"\textbf{{{h}}}" for h in headers) + r" \\",
            r"\midrule",
        ]

        for _, row_data in df_ablation.iterrows():
            group = _escape_latex(str(row_data[group_col]))
            val   = float(row_data[metric_col])
            val_str = _bold(val, best_val)
            row = [group, val_str]

            if has_n_features:
                row.append(str(int(row_data["n_features"])))
            if has_delta:
                delta = float(row_data["delta"])
                sign  = "+" if delta >= 0 else ""
                row.append(f"{sign}{delta:.4f}")

            lines.append(" & ".join(row) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 3. Per-class table
    # ------------------------------------------------------------------

    def per_class_table(
        self,
        result: "ExperimentResult",
        y_true: Optional["np.ndarray"] = None,
        y_pred: Optional["np.ndarray"] = None,
        caption: Optional[str] = None,
        label: str = "per_class",
    ) -> str:
        """Per-class precision / recall / F1 / support table.

        If y_true and y_pred are supplied, recomputes fresh metrics.
        Otherwise uses per_class_accuracy from the ExperimentResult.

        Args:
            result:  ExperimentResult object.
            y_true:  (N,) ground-truth labels (optional, enables P/R/F1).
            y_pred:  (N,) predicted labels (optional).
            caption: \\caption{} text (auto-generated if None).
            label:   \\label{} suffix.

        Returns:
            Multi-line LaTeX string.
        """
        import numpy as np

        model_esc = _escape_latex(result.model_name)
        if caption is None:
            caption = (
                f"Per-class classification performance for {result.model_name} "
                f"on JHMDB (accuracy = {result.accuracy:.4f})."
            )

        class_names = sorted(result.per_class_accuracy.keys())

        has_prf = (
            y_true is not None
            and y_pred is not None
            and HAS_SKLEARN
        )

        prec_dict: dict = {}
        rec_dict:  dict = {}
        f1_dict:   dict = {}
        sup_dict:  dict = {}

        if has_prf:
            prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
                y_true, y_pred, labels=class_names, zero_division=0
            )
            for i, cls in enumerate(class_names):
                prec_dict[cls] = float(prec_arr[i])
                rec_dict[cls]  = float(rec_arr[i])
                f1_dict[cls]   = float(f1_arr[i])
                sup_dict[cls]  = int(sup_arr[i])

        # Column spec
        if has_prf:
            col_spec = "l c c c c r"
            headers  = ["Class", "Precision", "Recall", "F1 Score", "Accuracy", "Support"]
        else:
            col_spec = "l c"
            headers  = ["Class", "Accuracy"]

        lines: List[str] = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{_escape_latex(caption)}}}",
            rf"\label{{{self.label_prefix}:{label}}}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            " & ".join(rf"\textbf{{{h}}}" for h in headers) + r" \\",
            r"\midrule",
        ]

        best_acc  = max(result.per_class_accuracy.values()) if result.per_class_accuracy else 0.0
        best_f1   = max(f1_dict.values())   if f1_dict  else 0.0
        best_prec = max(prec_dict.values()) if prec_dict else 0.0
        best_rec  = max(rec_dict.values())  if rec_dict  else 0.0

        for cls in class_names:
            cls_esc = _escape_latex(cls)
            acc_val = result.per_class_accuracy.get(cls, 0.0)
            acc_str = _bold(acc_val, best_acc)

            if has_prf:
                row = [
                    cls_esc,
                    _bold(prec_dict.get(cls, 0.0), best_prec),
                    _bold(rec_dict.get(cls, 0.0),  best_rec),
                    _bold(f1_dict.get(cls, 0.0),   best_f1),
                    acc_str,
                    str(sup_dict.get(cls, 0)),
                ]
            else:
                row = [cls_esc, acc_str]

            lines.append(" & ".join(row) + r" \\")

        # Macro averages footer
        lines.append(r"\midrule")
        if has_prf:
            macro_p = float(np.mean(list(prec_dict.values()))) if prec_dict else 0.0
            macro_r = float(np.mean(list(rec_dict.values())))  if rec_dict  else 0.0
            macro_f = result.macro_f1
            total_s = sum(sup_dict.values())
            lines.append(
                rf"\textbf{{Macro Avg}} & {macro_p:.4f} & {macro_r:.4f} "
                rf"& \textbf{{{macro_f:.4f}}} & {result.accuracy:.4f} & {total_s} \\"
            )
        else:
            lines.append(
                rf"\textbf{{Overall Accuracy}} & {result.accuracy:.4f} \\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# generate_latex_tables — convenience wrapper
# ─────────────────────────────────────────────────────────────

def generate_latex_tables(
    results: List["ExperimentResult"],
    ablation_df: Optional["pd.DataFrame"],
    output_dir: str,
    reporter: Optional[LaTeXReporter] = None,
    y_true_best: Optional["np.ndarray"] = None,
    y_pred_best: Optional["np.ndarray"] = None,
) -> List[str]:
    """Write all standard LaTeX table files to output_dir.

    Files written:
        - model_comparison.tex   — multi-model accuracy comparison
        - ablation.tex           — feature group ablation (if ablation_df given)
        - per_class.tex          — per-class metrics for the best model

    Args:
        results:      list of ExperimentResult objects.
        ablation_df:  pandas DataFrame for ablation table (or None to skip).
        output_dir:   directory path to write .tex files.
        reporter:     LaTeXReporter instance (created with defaults if None).
        y_true_best:  ground-truth labels for the best model's per-class table.
        y_pred_best:  predictions for the best model's per-class table.

    Returns:
        List of absolute paths to the written .tex files.
    """
    if reporter is None:
        reporter = LaTeXReporter()

    out_path   = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # 1. Model comparison
    if results:
        tex  = reporter.model_comparison_table(results)
        path = out_path / "model_comparison.tex"
        path.write_text(tex, encoding="utf-8")
        written.append(str(path))

    # 2. Ablation table
    if ablation_df is not None and HAS_PANDAS:
        try:
            tex  = reporter.ablation_table(ablation_df)
            path = out_path / "ablation.tex"
            path.write_text(tex, encoding="utf-8")
            written.append(str(path))
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Ablation table generation failed: {exc}", UserWarning, stacklevel=2
            )

    # 3. Per-class table for best model
    if results:
        best_result = max(results, key=lambda r: r.accuracy)
        tex  = reporter.per_class_table(
            best_result,
            y_true=y_true_best,
            y_pred=y_pred_best,
        )
        path = out_path / "per_class.tex"
        path.write_text(tex, encoding="utf-8")
        written.append(str(path))

    return written
