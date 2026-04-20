"""Publication-quality visualisation functions for HierPose.

All plots are designed for inclusion in IEEE/ACM papers:
    - 300 DPI output
    - Times New Roman–compatible fonts where available
    - Colour-blind–friendly palettes

Functions:
    plot_confusion_matrix  — normalised seaborn heatmap (16×14 inches)
    plot_tsne              — t-SNE 2D scatter coloured by class
    plot_ablation_curve    — dual-axis bar + line ablation plot
    plot_model_comparison  — grouped bar chart across models and metrics
    plot_shap_summary      — beeswarm-style SHAP feature importance plot
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# Matplotlib / Seaborn (required)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

# Optional heavy imports
try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─────────────────────────────────────────────────────────────
# Shared style helpers
# ─────────────────────────────────────────────────────────────

def _apply_base_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    matplotlib.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         100,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


def _save_or_show(
    fig: plt.Figure,
    save_path: Optional[str],
    dpi: int = 300,
) -> None:
    """Save figure if save_path is given, else call plt.show()."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
    dpi: int = 300,
    title: str = "Confusion Matrix (Normalised)",
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot a normalised confusion matrix as a seaborn heatmap.

    Args:
        cm:          (N, N) integer confusion matrix (raw counts).
        class_names: sequence of N class name strings.
        save_path:   path to save PNG/PDF (e.g. "figures/cm.png").
                     If None, plt.show() is called.
        dpi:         output resolution (default 300).
        title:       figure title.
        cmap:        seaborn colormap name.

    Returns:
        matplotlib Figure object.
    """
    _apply_base_style()

    # Normalise row-wise (per true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        where=(row_sums != 0),
        out=np.zeros_like(cm, dtype=float),
    )

    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="lightgrey",
        ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"label": "Proportion"},
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted Class", fontsize=13, labelpad=10)
    ax.set_ylabel("True Class",      fontsize=13, labelpad=10)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    # Overall accuracy in subtitle
    n_correct = int(np.trace(cm))
    n_total   = int(cm.sum())
    acc_str   = f"Overall accuracy: {n_correct}/{n_total} = {n_correct/max(n_total,1):.1%}"
    fig.text(0.5, 0.01, acc_str, ha="center", fontsize=11, style="italic")

    plt.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# ─────────────────────────────────────────────────────────────
# 2. t-SNE Feature Embedding
# ─────────────────────────────────────────────────────────────

def plot_tsne(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
    perplexity: float = 30.0,
    dpi: int = 300,
    title: str = "t-SNE Feature Embedding",
    random_state: int = 42,
    alpha: float = 0.7,
    marker_size: int = 18,
) -> plt.Figure:
    """2-D t-SNE scatter plot of feature vectors coloured by class.

    Args:
        X:            (N, n_features) feature matrix.
        y:            (N,) integer class labels (0-indexed).
        class_names:  class label strings.
        save_path:    output path or None (show interactively).
        perplexity:   t-SNE perplexity parameter.
        dpi:          output resolution.
        title:        figure title.
        random_state: reproducibility seed.
        alpha:        point transparency.
        marker_size:  scatter point size.

    Returns:
        matplotlib Figure.

    Raises:
        ImportError: if scikit-learn is not installed.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for t-SNE: pip install scikit-learn")

    _apply_base_style()

    # Reduce dimensionality
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_2d = tsne.fit_transform(X)

    n_classes  = len(class_names)
    palette    = sns.color_palette("tab20", n_colors=max(n_classes, 2))
    fig, ax    = plt.subplots(figsize=(14, 10))

    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=cls_name,
            color=palette[cls_idx % len(palette)],
            alpha=alpha,
            s=marker_size,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.18, 1.0),
        framealpha=0.9,
        fontsize=9,
        markerscale=1.5,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.text(
        0.5, 0.005,
        f"N={len(y)} samples, perplexity={perplexity}",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# ─────────────────────────────────────────────────────────────
# 3. Ablation Study Curve
# ─────────────────────────────────────────────────────────────

def plot_ablation_curve(
    df_ablation: "pd.DataFrame",
    metric: str = "cv_accuracy",
    save_path: Optional[str] = None,
    dpi: int = 300,
    title: str = "Feature Group Ablation Study",
    bar_color: str = "#4472C4",
    line_color: str = "#FF4444",
) -> plt.Figure:
    """Dual-axis bar + cumulative-line ablation plot.

    Args:
        df_ablation: DataFrame with columns including:
                         - "feature_group": group label
                         - metric: accuracy / F1 value per ablation step
                         - (optional) "n_features": feature count
        metric:      column name to plot as bar heights.
        save_path:   output path or None.
        dpi:         output resolution.
        title:       figure title.
        bar_color:   matplotlib colour for bars.
        line_color:  matplotlib colour for cumulative-line overlay.

    Returns:
        matplotlib Figure.

    Raises:
        ImportError: if pandas is not available.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for plot_ablation_curve: pip install pandas")

    if metric not in df_ablation.columns:
        raise ValueError(
            f"Column '{metric}' not found in df_ablation. "
            f"Available: {list(df_ablation.columns)}"
        )

    _apply_base_style()

    labels   = list(df_ablation["feature_group"].astype(str))
    values   = list(df_ablation[metric].astype(float))
    x_pos    = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar chart
    bars = ax1.bar(
        x_pos,
        values,
        color=bar_color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    # Value labels on bars
    for bar_obj, val in zip(bars, values):
        ax1.text(
            bar_obj.get_x() + bar_obj.get_width() / 2.0,
            bar_obj.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xlabel("Feature Group", fontsize=12)
    ax1.set_ylabel(metric.replace("_", " ").title(), fontsize=12, color=bar_color)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax1.set_ylim(0, min(1.05, max(values) * 1.15))
    ax1.yaxis.label.set_color(bar_color)
    ax1.tick_params(axis="y", colors=bar_color)

    # Secondary axis: cumulative / line overlay
    ax2 = ax1.twinx()
    ax2.plot(
        x_pos,
        values,
        color=line_color,
        marker="o",
        linewidth=2.0,
        markersize=6,
        label=f"Trend: {metric}",
        zorder=3,
    )

    # Optional: feature count on secondary axis if available
    if "n_features" in df_ablation.columns:
        n_feats = list(df_ablation["n_features"].astype(int))
        ax2.set_ylabel("# Features / Trend", fontsize=12, color=line_color)
        ax2_b = ax1.twinx()
        ax2_b.spines["right"].set_position(("outward", 50))
        ax2_b.plot(
            x_pos, n_feats,
            color="#888888",
            linestyle="--",
            linewidth=1.5,
            marker="s",
            markersize=4,
            label="# Features",
        )
        ax2_b.set_ylabel("Number of Features", fontsize=10, color="#888888")
        ax2_b.tick_params(axis="y", colors="#888888")

    ax2.tick_params(axis="y", colors=line_color)
    ax2.yaxis.label.set_color(line_color)

    ax1.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# ─────────────────────────────────────────────────────────────
# 4. Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    dpi: int = 300,
    title: str = "Model Performance Comparison",
    metrics: Optional[List[str]] = None,
    palette: Optional[List[str]] = None,
) -> plt.Figure:
    """Grouped bar chart comparing multiple models across metrics.

    Args:
        results_dict: dict mapping model_name → {metric: value}.
                      Example::

                          {
                              "LightGBM": {"accuracy": 0.89, "macro_f1": 0.87, "weighted_f1": 0.89},
                              "XGBoost":  {"accuracy": 0.87, "macro_f1": 0.85, "weighted_f1": 0.87},
                          }

        save_path:    output path or None.
        dpi:          output resolution.
        title:        figure title.
        metrics:      list of metric keys to plot. Defaults to
                      ["accuracy", "macro_f1", "weighted_f1"].
        palette:      list of hex/name colours (one per metric). Defaults to seaborn tab10.

    Returns:
        matplotlib Figure.
    """
    if metrics is None:
        metrics = ["accuracy", "macro_f1", "weighted_f1"]

    if palette is None:
        palette = list(sns.color_palette("tab10", n_colors=len(metrics)))

    _apply_base_style()

    model_names  = list(results_dict.keys())
    n_models     = len(model_names)
    n_metrics    = len(metrics)
    bar_width    = 0.8 / n_metrics
    x_base       = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.8), 7))

    for m_idx, (metric_name, color) in enumerate(zip(metrics, palette)):
        offsets  = x_base + (m_idx - n_metrics / 2.0 + 0.5) * bar_width
        heights  = [
            float(results_dict[model].get(metric_name, 0.0))
            for model in model_names
        ]
        bars = ax.bar(
            offsets,
            heights,
            width=bar_width * 0.9,
            color=color,
            label=metric_name.replace("_", " ").title(),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=2,
        )

        # Value annotations
        for bar_obj, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2.0,
                    h + 0.004,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=0,
                )

    ax.set_xticks(x_base)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# ─────────────────────────────────────────────────────────────
# 5. SHAP Summary (Beeswarm-style)
# ─────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    save_path: Optional[str] = None,
    max_display: int = 20,
    dpi: int = 300,
    title: str = "SHAP Feature Importance (Beeswarm)",
    X_background: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Beeswarm-style SHAP feature importance plot.

    If the shap library is available, delegates to shap.summary_plot for
    an authentic beeswarm.  Otherwise falls back to a horizontal bar chart
    of mean absolute SHAP values.

    Args:
        shap_values:    (N, n_features) SHAP value matrix
                        OR (n_features,) for a single sample.
        feature_names:  sequence of n_features feature name strings.
        save_path:      output path or None.
        max_display:    show only the top-k features by mean |SHAP|.
        dpi:            output resolution.
        title:          figure title.
        X_background:   (N, n_features) original feature matrix used as
                        colour background for shap.summary_plot (optional).

    Returns:
        matplotlib Figure.
    """
    _apply_base_style()

    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)

    n_features = shap_arr.shape[1]
    feat_names = list(feature_names)[:n_features]

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_arr).mean(axis=0)

    # Top-k feature selection
    top_idx    = np.argsort(mean_abs)[::-1][:max_display]
    top_names  = [feat_names[i] for i in top_idx]
    top_shap   = shap_arr[:, top_idx]
    top_abs    = mean_abs[top_idx]

    if HAS_SHAP and shap_arr.shape[0] > 1:
        # Use native shap beeswarm
        fig = plt.figure(figsize=(12, max(6, max_display * 0.35)))
        try:
            shap.summary_plot(
                top_shap,
                features=X_background[:, top_idx] if X_background is not None else top_shap,
                feature_names=top_names,
                max_display=max_display,
                show=False,
                plot_size=None,
            )
            fig = plt.gcf()
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        except Exception as exc:
            warnings.warn(
                f"shap.summary_plot failed ({exc}), falling back to bar chart.",
                UserWarning,
                stacklevel=2,
            )
            plt.close(fig)
            fig = _shap_bar_fallback(top_names, top_abs, title, max_display)
    else:
        fig = _shap_bar_fallback(top_names, top_abs, title, max_display)

    _save_or_show(fig, save_path, dpi)
    return fig


def _shap_bar_fallback(
    feature_names: List[str],
    mean_abs_shap: np.ndarray,
    title: str,
    max_display: int,
) -> plt.Figure:
    """Horizontal bar chart fallback when shap beeswarm is unavailable."""
    n = min(max_display, len(feature_names))
    names  = feature_names[:n]
    values = mean_abs_shap[:n]

    # Sort ascending for horizontal bar display (largest at top)
    order  = np.argsort(values)
    names  = [names[i]  for i in order]
    values = values[order]

    palette = sns.color_palette("coolwarm_r", n_colors=n)

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.38)))

    bars = ax.barh(
        names,
        values,
        color=[palette[i] for i in range(n)],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    for bar_obj, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.01,
            bar_obj.get_y() + bar_obj.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            fontsize=8.5,
        )

    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    return fig
