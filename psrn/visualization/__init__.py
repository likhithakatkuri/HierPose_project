"""HierPose visualization package.

Publication-quality matplotlib/seaborn plot utilities for:
    - Confusion matrices         (plot_confusion_matrix)
    - t-SNE feature embeddings   (plot_tsne)
    - Ablation study curves      (plot_ablation_curve)
    - Model comparison bars      (plot_model_comparison)
    - SHAP summary beeswarms     (plot_shap_summary)

All functions write 300 DPI PNG/PDF when save_path is supplied and
optionally display the figure via plt.show() otherwise.

Example::

    from psrn.visualization import plot_confusion_matrix, plot_model_comparison

    plot_confusion_matrix(cm, class_names, save_path="figures/cm.png")
    plot_model_comparison(results_dict, save_path="figures/models.png")
"""

from psrn.visualization.plots import (
    plot_confusion_matrix,
    plot_tsne,
    plot_ablation_curve,
    plot_model_comparison,
    plot_shap_summary,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_tsne",
    "plot_ablation_curve",
    "plot_model_comparison",
    "plot_shap_summary",
]
