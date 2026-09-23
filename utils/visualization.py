"""
Visualization utilities:
  - ROC curves (per-run + mean) saved as PDF
  - Confusion matrix heatmaps
  - CSV / Excel export (wrappers)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/script use
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_roc_curves(roc_records, figures_dir, filename="roc_curves.pdf"):
    """
    Plot all per-run ROC curves plus the interpolated mean curve.

    Args:
        roc_records: list of dicts with keys 'fpr', 'tpr'
        figures_dir: output directory
        filename: output file name
    """
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    base_fpr = np.linspace(0, 1, 201)
    tpr_interp = []

    for i, rd in enumerate(roc_records):
        fpr, tpr = rd["fpr"], rd["tpr"]
        run_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, alpha=0.35, lw=1.0, color="steelblue",
                label=f"Run {i+1} (AUC={run_auc:.3f})" if len(roc_records) <= 5 else None)
        tpr_i = np.interp(base_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tpr_interp.append(tpr_i)

    mean_tpr = np.mean(tpr_interp, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tpr_interp, axis=0)
    mean_auc = auc(base_fpr, mean_tpr)

    ax.plot(base_fpr, mean_tpr, color="crimson", lw=2.5,
            label=f"Mean ROC (AUC = {mean_auc:.3f})")
    ax.fill_between(
        base_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="crimson", alpha=0.15, label="± 1 SD"
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Chance")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Disentangled US Framework", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(figures_dir, filename)
    fig.savefig(path, format="pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, class_names, figures_dir, filename="confusion_matrix.pdf"):
    """
    Plot a confusion matrix heatmap and save as PDF.

    Args:
        cm: 2D list or numpy array
        class_names: list of class label strings
        figures_dir: output directory
        filename: output file name
    """
    import matplotlib.colors as mcolors
    os.makedirs(figures_dir, exist_ok=True)
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()

    path = os.path.join(figures_dir, filename)
    fig.savefig(path, format="pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_results_csv(summary: dict, path: str):
    """Quick helper to dump summary dict to CSV without pandas."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "ci_low", "ci_high"])
        for metric, stats in summary.items():
            writer.writerow([
                metric,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['ci_low']:.6f}",
                f"{stats['ci_high']:.6f}",
            ])
