"""
Evaluation module — computes all required metrics on a DataLoader.

Metrics:
  Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score, AUC-ROC
  Confusion matrix, ROC curve data
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import scipy.stats as st


class Evaluator:
    """
    Evaluates a DisentangledUSFramework checkpoint on a DataLoader.

    Uses Branch-1-only inference path (no masks required).
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Run inference on all batches.

        Returns:
            metrics: dict with scalar performance metrics
            roc_data: dict with fpr, tpr, thresholds arrays
        """
        self.model.eval()
        all_labels = []
        all_probs = []

        for full_img, bg_img, roi_img, mask_t, labels in loader:
            full_img = full_img.to(self.device, non_blocking=True)
            logits, _ = self.model.inference_forward(full_img)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp + 1e-8)

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        metrics = {
            "accuracy":    accuracy_score(y_true, y_pred),
            "precision":   precision_score(y_true, y_pred, zero_division=0),
            "recall":      recall_score(y_true, y_pred, zero_division=0),
            "specificity": specificity,
            "f1":          f1_score(y_true, y_pred, zero_division=0),
            "auc_roc":     roc_auc_score(y_true, y_prob),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        return metrics, roc_data, y_prob, y_true


def aggregate_run_metrics(all_metrics):
    """
    Compute mean ± std and 95% CI across multiple independent runs.

    Args:
        all_metrics: list of metric dicts, one per run

    Returns:
        summary: dict  metric_name → {mean, std, ci_low, ci_high}
    """
    keys = [k for k in all_metrics[0] if k != "confusion_matrix"]
    n = len(all_metrics)
    summary = {}

    for k in keys:
        vals = np.array([m[k] for m in all_metrics], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1)
        if n > 1:
            ci = st.t.interval(0.95, df=n - 1, loc=mean, scale=st.sem(vals))
        else:
            ci = (mean, mean)
        summary[k] = {
            "mean": float(mean),
            "std": float(std),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "values": vals.tolist(),
        }

    return summary
