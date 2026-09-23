"""
Metrics logger — tracks per-run and cross-run results,
exports to CSV/Excel, and prints formatted summary tables.
"""

import os
import json
import numpy as np
import pandas as pd


class MetricsLogger:
    """
    Accumulates metric dicts across multiple runs and provides
    export utilities.
    """

    def __init__(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        self.run_records = []        # list of per-run metric dicts
        self.roc_records = []        # list of (fpr, tpr) tuples per run

    def add_run(self, run_id: int, metrics: dict, roc_data: dict = None):
        record = {"run_id": run_id, **metrics}
        self.run_records.append(record)
        if roc_data is not None:
            self.roc_records.append(roc_data)

    def save_per_run_csv(self, filename="per_run_metrics.csv"):
        """Save one row per run."""
        flat = []
        for r in self.run_records:
            row = {k: v for k, v in r.items() if k != "confusion_matrix"}
            flat.append(row)
        df = pd.DataFrame(flat)
        path = os.path.join(self.results_dir, filename)
        df.to_csv(path, index=False)
        return path

    def save_summary_csv(self, summary: dict, filename="summary_metrics.csv"):
        """Save mean ± std ± CI for each metric."""
        rows = []
        for metric, stats in summary.items():
            rows.append({
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
            })
        df = pd.DataFrame(rows)
        path = os.path.join(self.results_dir, filename)
        df.to_csv(path, index=False)
        return path

    def save_excel(self, summary: dict, filename="results.xlsx"):
        """Export per-run and summary to a single Excel workbook."""
        path = os.path.join(self.results_dir, filename)
        flat = []
        for r in self.run_records:
            row = {k: v for k, v in r.items() if k != "confusion_matrix"}
            flat.append(row)
        per_run_df = pd.DataFrame(flat)

        summary_rows = []
        for metric, stats in summary.items():
            summary_rows.append({
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "ci_low (95%)": stats["ci_low"],
                "ci_high (95%)": stats["ci_high"],
            })
        summary_df = pd.DataFrame(summary_rows)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            per_run_df.to_excel(writer, sheet_name="Per-Run", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        return path

    def print_summary(self, summary: dict):
        print("\n" + "=" * 60)
        print("  Metric Summary (mean ± std  [95% CI])")
        print("=" * 60)
        for metric, stats in summary.items():
            print(
                f"  {metric:<20s}: "
                f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                f"  [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]"
            )
        print("=" * 60 + "\n")

    def save_json(self, summary: dict, filename="summary.json"):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path
