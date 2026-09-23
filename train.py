"""
Main training entry point.

Runs N independent experiments with different random seeds,
logs all metrics, and produces consolidated PDF reports.

Example usage:
    python train.py --config config/default.yaml --num_runs 10 --device cuda
    python train.py --encoder resnet18 --shared_encoder
    python train.py --encoder vit --epochs 50 --batch_size 16
"""

import os
import sys
import argparse
import logging
import yaml
import torch

from data import build_dataloaders
from models import build_model
from training import Trainer, Evaluator
from training.evaluator import aggregate_run_metrics
from utils.misc import set_seed, setup_logging, count_parameters
from utils.metrics import MetricsLogger
from utils.visualization import plot_roc_curves, plot_confusion_matrix
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

def _resolve_exp_name(exp_name: str | None, base_dir: str = "outputs") -> str:
    """Return exp_name as-is, or find the next expN that doesn't exist yet."""
    if exp_name:
        return exp_name
    i = 1
    while os.path.exists(os.path.join(base_dir, f"exp{i}")):
        i += 1
    return f"exp{i}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disentangled US Framework — Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")

    # --- Experiment ---
    parser.add_argument("--exp_name", default=None,
                        help="Experiment folder name (e.g. exp1). "
                             "Auto-increments (exp1, exp2, …) when omitted.")
    parser.add_argument("--num_runs", type=int, default=None,
                        help="Number of independent runs (overrides config)")
    parser.add_argument("--base_seed", type=int, default=None,
                        help="Base random seed (overrides config)")
    parser.add_argument("--device", default=None,
                        help="'cuda' | 'cpu' | 'cuda:0' etc.")

    # --- Model ---
    parser.add_argument("--model_type", choices=["disentangled", "baseline"],
                        default=None,
                        help="Model variant: disentangled (proposed) | baseline (ablation)")
    parser.add_argument("--encoder", choices=["resnet18", "vit", "debface"],
                        default=None, help="Encoder architecture")
    parser.add_argument("--shared_encoder", action="store_true",
                        help="Use shared encoder weights for all branches")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train encoder from scratch (no ImageNet init)")

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default=None)

    # --- Loss ---
    parser.add_argument("--lambda_sim", type=float, default=None)
    parser.add_argument("--lambda_dis", type=float, default=None)
    parser.add_argument("--lambda_orth", type=float, default=None)
    parser.add_argument("--lambda_adv", type=float, default=None)
    parser.add_argument("--sim_loss", choices=["ntxent", "cosine"], default=None)

    # --- Data ---
    parser.add_argument("--data_dir", default=None, help="Override data directory")

    # --- Output ---
    parser.add_argument("--no_gradcam", action="store_true",
                        help="Skip Grad-CAM generation")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")

    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Merge CLI argument overrides into config dict."""
    if args.num_runs is not None:
        cfg["experiment"]["num_runs"] = args.num_runs
    if args.base_seed is not None:
        cfg["experiment"]["base_seed"] = args.base_seed
    if args.model_type is not None:
        cfg["model"]["model_type"] = args.model_type
    if args.encoder is not None:
        cfg["model"]["encoder_type"] = args.encoder
    if args.shared_encoder:
        cfg["model"]["shared_encoder"] = True
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.optimizer is not None:
        cfg["training"]["optimizer"] = args.optimizer
    if args.lambda_sim is not None:
        cfg["loss"]["lambda_sim"] = args.lambda_sim
    if args.lambda_dis is not None:
        cfg["loss"]["lambda_dis"] = args.lambda_dis
    if args.lambda_orth is not None:
        cfg["loss"]["lambda_orth"] = args.lambda_orth
    if args.lambda_adv is not None:
        cfg["loss"]["lambda_adv"] = args.lambda_adv
    if args.sim_loss is not None:
        cfg["loss"]["sim_loss_type"] = args.sim_loss
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.no_wandb:
        cfg["experiment"]["use_wandb"] = False
    return cfg


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = apply_cli_overrides(cfg, args)

    exp_cfg = cfg["experiment"]
    num_runs = exp_cfg.get("num_runs", 10)
    base_seed = exp_cfg.get("base_seed", 42)

    exp_name = _resolve_exp_name(args.exp_name)
    exp_dir = os.path.join("outputs", exp_name)
    results_dir = os.path.join(exp_dir, "results")
    figures_dir = os.path.join(exp_dir, "figures")
    cfg["training"]["log_dir"] = os.path.join(exp_dir, "logs")
    cfg["training"]["checkpoint_dir"] = os.path.join(exp_dir, "checkpoints")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    setup_logging(cfg["training"]["log_dir"], run_id=0)
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment: {exp_name}  →  {exp_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Encoder: {cfg['model']['encoder_type']}")
    logger.info(f"Shared encoder: {cfg['model']['shared_encoder']}")
    logger.info(f"Num runs: {num_runs}")

    metrics_logger = MetricsLogger(results_dir)
    all_test_metrics = []

    for run_id in range(num_runs):
        seed = base_seed + run_id
        logger.info(f"\n{'='*60}")
        logger.info(f"  RUN {run_id+1}/{num_runs}  |  seed={seed}")
        logger.info(f"{'='*60}")

        set_seed(seed)

        train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=seed)

        # ── Sanity-check: pull one batch and log shapes ─────────────────
        logger.info("Checking first training batch …")
        for full_img, bg_img, roi_img, mask_t, batch_labels in train_loader:
            logger.info(
                "Batch OK — full=%s bg=%s roi=%s mask=%s labels=%s unique=%s",
                tuple(full_img.shape), tuple(bg_img.shape),
                tuple(roi_img.shape), tuple(mask_t.shape),
                tuple(batch_labels.shape), batch_labels.unique().tolist(),
            )
            # Save diagnostic images to disk (safe on headless servers).
            sample_idx = min(2, full_img.shape[0] - 1)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            for ax, t, title in zip(
                axes,
                [full_img, bg_img, roi_img],
                ["full", "bg", "roi"],
            ):
                ax.imshow(t[sample_idx, 0].cpu().numpy(), cmap="gray")
                ax.set_title(title)
                ax.axis("off")
            fig_path = os.path.join(
                cfg["training"].get("log_dir", "outputs/logs"),
                f"run_{run_id:02d}_sample_check.png",
            )
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            logger.info("Sample images saved → %s", fig_path)
            break  # only one batch needed



        model = build_model(cfg)
        logger.info(
            f"  Model: {cfg['model'].get('model_type', 'disentangled')} | "
            f"Parameters: {count_parameters(model):,}"
        )

        trainer = Trainer(model, cfg, device, run_id=run_id)
        trainer.fit(train_loader, val_loader)

        evaluator = Evaluator(model, device)
        test_metrics, roc_data, y_prob, y_true = evaluator.evaluate(test_loader)

        logger.info(
            f"  Test | AUC={test_metrics['auc_roc']:.4f} | "
            f"F1={test_metrics['f1']:.4f} | Acc={test_metrics['accuracy']:.4f}"
        )

        metrics_logger.add_run(run_id, test_metrics, roc_data)
        all_test_metrics.append(test_metrics)

        # Save per-run confusion matrix
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            class_names=["Benign", "Malignant"],
            figures_dir=os.path.join(figures_dir, f"run_{run_id:02d}"),
            filename="confusion_matrix.pdf",
        )

    # Aggregate across all runs
    summary = aggregate_run_metrics(all_test_metrics)
    metrics_logger.print_summary(summary)

    # Save all outputs
    metrics_logger.save_per_run_csv()
    metrics_logger.save_summary_csv(summary)
    metrics_logger.save_json(summary)
    try:
        metrics_logger.save_excel(summary)
    except ImportError:
        logger.warning("openpyxl not installed — skipping Excel export.")

    # ROC curves PDF
    plot_roc_curves(
        metrics_logger.roc_records,
        figures_dir=figures_dir,
        filename="roc_curves.pdf",
    )

    # Grad-CAM report (last run model)
    if not args.no_gradcam:
        from utils.gradcam import generate_gradcam_report
        gc_dir = cfg.get("gradcam", {}).get(
            "output_dir", os.path.join(figures_dir, "gradcam")
        )
        generate_gradcam_report(
            model=model,
            loader=test_loader,
            device=device,
            figures_dir=gc_dir,
            num_samples=cfg.get("gradcam", {}).get("num_samples", 20),
            encoder_type=cfg["model"]["encoder_type"],
        )

    logger.info("\nTraining complete. All outputs saved.")
    logger.info(f"  Experiment: {exp_dir}")
    logger.info(f"  Results:    {results_dir}")
    logger.info(f"  Figures:    {figures_dir}")


if __name__ == "__main__":
    main()
