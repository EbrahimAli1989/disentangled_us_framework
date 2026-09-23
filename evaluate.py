"""
Standalone evaluation script.

Loads a saved checkpoint and evaluates on the test set.
Also generates Grad-CAM visualizations and efficiency benchmarks.

Example:
    python evaluate.py --checkpoint outputs/checkpoints/run_00_best.pth
    python evaluate.py --checkpoint outputs/checkpoints/run_00_best.pth --no_gradcam
"""

import os
import sys
import argparse
import logging
import yaml
import torch

from data import build_dataloaders
from models import DisentangledUSFramework
from training import Evaluator
from utils.misc import set_seed, count_parameters, compute_flops, benchmark_inference
from utils.visualization import plot_roc_curves, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disentangled US Framework — Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_gradcam", action="store_true")
    parser.add_argument("--output_dir", default="outputs/eval_results")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    _, _, test_loader = build_dataloaders(cfg, seed=args.seed)

    model = DisentangledUSFramework(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")
    logger.info(f"Parameters: {count_parameters(model):,}")

    # Evaluate
    evaluator = Evaluator(model, device)
    metrics, roc_data, y_prob, y_true = evaluator.evaluate(test_loader)

    print("\nTest Metrics:")
    print("-" * 40)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:<20s}: {v:.4f}")
    print()

    # Confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=["Benign", "Malignant"],
        figures_dir=args.output_dir,
        filename="confusion_matrix.pdf",
    )

    # ROC curve
    plot_roc_curves(
        [roc_data],
        figures_dir=args.output_dir,
        filename="roc_curve.pdf",
    )

    # Efficiency benchmarks
    flops, _ = compute_flops(model.enc1, device=str(device))
    lat_mean, lat_std = benchmark_inference(model, device)
    print(f"  Inference latency: {lat_mean:.2f} ± {lat_std:.2f} ms/sample")
    if flops:
        print(f"  Branch-1 FLOPs:    {flops/1e9:.3f} GFLOPs")

    # Grad-CAM
    if not args.no_gradcam:
        from utils.gradcam import generate_gradcam_report
        generate_gradcam_report(
            model=model,
            loader=test_loader,
            device=device,
            figures_dir=os.path.join(args.output_dir, "gradcam"),
            num_samples=20,
            encoder_type=cfg["model"]["encoder_type"],
        )

    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
