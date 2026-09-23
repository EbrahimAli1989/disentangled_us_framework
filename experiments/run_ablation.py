"""
Ablation study runner.

Systematically evaluates all combinations of:
  - Encoder type:      resnet18 | vit | debface
  - Shared encoder:   true | false
  - Loss combination: full | no_orth | no_adv | cls_only

Each condition runs with a fixed seed to ensure comparability.
Results are saved to outputs/results/ablation/.
"""

import os
import sys
import copy
import json
import logging
import argparse

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import build_dataloaders
from models import build_model
from training import Trainer, Evaluator
from utils.misc import set_seed, setup_logging, count_parameters, compute_flops, benchmark_inference
from utils.metrics import MetricsLogger
from training.evaluator import aggregate_run_metrics

logger = logging.getLogger(__name__)


ABLATION_CONDITIONS = [
    # --- Baseline (no disentanglement) — primary ablation reference ---
    {"name": "Baseline",         "model.model_type": "baseline",   "model.encoder_type": "resnet18"},

    # --- Encoder comparison (proposed disentangled framework) ---
    {"name": "ResNet18_distinct",   "model.model_type": "disentangled", "model.encoder_type": "resnet18", "model.shared_encoder": False},
    {"name": "ResNet18_shared",     "model.model_type": "disentangled", "model.encoder_type": "resnet18", "model.shared_encoder": True},
    {"name": "ViT_distinct",        "model.model_type": "disentangled", "model.encoder_type": "vit",      "model.shared_encoder": False},
    {"name": "DebFace_distinct",    "model.model_type": "disentangled", "model.encoder_type": "debface",  "model.shared_encoder": False},

    # --- Loss combination comparison (ResNet18 distinct as base) ---
    {"name": "LossFull",    "model.model_type": "disentangled", "model.encoder_type": "resnet18", "loss.lambda_orth": 0.1, "loss.lambda_adv": 0.1},
    {"name": "LossNoOrth",  "model.model_type": "disentangled", "model.encoder_type": "resnet18", "loss.lambda_orth": 0.0, "loss.lambda_adv": 0.1},
    {"name": "LossNoAdv",   "model.model_type": "disentangled", "model.encoder_type": "resnet18", "loss.lambda_orth": 0.1, "loss.lambda_adv": 0.0},
    {"name": "LossNoDis",   "model.model_type": "disentangled", "model.encoder_type": "resnet18", "loss.lambda_dis":  0.0},
    {"name": "LossClsOnly", "model.model_type": "disentangled", "model.encoder_type": "resnet18",
     "loss.lambda_sim": 0.0, "loss.lambda_dis": 0.0, "loss.lambda_orth": 0.0, "loss.lambda_adv": 0.0},
]


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dot-notation key overrides to a nested config dict."""
    cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        if key == "name":
            continue
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def run_condition(base_cfg: dict, condition: dict, device: torch.device, seed: int):
    """Run a single ablation condition and return metrics."""
    name = condition["name"]
    cfg = _apply_overrides(base_cfg, condition)

    set_seed(seed)

    results_dir = os.path.join(
        cfg["experiment"]["results_dir"], "ablation", name
    )
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=seed)

    model = build_model(cfg)
    trainer = Trainer(model, cfg, device, run_id=0)
    trainer.fit(train_loader, val_loader)

    evaluator = Evaluator(model, device)
    metrics, roc_data, y_prob, y_true = evaluator.evaluate(test_loader)

    # Efficiency analysis
    params = count_parameters(model)
    flops, _ = compute_flops(model.enc1, device=str(device))
    lat_mean, lat_std = benchmark_inference(model, device)

    metrics["params"] = params
    metrics["flops_gflops"] = flops / 1e9 if flops else None
    metrics["latency_ms"] = lat_mean

    return metrics, roc_data, name


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Subset of condition names to run (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    log_dir = os.path.join(base_cfg["training"]["log_dir"], "ablation")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "ablation.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device(args.device)

    conditions = ABLATION_CONDITIONS
    if args.conditions:
        conditions = [c for c in conditions if c["name"] in args.conditions]

    ablation_results = {}

    for condition in conditions:
        name = condition["name"]
        logger.info(f"\n{'='*50}\nAblation condition: {name}\n{'='*50}")
        try:
            metrics, roc_data, _ = run_condition(base_cfg, condition, device, args.seed)
            ablation_results[name] = metrics
            logger.info(
                f"  AUC={metrics['auc_roc']:.4f} | F1={metrics['f1']:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | Params={metrics['params']:,}"
            )
        except Exception as e:
            logger.error(f"  Condition {name} failed: {e}", exc_info=True)
            ablation_results[name] = {"error": str(e)}

    # Save summary
    out_path = os.path.join(
        base_cfg["experiment"]["results_dir"], "ablation", "ablation_summary.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Condition':<28} {'AUC':>8} {'F1':>8} {'Acc':>8} {'Params':>12} {'FLOPs(G)':>10}")
    print("-" * 90)
    for name, m in ablation_results.items():
        if "error" in m:
            print(f"{name:<28}  ERROR: {m['error']}")
        else:
            print(
                f"{name:<28} {m.get('auc_roc', 0):8.4f} {m.get('f1', 0):8.4f} "
                f"{m.get('accuracy', 0):8.4f} {m.get('params', 0):12,} "
                f"{m.get('flops_gflops', 0) or 0:10.2f}"
            )
    print("=" * 90)
    print(f"\nAblation summary saved to {out_path}")


if __name__ == "__main__":
    main()
