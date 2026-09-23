"""
Reproducibility, model analysis, and benchmark utilities.
"""

import random
import time
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Estimate FLOPs using thop (if available) or a rough MACs estimate.

    Returns:
        flops: total FLOPs (float)
        params: total parameters (int)
    """
    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        model = model.to(device).eval()
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        flops = 2 * macs
        logger.info(f"FLOPs: {flops/1e9:.2f} GFLOPs | Params: {params/1e6:.2f}M")
        return flops, int(params)
    except ImportError:
        logger.warning("thop not installed — FLOPs estimate unavailable.")
        return None, count_parameters(model)


def benchmark_inference(model, device, input_size=(1, 3, 224, 224), n_warmup=50, n_runs=200):
    """
    Benchmark Branch-1-only inference latency.

    Returns:
        mean_ms:  mean latency per sample in milliseconds
        std_ms:   standard deviation
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_size).to(device)
    timings = []

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.inference_forward(dummy)

    if device.startswith("cuda") or (hasattr(device, "type") and device.type == "cuda"):
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.inference_forward(dummy)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    logger.info(f"Inference latency: {mean_ms:.2f} ± {std_ms:.2f} ms/sample")
    return mean_ms, std_ms


def setup_logging(log_dir: str, run_id: int):
    """Configure logging to file + console."""
    import os
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{run_id:02d}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
