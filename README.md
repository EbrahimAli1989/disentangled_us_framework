# CLASSIFICATION OF TUMORS IN BREAST ULTRASOUND IMAGES USING
DISENTANGLEMENT DEEP LEARNING FRAMEWORK
A research-grade deep learning framework for classifying breast tumors in ultrasound images using disentangled representation learning. Designed for reproducibility, publication-quality reporting, and systematic ablation studies.





## Overview

The core idea is **disentangled representation learning**: instead of training a single encoder on the full ultrasound image, the framework forces two distinct representations to emerge from each image — one capturing the tumor region of interest (ROI) features and one capturing background tissue features. These two representations are explicitly pushed apart during training, preventing the model from conflating lesion-specific and context-specific information.

**Why disentanglement for ultrasound?**
Ultrasound images contain both the lesion and surrounding tissue context. A standard classifier may learn spurious correlations from the background (e.g., probe angle artifacts, tissue depth). Disentangling ROI from background representations encourages the model to classify based on lesion-intrinsic features — improving generalization to new scanning conditions.



## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.1+. GPU strongly recommended.





## Quick Start

### Training

```bash
# Default: ResNet18, distinct encoders, 10 runs
python train.py --config config/default.yaml

# ViT encoder, shared weights, 5 runs
python train.py --encoder vit --shared_encoder --num_runs 5

# DebFace encoder, custom loss weights
python train.py --encoder debface --lambda_sim 0.5 --lambda_dis 1.0 --lambda_orth 0.2

# Override data directory
python train.py --data_dir /path/to/your/data --device cuda
```

### Evaluation

```bash
python evaluate.py --checkpoint outputs/checkpoints/run_00_best.pth
python evaluate.py --checkpoint outputs/checkpoints/run_00_best.pth --no_gradcam
```


