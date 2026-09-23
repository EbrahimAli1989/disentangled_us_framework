"""
Configurable classification MLP head.

Receives f_roi (256-dim) from Branch 1 during training.
At inference, only Branch 1 encoder + projection head (ROI) + this head is used.
"""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Variable-depth MLP classifier.

    Args:
        in_dim:       input feature dimension (default 256 = projection_dim)
        hidden_dims:  list of hidden layer sizes, e.g. [256, 128]
        num_classes:  number of output classes (2 for benign/malignant)
        dropout:      dropout probability applied after each hidden activation
        batchnorm:    whether to add BatchNorm1d after each hidden Linear
    """

    def __init__(
        self,
        in_dim=256,
        hidden_dims=None,
        num_classes=2,
        dropout=0.3,
        batchnorm=True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    # (B, num_classes) — raw logits


def build_classification_head(cfg):
    model_cfg = cfg["model"]
    return ClassificationHead(
        in_dim=model_cfg.get("projection_dim", 256),
        hidden_dims=model_cfg.get("classifier_hidden_dims", [256, 128]),
        num_classes=model_cfg.get("num_classes", 2),
        dropout=model_cfg.get("classifier_dropout", 0.3),
        batchnorm=model_cfg.get("classifier_batchnorm", True),
    )
