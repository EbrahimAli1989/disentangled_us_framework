"""
MLP projection heads that map encoder features → disentangled subspaces.

  ProjectionHead(in_dim=512, out_dim=256) used for:
    - f_b    (background from Branch 1)
    - f_roi  (ROI from Branch 1)
    - f_b2   (background from Branch 2)
    - f_roi2 (ROI from Branch 3)
"""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head:
        Linear(in_dim → hidden_dim) → BN → ReLU → Linear(hidden_dim → out_dim)

    L2-normalization is NOT applied here; the loss functions handle it if needed.
    """

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=256, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
