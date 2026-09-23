"""
Encoder architectures for the Disentangled US Framework.

Three options:
  1. resnet18   — standard ResNet-18 (11.2M params, ~1.8 GFLOPs)
  2. vit         — Vision Transformer ViT-B/16 adapted for 224×224
  3. debface     — Lightweight CNN inspired by the DebFace paper
                   (shared conv backbone + task-specific FC heads)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# ResNet-18 encoder
# ---------------------------------------------------------------------------

class ResNet18Encoder(nn.Module):
    """
    ResNet-18 with the final fully-connected layer replaced by a Linear
    projection to feature_dim (default 512).
    """

    def __init__(self, feature_dim=512, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features          # 512 for ResNet-18
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # Projection to desired feature_dim
        self.proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        ) if in_features != feature_dim else nn.Identity()
        self.feature_dim = feature_dim

    def forward(self, x):
        h = self.backbone(x)     # (B, 512)
        return self.proj(h)      # (B, feature_dim)

    def get_last_conv_layer(self):
        """Used by Grad-CAM."""
        return self.backbone.layer4[-1]


# ---------------------------------------------------------------------------
# Vision Transformer encoder
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """
    ViT-B/16 pretrained on ImageNet-21k via torchvision.
    The [CLS] token output is projected to feature_dim.
    """

    def __init__(self, feature_dim=512, pretrained=True):
        super().__init__()
        try:
            weights = (
                tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
            backbone = tv_models.vit_b_16(weights=weights)
        except AttributeError:
            # Older torchvision
            backbone = tv_models.vit_b_16(pretrained=pretrained)

        # Replace classification head with identity
        in_features = backbone.heads.head.in_features  # 768
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        h = self.backbone(x)     # (B, 768)
        return self.proj(h)      # (B, feature_dim)

    def get_attention_weights(self):
        """Returns the last transformer block's attention (for visualization)."""
        return self.backbone.encoder.layers[-1].self_attention


# ---------------------------------------------------------------------------
# DebFace-inspired lightweight CNN encoder
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Standard pre-activation residual block."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class DebFaceEncoder(nn.Module):
    """
    Lightweight CNN adapted from DebFace:
      - Shared convolutional backbone (similar to DebFace's EImg)
      - ~2.3M parameters vs ResNet-18's 11.2M
      - Lower FLOPs for efficient clinical inference

    Architecture:
      conv1 (3→64)  → pool
      ResBlock ×2 (64)
      conv2 (64→128) → pool
      ResBlock ×2 (128)
      conv3 (128→256) → pool
      ResBlock ×2 (256)
      GlobalAvgPool → Linear(256, feature_dim)
    """

    def __init__(self, feature_dim=512, pretrained=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(_ResBlock(64), _ResBlock(64))
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(_ResBlock(128), _ResBlock(128))
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(_ResBlock(256), _ResBlock(256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def get_last_conv_layer(self):
        """Used by Grad-CAM."""
        return self.stage3[-1].block[-1]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENCODERS = {
    "resnet18": ResNet18Encoder,
    "vit": ViTEncoder,
    "debface": DebFaceEncoder,
}


def build_encoder(cfg):
    """
    Build encoder(s) from config.

    Returns:
        If shared_encoder=True  → single encoder instance (shared weights)
        If shared_encoder=False → tuple (enc1, enc2, enc3) with independent weights
    """
    model_cfg = cfg["model"]
    enc_type = model_cfg.get("encoder_type", "resnet18")
    feature_dim = model_cfg.get("feature_dim", 512)
    pretrained = model_cfg.get("pretrained", True)
    shared = model_cfg.get("shared_encoder", False)

    if enc_type not in _ENCODERS:
        raise ValueError(
            f"Unknown encoder '{enc_type}'. Choose from {list(_ENCODERS.keys())}"
        )

    EncoderClass = _ENCODERS[enc_type]

    if shared:
        enc = EncoderClass(feature_dim=feature_dim, pretrained=pretrained)
        return enc, enc, enc       # same object → shared weights
    else:
        enc1 = EncoderClass(feature_dim=feature_dim, pretrained=pretrained)
        enc2 = EncoderClass(feature_dim=feature_dim, pretrained=pretrained)
        enc3 = EncoderClass(feature_dim=feature_dim, pretrained=pretrained)
        return enc1, enc2, enc3
