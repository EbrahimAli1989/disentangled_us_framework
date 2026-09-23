"""
Disentangled Multi-Branch US Tumor Classification Framework.

─────────────────────────────────────────────────────────────────────────
  PROPOSED METHOD — DisentangledUSFramework
─────────────────────────────────────────────────────────────────────────

  Input Image x  ──────────────────────────────────► Encoder 1 ──► h1
  + Mask m                                                              ├──► ProjHead_b   ──► f_b   (256)
                                                                        └──► ProjHead_roi ──► f_roi (256) ──► Classifier ──► logits

  Background extraction:
    x_bg = x ⊙ (1 − m)           [element-wise mask-out of tumor region]
         │
         ▼
  [Optional] InpaintingNetwork(x_bg, m)
         │    Lightweight U-Net fills tumor-region holes with realistic
         │    background tissue.  Active when inpainting_enabled=True.
         ▼
    x_bg_inp  ──────────────────────────────────────► Encoder 2 ──► h2
                                                                        └──► ProjHead_b2  ──► f_b2  (256)

  ROI extraction:
    x_roi = x ⊙ m
         │
         ▼
    x_roi ──────────────────────────────────────────► Encoder 3 ──► h3
                                                                        └──► ProjHead_roi2 ──► f_roi2 (256)

  Losses (training only):
    L_total = L_cls  +  λ_sim·L_sim  +  λ_dis·L_dis  +  λ_orth·L_orth  +  λ_adv·L_adv

  Inference:  Branch 1 encoder + ProjHead_roi + Classifier  (no mask needed)

─────────────────────────────────────────────────────────────────────────
  BASELINE — BaselineUSModel  (ablation comparison)
─────────────────────────────────────────────────────────────────────────

  Input Image x  ──► Encoder ──► h (512) ──► Classifier ──► logits

  Single branch, full image only.  No masking, no inpainting,
  no disentanglement losses, no projection heads.
  Loss: cross-entropy only.

─────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder
from .projection_heads import ProjectionHead
from .classification_head import ClassificationHead, build_classification_head


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DisentanglementOutput:
    logits:  torch.Tensor   # (B, num_classes)
    f_b:     torch.Tensor   # (B, 256) — background features, Branch 1
    f_roi:   torch.Tensor   # (B, 256) — ROI features, Branch 1
    f_b2:    torch.Tensor   # (B, 256) — background features, Branch 2
    f_roi2:  torch.Tensor   # (B, 256) — ROI features, Branch 3


# ─────────────────────────────────────────────────────────────────────────────
# Inpainting network  (lightweight U-Net)
# ─────────────────────────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class InpaintingNetwork(nn.Module):
    """
    Lightweight U-Net that fills tumor-region holes in the background image.

    Input:  x_bg  (B, 3, H, W) — masked background: x ⊙ (1 − m)
            mask  (B, 1, H, W) — binary tumor mask (1 = inpaint region)

    Output: (B, 3, H, W) — reconstructed background with holes filled.

    The known background pixels (mask == 0) are preserved exactly via the
    composite:  output = x_bg * (1 − mask) + prediction * mask

    Trained end-to-end; no explicit reconstruction loss required — the
    downstream disentanglement losses provide the supervision signal.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        # Encoder  (input = image + mask = in_channels+1 channels)
        self.enc1 = _DoubleConv(in_channels + 1, c)
        self.enc2 = _DoubleConv(c,     c * 2)
        self.enc3 = _DoubleConv(c * 2, c * 4)
        # Bottleneck
        self.bottleneck = _DoubleConv(c * 4, c * 8)
        # Decoder  (each level gets skip from corresponding encoder level)
        self.dec3 = _DoubleConv(c * 8 + c * 4, c * 4)
        self.dec2 = _DoubleConv(c * 4 + c * 2, c * 2)
        self.dec1 = _DoubleConv(c * 2 + c,     c)
        # Output projection — no activation; range learned to match normalised images
        self.out_conv = nn.Conv2d(c, in_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_bg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_bg : (B, 3, H, W) masked background tensor
            mask : (B, 1, H, W) float mask, 1 = tumor region to fill
        """
        x = torch.cat([x_bg, mask], dim=1)          # (B, 4, H, W)

        e1 = self.enc1(x)                            # (B, c,   H,   W)
        e2 = self.enc2(self.pool(e1))                # (B, 2c,  H/2, W/2)
        e3 = self.enc3(self.pool(e2))                # (B, 4c,  H/4, W/4)
        b  = self.bottleneck(self.pool(e3))          # (B, 8c,  H/8, W/8)

        up3 = F.interpolate(b,  size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3  = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2  = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1  = self.dec1(torch.cat([up1, e1], dim=1))

        pred = self.out_conv(d1)                     # (B, 3, H, W)

        # Composite: keep known background pixels; fill predicted in tumor region
        return x_bg * (1.0 - mask) + pred * mask


# ─────────────────────────────────────────────────────────────────────────────
# Proposed disentangled framework
# ─────────────────────────────────────────────────────────────────────────────

class DisentangledUSFramework(nn.Module):
    """
    Three-branch disentangled representation learning framework.

    During training all branches are active.
    During inference call ``inference_forward(full_img)`` — Branch 1 only,
    no mask required.
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        feature_dim = model_cfg.get("feature_dim", 512)
        proj_dim    = model_cfg.get("projection_dim", 256)

        enc1, enc2, enc3 = build_encoder(cfg)
        self.enc1 = enc1   # Branch 1: full US image
        self.enc2 = enc2   # Branch 2: background image (optionally inpainted)
        self.enc3 = enc3   # Branch 3: ROI image

        # Projection heads — Branch 1
        self.proj_b   = ProjectionHead(feature_dim, feature_dim, proj_dim)
        self.proj_roi = ProjectionHead(feature_dim, feature_dim, proj_dim)

        # Projection heads — Branch 2 & 3
        self.proj_b2   = ProjectionHead(feature_dim, feature_dim, proj_dim)
        self.proj_roi2 = ProjectionHead(feature_dim, feature_dim, proj_dim)

        # Classification head (uses f_roi from Branch 1)
        self.classifier = build_classification_head(cfg)

        # Optional learnable inpainting network for background branch
        if model_cfg.get("inpainting_enabled", False):
            self.inpainting = InpaintingNetwork(
                in_channels=3,
                base_channels=model_cfg.get("inpainting_channels", 32),
            )
        else:
            self.inpainting = None

    def forward(
        self,
        full_img: torch.Tensor,
        bg_img:   torch.Tensor,
        roi_img:  torch.Tensor,
        mask:     torch.Tensor | None = None,
    ) -> "DisentanglementOutput":
        """
        Full forward pass for training.

        Args:
            full_img : (B, 3, H, W) — original US image
            bg_img   : (B, 3, H, W) — x ⊙ (1−m), background with tumor zeroed
            roi_img  : (B, 3, H, W) — x ⊙ m, tumor ROI with background zeroed
            mask     : (B, 1, H, W) optional; required when inpainting_enabled=True

        Returns:
            DisentanglementOutput
        """
        # Branch 1 — full image
        h1    = self.enc1(full_img)
        f_b   = self.proj_b(h1)
        f_roi = self.proj_roi(h1)

        # Branch 2 — background (apply inpainting if enabled)
        if self.inpainting is not None:
            if mask is None:
                raise ValueError(
                    "mask must be provided when inpainting_enabled=True"
                )
            bg_inp = self.inpainting(bg_img, mask)
        else:
            bg_inp = bg_img
        h2   = self.enc2(bg_inp)
        f_b2 = self.proj_b2(h2)

        # Branch 3 — ROI only
        h3     = self.enc3(roi_img)
        f_roi2 = self.proj_roi2(h3)

        logits = self.classifier(f_roi)

        return DisentanglementOutput(
            logits=logits,
            f_b=f_b,
            f_roi=f_roi,
            f_b2=f_b2,
            f_roi2=f_roi2,
        )

    @torch.no_grad()
    def inference_forward(self, full_img: torch.Tensor):
        """
        Lightweight inference — Branch 1 only, no mask required.

        Returns:
            logits : (B, num_classes)
            f_roi  : (B, proj_dim)
        """
        h1     = self.enc1(full_img)
        f_roi  = self.proj_roi(h1)
        logits = self.classifier(f_roi)
        return logits, f_roi

    def get_branch1_encoder(self):
        """Return encoder 1 and ROI projection head (used by Grad-CAM)."""
        return self.enc1, self.proj_roi


# ─────────────────────────────────────────────────────────────────────────────
# Baseline model  (ablation: no disentanglement)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineUSModel(nn.Module):
    """
    Single-encoder baseline for ablation comparison.

    Uses only the full ultrasound image.  No mask, no background/ROI
    separation, no projection heads, no disentanglement losses.
    Loss: cross-entropy only.

    Provides the same ``inference_forward`` and ``get_branch1_encoder``
    interface as DisentangledUSFramework for drop-in compatibility.
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg   = cfg["model"]
        feature_dim = model_cfg.get("feature_dim", 512)

        enc1, _, _ = build_encoder(cfg)   # only one encoder needed
        self.encoder = enc1

        # Classifier takes encoder output directly (feature_dim, not proj_dim)
        self.classifier = ClassificationHead(
            in_dim=feature_dim,
            hidden_dims=model_cfg.get("classifier_hidden_dims", [256, 128]),
            num_classes=model_cfg.get("num_classes", 2),
            dropout=model_cfg.get("classifier_dropout", 0.3),
            batchnorm=model_cfg.get("classifier_batchnorm", True),
        )

    def forward(self, full_img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Only ``full_img`` is used; extra positional/keyword args are ignored
        so the baseline can be swapped in wherever DisentangledUSFramework is used.

        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(self.encoder(full_img))

    @torch.no_grad()
    def inference_forward(self, full_img: torch.Tensor):
        """
        Returns:
            logits   : (B, num_classes)
            features : (B, feature_dim)
        """
        features = self.encoder(full_img)
        logits   = self.classifier(features)
        return logits, features

    def get_branch1_encoder(self):
        return self.encoder, None


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg) -> nn.Module:
    """
    Instantiate the correct model from config.

    ``cfg["model"]["model_type"]`` controls which variant is built:
      "disentangled" (default) → DisentangledUSFramework
      "baseline"               → BaselineUSModel
    """
    model_type = cfg["model"].get("model_type", "disentangled")
    if model_type == "baseline":
        return BaselineUSModel(cfg)
    if model_type == "disentangled":
        return DisentangledUSFramework(cfg)
    raise ValueError(
        f"Unknown model_type '{model_type}'. Choose: disentangled | baseline"
    )
