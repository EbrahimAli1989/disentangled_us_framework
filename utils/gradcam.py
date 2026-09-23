"""
Grad-CAM and Grad-CAM++ for CNN encoders (ResNet18, DebFaceEncoder).
ViT attention-map visualization is included as a separate helper.

References:
  Selvaraju et al. "Grad-CAM" (ICCV 2017)
  Chattopadhyay et al. "Grad-CAM++" (WACV 2018)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.

    Usage:
        gcam = GradCAM(model.enc1, target_layer=model.enc1.get_last_conv_layer())
        heatmap = gcam(input_tensor, class_idx)
    """

    def __init__(self, encoder, target_layer):
        self.encoder = encoder
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(fwd_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(bwd_hook)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, x, class_idx=None):
        """
        Args:
            x: (1, C, H, W) input tensor (requires grad)
            class_idx: target class index; if None, uses argmax

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.encoder.eval()
        x = x.requires_grad_(True)

        # Forward through full framework encoder
        features = self.encoder(x)  # (1, feature_dim)

        if class_idx is None:
            class_idx = features.argmax(dim=-1).item()

        score = features[0, class_idx]
        self.encoder.zero_grad()
        score.backward()

        # Weights = global average pooled gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample and normalise to [0, 1]
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ — improved gradient weighting with second-order derivatives.
    Chattopadhyay et al., WACV 2018.
    """

    def __call__(self, x, class_idx=None):
        self.encoder.eval()
        x = x.requires_grad_(True)

        features = self.encoder(x)

        if class_idx is None:
            class_idx = features.argmax(dim=-1).item()

        score = features[0, class_idx]
        self.encoder.zero_grad()
        score.backward()

        grads = self._gradients            # (1, C, H, W)
        acts = self._activations           # (1, C, H, W)

        grads_sq = grads ** 2
        grads_cu = grads ** 3
        denom = 2 * grads_sq + (acts * grads_cu).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha = grads_sq / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(image_np, heatmap, alpha=0.5, colormap="jet"):
    """
    Overlay a Grad-CAM heatmap onto the original image.

    Args:
        image_np: (H, W, 3) uint8 RGB image
        heatmap:  (H, W) float [0, 1]
        alpha:    blending weight for heatmap
        colormap: matplotlib colormap name

    Returns:
        overlaid: (H, W, 3) uint8
    """
    cmap = cm_module.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlaid = (1 - alpha) * image_np + alpha * colored
    return np.clip(overlaid, 0, 255).astype(np.uint8)


def vit_attention_map(vit_encoder, x, head_fusion="mean"):
    """
    Extract attention rollout from a ViT encoder.

    Args:
        vit_encoder: ViTEncoder instance
        x: (1, 3, H, W) input tensor
        head_fusion: 'mean' | 'max' | 'min' across attention heads

    Returns:
        attn_map: (H, W) numpy array in [0, 1]
    """
    vit_encoder.eval()
    attn_weights = []

    def hook_fn(module, inp, output):
        # output from MultiheadAttention is (attn_output, attn_weights)
        # torchvision ViT uses nn.MultiheadAttention
        pass

    # Register hooks on each attention layer
    hooks = []
    for layer in vit_encoder.backbone.encoder.layers:
        hooks.append(
            layer.self_attention.register_forward_hook(
                lambda m, i, o: attn_weights.append(o[1].detach() if o[1] is not None else None)
            )
        )

    with torch.no_grad():
        _ = vit_encoder(x)

    for h in hooks:
        h.remove()

    # Rollout: process attention weight matrices
    valid = [w for w in attn_weights if w is not None]
    if not valid:
        H = W = int(x.shape[-1] // 16)
        return np.ones((H, W)) / (H * W)

    rollout = torch.eye(valid[0].shape[-1], device=x.device)
    for attn in valid:
        if head_fusion == "mean":
            fused = attn.mean(dim=1)
        elif head_fusion == "max":
            fused = attn.max(dim=1).values
        else:
            fused = attn.min(dim=1).values
        fused = fused + torch.eye(fused.size(-1), device=fused.device)
        fused = fused / fused.sum(dim=-1, keepdim=True)
        rollout = fused @ rollout

    # Take CLS token row
    num_patches = rollout.shape[-1] - 1
    H = W = int(num_patches ** 0.5)
    mask = rollout[0, 0, 1:].reshape(H, W).cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def generate_gradcam_report(
    model,
    loader,
    device,
    figures_dir,
    num_samples=20,
    method="gradcam++",
    encoder_type="resnet18",
):
    """
    Generate and save Grad-CAM / attention overlays for a set of samples.

    Saves per-sample PDFs and a summary grid PDF.
    """
    os.makedirs(figures_dir, exist_ok=True)
    enc = model.enc1

    if encoder_type == "vit":
        use_vit = True
        gcam = None
    else:
        use_vit = False
        target_layer = enc.get_last_conv_layer()
        GCAMClass = GradCAMPlusPlus if method == "gradcam++" else GradCAM
        gcam = GCAMClass(enc, target_layer)

    model.eval()
    samples_done = 0
    fig_paths = []

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    for full_img, bg_img, roi_img, labels in loader:
        for b in range(full_img.size(0)):
            if samples_done >= num_samples:
                break

            inp = full_img[b:b+1].to(device)
            label = labels[b].item()

            if use_vit:
                heatmap = vit_attention_map(enc, inp)
                h, w = heatmap.shape
                inp_up = F.interpolate(
                    inp, size=(h * 16, w * 16), mode="bilinear", align_corners=False
                )
            else:
                heatmap = gcam(inp)

            # Denormalise for display
            img_np = full_img[b].permute(1, 2, 0).numpy()
            img_np = (img_np * imagenet_std + imagenet_mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            if heatmap.shape != img_np.shape[:2]:
                from PIL import Image
                heatmap = np.array(
                    Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                        (img_np.shape[1], img_np.shape[0])
                    )
                ) / 255.0

            overlaid = overlay_heatmap(img_np, heatmap)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Heatmap")
            axes[1].axis("off")
            axes[2].imshow(overlaid)
            axes[2].set_title(f"Overlay | Label: {'Malignant' if label else 'Benign'}")
            axes[2].axis("off")

            plt.tight_layout()
            path = os.path.join(figures_dir, f"gradcam_{samples_done:03d}.pdf")
            fig.savefig(path, format="pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)
            fig_paths.append(path)
            samples_done += 1

        if samples_done >= num_samples:
            break

    if gcam is not None:
        gcam.remove_hooks()

    return fig_paths
