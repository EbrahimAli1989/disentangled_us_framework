"""
Multi-objective disentanglement loss functions.

Total loss:
    L = L_cls
      + λ_sim  · L_sim
      + λ_dis  · L_dis
      + λ_orth · L_orth
      + λ_adv  · L_adv    (optional adversarial disentanglement)

──────────────────────────────────────────────────────────────────
L_cls   Cross-entropy classification loss
L_sim   NT-Xent / cosine similarity loss encouraging
            f_b ↔ f_b2  and  f_roi ↔ f_roi2  alignment
L_dis   KL-divergence between ROI and background feature distributions
            KL(f_roi ‖ f_b) + KL(f_roi ‖ f_b2)
          + KL(f_roi2 ‖ f_b) + KL(f_roi2 ‖ f_b2)
L_orth  Orthogonality constraint on the ROI/background feature pair
            ‖ f_roi · f_b^T ‖_F²  (Barlow-Twins-inspired)
            Justified by: Zbontar et al. 2021 "Barlow Twins" (ICML 2021)
L_adv   Gradient-reversal adversarial head trained to confuse ROI/BG identity,
            similar to DebFace (Gong et al., ECCV 2020)
──────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: soft probability for KL computation
# ---------------------------------------------------------------------------

def _to_prob(x, eps=1e-8):
    """Convert a feature vector to a soft probability distribution via softmax."""
    return F.softmax(x, dim=-1) + eps


# ---------------------------------------------------------------------------
# NT-Xent (Normalized Temperature-scaled Cross Entropy)
# Chen et al., SimCLR (ICML 2020)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    NT-Xent loss between two views of the same sample.

    For each (a_i, b_i) pair, maximise similarity and push apart cross-sample pairs.
    temperature: controls the concentration of the distribution.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a, z_b):
        """
        Args:
            z_a, z_b: (B, D) feature tensors — un-normalized
        """
        B = z_a.size(0)
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        # (2B, D)
        z = torch.cat([z_a, z_b], dim=0)
        # (2B, 2B) cosine similarity matrix
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarities on the diagonal
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Labels: for each i in [0, B), the positive is at i+B and vice versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Cosine similarity loss (direct alignment)
# ---------------------------------------------------------------------------

def cosine_similarity_loss(z_a, z_b):
    """1 - mean cosine similarity, encouraging alignment."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    return 1.0 - (z_a * z_b).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# KL divergence loss for disentanglement
# ---------------------------------------------------------------------------

def kl_divergence_loss(p, q):
    """
    KL(P ‖ Q) averaged over the batch.
    Inputs are raw feature vectors; softmax is applied internally.
    """
    p_prob = _to_prob(p)
    q_prob = _to_prob(q)
    return F.kl_div(q_prob.log(), p_prob, reduction="batchmean")


def symmetric_kl(a, b):
    """Symmetric KL: KL(A‖B) + KL(B‖A)."""
    return 0.5 * (kl_divergence_loss(a, b) + kl_divergence_loss(b, a))


# ---------------------------------------------------------------------------
# Orthogonality constraint
# Inspired by: Zbontar et al. "Barlow Twins" (ICML 2021)
#              and Kumar et al. "Disentangled Representations" (NeurIPS 2017)
# ---------------------------------------------------------------------------

def orthogonality_loss(z_a, z_b):
    """
    Penalise cross-correlation between z_a and z_b.

    For two (B, D) tensors, compute the cross-correlation matrix C = z_a^T z_b / B
    and penalise the Frobenius norm: ‖C‖_F².

    When z_a ⊥ z_b, C ≈ 0 and the loss vanishes.
    """
    z_a = F.normalize(z_a, dim=0)   # (B, D) — column-wise normalisation
    z_b = F.normalize(z_b, dim=0)
    C = torch.mm(z_a.T, z_b) / z_a.size(0)  # (D, D)
    return (C ** 2).sum()


# ---------------------------------------------------------------------------
# Adversarial disentanglement head (gradient reversal)
# DebFace (Gong et al., ECCV 2020): adversarial branch confuses ROI/BG identity
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients during backprop, passes forward unchanged."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class AdversarialDiscriminator(nn.Module):
    """
    Binary discriminator trained to distinguish ROI vs background features.
    The gradient reversal layer makes the encoder learn features
    that *fool* the discriminator — pushing ROI and BG apart.
    """

    def __init__(self, in_dim=256, hidden_dim=128, alpha=1.0):
        super().__init__()
        self.grl = GradientReversal(alpha=alpha)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),   # binary: is this ROI or background?
        )

    def forward(self, x):
        x = self.grl(x)
        return self.net(x)


# ---------------------------------------------------------------------------
# Unified loss module
# ---------------------------------------------------------------------------

class DisentanglementLoss(nn.Module):
    """
    Combines all loss components into one callable.

    Loss = L_cls
         + λ_sim  * L_sim    (NT-Xent or cosine)
         + λ_dis  * L_dis    (symmetric KL between ROI/BG pairs)
         + λ_orth * L_orth   (orthogonality constraint)
         + λ_adv  * L_adv    (adversarial confusion; optional)
    """

    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg.get("loss", {})
        model_cfg = cfg.get("model", {})
        proj_dim = model_cfg.get("projection_dim", 256)

        self.lambda_sim = loss_cfg.get("lambda_sim", 1.0)
        self.lambda_dis = loss_cfg.get("lambda_dis", 0.5)
        self.lambda_orth = loss_cfg.get("lambda_orth", 0.1)
        self.lambda_adv = loss_cfg.get("lambda_adv", 0.1)

        self.sim_loss_type = loss_cfg.get("sim_loss_type", "ntxent")
        self.dis_loss_type = loss_cfg.get("dis_loss_type", "kl+orth")

        self.ce_loss = nn.CrossEntropyLoss()
        self.ntxent = NTXentLoss(
            temperature=loss_cfg.get("ntxent_temperature", 0.07)
        )

        # Optional adversarial discriminator
        self.use_adv = self.lambda_adv > 0
        if self.use_adv:
            self.disc = AdversarialDiscriminator(in_dim=proj_dim)
        else:
            self.disc = None

    def _similarity_loss(self, f_roi, f_roi2, f_b, f_b2):
        """Encourage alignment within semantically identical views."""
        if self.sim_loss_type == "ntxent":
            roi_sim = self.ntxent(f_roi, f_roi2)
            bg_sim = self.ntxent(f_b, f_b2)
        else:  # cosine
            roi_sim = cosine_similarity_loss(f_roi, f_roi2)
            bg_sim = cosine_similarity_loss(f_b, f_b2)
        return 0.5 * (roi_sim + bg_sim)

    def _disentanglement_loss(self, f_roi, f_roi2, f_b, f_b2):
        """Push ROI and background distributions apart."""
        pairs = [
            (f_roi, f_b),
            (f_roi, f_b2),
            (f_roi2, f_b),
            (f_roi2, f_b2),
        ]
        total = torch.tensor(0.0, device=f_roi.device)
        if "kl" in self.dis_loss_type:
            for a, b in pairs:
                total = total + symmetric_kl(a, b)
            total = total / len(pairs)
        if "orth" in self.dis_loss_type:
            orth = sum(orthogonality_loss(a, b) for a, b in pairs) / len(pairs)
            total = total + orth
        return total

    def _adversarial_loss(self, f_roi, f_b):
        """
        Adversarial confusion loss.

        The discriminator is trained to separate ROI (label=1) from BG (label=0).
        Via gradient reversal, the encoder is trained to confuse the discriminator,
        forcing f_roi and f_b to share no discriminative structure.
        """
        B = f_roi.size(0)
        features = torch.cat([f_roi, f_b], dim=0)          # (2B, D)
        labels = torch.cat([
            torch.ones(B, dtype=torch.long, device=f_roi.device),
            torch.zeros(B, dtype=torch.long, device=f_roi.device),
        ])
        logits = self.disc(features)
        return self.ce_loss(logits, labels)

    def forward(self, output, labels):
        """
        Args:
            output: DisentanglementOutput (logits, f_b, f_roi, f_b2, f_roi2)
            labels: (B,) ground-truth class indices

        Returns:
            total_loss: scalar
            loss_dict:  dict with individual component values (for logging)
        """
        l_cls = self.ce_loss(output.logits, labels)

        l_sim = self._similarity_loss(
            output.f_roi, output.f_roi2, output.f_b, output.f_b2
        )

        l_dis = self._disentanglement_loss(
            output.f_roi, output.f_roi2, output.f_b, output.f_b2
        )

        l_orth = orthogonality_loss(output.f_roi, output.f_b)

        total = l_cls
        total = total + self.lambda_sim * l_sim
        total = total + self.lambda_dis * l_dis
        total = total + self.lambda_orth * l_orth

        loss_dict = {
            "cls": l_cls.item(),
            "sim": l_sim.item(),
            "dis": l_dis.item(),
            "orth": l_orth.item(),
        }

        if self.use_adv and self.disc is not None:
            l_adv = self._adversarial_loss(output.f_roi, output.f_b)
            total = total + self.lambda_adv * l_adv
            loss_dict["adv"] = l_adv.item()

        loss_dict["total"] = total.item()
        return total, loss_dict


# ---------------------------------------------------------------------------
# Baseline loss  (cross-entropy only — used with BaselineUSModel)
# ---------------------------------------------------------------------------

class BaselineLoss(nn.Module):
    """
    Plain cross-entropy loss for the single-encoder baseline model.

    Provides the same (total_loss, loss_dict) interface as
    DisentanglementLoss so the Trainer can call both identically.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits : (B, num_classes) raw logits from BaselineUSModel
            labels : (B,) ground-truth class indices

        Returns:
            total_loss : scalar tensor
            loss_dict  : {"cls": float, "total": float}
        """
        loss = self.ce_loss(logits, labels)
        loss_dict = {"cls": loss.item(), "total": loss.item()}
        return loss, loss_dict
