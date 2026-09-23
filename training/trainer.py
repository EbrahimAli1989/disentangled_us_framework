"""
Training loop with:
  - Mixed-precision (AMP)
  - Gradient clipping
  - LR scheduling (cosine / step / plateau / none)
  - Early stopping
  - TensorBoard logging
  - W&B logging (optional)
  - Checkpoint saving / loading
"""

import os
import math
import time
import logging

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau
)

from losses import DisentanglementLoss, BaselineLoss
from utils.misc import set_seed, count_parameters

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """
    Full training harness for DisentangledUSFramework.

    Usage:
        trainer = Trainer(model, cfg, device)
        best_val_loss = trainer.fit(train_loader, val_loader)
    """

    def __init__(self, model, cfg, device, run_id=0):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.run_id = run_id

        # Variant detection — drives forward-call and loss selection
        self.model_type = cfg["model"].get("model_type", "disentangled")
        self.use_inpainting = (
            self.model_type == "disentangled"
            and cfg["model"].get("inpainting_enabled", False)
        )

        train_cfg = cfg["training"]
        self.epochs = train_cfg.get("epochs", 100)
        self.grad_clip = train_cfg.get("gradient_clip", 1.0)
        self.mixed_precision = train_cfg.get("mixed_precision", True)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.ckpt_dir = train_cfg.get("checkpoint_dir", "outputs/checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.model_type == "baseline":
            self.criterion = BaselineLoss(cfg).to(device)
        else:
            self.criterion = DisentanglementLoss(cfg).to(device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.early_stop = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 20)
        )

        self.writer = self._build_tb_writer(cfg)
        self.wandb = self._init_wandb(cfg)

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def _build_optimizer(self):
        train_cfg = self.cfg["training"]
        params = list(self.model.parameters())
        if hasattr(self, "criterion"):
            params += list(self.criterion.parameters())
        opt_name = train_cfg.get("optimizer", "adamw").lower()
        lr = train_cfg.get("lr", 1e-4)
        lr = float(lr)
        wd = train_cfg.get("weight_decay", 1e-4)
        wd = float(wd)
        if opt_name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self):
        train_cfg = self.cfg["training"]
        sched = train_cfg.get("scheduler", "cosine").lower()
        if sched == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.get("scheduler_T_max", self.epochs),
                eta_min=1e-7,
            )
        elif sched == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif sched == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="min", patience=10, factor=0.5
            )
        return None

    def _build_tb_writer(self, cfg):
        if not cfg.get("experiment", {}).get("use_tensorboard", True):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(
                cfg["training"].get("log_dir", "outputs/logs"),
                f"run_{self.run_id:02d}",
            )
            return SummaryWriter(log_dir=log_dir)
        except ImportError:
            return None

    def _init_wandb(self, cfg):
        exp_cfg = cfg.get("experiment", {})
        if not exp_cfg.get("use_wandb", False):
            return None
        try:
            import wandb
            wandb.init(
                project=exp_cfg.get("wandb_project", "disentangled-us"),
                name=f"run_{self.run_id:02d}",
                config=cfg,
            )
            return wandb
        except ImportError:
            return None

    # -----------------------------------------------------------------------
    # Warmup schedule
    # -----------------------------------------------------------------------

    def _warmup_lr(self, epoch):
        """Linear warmup for the first `warmup_epochs` epochs."""
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            base_lr = self.cfg["training"].get("lr", 1e-4)
            for pg in self.optimizer.param_groups:
                pg["lr"] = float(base_lr) * (epoch + 1) / self.warmup_epochs

    # -----------------------------------------------------------------------
    # Single epoch
    # -----------------------------------------------------------------------

    def _train_epoch(self, loader, epoch):
        self.model.train()
        self.criterion.train()
        running = {}
        n = 0

        for full_img, bg_img, roi_img, mask_t, labels in loader:
            full_img = full_img.to(self.device, non_blocking=True)
            bg_img   = bg_img.to(self.device, non_blocking=True)
            roi_img  = roi_img.to(self.device, non_blocking=True)
            mask_t   = mask_t.to(self.device, non_blocking=True)
            labels   = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                if self.model_type == "baseline":
                    output = self.model(full_img)
                    loss, loss_dict = self.criterion(output, labels)
                else:
                    mask_arg = mask_t if self.use_inpainting else None
                    output = self.model(full_img, bg_img, roi_img, mask_arg)
                    loss, loss_dict = self.criterion(output, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = labels.size(0)
            n += bs
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v * bs

        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.model.eval()
        self.criterion.eval()
        running = {}
        n = 0

        for full_img, bg_img, roi_img, mask_t, labels in loader:
            full_img = full_img.to(self.device, non_blocking=True)
            bg_img   = bg_img.to(self.device, non_blocking=True)
            roi_img  = roi_img.to(self.device, non_blocking=True)
            mask_t   = mask_t.to(self.device, non_blocking=True)
            labels   = labels.to(self.device, non_blocking=True)

            if self.model_type == "baseline":
                output = self.model(full_img)
                _, loss_dict = self.criterion(output, labels)
            else:
                mask_arg = mask_t if self.use_inpainting else None
                output = self.model(full_img, bg_img, roi_img, mask_arg)
                _, loss_dict = self.criterion(output, labels)

            bs = labels.size(0)
            n += bs
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v * bs

        return {k: v / n for k, v in running.items()}

    # -----------------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, epoch, val_loss, tag="best"):
        path = os.path.join(
            self.ckpt_dir, f"run_{self.run_id:02d}_{tag}.pth"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)
        return path

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from {path} (epoch {ckpt['epoch']}, "
            f"val_loss {ckpt['val_loss']:.4f})"
        )
        return ckpt

    # -----------------------------------------------------------------------
    # Main fit loop
    # -----------------------------------------------------------------------

    def fit(self, train_loader, val_loader):
        logger.info(
            f"Run {self.run_id} | Parameters: "
            f"{count_parameters(self.model):,}"
        )
        best_val_loss = math.inf
        best_ckpt_path = None

        for epoch in range(self.epochs):
            t0 = time.time()
            self._warmup_lr(epoch)

            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._val_epoch(val_loader)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["total"])
                elif epoch >= self.warmup_epochs:
                    self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            log_str = (
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"LR {lr:.2e} | "
                f"Train loss {train_metrics['total']:.4f} | "
                f"Val loss {val_metrics['total']:.4f} | "
                f"{elapsed:.1f}s"
            )
            logger.info(log_str)

            # TensorBoard
            if self.writer:
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("lr", lr, epoch)

            # W&B
            if self.wandb:
                self.wandb.log({
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "lr": lr,
                    "epoch": epoch,
                })

            # Checkpoint best model
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_ckpt_path = self._save_checkpoint(
                    epoch, best_val_loss, tag="best"
                )

            # Early stopping
            if self.early_stop(val_metrics["total"]):
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1}."
                )
                break

        if self.writer:
            self.writer.close()

        # Reload best weights before returning
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            self.load_checkpoint(best_ckpt_path)

        return best_val_loss
