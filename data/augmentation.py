"""
Augmentation pipeline for ultrasound medical imaging.

Augmentations preserve medical image realism — no color jitter
that alters grayscale semantics, elastic deformation bounded to
avoid anatomical distortion.
"""

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization on PIL Image."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        try:
            import cv2
            img_np = np.array(img)
            if img_np.ndim == 2:
                clahe = cv2.createCLAHE(
                    clipLimit=self.clip_limit,
                    tileGridSize=self.tile_grid_size
                )
                img_np = clahe.apply(img_np)
            else:
                # Apply to L channel in LAB
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(
                    clipLimit=self.clip_limit,
                    tileGridSize=self.tile_grid_size
                )
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_np)
        except ImportError:
            return img


class ElasticDeformation:
    """
    Elastic deformation via random displacement fields.
    Bounded alpha/sigma to prevent clinically unrealistic warping.
    """

    def __init__(self, alpha=34, sigma=4, prob=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates
            img_np = np.array(img).astype(np.float32)
            shape = img_np.shape[:2]
            dx = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), self.sigma
            ) * self.alpha
            dy = gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), self.sigma
            ) * self.alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = (
                np.reshape(y + dy, (-1, 1)),
                np.reshape(x + dx, (-1, 1)),
            )
            if img_np.ndim == 3:
                channels = [
                    map_coordinates(img_np[:, :, c], indices, order=1, mode='reflect')
                    .reshape(shape)
                    for c in range(img_np.shape[2])
                ]
                distorted = np.stack(channels, axis=-1)
            else:
                distorted = map_coordinates(
                    img_np, indices, order=1, mode='reflect'
                ).reshape(shape)
            return Image.fromarray(np.clip(distorted, 0, 255).astype(np.uint8))
        except ImportError:
            return img


class GaussianNoise:
    """Additive Gaussian noise in pixel space (medical-realistic std)."""

    def __init__(self, std=0.02, prob=0.5):
        self.std = std
        self.prob = prob

    def __call__(self, tensor):
        if random.random() < self.prob:
            noise = torch.randn_like(tensor) * self.std
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


class RandomBlur:
    """Random Gaussian blur — simulates transducer focus variation."""

    def __init__(self, kernel_range=(3, 7), prob=0.3):
        self.kernel_range = kernel_range
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            k = random.choice(
                range(self.kernel_range[0], self.kernel_range[1] + 1, 2)
            )
            return img.filter(ImageFilter.GaussianBlur(radius=k // 2))
        return img


def build_transforms(cfg, split="train"):
    """
    Build torchvision transform pipeline from config.

    Args:
        cfg: augmentation config dict (from default.yaml augmentation section)
        split: 'train' | 'val' | 'test'

    Returns:
        transform callable
    """
    aug = cfg.get("augmentation", {})
    image_size = cfg.get("data", {}).get("image_size", 224)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split != "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    train_cfg = aug.get("train", {})
    transforms = []

    transforms.append(T.Resize((image_size, image_size)))

    if train_cfg.get("elastic_deform", False):
        transforms.append(ElasticDeformation(
            alpha=train_cfg.get("elastic_alpha", 34),
            sigma=train_cfg.get("elastic_sigma", 4),
            prob=0.4,
        ))

    if train_cfg.get("clahe", False):
        transforms.append(CLAHE())

    if train_cfg.get("blur_prob", 0) > 0:
        transforms.append(RandomBlur(
            kernel_range=train_cfg.get("blur_kernel", [3, 7]),
            prob=train_cfg.get("blur_prob", 0.3),
        ))

    if train_cfg.get("horizontal_flip", False):
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    if train_cfg.get("vertical_flip", False):
        transforms.append(T.RandomVerticalFlip(p=0.3))

    if train_cfg.get("rotation_degrees", 0) > 0:
        transforms.append(T.RandomRotation(
            degrees=train_cfg.get("rotation_degrees", 15)
        ))

    if train_cfg.get("affine", False):
        transforms.append(T.RandomAffine(
            degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
        ))

    if train_cfg.get("random_crop", False):
        scale = train_cfg.get("crop_scale", [0.8, 1.0])
        transforms.append(T.RandomResizedCrop(
            size=image_size, scale=tuple(scale), ratio=(0.9, 1.1)
        ))

    transforms.append(T.ToTensor())

    transforms.append(GaussianNoise(
        std=train_cfg.get("gaussian_noise_std", 0.02), prob=0.4
    ))

    bj = train_cfg.get("brightness_jitter", 0)
    cj = train_cfg.get("contrast_jitter", 0)
    if bj > 0 or cj > 0:
        transforms.append(T.ColorJitter(brightness=bj, contrast=cj))

    transforms.append(T.Normalize(mean=imagenet_mean, std=imagenet_std))

    return T.Compose(transforms)
