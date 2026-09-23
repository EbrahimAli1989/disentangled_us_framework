"""
Dataset and dataloader builders for the Disentangled US Framework.

Expects:
    data_dir/images.npy   — (N, H, W) or (N, H, W, C) uint8 ultrasound images
    data_dir/masks.npy    — (N, H, W) binary masks (1=tumor ROI)
    data_dir/labels.npy   — (N,) int labels  0=benign  1=malignant
"""

import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

from .augmentation import build_transforms
import cv2
from simple_lama_inpainting import SimpleLama

# Initialize the model globally so it doesn't reload the weights for every image
# Note: The first time you run this, it will take a minute to download the LaMa model weights.
logger = logging.getLogger(__name__)
lama_model = SimpleLama()


class UltrasoundDataset(Dataset):
    """
    Returns (full_img, bg_img, roi_img, label) per sample.

    Two extraction modes are selected via `extract_bg_roi`:

    extract_bg_roi=True  — mask-zeroing mode (default / original behaviour)
        bg_img : full image with the tumor ROI pixels zeroed out
        roi_img: full image with the background pixels zeroed out

    extract_bg_roi=False — bounding-rectangle mode
        roi_img: ROI bounding-rect cropped from the image, resized to 224×224
        bg_img : background tissue patch (same size as ROI box) sampled from
                 a region that does not overlap the mask, resized to 224×224
    """

    def __init__(self, images, masks, labels, transform=None, extract_bg_roi=True):
        self.images = images      # (N, H, W) or (N, H, W, 3)
        self.masks = masks        # (N, H, W)  binary
        self.labels = labels      # (N,)
        self.transform = transform
        self.extract_bg_roi = extract_bg_roi

        mode = "mask-zeroing (extract_bg_roi=True)" if extract_bg_roi \
               else "bounding-rect (extract_bg_roi=False)"
        n_pos = int((labels == 1).sum())
        logger.info(
            "UltrasoundDataset init: %d samples (%d benign / %d malignant) | "
            "mode=%s | img_shape=%s | transform=%s",
            len(labels), len(labels) - n_pos, n_pos,
            mode, str(images.shape), "yes" if transform else "no",
        )


    def __len__(self):
        return len(self.labels)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_pil_rgb(self, arr):
        """Convert numpy array to 3-channel PIL Image."""
        if arr.ndim == 3 and arr.shape[-1] not in (1, 3):
            # CHW format (C, H, W) — PIL expects HWC
            arr = arr.transpose(1, 2, 0)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        return Image.fromarray(arr.astype(np.uint8))

    def _extract_rect_roi(self,img, mask):
        """
        Crop the bounding rectangle of the largest ROI contour, padded by a margin,
        and resize to (224, 224).

        Args:
            img (np.ndarray): The source image (H, W) or (H, W, C).
            mask (np.ndarray): The binary mask (H, W).
            margin (int): Number of pixels to expand the bounding box by.

        Returns:
            roi_crop : (224, 224[, C]) uint8 array
            roi_box  : (x, y, w, h) padded bounding rectangle
        """
        # Get image dimensions up front to ensure we don't pad outside the image bounds
        margin = np.random.randint(0, 10)  # 21 is excluded
        img_h, img_w = img.shape[:2]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # No ROI found — fall back to a centre crop
            size = min(img_h, img_w) // 2
            y0, x0 = (img_h - size) // 2, (img_w - size) // 2
            return cv2.resize(img[y0:y0 + size, x0:x0 + size], (224, 224)), \
                (x0, y0, size, size)

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # 1. Apply the margin to expand the bounding box coordinates
        x1 = x - margin
        y1 = y - margin
        x2 = x + w + margin
        y2 = y + h + margin

        # 2. Clamp the new coordinates to ensure they stay within the image frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        # 3. Calculate the new width and height
        new_w = x2 - x1
        new_h = y2 - y1

        # Clamp to avoid a zero-dimension crop from degenerate contours
        new_w, new_h = max(new_w, 1), max(new_h, 1)

        roi_crop = img[y1:y1 + new_h, x1:x1 + new_w]

        if roi_crop.size == 0:
            # Bounding rect fell outside the image — fall back to centre crop
            size = min(img_h, img_w) // 2
            y0, x0 = (img_h - size) // 2, (img_w - size) // 2
            return cv2.resize(img[y0:y0 + size, x0:x0 + size], (224, 224)), \
                (x0, y0, size, size)

        return cv2.resize(roi_crop, (224, 224)), (x1, y1, new_w, new_h)


    def inpaint_tumor_with_lama(self, image, mask, dilate_iterations=2):
        """
        Removes the tumor using the Deep Learning LaMa model and hallucinates
        highly realistic background tissue.

        Args:
            image (np.ndarray): The source ultrasound image (OpenCV format).
            mask (np.ndarray): The binary mask of the tumor (OpenCV format).
            dilate_iterations (int): Number of times to expand the mask to ensure
                                     border artifacts are completely removed.

        Returns:
            inpainted_image (np.ndarray): The realistically inpainted image (OpenCV format).
        """
        # 1. Standardize the mask to 8-bit single channel (0 and 255)
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 2. Dilate the mask to cover the fuzzy borders of the tumor
        if dilate_iterations > 0:
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=dilate_iterations)

        # 3. Convert OpenCV arrays (BGR/Grayscale) to PIL Images (RGB/L)
        # Convert image from BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Convert mask to single channel 'L' mode for PIL
        pil_mask = Image.fromarray(binary_mask).convert('L')

        # 4. Perform Deep Learning Inpainting using LaMa
        # The model looks at the whole image context and fills the mask region
        result_pil = lama_model(pil_image, pil_mask)

        # 5. Convert the resulting PIL image back to an OpenCV array (BGR)
        result_np = np.array(result_pil)
        result_np = cv2.resize(result_np, (224, 224))
        inpainted_image = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return inpainted_image
    def _extract_bg_patch(self, image, mask, patch_size=(64, 64), target_size=(224, 224), stride=32,
                                          min_tissue_intensity=10):
        """
        Extracts a background patch furthest from the tumor, ensuring it contains tissue, and resizes it.

        Args:
            image (np.ndarray): The source ultrasound image.
            mask (np.ndarray): The tumor mask (same height/width as image).
            patch_size (tuple): The size of the crop to take from the original image (H, W).
            target_size (tuple): The final size to resize the patch to (H, W).
            stride (int): Step size for the sliding window.
            min_tissue_intensity (float): Minimum mean pixel value to be considered tissue.

        Returns:
            resized_patch (np.ndarray): The final 224x224 image patch.
            best_coords (tuple): The (y1, x1, y2, x2) coordinates of the original extraction.
        """

        # 1. Standardize the mask to 8-bit single channel (0 for background, 255 for tumor)
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 2. Compute the Distance Transform
        inverted_mask = cv2.bitwise_not(binary_mask)
        dist_map = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)

        img_h, img_w = image.shape[:2]
        patch_h, patch_w = patch_size

        valid_patches = []

        # 3. Sliding window to find valid background patches
        for y in range(0, img_h - patch_h + 1, stride):
            for x in range(0, img_w - patch_w + 1, stride):

                # Extract the corresponding patches from mask and image
                mask_patch = binary_mask[y:y + patch_h, x:x + patch_w]
                img_patch = image[y:y + patch_h, x:x + patch_w]

                # 4. Check for tumor overlap AND sufficient tissue intensity
                if np.max(mask_patch) == 0 and np.mean(img_patch) >= min_tissue_intensity:
                    # Get the distance score at the center of this patch
                    center_y = y + (patch_h // 2)
                    center_x = x + (patch_w // 2)
                    dist_score = dist_map[center_y, center_x]

                    valid_patches.append({
                        'coords': (y, x, y + patch_h, x + patch_w),
                        'score': dist_score
                    })

        if not valid_patches:
            raise ValueError(
                f"No patches found with mean intensity >= {min_tissue_intensity} that also exclude the tumor. Try reducing 'patch_size' or 'min_tissue_intensity'.")

        # 5. Sort patches by distance score (descending) and select the furthest one
        valid_patches.sort(key=lambda item: item['score'], reverse=True)
        num = np.random.randint(0, max(len(valid_patches) // 20, 1))  # 21 is excluded
        best_patch_info = valid_patches[num]
        y1, x1, y2, x2 = best_patch_info['coords']

        # Extract the patch from the original image
        best_patch = image[y1:y2, x1:x2]

        # 6. Resize the patch to the target size (224x224)
        resized_patch = cv2.resize(best_patch, target_size, interpolation=cv2.INTER_CUBIC)

        return resized_patch #, best_patch_info['coords']
    '''
    def _extract_rect_roi(self, img, mask):
        """
        Crop the bounding rectangle of the largest ROI contour and resize
        to (224, 224).

        Returns:
            roi_crop : (224, 224[, C]) uint8 array
            roi_box  : (x, y, w, h) bounding rectangle
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            # No ROI found — fall back to a centre crop
            H, W = mask.shape
            size = min(H, W) // 2
            y0, x0 = (H - size) // 2, (W - size) // 2
            return cv2.resize(img[y0:y0 + size, x0:x0 + size], (224, 224)), \
                   (x0, y0, size, size)

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Clamp to avoid a zero-dimension crop from degenerate contours
        w, h = max(w, 1), max(h, 1)
        roi_crop = img[y:y + h, x:x + w]
        if roi_crop.size == 0:
            # Bounding rect fell outside the image — fall back to centre crop
            H, W = mask.shape
            size = min(H, W) // 2
            y0, x0 = (H - size) // 2, (W - size) // 2
            return cv2.resize(img[y0:y0 + size, x0:x0 + size], (224, 224)), \
                   (x0, y0, size, size)
        return cv2.resize(roi_crop, (224, 224)), (x, y, w, h)

    def _extract_bg_patch(self, img, mask, roi_box,
                          patch_w, patch_h,
                          margin=50, max_tries=300):
        """
        Three-phase background patch extraction.  Each phase tries up to
        `max_tries` random positions (total budget ≤ 3 × max_tries).

        Phase 1a — strict:   no mask overlap  +  far from ROI  +  tissue
        Phase 1b — relaxed:  no mask overlap  +  tissue          (distance
                             constraint dropped for large/central ROIs where
                             Phase 1a is geometrically impossible)
        Phase 2  — fallback: sample freely, keep minimum-overlap position,
                             zero out the tumor pixels that remain.

        Mask values are binarised (> 0 → 1) before any check so the method
        is correct regardless of whether the stored mask uses 0/1 or 0/255.
        """
        x, y, w, h = roi_box
        H_img, W_img = mask.shape

        patch_w = min(max(patch_w, 1), W_img)
        patch_h = min(max(patch_h, 1), H_img)

        max_rx = max(1, W_img - patch_w + 1)
        max_ry = max(1, H_img - patch_h + 1)

        # Binarise once — handles 0/1, 0/255, and float masks uniformly.
        bin_mask = (mask > 0).astype(np.uint8)

        def _sample():
            return np.random.randint(0, max_rx), np.random.randint(0, max_ry)

        def _overlap(rx, ry):
            return int(bin_mask[ry:ry + patch_h, rx:rx + patch_w].sum())

        def _is_tissue(rx, ry):
            return np.mean(img[ry:ry + patch_h, rx:rx + patch_w]) >= 10

        def _is_far(rx, ry):
            return (abs(rx - x) > w + margin) or (abs(ry - y) > h + margin)

        # ── Phase 1a: strict (no overlap + far + tissue) ────────────────
        for _ in range(max_tries):
            rx, ry = _sample()
            if _overlap(rx, ry) == 0 and _is_far(rx, ry) and _is_tissue(rx, ry):
                return cv2.resize(
                    img[ry:ry + patch_h, rx:rx + patch_w], (224, 224)
                )

        # ── Phase 1b: relaxed (no overlap + tissue, no distance req.) ───
        # Needed when the ROI is large relative to the image and no position
        # can satisfy both zero-overlap AND the spatial margin simultaneously.
        for _ in range(max_tries):
            rx, ry = _sample()
            if _overlap(rx, ry) == 0 and _is_tissue(rx, ry):
                return cv2.resize(
                    img[ry:ry + patch_h, rx:rx + patch_w], (224, 224)
                )

        # ── Phase 2: minimum-overlap fallback ───────────────────────────
        logger.warning(
            "bg_patch: no clean patch after %d attempts "
            "(roi_box=(%d,%d,%d,%d), patch=%dx%d, image=%dx%d). "
            "Using minimum-overlap fallback.",
            2 * max_tries, x, y, w, h, patch_w, patch_h, W_img, H_img,
        )

        best_rx, best_ry = 0, 0
        best_overlap = _overlap(0, 0)

        for _ in range(max_tries):
            rx, ry = _sample()
            if not _is_tissue(rx, ry):
                continue
            ov = _overlap(rx, ry)
            if ov < best_overlap:
                best_overlap = ov
                best_rx, best_ry = rx, ry
                if ov == 0:
                    break

        patch = img[best_ry:best_ry + patch_h,
                    best_rx:best_rx + patch_w].copy()
        patch_mask = bin_mask[best_ry:best_ry + patch_h,
                               best_rx:best_rx + patch_w]
        total_px = patch_w * patch_h
        overlap_pct = 100.0 * best_overlap / total_px

        logger.info(
            "bg_patch: fallback at (%d,%d), overlap=%d/%d px (%.1f%%)%s.",
            best_rx, best_ry, best_overlap, total_px, overlap_pct,
            " — tumor pixels zeroed" if best_overlap > 0 else " — no overlap",
        )

        if best_overlap > 0:
            patch[patch_mask == 1] = 0  # works for HW and HWC

        return cv2.resize(patch, (224, 224))
    '''
    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        img   = self.images[idx]
        mask  = self.masks[idx]       # H × W, values 0/1
        label = int(self.labels[idx])

        logger.debug(
            "getitem[%d]: img=%s dtype=%s mask_sum=%d label=%d",
            idx, img.shape, img.dtype, int((mask > 0).sum()), label,
        )

        # mask_f: (H, W) float for broadcasting; 1 = tumor ROI
        mask_f = mask.astype(np.float32)

        if self.extract_bg_roi:
            # ── Mask-zeroing mode: x_bg = x ⊙ (1−m),  x_roi = x ⊙ m ──
            if img.ndim == 3 and img.shape[0] != mask.shape[0]:
                # CHW: broadcast over channel dim
                bg_arr  = (img * (1.0 - mask_f[np.newaxis])).astype(img.dtype)
                roi_arr = (img *        mask_f[np.newaxis] ).astype(img.dtype)
            elif img.ndim == 3:
                # HWC
                bg_arr  = (img * (1.0 - mask_f[:, :, np.newaxis])).astype(img.dtype)
                roi_arr = (img *        mask_f[:, :, np.newaxis] ).astype(img.dtype)
            else:
                # HW grayscale
                bg_arr  = (img * (1.0 - mask_f)).astype(img.dtype)
                roi_arr = (img *        mask_f ).astype(img.dtype)
        else:
            # ── Bounding-rectangle mode ─────────────────────────────────
            roi_arr, roi_box = self._extract_rect_roi(img, mask)
            #bg_arr = self._extract_bg_patch(
            #    img, mask, roi_box,
            #    patch_w=roi_box[2], patch_h=roi_box[3],
            #)
            bg_arr = self.inpaint_tumor_with_lama(img, mask)

        full_pil = self._to_pil_rgb(img)
        bg_pil   = self._to_pil_rgb(bg_arr)
        roi_pil  = self._to_pil_rgb(roi_arr)

        logger.debug(
            "getitem[%d]: PIL sizes full=%s bg=%s roi=%s — applying transform",
            idx, full_pil.size, bg_pil.size, roi_pil.size,
        )

        if self.transform is not None:
            full_t = self.transform(full_pil)
            bg_t   = self.transform(bg_pil)
            roi_t  = self.transform(roi_pil)
        else:
            full_t = torch.from_numpy(
                np.array(full_pil).transpose(2, 0, 1)
            ).float() / 255.0
            bg_t = torch.from_numpy(
                np.array(bg_pil).transpose(2, 0, 1)
            ).float() / 255.0
            roi_t = torch.from_numpy(
                np.array(roi_pil).transpose(2, 0, 1)
            ).float() / 255.0

        logger.debug(
            "getitem[%d]: tensors full=%s bg=%s roi=%s label=%d",
            idx, tuple(full_t.shape), tuple(bg_t.shape),
            tuple(roi_t.shape), label,
        )

        # mask_t: (1, H, W) float — available for learnable inpainting in the model
        mask_t = torch.from_numpy(mask_f).unsqueeze(0)

        return full_t, bg_t, roi_t, mask_t, torch.tensor(label, dtype=torch.long)


def _worker_init_fn(worker_id):
    """
    Called once per DataLoader worker process at startup.

    1. Disables OpenCV's internal TBB thread pool so that cv2 operations
       (findContours, resize, CLAHE …) are single-threaded inside forked
       workers.  Without this, forked workers inherit a partially-initialised
       TBB state that can deadlock silently inside _train_epoch.
    2. Re-seeds NumPy so each worker draws different random samples.
    """
    cv2.setNumThreads(0)
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    logger.debug("DataLoader worker %d ready (seed=%d).", worker_id, worker_seed)


def build_dataloaders(cfg, seed=42):
    """
    Build stratified train/val/test DataLoaders.

    The extraction mode is controlled by ``cfg["data"]["extract_bg_roi"]``:
        True  (default) — mask-zeroing mode (fast, no cv2 in __getitem__)
        False           — bounding-rectangle + background-patch mode

    Args:
        cfg: full configuration dict
        seed: random seed for splitting

    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg   = cfg["data"]
    data_dir   = data_cfg["data_dir"]
    # Read extraction mode from config; default to True (mask-zeroing).
    extract_bg_roi = data_cfg.get("extract_bg_roi", True)

    logger.info(
        "build_dataloaders: data_dir=%s | extract_bg_roi=%s | seed=%d",
        data_dir, extract_bg_roi, seed,
    )

    images = np.load(os.path.join(data_dir, "Breast_images_B.npy"))
    masks  = np.load(os.path.join(data_dir, "Breast_masks_B.npy"))
    labels = np.load(os.path.join(data_dir, "Breast_label_B.npy"))

    logger.info(
        "Loaded arrays — images=%s masks=%s labels=%s unique_labels=%s",
        images.shape, masks.shape, labels.shape, np.unique(labels).tolist(),
    )

    train_r = data_cfg.get("train_ratio", 0.70)
    val_r   = data_cfg.get("val_ratio",   0.15)
    test_r  = data_cfg.get("test_ratio",  0.15)

    idx = np.arange(len(labels))

    # First split: train vs (val+test)
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, labels, test_size=(val_r + test_r), stratify=labels, random_state=seed
    )

    # Second split: val vs test
    val_frac = val_r / (val_r + test_r)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=(1 - val_frac), stratify=y_tmp, random_state=seed
    )

    logger.info(
        "Split sizes — train=%d val=%d test=%d",
        len(idx_train), len(idx_val), len(idx_test),
    )

    train_tf = build_transforms(cfg, split="train")
    val_tf   = build_transforms(cfg, split="val")

    train_ds = UltrasoundDataset(
        images[idx_train], masks[idx_train], labels[idx_train],
        transform=train_tf, extract_bg_roi=extract_bg_roi,
    )
    val_ds = UltrasoundDataset(
        images[idx_val], masks[idx_val], labels[idx_val],
        transform=val_tf, extract_bg_roi=extract_bg_roi,
    )
    test_ds = UltrasoundDataset(
        images[idx_test], masks[idx_test], labels[idx_test],
        transform=val_tf, extract_bg_roi=extract_bg_roi,
    )

    num_workers = data_cfg.get("num_workers", 4)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        worker_init_fn=_worker_init_fn,
        # persistent_workers keeps the pool alive between epochs,
        # avoiding repeated fork overhead (requires num_workers > 0).
        persistent_workers=(num_workers > 0),
    )

    logger.info("DataLoader: num_workers=%d pin_memory=%s",
                num_workers, loader_kwargs["pin_memory"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    logger.info(
        "DataLoaders ready — train=%d batches | val=%d batches | test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader
