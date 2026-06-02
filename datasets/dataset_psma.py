import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size  # (H, W)
        self.low_res = low_res          # (H, W)

    def _resize_image(self, image, size):
        # size: (H, W), cv2 expects (W, H)
        return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    def _resize_mask(self, mask, size):
        return cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = np.asarray(image)
        label = np.asarray(label)

        image = self._resize_image(image, self.output_size)
        label_hr = self._resize_mask(label, self.output_size)
        label_lr = self._resize_mask(label, self.low_res)

        image = image.astype(np.float32)
        label_hr = label_hr.astype(np.int64)
        label_lr = label_lr.astype(np.int64)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)   # H,W -> 1,H,W
        elif image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label_hr).long(),
            'low_res_label': torch.from_numpy(label_lr).long()
        }


class PSMADataset(Dataset):
    """
    Supports both:
      1) flat:
         split/images/*.{jpg,jpeg,png,npy}
         split/masks/*.{png,jpg,jpeg,npy}
      2) nested:
         split/images/<case>/*.{jpg,jpeg,png,npy}
         split/masks/<case>/*.{png,jpg,jpeg,npy}

    Notes:
    - Images are loaded as numpy arrays with original channel structure.
    - No RGB conversion is applied.
    - Masks are loaded as single-channel label maps when possible.
    """
    IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".npy"]
    MASK_EXTS = [".png", ".jpg", ".jpeg", ".npy"]

    def __init__(self, base_dir, split="train", transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(base_dir, split, "images")
        self.mask_dir = os.path.join(base_dir, split, "masks")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_paths = self._collect_files(self.image_dir, self.IMAGE_EXTS)
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No image files with extensions {self.IMAGE_EXTS} found under {self.image_dir}"
            )

        self.samples = []
        missing_masks = []

        for image_path in self.image_paths:
            mask_path = self._find_matching_mask(image_path)
            if mask_path is None:
                missing_masks.append(image_path)
                continue
            self.samples.append((image_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("No matched image-mask pairs found.")

        if len(missing_masks) > 0:
            print(f"[WARN] {len(missing_masks)} images do not have matching masks and were skipped.")
            print(f"[WARN] Example missing image: {missing_masks[0]}")

    def _collect_files(self, root_dir, extensions):
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
        return sorted(files)

    def _find_matching_mask(self, image_path):
        rel_path = os.path.relpath(image_path, self.image_dir)
        rel_stem = os.path.splitext(rel_path)[0]
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Preferred: preserve relative subfolder structure
        for ext in self.MASK_EXTS:
            candidate = os.path.join(self.mask_dir, rel_stem + ext)
            if os.path.isfile(candidate):
                return candidate

        # Fallback: same basename directly under masks/
        for ext in self.MASK_EXTS:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.isfile(candidate):
                return candidate

        return None

    def _load_image(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            image = np.load(path)
        else:
            # Keep native channel layout from file; no RGB conversion
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to read image: {path}")

            # cv2 loads color images as BGR. Since you said no RGB conversion,
            # we keep the array as loaded.
            # Grayscale stays HxW, color stays HxWxC.

        if image.ndim not in [2, 3]:
            raise ValueError(f"Unsupported image shape {image.shape} for file: {path}")

        # If CHW npy is provided, convert to HWC for consistency before transform
        if image.ndim == 3 and image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3]:
            image = np.transpose(image, (1, 2, 0))

        return image

    def _load_mask(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            mask = np.load(path)
        else:
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"Failed to read mask: {path}")

        if mask.ndim == 3:
            # Expecting label map; use first channel if multi-channel
            mask = mask[..., 0]

        if mask.ndim != 2:
            raise ValueError(f"Unsupported mask shape {mask.shape} for file: {path}")

        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        sample = {
            'image': image,
            'label': mask
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['case_name'] = os.path.basename(image_path)
        return sample