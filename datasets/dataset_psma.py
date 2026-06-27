import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    # image: HWC, rotate over spatial dims only
    image = ndimage.rotate(image, angle, axes=(0, 1), order=1, reshape=False)
    # label: HW
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def _resize_image(image, output_size):
    """
    image: numpy array, HWC
    output_size: (H, W)
    """
    h, w = image.shape[:2]
    if (h, w) == tuple(output_size):
        return image
    zoom_factors = (output_size[0] / h, output_size[1] / w, 1)
    return zoom(image, zoom_factors, order=3)


def _resize_label(label, output_size):
    """
    label: numpy array, HW
    output_size: (H, W)
    """
    h, w = label.shape
    if (h, w) == tuple(output_size):
        return label
    zoom_factors = (output_size[0] / h, output_size[1] / w)
    return zoom(label, zoom_factors, order=0)


class TrainTransform(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = _resize_image(image, self.output_size)
        label = _resize_label(label, self.output_size)

        low_res_label = _resize_label(label, self.low_res)

        # HWC -> CHW
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        if image.max() > 1.0:
            image = image / 255.0

        label = torch.from_numpy(label.astype(np.int64))
        low_res_label = torch.from_numpy(low_res_label.astype(np.int64))

        return {
            "image": image,
            "label": label,
            "low_res_label": low_res_label,
        }


class ValTransform(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        image = _resize_image(image, self.output_size)
        label = _resize_label(label, self.output_size)
        low_res_label = _resize_label(label, self.low_res)

        # HWC -> CHW
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        if image.max() > 1.0:
            image = image / 255.0

        label = torch.from_numpy(label.astype(np.int64))
        low_res_label = torch.from_numpy(low_res_label.astype(np.int64))

        return {
            "image": image,
            "label": label,
            "low_res_label": low_res_label,
        }


class PSMADataset(Dataset):
    """
    Expected folder structure:
    dataset/
        train/
            images/
                cats/cat_001.jpg
                dogs/dog_001.png
            masks/
                cats/cat_001.png
                dogs/dog_001.png
        val/
            images/
                cats/cat_101.jpg
                dogs/dog_101.jpg
            masks/
                cats/cat_101.png
                dogs/dog_101.png

    Notes:
    - Image files can be .jpg/.jpeg/.png
    - Mask files are matched by relative path stem, regardless of extension
    - Masks are loaded as single-channel integer label maps
    """

    def __init__(self, base_dir, split="train", transform=None):
        assert split in ["train", "val"], f"Unsupported split: {split}"
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform

        self.image_dir = self.base_dir / split / "images"
        self.mask_dir = self.base_dir / split / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        image_paths = []

        for p in self.image_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                image_paths.append(p)

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise RuntimeError(f"No image files found in {self.image_dir}")

        for img_path in image_paths:
            rel_path = img_path.relative_to(self.image_dir)
            rel_stem = rel_path.with_suffix("")

            # search corresponding mask with any supported extension
            mask_path = None
            for ext in IMG_EXTENSIONS:
                candidate = (self.mask_dir / rel_stem).with_suffix(ext)
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                raise FileNotFoundError(
                    f"No matching mask found for image: {img_path}. "
                    f"Expected under {self.mask_dir / rel_stem} with one of {IMG_EXTENSIONS}"
                )

            samples.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "case_name": str(rel_stem),
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        image = Image.open(sample_info["image_path"]).convert("RGB")
        mask = Image.open(sample_info["mask_path"])

        image = np.array(image)          # HWC, uint8
        label = np.array(mask)           # HW, uint8/int

        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        sample["case_name"] = sample_info["case_name"]
        return sample