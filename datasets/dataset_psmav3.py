import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom, gaussian_filter, map_coordinates
from torch.utils.data import Dataset
from torchvision import transforms as T1

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ----------------------------
# Basic spatial helpers
# ----------------------------
def random_rotate(image, label, angle_range=(-30, 30)):
    """
    image: HWC
    label: HW
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    image = ndimage.rotate(image, angle, axes=(0, 1), order=1, reshape=False, mode="nearest")
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="nearest")
    return image, label


def random_mirror(image, label):
    """
    2D axis-wise flipping.
    image: HWC
    label: HW
    """
    axis = np.random.randint(0, 2)  # 0: vertical, 1: horizontal
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_scale(image, label, scale_range=(0.7, 1.4), anisotropic=False):
    """
    Random scaling then resize back to original size.
    image: HWC
    label: HW
    """
    h, w = image.shape[:2]

    if anisotropic:
        sy = np.random.uniform(scale_range[0], scale_range[1])
        sx = np.random.uniform(scale_range[0], scale_range[1])
    else:
        s = np.random.uniform(scale_range[0], scale_range[1])
        sy, sx = s, s

    scaled_img = zoom(image, (sy, sx, 1), order=1)
    scaled_lbl = zoom(label, (sy, sx), order=0)

    scaled_img = _fit_to_size_image(scaled_img, (h, w))
    scaled_lbl = _fit_to_size_label(scaled_lbl, (h, w))

    return scaled_img, scaled_lbl


def elastic_deformation(image, label, alpha=20.0, sigma=4.0):
    """
    2D elastic deformation.
    image: HWC
    label: HW
    """
    h, w = label.shape

    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode="reflect") * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (y + dy, x + dx)

    deformed_img = np.empty_like(image)
    for c in range(image.shape[2]):
        deformed_img[..., c] = map_coordinates(
            image[..., c],
            indices,
            order=1,
            mode="reflect"
        )

    deformed_lbl = map_coordinates(label, indices, order=0, mode="reflect")

    return deformed_img, deformed_lbl


def random_crop(image, label, crop_size, foreground_oversample_prob=0.0):
    """
    image: HWC
    label: HW
    crop_size: (crop_h, crop_w)

    If foreground oversampling is triggered and foreground exists,
    choose crop center around a foreground pixel.
    """
    h, w = label.shape
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        image = _pad_if_needed_image(image, crop_size)
        label = _pad_if_needed_label(label, crop_size)
        h, w = label.shape

    use_fg = random.random() < foreground_oversample_prob
    fg_coords = np.argwhere(label > 0)

    if use_fg and len(fg_coords) > 0:
        cy, cx = fg_coords[np.random.randint(len(fg_coords))]
        y1 = max(0, cy - crop_h // 2)
        x1 = max(0, cx - crop_w // 2)
        y1 = min(y1, h - crop_h)
        x1 = min(x1, w - crop_w)
    else:
        y1 = np.random.randint(0, h - crop_h + 1)
        x1 = np.random.randint(0, w - crop_w + 1)

    y2 = y1 + crop_h
    x2 = x1 + crop_w

    image = image[y1:y2, x1:x2, :]
    label = label[y1:y2, x1:x2]
    return image, label


# ----------------------------
# Intensity transforms
# ----------------------------
def add_gaussian_noise(image, std_range=(0.0, 0.05)):
    """
    image expected in [0, 255] uint8 or float-like image.
    Applied in float32, returned as float32 in [0, 255].
    """
    img = image.astype(np.float32)
    std = np.random.uniform(std_range[0], std_range[1]) * 255.0
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    return img


def add_gaussian_blur(image, sigma_range=(0.5, 1.5)):
    img = image.astype(np.float32)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    if img.ndim == 3:
        blurred = np.empty_like(img)
        for c in range(img.shape[2]):
            blurred[..., c] = gaussian_filter(img[..., c], sigma=sigma)
        return np.clip(blurred, 0, 255)
    else:
        return np.clip(gaussian_filter(img, sigma=sigma), 0, 255)


def adjust_brightness_contrast(image, brightness_range=(0.75, 1.25), contrast_range=(0.75, 1.25)):
    img = image.astype(np.float32)
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])

    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast + mean
    img = img * brightness
    img = np.clip(img, 0, 255)
    return img


def gamma_correction(image, gamma_range=(0.7, 1.5)):
    img = image.astype(np.float32) / 255.0
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img = np.power(np.clip(img, 0, 1), gamma)
    img = np.clip(img * 255.0, 0, 255)
    return img


# ----------------------------
# Resize / fit / pad helpers
# ----------------------------
def _resize_image(image, output_size):
    """
    image: HWC
    output_size: (H, W)
    """
    h, w = image.shape[:2]
    if (h, w) == tuple(output_size):
        return image
    zoom_factors = (output_size[0] / h, output_size[1] / w, 1)
    return zoom(image, zoom_factors, order=3)


def _resize_label(label, output_size):
    """
    label: HW
    output_size: (H, W)
    """
    h, w = label.shape
    if (h, w) == tuple(output_size):
        return label
    zoom_factors = (output_size[0] / h, output_size[1] / w)
    return zoom(label, zoom_factors, order=0)


def _pad_if_needed_image(image, output_size):
    h, w = image.shape[:2]
    out_h, out_w = output_size
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)

    if pad_h == 0 and pad_w == 0:
        return image

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="reflect"
    )


def _pad_if_needed_label(label, output_size):
    h, w = label.shape
    out_h, out_w = output_size
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)

    if pad_h == 0 and pad_w == 0:
        return label

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return np.pad(
        label,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    )


def _fit_to_size_image(image, output_size):
    """
    Center crop or pad to target size.
    """
    image = _pad_if_needed_image(image, output_size)
    h, w = image.shape[:2]
    out_h, out_w = output_size

    y1 = max(0, (h - out_h) // 2)
    x1 = max(0, (w - out_w) // 2)
    return image[y1:y1 + out_h, x1:x1 + out_w, :]


def _fit_to_size_label(label, output_size):
    """
    Center crop or pad to target size.
    """
    label = _pad_if_needed_label(label, output_size)
    h, w = label.shape
    out_h, out_w = output_size

    y1 = max(0, (h - out_h) // 2)
    x1 = max(0, (w - out_w) // 2)
    return label[y1:y1 + out_h, x1:x1 + out_w]


# ----------------------------
# Train / val transforms
# ----------------------------
class TrainTransform(object):
    def __init__(
        self,
        output_size,
        low_res,
        # spatial probabilities
        p_random_crop=1.0,
        p_rotation=0.5,
        p_scaling=0.5,
        p_elastic=0.1,
        p_mirroring=0.5,
        # intensity probabilities
        p_gaussian_noise=0.15,
        p_gaussian_blur=0.15,
        p_brightness_contrast=0.1,
        p_gamma=0.2,
        # parameter ranges
        crop_size=None,
        foreground_oversample_prob=0.8,
        rotation_range=(-10, 10),
        scale_range=(0.7, 1.4),
        anisotropic_scale=False,
        elastic_alpha=20.0,
        elastic_sigma=4.0,
        noise_std_range=(0.0, 0.05),
        blur_sigma_range=(0.5, 1.5),
        brightness_range=(0.75, 1.25),
        contrast_range=(0.75, 1.25),
        gamma_range=(0.7, 1.5),
    ):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)
        self.crop_size = tuple(crop_size) if crop_size is not None else tuple(output_size)

        self.p_random_crop = p_random_crop
        self.p_rotation = p_rotation
        self.p_scaling = p_scaling
        self.p_elastic = p_elastic
        self.p_mirroring = p_mirroring

        self.p_gaussian_noise = p_gaussian_noise
        self.p_gaussian_blur = p_gaussian_blur
        self.p_brightness_contrast = p_brightness_contrast
        self.p_gamma = p_gamma

        self.foreground_oversample_prob = foreground_oversample_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.anisotropic_scale = anisotropic_scale
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.noise_std_range = noise_std_range
        self.blur_sigma_range = blur_sigma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range

        self.to_tensor = T1.ToTensor()

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # 1) Sampling / patch op: random crop with optional foreground oversampling
        if random.random() < self.p_random_crop:
            image, label = random_crop(
                image,
                label,
                crop_size=self.crop_size,
                foreground_oversample_prob=self.foreground_oversample_prob,
            )

        # 2) Spatial transforms
        if random.random() < self.p_rotation:
            image, label = random_rotate(image, label, angle_range=self.rotation_range)

        if random.random() < self.p_scaling:
            image, label = random_scale(
                image,
                label,
                scale_range=self.scale_range,
                anisotropic=self.anisotropic_scale,
            )

        if random.random() < self.p_elastic:
            image, label = elastic_deformation(
                image,
                label,
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma,
            )

        if random.random() < self.p_mirroring:
            image, label = random_mirror(image, label)

        # 3) Intensity transforms (image only)
        if random.random() < self.p_gaussian_noise:
            image = add_gaussian_noise(image, std_range=self.noise_std_range)

        if random.random() < self.p_gaussian_blur:
            image = add_gaussian_blur(image, sigma_range=self.blur_sigma_range)

        if random.random() < self.p_brightness_contrast:
            image = adjust_brightness_contrast(
                image,
                brightness_range=self.brightness_range,
                contrast_range=self.contrast_range,
            )

        if random.random() < self.p_gamma:
            image = gamma_correction(image, gamma_range=self.gamma_range)

        # final resize to standard shape
        image = _resize_image(image, self.output_size)
        label = _resize_label(label, self.output_size)
        low_res_label = _resize_label(label, self.low_res)

        # image: use ToTensor as requested
        image = self.to_tensor(image.astype(np.uint8) if image.dtype != np.uint8 else image)

        # labels should remain integer class maps
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
        self.to_tensor = T1.ToTensor()

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        image = _resize_image(image, self.output_size)
        label = _resize_label(label, self.output_size)
        low_res_label = _resize_label(label, self.low_res)

        image = self.to_tensor(image.astype(np.uint8) if image.dtype != np.uint8 else image)
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

        image = np.array(image)   # HWC, uint8
        label = np.array(mask)    # HW, uint8/int

        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        sample["case_name"] = sample_info["case_name"]
        return sample