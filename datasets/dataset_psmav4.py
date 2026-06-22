import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom, gaussian_filter, map_coordinates, binary_dilation
from torch.utils.data import Dataset
from torchvision import transforms as T1

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ----------------------------
# Basic spatial helpers
# ----------------------------
def random_rotate(image, label, angle_range=(-7, 7)):
    """
    image: HWC
    label: HW
    Small-angle rotation to avoid destroying tiny lesions.
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    image = ndimage.rotate(
        image, angle, axes=(0, 1), order=1, reshape=False, mode="nearest"
    )
    label = ndimage.rotate(
        label, angle, order=0, reshape=False, mode="nearest"
    )
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


def random_scale(image, label, scale_range=(0.9, 1.1), anisotropic=False):
    """
    Random scaling then resize back to original size.
    Conservative scaling for small-lesion preservation.
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

    new_h = max(1, int(round(h * sy)))
    new_w = max(1, int(round(w * sx)))

    scaled_img = _resize_image(image, (new_h, new_w))
    scaled_lbl = _resize_label(label, (new_h, new_w))

    scaled_img = _fit_to_size_image(scaled_img, (h, w))
    scaled_lbl = _fit_to_size_label(scaled_lbl, (h, w))

    return scaled_img, scaled_lbl


def elastic_deformation(image, label, alpha=6.0, sigma=8.0):
    """
    Very mild 2D elastic deformation.
    For small-lesion tasks this should generally be low-probability or disabled.
    image: HWC
    label: HW
    """
    h, w = label.shape

    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode="reflect") * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (y + dy, x + dx)

    deformed_img = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        deformed_img[..., c] = map_coordinates(
            image[..., c],
            indices,
            order=1,
            mode="reflect"
        )

    deformed_lbl = map_coordinates(label, indices, order=0, mode="reflect")

    return deformed_img, deformed_lbl


def random_crop_lesion_aware(
    image,
    label,
    crop_size,
    foreground_oversample_prob=0.9,
    lesion_dilation_radius=5,
    force_all_fg_classes=False,
):
    """
    image: HWC
    label: HW
    crop_size: (crop_h, crop_w)

    Lesion-aware crop:
    - with probability foreground_oversample_prob, sample crop around lesion
    - optionally dilate lesion map so tiny lesions get contextual sampling
    - if no foreground exists, fallback to random crop
    """
    h, w = label.shape
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        image = _pad_if_needed_image(image, crop_size)
        label = _pad_if_needed_label(label, crop_size)
        h, w = label.shape

    use_fg = random.random() < foreground_oversample_prob

    if use_fg:
        fg_mask = label > 0

        if lesion_dilation_radius > 0 and fg_mask.any():
            structure = np.ones((2 * lesion_dilation_radius + 1, 2 * lesion_dilation_radius + 1), dtype=bool)
            fg_mask = binary_dilation(fg_mask, structure=structure)

        fg_coords = np.argwhere(fg_mask)
    else:
        fg_coords = np.empty((0, 2), dtype=np.int64)

    if len(fg_coords) > 0:
        cy, cx = fg_coords[np.random.randint(len(fg_coords))]
        # add random offset so crop is not always lesion-centered
        offset_y = np.random.randint(-crop_h // 6, crop_h // 6 + 1)
        offset_x = np.random.randint(-crop_w // 6, crop_w // 6 + 1)
        cy = np.clip(cy + offset_y, 0, h - 1)
        cx = np.clip(cx + offset_x, 0, w - 1)

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
def add_gaussian_noise(image, std_range=(0.0, 0.02)):
    """
    Mild noise for small-lesion preservation.
    Works on float or uint8-like inputs.
    Returns float32 in original dynamic range assumption [0, 255] if image.max()>1.5,
    otherwise approximately [0, 1].
    """
    img = image.astype(np.float32)
    scale = 255.0 if img.max() > 1.5 else 1.0
    std = np.random.uniform(std_range[0], std_range[1]) * scale
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, scale)
    return img


def add_gaussian_blur(image, sigma_range=(0.2, 0.8)):
    """
    Mild blur only. Strong blur can erase tiny lesions.
    """
    img = image.astype(np.float32)
    scale = 255.0 if img.max() > 1.5 else 1.0
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    if img.ndim == 3:
        blurred = np.empty_like(img)
        for c in range(img.shape[2]):
            blurred[..., c] = gaussian_filter(img[..., c], sigma=sigma)
        return np.clip(blurred, 0, scale)
    else:
        return np.clip(gaussian_filter(img, sigma=sigma), 0, scale)


def adjust_brightness_contrast(
    image,
    brightness_range=(0.9, 1.1),
    contrast_range=(0.9, 1.1)
):
    """
    Conservative intensity perturbation.
    """
    img = image.astype(np.float32)
    scale = 255.0 if img.max() > 1.5 else 1.0

    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])

    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast + mean
    img = img * brightness
    img = np.clip(img, 0, scale)
    return img


def gamma_correction(image, gamma_range=(0.9, 1.1)):
    """
    Mild gamma correction only.
    """
    img = image.astype(np.float32)
    scale = 255.0 if img.max() > 1.5 else 1.0
    img = img / scale
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img = np.power(np.clip(img, 0, 1), gamma)
    img = np.clip(img * scale, 0, scale)
    return img


# ----------------------------
# Resize / fit / pad helpers
# ----------------------------
def _resize_image(image, output_size):
    """
    image: HWC
    output_size: (H, W)

    Uses bicubic interpolation for image resizing.
    """
    h, w = image.shape[:2]
    if (h, w) == tuple(output_size):
        return image

    out_h, out_w = output_size

    # PIL expects uint8/float-compatible arrays; keep dtype behavior simple and safe
    if image.shape[2] == 1:
        img_2d = image[..., 0]
        pil_img = Image.fromarray(img_2d.astype(np.float32), mode="F")
        resized = pil_img.resize((out_w, out_h), resample=Image.BICUBIC)
        resized = np.array(resized, dtype=np.float32)[..., None]
    else:
        # If image is float, PIL RGB handling is awkward; convert channel-wise
        channels = []
        for c in range(image.shape[2]):
            pil_ch = Image.fromarray(image[..., c].astype(np.float32), mode="F")
            ch_resized = pil_ch.resize((out_w, out_h), resample=Image.BICUBIC)
            channels.append(np.array(ch_resized, dtype=np.float32))
        resized = np.stack(channels, axis=-1)

    return resized


def _resize_label(label, output_size):
    """
    label: HW
    output_size: (H, W)

    Uses nearest-neighbor interpolation for segmentation masks.
    """
    h, w = label.shape
    if (h, w) == tuple(output_size):
        return label

    out_h, out_w = output_size
    pil_lbl = Image.fromarray(label.astype(np.int32), mode="I")
    resized = pil_lbl.resize((out_w, out_h), resample=Image.NEAREST)
    return np.array(resized, dtype=label.dtype)


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
# Tensor conversion helper
# ----------------------------
def image_to_tensor(image):
    """
    Converts HWC numpy image to CHW torch.float32 tensor.
    Keeps float precision instead of forcing uint8.
    If input looks like [0,255], normalize to [0,1].
    """
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


# ----------------------------
# Train / val transforms
# ----------------------------
class TrainTransformWholeBodyLesion(object):
    def __init__(
        self,
        output_size,
        low_res,
        # sampling
        crop_size=None,
        p_random_crop=1.0,
        foreground_oversample_prob=0.9,
        lesion_dilation_radius=5,
        # spatial probabilities
        p_rotation=0.4,
        p_scaling=0.3,
        p_elastic=0.0,   # default off for small lesions
        p_mirroring=0.,
        # intensity probabilities
        p_gaussian_noise=0.10,
        p_gaussian_blur=0.08,
        p_brightness_contrast=0.10,
        p_gamma=0.10,
        # parameter ranges
        rotation_range=(-7, 7),
        scale_range=(0.9, 1.1),
        anisotropic_scale=False,
        elastic_alpha=6.0,
        elastic_sigma=8.0,
        noise_std_range=(0.0, 0.02),
        blur_sigma_range=(0.2, 0.8),
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        gamma_range=(0.9, 1.1),
    ):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)
        self.crop_size = tuple(crop_size) if crop_size is not None else tuple(output_size)

        self.p_random_crop = p_random_crop
        self.foreground_oversample_prob = foreground_oversample_prob
        self.lesion_dilation_radius = lesion_dilation_radius

        self.p_rotation = p_rotation
        self.p_scaling = p_scaling
        self.p_elastic = p_elastic
        self.p_mirroring = p_mirroring

        self.p_gaussian_noise = p_gaussian_noise
        self.p_gaussian_blur = p_gaussian_blur
        self.p_brightness_contrast = p_brightness_contrast
        self.p_gamma = p_gamma

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

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # 0) Ensure image has channel dim
        if image.ndim == 2:
            image = image[..., None]

        # 1) Lesion-aware crop
        if random.random() < self.p_random_crop:
            image, label = random_crop_lesion_aware(
                image=image,
                label=label,
                crop_size=self.crop_size,
                foreground_oversample_prob=self.foreground_oversample_prob,
                lesion_dilation_radius=self.lesion_dilation_radius,
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

        if self.p_elastic > 0 and random.random() < self.p_elastic:
            image, label = elastic_deformation(
                image,
                label,
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma,
            )

        if random.random() < self.p_mirroring:
            image, label = random_mirror(image, label)

        # 3) Intensity transforms (image only, conservative)
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

        # 4) Final resize
        image = _resize_image(image, self.output_size)
        label = _resize_label(label, self.output_size)
        low_res_label = _resize_label(label, self.low_res)

        # 5) To tensor
        image = image_to_tensor(image)
        label = torch.from_numpy(label.astype(np.int64))
        low_res_label = torch.from_numpy(low_res_label.astype(np.int64))

        return {
            "image": image,
            "label": label,
            "low_res_label": low_res_label,
        }


class ValTransformWholeBodyLesion(object):
    def __init__(self, output_size, low_res):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if image.ndim == 2:
            image = image[..., None]

        image = _resize_image(image, self.output_size)
        label = _resize_label(label, self.output_size)
        low_res_label = _resize_label(label, self.low_res)

        image = image_to_tensor(image)
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
        mask = Image.open(sample_info["mask_path"]).convert("L")

        image = np.array(image)   # HWC, uint8
        label = np.array(mask)    # HW, uint8/int

        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        sample["case_name"] = sample_info["case_name"]
        return sample