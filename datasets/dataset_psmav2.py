import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ----------------------------
# Basic spatial helpers
# ----------------------------
def random_rotate(image, label, angle_range=(-7, 7)):
    """
    image: HWC
    label: HW
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
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_scale(image, label, scale_range=(0.9, 1.1), anisotropic=False):
    """
    Random scaling then fit back to original size.
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
    Mild 2D elastic deformation.
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


def random_crop(image, label, crop_size):
    """
    Standard random crop.
    image: HWC
    label: HW
    crop_size: (crop_h, crop_w)
    """
    h, w = label.shape
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        image = _pad_if_needed_image(image, crop_size)
        label = _pad_if_needed_label(label, crop_size)
        h, w = label.shape

    y1 = np.random.randint(0, h - crop_h + 1)
    x1 = np.random.randint(0, w - crop_w + 1)

    image = image[y1:y1 + crop_h, x1:x1 + crop_w, :]
    label = label[y1:y1 + crop_h, x1:x1 + crop_w]
    return image, label


def random_short_side_resize(image, label, short_size_range=(256, 512), max_size=1024):
    """
    Resize image/label so that the short side is randomly sampled
    from short_size_range, preserving aspect ratio.
    """
    h, w = label.shape
    target_short = random.randint(short_size_range[0], short_size_range[1])
    scale = min(target_short / float(min(h, w)), max_size / float(max(h, w)))

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    image = _resize_image(image, (new_h, new_w))
    label = _resize_label(label, (new_h, new_w))
    return image, label


# ----------------------------
# Intensity transforms
# ----------------------------
def add_gaussian_noise(image, std_range=(0.0, 0.02)):
    img = image.astype(np.float32)
    scale = 255.0 if img.max() > 1.5 else 1.0
    std = np.random.uniform(std_range[0], std_range[1]) * scale
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, scale)
    return img


def add_gaussian_blur(image, sigma_range=(0.2, 0.8)):
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
    """
    h, w = image.shape[:2]
    if (h, w) == tuple(output_size):
        return image

    out_h, out_w = output_size

    if image.shape[2] == 1:
        img_2d = image[..., 0]
        pil_img = Image.fromarray(img_2d.astype(np.float32), mode="F")
        resized = pil_img.resize((out_w, out_h), resample=Image.BICUBIC)
        resized = np.array(resized, dtype=np.float32)[..., None]
    else:
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


def _pad_to_aspect_image(image, target_size):
    """
    Pad image to match target aspect ratio without stretching.
    image: HWC
    target_size: (H, W)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    target_ratio = target_h / float(target_w)
    ratio = h / float(w)

    if abs(ratio - target_ratio) < 1e-8:
        return image

    if ratio > target_ratio:
        # image too tall -> pad width
        new_w = int(round(h / target_ratio))
        pad_w = max(0, new_w - w)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return np.pad(
            image,
            ((0, 0), (pad_left, pad_right), (0, 0)),
            mode="reflect"
        )
    else:
        # image too wide -> pad height
        new_h = int(round(w * target_ratio))
        pad_h = max(0, new_h - h)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        return np.pad(
            image,
            ((pad_top, pad_bottom), (0, 0), (0, 0)),
            mode="reflect"
        )


def _pad_to_aspect_label(label, target_size):
    """
    Pad label to match target aspect ratio without stretching.
    label: HW
    target_size: (H, W)
    """
    h, w = label.shape
    target_h, target_w = target_size
    target_ratio = target_h / float(target_w)
    ratio = h / float(w)

    if abs(ratio - target_ratio) < 1e-8:
        return label

    if ratio > target_ratio:
        # label too tall -> pad width
        new_w = int(round(h / target_ratio))
        pad_w = max(0, new_w - w)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return np.pad(
            label,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )
    else:
        # label too wide -> pad height
        new_h = int(round(w * target_ratio))
        pad_h = max(0, new_h - h)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        return np.pad(
            label,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=0
        )


def _fit_to_size_image(image, output_size):
    image = _pad_if_needed_image(image, output_size)
    h, w = image.shape[:2]
    out_h, out_w = output_size

    y1 = max(0, (h - out_h) // 2)
    x1 = max(0, (w - out_w) // 2)
    return image[y1:y1 + out_h, x1:x1 + out_w, :]


def _fit_to_size_label(label, output_size):
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
    If input looks like [0,255], normalize to [0,1].
    """
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def make_divisible(value, divisor):
    if divisor == 0:
        raise ValueError("divisor cannot be 0")
    return value + ((-value) % divisor)


# ----------------------------
# Train / val transforms
# ----------------------------
class TrainTransform(object):
    def __init__(
        self,
        output_size,
        low_res,
        short_size_range=None,
        max_size=None,
        # spatial probabilities
        p_rotation=0.5,
        p_scaling=0.,
        p_elastic=0.,
        p_mirroring=0.5,
        # intensity probabilities
        p_gaussian_noise=0.,
        p_gaussian_blur=0.,
        p_brightness_contrast=0.,
        p_gamma=0.,
        # parameter ranges
        rotation_range=(-10, 10),
        scale_range=(0.95, 1.05),
        anisotropic_scale=False,
        elastic_alpha=6.0,
        elastic_sigma=8.0,
        noise_std_range=(0.0, 0.02),
        blur_sigma_range=(0.3, 0.8),
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        gamma_range=(0.9, 1.1),
    ):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)
        
        if short_size_range is None:
            based_size = np.mean(output_size)
            short_size_range = (
                make_divisible(based_size / 1.07, 4), # 480
                make_divisible(based_size * 1.25, 4) # 640
            )
        
        self.short_size_range = tuple(short_size_range)
        
        if max_size is None:
            based_size = np.mean(self.output_size)
            max_size = based_size * 2
        self.max_size = max_size
        
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

        if image.ndim == 2:
            image = image[..., None]

        # 1) Random short-side resize
        image, label = random_short_side_resize(
            image,
            label,
            short_size_range=self.short_size_range,
            max_size=self.max_size,
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

        # 3) Normal random crop to output size
        image, label = random_crop(image, label, self.output_size)

        # 4) Intensity transforms
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

        # No final resize; crop already guarantees output_size
        # low_res_label = _resize_label(label, self.low_res)

        image = image_to_tensor(image)
        label = torch.from_numpy(label.astype(np.int64))
        # low_res_label = torch.from_numpy(low_res_label.astype(np.int64))

        return {
            "image": image,
            "label": label,
            # "low_res_label": low_res_label,
        }


class ValTransform(object):
    def __init__(self, output_size, low_res):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if image.ndim == 2:
            image = image[..., None]

        # keep scale of hw in validation
        image = _pad_to_aspect_image(image, self.output_size)
        label = _pad_to_aspect_label(label, self.output_size)

        image = _resize_image(image, self.output_size)
        label = _resize_label(label, self.output_size)

        # low_res_label = _resize_label(label, self.low_res)

        image = image_to_tensor(image)
        label = torch.from_numpy(label.astype(np.int64))
        # low_res_label = torch.from_numpy(low_res_label.astype(np.int64))

        return {
            "image": image,
            "label": label,
            # "low_res_label": low_res_label,
        }


# ----------------------------
# Dataset
# ----------------------------
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

        image = np.array(image)
        label = np.array(mask)

        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        sample["case_name"] = sample_info["case_name"]
        return sample