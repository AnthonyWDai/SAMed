import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _prepare_image_for_pil(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 HxW or HxWxC for PIL.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Handle CHW -> HWC if needed
    if image.ndim == 3 and image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            # If already in [0,1], scale to [0,255]
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255)
        else:
            image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

    return image


def _prepare_binary_label(label: np.ndarray) -> np.ndarray:
    """
    Ensure binary segmentation mask is strictly 0/1.
    Accepts masks like {0,1}, {0,255}, or arbitrary positive foreground.
    """
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    # Squeeze singleton channel if present
    if label.ndim == 3 and label.shape[-1] == 1:
        label = label[..., 0]
    if label.ndim == 3 and label.shape[0] == 1:
        label = label[0]

    # Convert any positive value to 1
    label = (label > 0).astype(np.uint8)
    return label


def _to_tensor_image(image: np.ndarray) -> torch.Tensor:
    """
    Convert HxW or HxWxC float image in [0,1] to CHW torch.float32.
    """
    if image.ndim == 2:
        image = image[:, :, None]

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim 2 or 3, got shape {image.shape}")

    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(image.astype(np.float32))


class TrainTransform(object):
    def __init__(self, output_size, low_res):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)
        self.resize_img = transforms.Resize(self.output_size, interpolation=Image.BILINEAR)
        self.resize_mask_hr = transforms.Resize(self.output_size, interpolation=Image.NEAREST)
        self.resize_mask_lr = transforms.Resize(self.low_res, interpolation=Image.NEAREST)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # --- image ---
        image = _prepare_image_for_pil(image)
        image_pil = Image.fromarray(image)

        # --- label ---
        label = _prepare_binary_label(label)
        label_pil = Image.fromarray(label)

        # Resize
        image_resized = self.resize_img(image_pil)
        label_hr = self.resize_mask_hr(label_pil)

        # IMPORTANT: derive low-res label from high-res resized label for consistency
        label_lr = self.resize_mask_lr(label_hr)

        # Convert back to numpy
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        label_hr_np = np.array(label_hr).astype(np.uint8)
        label_lr_np = np.array(label_lr).astype(np.uint8)

        # Final label sanitization after resize
        label_hr_np = (label_hr_np > 0).astype(np.uint8)
        label_lr_np = (label_lr_np > 0).astype(np.uint8)

        return {
            "image": _to_tensor_image(image_np),
            "label": torch.from_numpy(label_hr_np).long(),
            "low_res_label": torch.from_numpy(label_lr_np).long(),
        }


class ValTransform(object):
    def __init__(self, output_size, low_res):
        self.output_size = tuple(output_size)
        self.low_res = tuple(low_res)
        self.resize_img = transforms.Resize(self.output_size, interpolation=Image.BILINEAR)
        self.resize_mask_hr = transforms.Resize(self.output_size, interpolation=Image.NEAREST)
        self.resize_mask_lr = transforms.Resize(self.low_res, interpolation=Image.NEAREST)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Deterministic preprocessing only
        image = _prepare_image_for_pil(image)
        label = _prepare_binary_label(label)

        image_pil = Image.fromarray(image)
        label_pil = Image.fromarray(label)

        image_resized = self.resize_img(image_pil)
        label_hr = self.resize_mask_hr(label_pil)
        label_lr = self.resize_mask_lr(label_hr)

        image_np = np.array(image_resized).astype(np.float32) / 255.0
        label_hr_np = np.array(label_hr).astype(np.uint8)
        label_lr_np = np.array(label_lr).astype(np.uint8)

        label_hr_np = (label_hr_np > 0).astype(np.uint8)
        label_lr_np = (label_lr_np > 0).astype(np.uint8)

        return {
            "image": _to_tensor_image(image_np),
            "label": torch.from_numpy(label_hr_np).long(),
            "low_res_label": torch.from_numpy(label_lr_np).long(),
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

        # Fallback: same basename directly under mask_dir
        for ext in self.MASK_EXTS:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.isfile(candidate):
                return candidate

        return None

    def _load_image(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            image = np.load(path)

            if image.ndim == 2:
                pass
            elif image.ndim == 3:
                if image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))

                if image.shape[-1] == 1:
                    image = image[..., 0]
                elif image.shape[-1] == 3:
                    image = image.astype(np.float32)
                    lo, hi = np.percentile(image, [1, 99])
                    if hi > lo:
                        image = np.clip(image, lo, hi)
                        image = (image - lo) / (hi - lo)
                    else:
                        image = np.zeros_like(image, dtype=np.float32)
                    return (image * 255).astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported channel dimension in .npy image shape {image.shape} for file: {path}")
            else:
                raise ValueError(f"Unsupported .npy image shape {image.shape} for file: {path}")

            image = image.astype(np.float32)
            lo, hi = np.percentile(image, [1, 99])
            if hi > lo:
                image = np.clip(image, lo, hi)
                image = (image - lo) / (hi - lo)
            else:
                image = np.zeros_like(image, dtype=np.float32)

            image = (image * 255).astype(np.uint8)
            image = np.stack([image] * 3, axis=-1)
            return image

        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _load_mask(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            mask = np.load(path)
        else:
            mask = np.array(Image.open(path))

        # Reduce mask to single channel if needed
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask[:, :, 0]

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