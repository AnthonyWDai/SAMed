import os
import sys
import argparse
import logging
from glob import glob
from importlib import import_module

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import nibabel as nib

from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from segment_anything import sam_model_registry


def setup_nnunet_objects(dataset_json_file, plans_json_file, configuration_name):
    dataset_json = load_json(dataset_json_file)
    plans = load_json(plans_json_file)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    return dataset_json, plans_manager, configuration_manager


def collect_rgb_cases(input_dir, gt_dir, image_exts=(".png", ".jpg", ".jpeg", ".rgb")):
    """
    Assumes:
      input_dir/
        caseA/
          slice_0000.png
          slice_0001.png
          ...
        caseB/
          ...
      gt_dir/
        caseA.nii.gz
        caseB.nii.gz

    Returns:
      [
        {
          "case_id": "caseA",
          "slice_files": [...],
          "gt_nii": ".../caseA.nii.gz"
        },
        ...
      ]
    """
    cases = []
    case_dirs = sorted([d for d in glob(os.path.join(input_dir, "*")) if os.path.isdir(d)])

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)

        slice_files = []
        for ext in image_exts:
            slice_files.extend(glob(os.path.join(case_dir, f"*{ext}")))
        slice_files = sorted(slice_files)

        if len(slice_files) == 0:
            logging.warning(f"No RGB slices found in {case_dir}, skipping.")
            continue

        gt_nii = os.path.join(gt_dir, f"{case_id}.nii.gz")
        if not os.path.isfile(gt_nii):
            logging.warning(f"Missing GT NIfTI for case {case_id}: {gt_nii}, skipping.")
            continue

        cases.append({
            "case_id": case_id,
            "slice_files": slice_files,
            "gt_nii": gt_nii,
        })

    return cases


def truncate_output_filename(output_dir, case_id):
    return os.path.join(output_dir, case_id)


def read_rgb_image(image_file: str) -> np.ndarray:
    """
    Reads image as RGB HWC float32.
    Supports jpg/png directly.
    For .rgb, this implementation tries PIL first; if your .rgb is raw binary,
    you may need a custom reader beyond this code.
    """
    img = Image.open(image_file).convert("RGB")
    image = np.asarray(img).astype(np.float32)
    return image


def preprocess_slice_like_val_transform(image_hwc: np.ndarray, output_size: int, device: torch.device):
    """
    image_hwc: (H, W, 3), float32
    returns: (1, 3, output_size, output_size)
    """
    x = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float().to(device)
    x = F.interpolate(x, size=(output_size, output_size), mode='bilinear', align_corners=False)

    if x.max() > 1.0:
        x = x / 255.0

    return x


def standardize_model_output(pred):
    """
    Standardize possible SAMed outputs to (1, C, H, W)
    """
    if isinstance(pred, dict):
        if 'masks' in pred:
            pred = pred['masks']
        else:
            raise RuntimeError(f"Unexpected dict output keys: {pred.keys()}")
    elif isinstance(pred, (list, tuple)):
        pred = pred[0]

    if not torch.is_tensor(pred):
        raise RuntimeError(f"Model output is not a tensor. Got type: {type(pred)}")

    if pred.ndim != 4:
        raise RuntimeError(f"Expected model output shape (B, C, H, W), got {pred.shape}")

    return pred


def ensure_export_compatible_logits(predicted_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    predicted_logits: (C, Z, H, W)
    """
    c = predicted_logits.shape[0]

    if num_classes == 1:
        if c == 2:
            pass
        elif c == 1:
            fg = predicted_logits
            bg = -fg
            predicted_logits = torch.cat([bg, fg], dim=0)
        else:
            raise RuntimeError(f"Binary task but model returned {c} channels.")
    else:
        if c != num_classes:
            raise RuntimeError(
                f"Multiclass task expects {num_classes} output channels, but model returned {c}."
            )

    return predicted_logits


def build_data_properties_from_reference_nii(reference_nii_file: str):
    """
    Build the minimal data_properties needed by nnU-Net export from a reference NIfTI.

    This is based partly on knowledge beyond the provided code context:
    nnU-Net export uses fields such as shape/original spacing/original affine-compatible metadata.
    Depending on your nnU-Net version, you may need to add more keys.
    """
    nii = nib.load(reference_nii_file)
    shape = nii.shape
    spacing = nii.header.get_zooms()[:3]

    # nnU-Net typically uses z, y, x ordering internally for spacing/shape
    shape_zyx = tuple(shape[::-1])
    spacing_zyx = tuple(spacing[::-1])

    data_properties = {
        "sitk_stuff": {
            "spacing": tuple(float(x) for x in spacing),
            "origin": (0.0, 0.0, 0.0),
            "direction": tuple(np.eye(3).reshape(-1).tolist())
        },
        "spacing": spacing_zyx,
        "original_spacing": spacing_zyx,
        "shape_before_cropping": shape_zyx,
        "bbox_used_for_cropping": [[0, shape_zyx[0]], [0, shape_zyx[1]], [0, shape_zyx[2]]],
        "shape_after_cropping_and_before_resampling": shape_zyx,
        "nibabel_stuff": {
            "original_affine": nii.affine,
        }
    }

    return data_properties, shape


@torch.inference_mode()
def predict_case_from_rgb_slices(
    model,
    slice_files,
    reference_nii_file,
    num_classes: int,
    img_size: int,
    input_size: int,
    device: torch.device,
    multimask_output: bool,
    model_image_size_mode: str = "list",
):
    """
    Predict from RGB slices and stack into nnU-Net-style logits: (C, Z, H, W)

    The number/order of slices must correspond to the reference NIfTI z-dimension.
    """
    ref_nii = nib.load(reference_nii_file)
    ref_shape = ref_nii.shape  # (X, Y, Z) or sometimes (H, W, Z) depending on convention
    if len(ref_shape) != 3:
        raise RuntimeError(f"Expected 3D reference NIfTI, got shape {ref_shape}")

    target_h = ref_shape[1]
    target_w = ref_shape[0]
    target_z = ref_shape[2]

    if len(slice_files) != target_z:
        raise RuntimeError(
            f"Number of RGB slices ({len(slice_files)}) does not match reference NIfTI depth ({target_z}) "
            f"for {reference_nii_file}"
        )

    preds = []
    model.eval()

    for s, image_file in enumerate(tqdm(slice_files, desc="Predicting slices")):
        image_hwc = read_rgb_image(image_file)
        original_h, original_w = image_hwc.shape[:2]

        x = preprocess_slice_like_val_transform(image_hwc, img_size, device)

        if model_image_size_mode == "list":
            image_size_arg = [input_size, input_size]
        elif model_image_size_mode == "scalar":
            image_size_arg = input_size
        else:
            raise ValueError("model_image_size_mode must be 'list' or 'scalar'")

        pred = model(x, multimask_output, image_size_arg)
        pred = standardize_model_output(pred)

        # resize back to original RGB slice resolution first
        pred = F.interpolate(pred, size=(original_h, original_w), mode='bilinear', align_corners=False)[0]

        # then resize to reference NIfTI in-plane shape if needed
        if (original_h, original_w) != (target_h, target_w):
            pred = F.interpolate(
                pred.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )[0]

        preds.append(pred.cpu())

    predicted_logits = torch.stack(preds, dim=1)  # (C, Z, H, W)
    predicted_logits = ensure_export_compatible_logits(predicted_logits, num_classes)
    return predicted_logits


def inference_on_folder(args):
    maybe_mkdir_p(args.output_dir)

    dataset_json, plans_manager, configuration_manager = setup_nnunet_objects(
        args.dataset_json,
        args.plans_json,
        args.configuration
    )

    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1]
    )

    pkg = import_module(args.module)
    model = pkg.LoRA_Sam(sam, args.rank).to(args.device)

    if args.lora_ckpt is None:
        raise ValueError("--lora_ckpt must be provided")
    model.load_lora_parameters(args.lora_ckpt)
    model.eval()

    multimask_output = args.num_classes > 1

    cases = collect_rgb_cases(args.input_dir, args.gt_dir, tuple(args.image_exts))
    logging.info(f"Found {len(cases)} valid RGB cases in {args.input_dir}")

    for case in cases:
        case_id = case["case_id"]
        slice_files = case["slice_files"]
        gt_nii = case["gt_nii"]

        logging.info(f"Processing case: {case_id}")

        predicted_logits = predict_case_from_rgb_slices(
            model=model,
            slice_files=slice_files,
            reference_nii_file=gt_nii,
            num_classes=args.num_classes,
            img_size=args.img_size,
            input_size=args.input_size,
            device=args.device,
            multimask_output=multimask_output,
            model_image_size_mode=args.model_image_size_mode
        )

        data_properties, _ = build_data_properties_from_reference_nii(gt_nii)

        output_file_truncated = truncate_output_filename(args.output_dir, case_id)

        export_prediction_from_logits(
            predicted_logits.numpy(),
            data_properties,
            configuration_manager,
            plans_manager,
            dataset_json,
            output_file_truncated,
            args.save_probabilities
        )

        logging.info(f"Saved: {output_file_truncated}.nii.gz")

    logging.info("Inference finished.")


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder with per-case subfolders of RGB slices')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Folder containing reference GT nii.gz files used for reconstruction metadata')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Folder to save output segmentations')

    parser.add_argument('--dataset_json', type=str, required=True,
                        help='Path to nnU-Net dataset.json')
    parser.add_argument('--plans_json', type=str, required=True,
                        help='Path to nnU-Net plans.json')
    parser.add_argument('--configuration', type=str, required=True,
                        help='nnU-Net configuration name, e.g. 3d_fullres')

    parser.add_argument('--num_classes', type=int, required=True,
                        help='Use 1 for binary. Multiclass supported if model outputs matching channels')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Resize size matching validation transform output_size')
    parser.add_argument('--input_size', type=int, default=512,
                        help='Size argument passed into SAMed forward')
    parser.add_argument('--deterministic', type=int, default=1)

    parser.add_argument('--ckpt', type=str, required=True,
                        help='SAM pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, required=True,
                        help='LoRA checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_b',
                        help='SAM ViT type')
    parser.add_argument('--rank', type=int, default=4,
                        help='LoRA rank')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder',
                        help='Module containing LoRA_Sam')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda / cpu / mps')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Save probabilities if supported by nnU-Net export')

    parser.add_argument('--model_image_size_mode', type=str, default='scalar', choices=['list', 'scalar'],
                        help="How to pass image_size into model(x, multimask_output, image_size)")

    parser.add_argument('--image_exts', nargs='+', default=['.png', '.jpg', '.jpeg', '.rgb'],
                        help='Allowed RGB slice extensions')

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
    elif args.device == 'mps':
        device = torch.device('mps')
    else:
        raise ValueError(f"Unsupported device: {args.device}")

    args.device = device

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    os.makedirs(args.output_dir, exist_ok=True)

    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    inference_on_folder(args)


if __name__ == '__main__':
    # python test_nnunetv1.py \
    # --input_dir /path/to/input_cases \
    # --output_dir /path/to/output_preds \
    # --dataset_json /path/to/dataset.json \
    # --plans_json /path/to/plans.json \
    # --configuration 3d_fullres \
    # --num_classes 2 \
    # --img_size 512 \
    # --input_size 512 \
    # --ckpt /path/to/sam_checkpoint.pth \
    # --lora_ckpt /path/to/lora_checkpoint.pth \
    # --vit_name vit_b \
    # --rank 4 \
    # --module sam_lora_image_encoder \
    # --device cuda \
    # --model_image_size_mode scalar
    main()