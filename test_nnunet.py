import os
import sys
import argparse
import logging
import random
from glob import glob
from importlib import import_module

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from segment_anything import sam_model_registry


def setup_nnunet_objects(dataset_json_file, plans_json_file, configuration_name):
    dataset_json = load_json(dataset_json_file)
    plans = load_json(plans_json_file)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    preprocessor = configuration_manager.preprocessor_class(verbose=False)

    return dataset_json, plans_manager, configuration_manager, preprocessor


def collect_cases(input_dir):
    """
    Collect paired 2-modality cases:
      case_001_0000.nii.gz
      case_001_0001.nii.gz
    Returns:
      [
        [path/to/case_001_0000.nii.gz, path/to/case_001_0001.nii.gz],
        ...
      ]
    """
    mod0_files = sorted(glob(os.path.join(input_dir, "*_0000.nii.gz")))
    cases = []
    for f0 in mod0_files:
        f1 = f0.replace("_0000.nii.gz", "_0001.nii.gz")
        if os.path.isfile(f1):
            cases.append([f0, f1])
        else:
            logging.warning(f"Missing paired modality for {f0}, skipping.")
    return cases


def truncate_output_filename(output_dir, case_files):
    case_id = os.path.basename(case_files[0]).replace("_0000.nii.gz", "")
    return os.path.join(output_dir, case_id)


def preprocess_case_nnunet(case_files, plans_manager, configuration_manager, dataset_json, preprocessor):
    """
    Mimics nnU-Net sequential preprocessing:
        data, seg, data_properties = preprocessor.run_case(...)
    Returns:
        data: np.ndarray (C, Z, Y, X)
        data_properties: dict
    """
    data, seg, data_properties = preprocessor.run_case(
        case_files,
        None,
        plans_manager,
        configuration_manager,
        dataset_json
    )
    return data, data_properties


# def build_rgb_from_two_modalities(slice_2ch: np.ndarray) -> np.ndarray:
#     """
#     slice_2ch: (2, H, W)
#     return: (H, W, 3) float32
#     channel0 = mod0
#     channel1 = mod1
#     channel2 = average(mod0, mod1)
#     """
#     c0 = slice_2ch[0]
#     c1 = slice_2ch[1]
#     c2 = 0.5 * (c0 + c1)
#     image = np.stack([c0, c1, c2], axis=-1).astype(np.float32)  # HWC
#     return image


# TODO: change to the following after new training scheme
def build_rgb_from_two_modalities(slice_2ch: np.ndarray) -> np.ndarray:
    """
    slice_2ch: (2, H, W)
    return: (H, W, 3) float32

    channel0 = mod0
    channel1 = mod1 mapped to mod0 value range
    channel2 = average(channel0, channel1)
    """
    if slice_2ch.shape[0] != 2:
        raise ValueError(f"Expected input shape (2, H, W), got {slice_2ch.shape}")

    c0 = slice_2ch[0].astype(np.float32)
    c1 = slice_2ch[1].astype(np.float32)

    c0_min, c0_max = c0.min(), c0.max()
    c1_min, c1_max = c1.min(), c1.max()

    # Map c1 to c0's value range
    if c1_max > c1_min:
        c1_mapped = (c1 - c1_min) / (c1_max - c1_min)  # [0, 1]
        c1_mapped = c1_mapped * (c0_max - c0_min) + c0_min
    else:
        # Constant channel fallback
        c1_mapped = np.full_like(c1, fill_value=c0_min)

    c2 = 0.5 * (c0 + c1_mapped)

    image = np.stack([c0, c1_mapped, c2], axis=-1).astype(np.float32)
    return image


def preprocess_slice_like_val_transform(image_hwc: np.ndarray, output_size: int, device: torch.device):
    """
    Match your ValTransform behavior as closely as possible:
      image = _resize_image(image, output_size)
      image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
      if image.max() > 1.0:
          image = image / 255.0

    Here we implement resize with torch interpolate.
    Input:
      image_hwc: (H, W, 3), float32
    Output:
      x: (1, 3, output_size, output_size)
    """
    x = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float().to(device)  # 1,3,H,W
    x = F.interpolate(x, size=(output_size, output_size), mode='bilinear', align_corners=False)

    if x.max() > 1.0:
        x = x / 255.0

    return x


def standardize_model_output(pred):
    """
    Standardize possible SAMed outputs to a tensor of shape (1, C, H, W).
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

    For binary tasks:
      - if C == 1, convert to 2-channel logits [background, foreground]
      - if C == 2, keep as is

    For multiclass tasks:
      - require C == num_classes
    """
    c = predicted_logits.shape[0]

    if num_classes == 1:
        # if c == 1:
            # fg = predicted_logits
            # bg = -fg
            # predicted_logits = torch.cat([bg, fg], dim=0)
        if c == 2:
            pass
        else:
            raise RuntimeError(f"Binary task but model returned {c} channels.")
    else:
        if c != num_classes:
            raise RuntimeError(
                f"Multiclass task expects {num_classes} output channels, but model returned {c}."
            )

    return predicted_logits


@torch.inference_mode()
def predict_case_with_samed_preprocessed(
    model,
    data: np.ndarray,
    num_classes: int,
    img_size: int,
    input_size: int,
    device: torch.device,
    multimask_output: bool,
    model_image_size_mode: str = "list",
):
    """
    data: nnU-Net preprocessed array, shape (C, Z, Y, X)
    Uses first two channels as modalities.
    Predicts slice by slice and returns logits of shape (C_out, Z, Y, X).
    """
    assert data.ndim == 4, f"Expected data shape (C, Z, Y, X), got {data.shape}"
    assert data.shape[0] >= 2, "Expected at least 2 channels/modalities in preprocessed data."

    _, z, h, w = data.shape
    preds = []

    model.eval()

    for s in tqdm(range(z), desc="Predicting slices"):
        slice_2ch = data[:2, s]  # (2, H, W)

        # preprocess from rgb conversion is correct
        image_hwc = build_rgb_from_two_modalities(slice_2ch)
        x = preprocess_slice_like_val_transform(image_hwc, img_size, device)

        if model_image_size_mode == "list":
            image_size_arg = [input_size, input_size]
        elif model_image_size_mode == "scalar":
            image_size_arg = input_size
        else:
            raise ValueError("model_image_size_mode must be 'list' or 'scalar'")
        
        pred = model(x, multimask_output, image_size_arg)
        pred = standardize_model_output(pred)  # (1, C, h, w)

        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)[0]  # (C, H, W)
        preds.append(pred.cpu())

    predicted_logits = torch.stack(preds, dim=1)  # (C, Z, H, W)
    predicted_logits = ensure_export_compatible_logits(predicted_logits, num_classes)
    return predicted_logits


def inference_on_folder(args):
    maybe_mkdir_p(args.output_dir)

    dataset_json, plans_manager, configuration_manager, preprocessor = setup_nnunet_objects(
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

    cases = collect_cases(args.input_dir)
    logging.info(f"Found {len(cases)} valid cases in {args.input_dir}")

    for case_files in cases:
        case_id = os.path.basename(case_files[0]).replace("_0000.nii.gz", "")
        logging.info(f"Processing case: {case_id}")

        # preprocess from nnunet is correct
        data, data_properties = preprocess_case_nnunet(
            case_files,
            plans_manager,
            configuration_manager,
            dataset_json,
            preprocessor
        )

        predicted_logits = predict_case_with_samed_preprocessed(
            model=model,
            data=data,
            num_classes=args.num_classes,
            img_size=args.img_size,
            input_size=args.input_size,
            device=args.device,
            multimask_output=multimask_output,
            model_image_size_mode=args.model_image_size_mode
        )

        output_file_truncated = truncate_output_filename(args.output_dir, case_files)

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
                        help='Folder containing paired *_0000.nii.gz and *_0001.nii.gz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Folder to save output segmentations')

    parser.add_argument('--dataset_json', type=str, required=True,
                        help='Path to nnU-Net dataset.json')
    parser.add_argument('--plans_json', type=str, required=True,
                        help='Path to nnU-Net plans.json')
    parser.add_argument('--configuration', type=str, required=True,
                        help='nnU-Net configuration name, e.g. 3d_fullres')

    parser.add_argument('--num_classes', type=int, required=True,
                        help='Current task: use 1 for binary. Multiclass supported later')
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
                        help="How to pass image_size into model(x, multimask_output, image_size). "
                             "'list' -> [input_size, input_size], 'scalar' -> input_size")

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
    main()