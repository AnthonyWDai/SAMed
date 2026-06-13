import os
import sys
import argparse
import logging
import random
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


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


def preprocess_case_nnunet(case_files, plans_manager, configuration_manager, dataset_json, preprocessor):
    """
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


def build_rgb_from_two_modalities(slice_2ch: np.ndarray) -> np.ndarray:
    """
    slice_2ch: (2, H, W)
    return: (H, W, 3) float32
    channel0 = mod0
    channel1 = mod1
    channel2 = average(mod0, mod1)
    """
    c0 = slice_2ch[0]
    c1 = slice_2ch[1]
    c2 = 0.5 * (c0 + c1)
    image = np.stack([c0, c1, c2], axis=-1).astype(np.float32)  # HWC
    return image


def preprocess_slice_like_val_transform(image_hwc: np.ndarray, output_size: int, device: torch.device):
    """
    Input:
      image_hwc: (H, W, 3), float32
    Output:
      x: (1, 3, output_size, output_size)
    """
    x = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).float().to(device)
    x = F.interpolate(x, size=(output_size, output_size), mode='bilinear', align_corners=False)

    if x.max() > 1.0:
        x = x / 255.0

    return x


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalize arbitrary float/int array to uint8 [0, 255].
    """
    arr = arr.astype(np.float32)
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def save_grayscale_image(arr: np.ndarray, save_path: str):
    arr_uint8 = normalize_to_uint8(arr)
    img = Image.fromarray(arr_uint8, mode='L')
    ext = os.path.splitext(save_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        img.save(save_path, quality=95)
    else:
        img.save(save_path)


def save_rgb_image(arr: np.ndarray, save_path: str):
    """
    arr: (H, W, 3)
    Save JPG/JPEG with quality=95.
    """
    arr_uint8 = normalize_to_uint8(arr)
    img = Image.fromarray(arr_uint8, mode='RGB')
    ext = os.path.splitext(save_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        img.save(save_path, quality=95)
    else:
        img.save(save_path)


def save_tensor_image(tensor_chw: torch.Tensor, save_path: str):
    """
    tensor_chw: (3, H, W) or (1, H, W)
    """
    arr = tensor_chw.detach().cpu().numpy()

    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {arr.shape}")

    if arr.shape[0] == 1:
        save_grayscale_image(arr[0], save_path)
    elif arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))  # HWC
        save_rgb_image(arr, save_path)
    else:
        raise ValueError(f"Expected 1 or 3 channels, got {arr.shape[0]}")


def export_case_images(
    case_files,
    case_id,
    data,
    output_dir,
    img_size,
    device,
    save_ext="jpg"
):
    """
    Save outputs from:
      1. preprocess_case_nnunet (modalities) as .npy
      2. build_rgb_from_two_modalities
      3. preprocess_slice_like_val_transform
    """
    assert data.ndim == 4, f"Expected data shape (C, Z, Y, X), got {data.shape}"
    assert data.shape[0] >= 2, "Expected at least 2 channels/modalities."

    _, z, h, w = data.shape

    case_out_dir = os.path.join(output_dir, case_id)
    mod0_dir = os.path.join(case_out_dir, "01_preprocess_case_nnunet_mod0_npy")
    mod1_dir = os.path.join(case_out_dir, "01_preprocess_case_nnunet_mod1_npy")
    rgb_dir = os.path.join(case_out_dir, "02_build_rgb_from_two_modalities")
    valtf_dir = os.path.join(case_out_dir, "03_preprocess_slice_like_val_transform")

    for d in [mod0_dir, mod1_dir, rgb_dir, valtf_dir]:
        os.makedirs(d, exist_ok=True)

    for s in tqdm(range(z), desc=f"Exporting slices for {case_id}"):
        slice_2ch = data[:2, s]  # (2, H, W)

        # Save preprocess_case_nnunet outputs as .npy
        mod0 = slice_2ch[0]
        mod1 = slice_2ch[1]
        np.save(os.path.join(mod0_dir, f"slice_{s:04d}.npy"), mod0)
        np.save(os.path.join(mod1_dir, f"slice_{s:04d}.npy"), mod1)

        # Save build_rgb_from_two_modalities output as JPG with quality=95
        image_hwc = build_rgb_from_two_modalities(slice_2ch)
        rgb_save_path = os.path.join(rgb_dir, f"slice_{s:04d}.jpg")
        save_rgb_image(image_hwc, rgb_save_path)

        # Save preprocess_slice_like_val_transform output
        x = preprocess_slice_like_val_transform(image_hwc, img_size, device)  # (1, 3, H, W)
        save_tensor_image(x[0], os.path.join(valtf_dir, f"slice_{s:04d}.{save_ext}"))

    logging.info(f"Saved exported images and npy arrays for case: {case_id}")


def export_on_folder(args):
    maybe_mkdir_p(args.output_dir)

    dataset_json, plans_manager, configuration_manager, preprocessor = setup_nnunet_objects(
        args.dataset_json,
        args.plans_json,
        args.configuration
    )

    cases = collect_cases(args.input_dir)
    cases = cases[:args.max_cases]
    logging.info(f"Processing first {len(cases)} cases from {args.input_dir}")
    logging.info(f"Found {len(cases)} valid cases in {args.input_dir}")

    for case_files in cases:
        case_id = os.path.basename(case_files[0]).replace("_0000.nii.gz", "")
        logging.info(f"Processing case: {case_id}")

        data, data_properties = preprocess_case_nnunet(
            case_files,
            plans_manager,
            configuration_manager,
            dataset_json,
            preprocessor
        )

        export_case_images(
            case_files=case_files,
            case_id=case_id,
            data=data,
            output_dir=args.output_dir,
            img_size=args.img_size,
            device=args.device,
            save_ext=args.save_ext
        )

    logging.info("Export finished.")


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing paired *_0000.nii.gz and *_0001.nii.gz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Folder to save exported images')

    parser.add_argument('--dataset_json', type=str, required=True,
                        help='Path to nnU-Net dataset.json')
    parser.add_argument('--plans_json', type=str, required=True,
                        help='Path to nnU-Net plans.json')
    parser.add_argument('--configuration', type=str, required=True,
                        help='nnU-Net configuration name, e.g. 3d_fullres')

    parser.add_argument('--img_size', type=int, default=512,
                        help='Resize size matching validation transform output_size')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--deterministic', type=int, default=1)

    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda / cpu / mps')
    parser.add_argument('--save_ext', type=str, default='jpg', choices=['png', 'jpg', 'jpeg'],
                        help='Image format for exported validation slices')
    parser.add_argument('--max_cases', type=int, default=5,
                        help='Number of 3D cases to process')

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

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

    export_on_folder(args)


if __name__ == '__main__':
    # python test_check.py \
    # --input_dir "${INFERENCE_INPUT}" \
    # --output_dir "${INFERENCE_OUTPUT}/check_image" \
    # --dataset_json "${SAMED_PREPROCESSED}/dataset.json" \
    # --plans_json "${SAMED_PREPROCESSED}/nnUNetResEncUNetLPlans.json" \
    # --configuration 2d \
    # --img_size 512 \
    # --device cpu \
    # --save_ext jpg
    main()