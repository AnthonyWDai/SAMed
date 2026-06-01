import argparse
import random
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from segment_anything import sam_model_registry
from trainerV2 import trainer_psma


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LoRA-SAM on medical segmentation datasets.")

    parser.add_argument(
        "--root_path",
        type=str,
        default="/data/LarryXu/Synapse/preprocessed_data/train_npz",
        help="Root directory for training data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/output/sam/results",
        help="Directory to save training outputs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PSMA",
        help="Dataset name.",
    )
    parser.add_argument(
        "--list_dir",
        type=str,
        default="./lists/lists_Synapse",
        help="Directory containing dataset split lists.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="Number of output segmentation classes. Exclude background",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=30000,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--stop_epoch",
        type=int,
        default=160,
        help="Epoch to stop training early, if applicable.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=2,
        help="Number of GPUs.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic training for reproducibility.",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.005,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input image size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    parser.add_argument(
        "--vit_name",
        type=str,
        default="vit_b",
        help="Vision Transformer backbone name.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/sam_vit_b_01ec64.pth",
        help="Path to pretrained SAM checkpoint.",
    )
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        default=None,
        help="Path to finetuned LoRA checkpoint.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Enable learning rate warmup.",
    )
    parser.add_argument(
        "--warmup_period",
        type=int,
        default=250,
        help="Warmup iterations when warmup is enabled.",
    )
    parser.add_argument(
        "--AdamW",
        action="store_true",
        help="Use AdamW optimizer.",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="sam_lora_image_encoder",
        help="Module containing LoRA_Sam implementation.",
    )
    parser.add_argument(
        "--dice_param",
        type=float,
        default=0.8,
        help="Weight for Dice-related loss term.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=1,
        help="Whether to freeze parts of the backbone.",
    )

    return parser


def set_seed(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_snapshot_path(args) -> Path:
    exp_name = f"{args.dataset}_{args.img_size}"
    suffixes = []

    if getattr(args, "is_pretrain", False):
        suffixes.append("pretrain")

    suffixes.append(args.vit_name)

    if args.max_iterations != 30000:
        suffixes.append(f"{str(args.max_iterations)[:2]}k")

    if args.max_epochs != 30:
        suffixes.append(f"epo{args.max_epochs}")

    suffixes.append(f"bs{args.batch_size}")

    if args.base_lr != 0.01:
        suffixes.append(f"lr{args.base_lr}")

    if args.seed != 1234:
        suffixes.append(f"s{args.seed}")

    suffixes.append(f"freeze{args.freeze}")

    snapshot_name = "_".join([exp_name] + suffixes)
    snapshot_path = Path(args.output) / snapshot_name
    snapshot_path.mkdir(parents=True, exist_ok=True)

    return snapshot_path


def save_config(args, snapshot_path: Path) -> None:
    config_file = snapshot_path / "config.txt"
    with config_file.open("w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def build_model(args):
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )

    module = import_module(args.module)
    net = module.LoRA_Sam(sam, args.rank).cuda()

    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    # Freeze image encoder
    if args.freeze >= 1:
        for p in net.sam.image_encoder.parameters():
            p.requires_grad = False

    # Freeze prompt encoder (SAMed change prompt encoder)
    if args.freeze >= 2:
        for p in net.sam.prompt_encoder.parameters():
            p.requires_grad = False

    # Train mask decoder
    if args.freeze >= 3:
        for p in net.sam.mask_decoder.iou_token.parameters():
            p.requires_grad = False

        for p in net.sam.mask_decoder.mask_tokens.parameters():
            p.requires_grad = False

    if args.freeze >= 4:
        for p in net.sam.mask_decoder.transformer.parameters():
            p.requires_grad = False

    return net, img_embedding_size


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed, deterministic=bool(args.deterministic))

    dataset_config = {
        "PSMA": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": args.num_classes,
        }
    }

    if args.dataset not in dataset_config:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    args.is_pretrain = True
    snapshot_path = build_snapshot_path(args)

    net, img_embedding_size = build_model(args)

    multimask_output = args.num_classes > 1
    low_res = img_embedding_size * 4

    save_config(args, snapshot_path)

    trainers = {
        "PSMA": trainer_psma,
    }

    trainers[args.dataset](args, net, str(snapshot_path), multimask_output, low_res)


if __name__ == "__main__":
    main()