import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from utils import DiceLossV2
from datasets.dataset_psma import PSMADataset, TrainTransform, ValTransform


class SimpleWriter:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.scalar_file = self.log_dir / "scalars.csv"
        self.image_dir = self.log_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self._scalar_fp = open(self.scalar_file, "a", newline="")
        self._scalar_writer = csv.writer(self._scalar_fp)
        if self.scalar_file.stat().st_size == 0:
            self._scalar_writer.writerow(["tag", "step", "value"])

    def add_scalar(self, tag, value, step):
        self._scalar_writer.writerow([tag, step, float(value)])
        self._scalar_fp.flush()

    def add_image(self, tag, image_tensor, step):
        """
        image_tensor expected shape:
          - [H, W]
          - [1, H, W]
          - [3, H, W]
        """
        safe_tag = tag.replace("/", "_")
        save_path = self.image_dir / f"{safe_tag}_step_{step}.png"

        img = image_tensor.detach().cpu()
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.dtype != torch.float32:
            img = img.float()

        img_min, img_max = img.min(), img.max()
        if img_max > 1.0 or img_min < 0.0:
            img = (img - img_min) / (img_max - img_min + 1e-8)

        save_image(img, str(save_path))

    def close(self):
        if self._scalar_fp is not None:
            self._scalar_fp.close()
            self._scalar_fp = None


def calc_loss(outputs, target_labels, ce_loss, dice_loss, dice_weight: float = 0.8):
    pred_logits = outputs["masks"]
    loss_ce = ce_loss(pred_logits, target_labels.long())
    loss_dice = dice_loss(pred_logits, target_labels)
    total_loss = (1.0 - dice_weight) * loss_ce + dice_weight * loss_dice
    return total_loss, loss_ce, loss_dice


def save_lora_checkpoint(model, save_path):
    try:
        model.save_lora_parameters(save_path)
    except AttributeError:
        model.module.save_lora_parameters(save_path)


def set_optimizer(args, model, initial_lr):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.AdamW:
        return optim.AdamW(
            params,
            lr=initial_lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
    return optim.SGD(
        params,
        lr=initial_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )


def update_learning_rate(optimizer, base_lr, iter_num, max_iterations, warmup, warmup_period):
    if warmup and iter_num < warmup_period:
        lr = base_lr * (iter_num + 1) / warmup_period
    else:
        shift_iter = iter_num - warmup_period if warmup else iter_num
        effective_max_iterations = max(1, max_iterations - warmup_period) if warmup else max_iterations
        progress = min(max(shift_iter / effective_max_iterations, 0.0), 1.0)
        lr = base_lr * (1.0 - progress) ** 0.9

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def compute_seg_dice_stats(pred, target, num_classes, eps=1e-5):
    """
    pred:   [B, H, W] predicted class ids
    target: [B, H, W] ground truth class ids

    Returns:
        dice_sum: sum of Dice scores over valid (sample, class) pairs
        valid_count: number of valid (sample, class) pairs

    Valid means the class is present in pred or target.
    Absent-in-both cases are excluded from aggregation.
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"

    reduce_dims = tuple(range(1, pred.ndim))
    dice_sum = 0.0
    valid_count = 0

    for cls in range(1, num_classes + 1):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()

        intersect = (pred_c * target_c).sum(dim=reduce_dims)
        pred_sum = pred_c.sum(dim=reduce_dims)
        target_sum = target_c.sum(dim=reduce_dims)
        denom = pred_sum + target_sum

        valid = denom > 0
        if valid.any():
            dice = (2.0 * intersect[valid] + eps) / (denom[valid] + eps)
            dice_sum += dice.sum().item()
            valid_count += valid.sum().item()

    return dice_sum, valid_count


def _extract_mask_from_sample(sample):
    """
    Helper for class-weight computation before DataLoader batching.
    Expected dataset item:
      - dict with "label" key
      - or tuple/list where mask is second item

    Mask shape:
      - [H, W]
      - [1, H, W]
    """
    if isinstance(sample, dict):
        if "label" not in sample:
            raise KeyError("Dataset sample dict must contain key 'label'")
        mask = sample["label"]
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        mask = sample[1]
    else:
        raise TypeError("Unsupported dataset sample format for mask extraction")

    mask = torch.as_tensor(mask)
    if mask.dim() == 3 and mask.size(0) == 1:
        mask = mask.squeeze(0)
    if mask.dim() != 2:
        raise ValueError(f"Expected mask with shape [H, W] or [1, H, W], got {tuple(mask.shape)}")
    return mask.long()


def compute_class_pixel_counts(dataset, num_classes_total):
    """
    Computes per-class pixel counts over the training dataset.

    num_classes_total:
        total number of label ids including background.
        Example:
          foreground classes = args.num_classes
          background = 1
          => num_classes_total = args.num_classes + 1
    """
    counts = torch.zeros(num_classes_total, dtype=torch.long)
    for idx in tqdm(range(len(dataset)), desc="Computing class pixel counts", ncols=80):
        sample = dataset[idx]
        mask = _extract_mask_from_sample(sample)
        vals, cnts = torch.unique(mask, return_counts=True)
        valid = (vals >= 0) & (vals < num_classes_total)
        vals = vals[valid]
        cnts = cnts[valid]
        counts[vals] += cnts.cpu()
    return counts


def make_ce_weights_inverse_sqrt(
    counts,
    clamp_max=5.0,
    normalize=True,
    background_scale=1.0,
    eps=1e-6,
):
    """
    Stable segmentation class weights:
        w_c = 1 / sqrt(count_c)

    Then optionally:
      - downweight background
      - clamp max
      - normalize mean weight to ~1
    """
    counts = counts.float()
    weights = torch.zeros_like(counts)

    nonzero = counts > 0
    weights[nonzero] = 1.0 / torch.sqrt(counts[nonzero] + eps)

    # background assumed class 0
    if len(weights) > 0:
        weights[0] = weights[0] * background_scale

    if clamp_max is not None:
        weights = torch.clamp(weights, max=clamp_max)

    if normalize:
        nz = weights > 0
        if nz.any():
            weights[nz] = weights[nz] / weights[nz].mean()

    return weights


def get_class_weights_csv_path(root_path):
    return os.path.join(root_path, "class_weights.csv")


def save_class_weights_csv(csv_path, class_counts, class_weights):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "pixel_count", "weight"])
        for class_id, (cnt, w) in enumerate(zip(class_counts.tolist(), class_weights.tolist())):
            writer.writerow([class_id, int(cnt), float(w)])


def load_class_weights_csv(csv_path, num_classes_total):
    class_counts = torch.zeros(num_classes_total, dtype=torch.long)
    class_weights = torch.zeros(num_classes_total, dtype=torch.float32)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != num_classes_total:
        raise ValueError(
            f"Class weight CSV at {csv_path} has {len(rows)} rows, "
            f"but expected {num_classes_total} classes."
        )

    for row in rows:
        class_id = int(row["class_id"])
        if class_id < 0 or class_id >= num_classes_total:
            raise ValueError(f"Invalid class_id={class_id} in {csv_path}")

        class_counts[class_id] = int(float(row["pixel_count"]))
        class_weights[class_id] = float(row["weight"])

    return class_counts, class_weights


def get_or_create_class_weights(root_path, train_dataset, num_classes_total, args):
    csv_path = get_class_weights_csv_path(root_path)
    force_recompute = getattr(args, "recompute_class_weights", False)

    if os.path.exists(csv_path) and not force_recompute:
        logging.info(f"Found existing class weights CSV: {csv_path}. Loading without recomputation.")
        class_counts, class_weights = load_class_weights_csv(csv_path, num_classes_total)
    else:
        if force_recompute and os.path.exists(csv_path):
            logging.info(f"Recomputing class weights despite existing CSV: {csv_path}")
        else:
            logging.info(f"No class weights CSV found at {csv_path}. Computing class weights.")

        class_counts = compute_class_pixel_counts(train_dataset, num_classes_total)
        class_weights = make_ce_weights_inverse_sqrt(
            class_counts,
            clamp_max=getattr(args, "ce_weight_clamp_max", 5.0),
            normalize=True,
            background_scale=getattr(args, "ce_background_scale", 1.0),
        )
        save_class_weights_csv(csv_path, class_counts, class_weights)
        logging.info(f"Saved class weights CSV to: {csv_path}")

    return class_counts, class_weights


@torch.no_grad()
def validate_psma(args, model, valloader, ce_loss, dice_loss, multimask_output):
    model.eval()

    val_loss_total = 0.0
    val_ce_total = 0.0
    val_dice_loss_total = 0.0
    val_metric_dice_total = 0.0
    val_metric_dice_count = 0
    num_batches = 0

    for sampled_batch in valloader:
        image_batch = sampled_batch["image"].cuda(non_blocking=True)
        label_batch = sampled_batch["label"].cuda(non_blocking=True)

        outputs = model(image_batch, multimask_output, args.img_size)
        loss, loss_ce, loss_dice = calc_loss(
            outputs,
            label_batch,
            ce_loss,
            dice_loss,
            args.dice_param,
        )

        pred_masks = outputs["masks"]
        pred_masks = torch.argmax(torch.softmax(pred_masks, dim=1), dim=1)

        dice_sum, dice_count = compute_seg_dice_stats(
            pred_masks,
            label_batch,
            args.num_classes,
        )

        val_loss_total += loss.item()
        val_ce_total += loss_ce.item()
        val_dice_loss_total += loss_dice.item()
        val_metric_dice_total += dice_sum
        val_metric_dice_count += dice_count
        num_batches += 1

    results = {
        "loss": val_loss_total / max(1, num_batches),
        "loss_ce": val_ce_total / max(1, num_batches),
        "loss_dice": val_dice_loss_total / max(1, num_batches),
        "mean_dice": (
            val_metric_dice_total / val_metric_dice_count
            if val_metric_dice_count > 0 else 0.0
        ),
    }

    model.train()
    return results


def trainer_psma(args, model, snapshot_path, multimask_output, low_res):
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    num_classes = args.num_classes
    num_classes_total = num_classes + 1  # include background

    train_dataset = PSMADataset(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([
            TrainTransform(
                output_size=[args.img_size, args.img_size],
                low_res=[low_res, low_res],
            )
        ]),
    )

    val_dataset = PSMADataset(
        base_dir=args.root_path,
        split="val",
        transform=transforms.Compose([
            ValTransform(
                output_size=[args.img_size, args.img_size],
                low_res=[low_res, low_res],
            )
        ]),
    )

    print(f"The length of train set is: {len(train_dataset)}")
    print(f"The length of val set is: {len(val_dataset)}")

    class_counts, class_weights = get_or_create_class_weights(
        root_path=args.root_path,
        train_dataset=train_dataset,
        num_classes_total=num_classes_total,
        args=args,
    )
    logging.info(f"class pixel counts: {class_counts.tolist()}")
    logging.info(f"class CE weights: {class_weights.tolist()}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_loss = CrossEntropyLoss(weight=class_weights.cuda())
    dice_loss = DiceLossV2(
        num_classes_total,
        include_background=True if num_classes == 1 else False,
    )

    initial_lr = base_lr / args.warmup_period if args.warmup else base_lr
    optimizer = set_optimizer(args, model, initial_lr)

    writer = SimpleWriter(os.path.join(snapshot_path, "log"))

    for c, (cnt, w) in enumerate(zip(class_counts.tolist(), class_weights.tolist())):
        writer.add_scalar(f"class_stats/count_class_{c}", cnt, 0)
        writer.add_scalar(f"class_stats/weight_class_{c}", w, 0)

    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    save_interval = int(0.1 * max_epoch)
    val_interval = getattr(args, "val_interval", save_interval)
    best_val_dice = -1.0

    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    epoch_iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in epoch_iterator:
        model.train()

        for _, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].cuda(non_blocking=True)
            label_batch = sampled_batch["label"].cuda(non_blocking=True)
            low_res_label_batch = sampled_batch["low_res_label"].cuda(non_blocking=True)

            assert image_batch.max() <= 3, f"image_batch max: {image_batch.max()}"

            outputs = model(image_batch, multimask_output, args.img_size)

            loss, loss_ce, loss_dice = calc_loss(
                outputs,
                label_batch,
                ce_loss,
                dice_loss,
                args.dice_param,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = update_learning_rate(
                optimizer=optimizer,
                base_lr=base_lr,
                iter_num=iter_num,
                max_iterations=max_iterations,
                warmup=args.warmup,
                warmup_period=args.warmup_period,
            )
            iter_num += 1

            weighted_ce = (1.0 - args.dice_param) * loss_ce.item()
            weighted_dice = args.dice_param * loss_dice.item()

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("train/total_loss", loss.item(), iter_num)
            writer.add_scalar("train/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("train/loss_dice", loss_dice.item(), iter_num)
            writer.add_scalar("train/weighted_loss_ce", weighted_ce, iter_num)
            writer.add_scalar("train/weighted_loss_dice", weighted_dice, iter_num)
            writer.add_scalar("train/soft_dice_score", 1.0 - loss_dice.item(), iter_num)

            logging.info(
                "iteration %d : loss=%f, loss_ce=%f, loss_dice=%f, weighted_ce=%f, weighted_dice=%f",
                iter_num,
                loss.item(),
                loss_ce.item(),
                loss_dice.item(),
                weighted_ce,
                weighted_dice,
            )

            if iter_num % 1000 == 0 and image_batch.shape[0] > 1:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                writer.add_image("train/Image", image, iter_num)

                pred_masks = outputs["masks"]
                pred_masks = torch.argmax(
                    torch.softmax(pred_masks, dim=1),
                    dim=1,
                    keepdim=True,
                )
                writer.add_image("train/Prediction", pred_masks[1] * 50, iter_num)

                gt_mask = label_batch[1].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", gt_mask, iter_num)

        if (epoch_num + 1) % val_interval == 0:
            val_results = validate_psma(
                args=args,
                model=model,
                valloader=valloader,
                ce_loss=ce_loss,
                dice_loss=dice_loss,
                multimask_output=multimask_output,
            )

            writer.add_scalar("val/total_loss", val_results["loss"], epoch_num + 1)
            writer.add_scalar("val/loss_ce", val_results["loss_ce"], epoch_num + 1)
            writer.add_scalar("val/loss_dice", val_results["loss_dice"], epoch_num + 1)
            writer.add_scalar("val/mean_dice", val_results["mean_dice"], epoch_num + 1)

            logging.info(
                "epoch %d validation : val_loss=%f, val_ce=%f, val_dice_loss=%f, val_mean_dice=%f",
                epoch_num + 1,
                val_results["loss"],
                val_results["loss_ce"],
                val_results["loss_dice"],
                val_results["mean_dice"],
            )

            if val_results["mean_dice"] > best_val_dice:
                best_val_dice = val_results["mean_dice"]
                best_path = os.path.join(snapshot_path, "best_model.pth")
                save_lora_checkpoint(model, best_path)
                logging.info(f"save best model to {best_path}, best_val_dice={best_val_dice:.6f}")

        if (epoch_num + 1) % save_interval == 0:
            save_path = os.path.join(snapshot_path, f"epoch_{epoch_num}.pth")
            save_lora_checkpoint(model, save_path)
            logging.info(f"save model to {save_path}")

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_path = os.path.join(snapshot_path, f"epoch_{epoch_num}.pth")
            save_lora_checkpoint(model, save_path)
            logging.info(f"save model to {save_path}")
            epoch_iterator.close()
            break

    writer.close()
    return "Training Finished!"
