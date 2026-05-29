import argparse
import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


def calc_loss(outputs, low_res_labels, ce_loss, dice_loss, dice_weight: float = 0.8):
    low_res_logits = outputs["low_res_logits"]
    loss_ce = ce_loss(low_res_logits, low_res_labels.long())
    loss_dice = dice_loss(low_res_logits, low_res_labels, softmax=True)
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
        lr = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def trainer_psma(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_psma import PSMADataset, RandomGenerator

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

    train_dataset = PSMADataset(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(
                output_size=[args.img_size, args.img_size],
                low_res=[low_res, low_res],
            )
        ]),
    )
    print(f"The length of train set is: {len(train_dataset)}")

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

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    initial_lr = base_lr / args.warmup_period if args.warmup else base_lr
    optimizer = set_optimizer(args, model, initial_lr)

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    save_interval = 20

    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    epoch_iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in epoch_iterator:
        for _, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
            low_res_label_batch = sampled_batch["low_res_label"].cuda()

            assert image_batch.max() <= 3, f"image_batch max: {image_batch.max()}"

            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(
                outputs,
                low_res_label_batch,
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

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)
            writer.add_scalar("info/loss_dice", loss_dice, iter_num)

            logging.info(
                "iteration %d : loss : %f, loss_ce: %f, loss_dice: %f",
                iter_num,
                loss.item(),
                loss_ce.item(),
                loss_dice.item(),
            )

            if iter_num % 20 == 0 and image_batch.shape[0] > 1:
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