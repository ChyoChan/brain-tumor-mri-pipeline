#!/usr/bin/env python3
"""Train ResNet50 on full images, cropped patches, or dual-input (patch + original)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification import (
    BrainTumorDataset,
    DualInputBrainTumorDataset,
    build_original_lookup,
    build_resnet50,
    build_resnet50_dual_input,
    compute_class_weights,
    get_train_transforms,
    get_val_transforms,
    train_classifier,
)
from src.constants import CLASS_NAMES
from src.data_splits import load_classification_manifest
from src.checkpoints import classification_path, sha256_file
from src.segmentation import build_segmentation_model
from src.utils import get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pipeline_config.yaml"))
    parser.add_argument("--mode", choices=["full", "cropped", "dual"], default="full")
    parser.add_argument("--seg-checkpoint", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--manifest", required=True, help="Frozen classification split JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device()

    cls_cfg = cfg["classification"]
    if cfg["dataset"]["classes"] != CLASS_NAMES or cfg["dataset"]["num_classes"] != len(CLASS_NAMES):
        raise ValueError("Configured class order must match the deployed class order")
    data_dir = cfg["paths"]["raw_data_dir"]
    split = load_classification_manifest(args.manifest, data_dir=ROOT / data_dir)
    train_samples, val_samples = split.samples["train"], split.samples["val"]

    input_size = cls_cfg["input_size"]
    train_tf = get_train_transforms(input_size, cls_cfg.get("augmentation"))
    val_tf = get_val_transforms(input_size)
    save_path = args.save_path or str(classification_path(ROOT, args.mode))
    if input_size != 224:
        raise ValueError("The deployed classifier preprocessing requires input_size=224")
    metadata = {
        "mode": args.mode, "architecture": "resnet50_dual" if args.mode == "dual" else "resnet50",
        "class_names": CLASS_NAMES, "input_size": input_size,
        "normalization": "grayscale_imagenet_v1",
        "crop_contract": "full_image" if args.mode == "full" else "roi_v1",
        "split_fingerprint": split.fingerprint, "dataset_fingerprint": split.dataset_fingerprint,
    }
    epochs = args.epochs if args.epochs is not None else cls_cfg["epochs"]

    if args.mode == "full":
        train_ds = BrainTumorDataset(train_samples, train_tf, grayscale=True)
        val_ds = BrainTumorDataset(val_samples, val_tf, grayscale=True)
        model = build_resnet50(cfg["dataset"]["num_classes"], cls_cfg["pretrained"])
        train_labels = [label for _, label in train_ds.samples]
    else:
        from src.preprocessing import generate_classification_patches

        ckpt = args.seg_checkpoint or str(ROOT / cfg["segmentation"]["save_path"])
        seg_model = build_segmentation_model(
            cfg["segmentation"].get("model", "UNet"),
            in_channels=cfg["segmentation"]["in_channels"],
            out_channels=cfg["segmentation"]["out_channels"],
            base_filters=cfg["segmentation"]["base_filters"],
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        seg_model.load_state_dict(state)
        seg_model.eval()
        seg_name = cfg["segmentation"].get("model", "unet").lower()
        if seg_name in {"attentionunet", "attention-unet"}:
            seg_name = "attention_unet"
        metadata["segmentation"] = {
            "model": seg_name,
            "sha256": sha256_file(ckpt),
        }
        processed = ROOT / cfg["paths"]["processed_data_dir"] / split.fingerprint[:12]
        train_patches = generate_classification_patches(
            train_samples, seg_model, device, processed / "patches" / "train"
        )
        val_patches = generate_classification_patches(
            val_samples, seg_model, device, processed / "patches" / "val"
        )
        if args.mode == "cropped":
            train_ds = BrainTumorDataset(train_patches, train_tf, grayscale=True)
            val_ds = BrainTumorDataset(val_patches, val_tf, grayscale=True)
            model = build_resnet50(cfg["dataset"]["num_classes"], cls_cfg["pretrained"])
            train_labels = [label for _, label in train_ds.samples]
        else:
            train_ds = DualInputBrainTumorDataset(
                train_patches,
                build_original_lookup(train_samples),
                patch_transform=train_tf,
                original_transform=val_tf,
                grayscale=True,
            )
            val_ds = DualInputBrainTumorDataset(
                val_patches,
                build_original_lookup(val_samples),
                patch_transform=val_tf,
                original_transform=val_tf,
                grayscale=True,
            )
            model = build_resnet50_dual_input(
                cfg["dataset"]["num_classes"], cls_cfg["pretrained"]
            )
            train_labels = [label for _, label in train_ds.samples]

    train_loader = DataLoader(
        train_ds, batch_size=cls_cfg["batch_size"], shuffle=True,
        num_workers=cls_cfg.get("num_workers", 2),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cls_cfg["batch_size"], shuffle=False,
        num_workers=cls_cfg.get("num_workers", 2),
    )
    class_weights = compute_class_weights(train_labels, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cls_cfg["learning_rate"],
        weight_decay=cls_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    train_classifier(
        model.to(device),
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=epochs,
        patience=cls_cfg["patience"],
        save_path=save_path,
        checkpoint_metadata=metadata,
    )
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
