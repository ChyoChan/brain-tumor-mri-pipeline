#!/usr/bin/env python3
"""Train UNet, ResUNet, or Attention U-Net from configs/pipeline_config.yaml."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from src.preprocessing import (
    SegmentationDataset,
    discover_image_mask_pairs,
    stratified_split_pairs,
)
from src.segmentation import build_segmentation_model, train_unet
from src.utils import get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pipeline_config.yaml"))
    parser.add_argument(
        "--model",
        default="unet",
        choices=["unet", "resunet", "attention_unet"],
    )
    parser.add_argument("--data-dir", default=None, help="Override paths.raw_data_dir")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device()

    data_dir = args.data_dir or cfg["paths"]["raw_data_dir"]
    seg_cfg = cfg["segmentation"]
    ratios = tuple(cfg["dataset"]["split_ratios"])
    save_path = args.save_path or str(
        ROOT / "models" / "segmentation" / f"best_{args.model}.pth"
    )
    epochs = args.epochs if args.epochs is not None else seg_cfg["epochs"]

    pairs = discover_image_mask_pairs(data_dir, allow_pseudo_fallback=True)
    train_pairs, val_pairs, _test_pairs = stratified_split_pairs(
        pairs, ratios=ratios, seed=cfg.get("seed", 42)
    )
    train_loader = DataLoader(
        SegmentationDataset(train_pairs, img_size=seg_cfg["img_size"], augment=True),
        batch_size=seg_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_pairs, img_size=seg_cfg["img_size"], augment=False),
        batch_size=seg_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    model = build_segmentation_model(
        args.model,
        in_channels=seg_cfg["in_channels"],
        out_channels=seg_cfg["out_channels"],
        base_filters=seg_cfg["base_filters"],
    )
    history = train_unet(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        lr=seg_cfg["learning_rate"],
        patience=seg_cfg["patience"],
        save_path=save_path,
    )
    metrics_path = ROOT / "outputs" / "metrics" / f"{args.model}_train_history.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(history, indent=2))
    print(f"Saved history to {metrics_path}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
