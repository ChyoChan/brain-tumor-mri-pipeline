#!/usr/bin/env python3
"""Evaluate a segmentation checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    SegmentationDataset,
    discover_image_mask_pairs,
    stratified_split_pairs,
)
from src.segmentation import build_segmentation_model, evaluate_segmentation
from src.utils import get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pipeline_config.yaml"))
    parser.add_argument("--model", default="unet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    seg_cfg = cfg["segmentation"]
    data_dir = args.data_dir or cfg["paths"]["raw_data_dir"]

    pairs = discover_image_mask_pairs(data_dir, allow_pseudo_fallback=True)
    _train, _val, test_pairs = stratified_split_pairs(
        pairs, tuple(cfg["dataset"]["split_ratios"]), seed=cfg.get("seed", 42)
    )
    loader = DataLoader(
        SegmentationDataset(test_pairs, img_size=seg_cfg["img_size"], augment=False),
        batch_size=seg_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
    )
    model = build_segmentation_model(
        args.model,
        in_channels=seg_cfg["in_channels"],
        out_channels=seg_cfg["out_channels"],
        base_filters=seg_cfg["base_filters"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    avg, std = evaluate_segmentation(model, loader, device)
    print(json.dumps({"avg": avg, "std": std}, indent=2))


if __name__ == "__main__":
    main()
