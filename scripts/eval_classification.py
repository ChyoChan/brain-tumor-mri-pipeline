#!/usr/bin/env python3
"""Evaluate deployed classification preprocessing on a frozen val/test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import (
    calibration_metrics, fit_temperature, load_temperature_artifact,
    save_temperature_artifact, temperature_probabilities,
)
from src.checkpoints import load_metadata
from src.constants import CLASS_NAMES
from src.data_splits import file_sha256, load_classification_manifest


def _write_reliability_plot(path: Path, before: dict, after: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, title, measures in zip(axes, ("Before temperature scaling", "After temperature scaling"), (before, after)):
        bins = [row for row in measures["bins"] if row["count"]]
        scores = [row["confidence"] for row in bins]
        accuracy = [row["accuracy"] for row in bins]
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        axis.plot(scores, accuracy, "o-", label="Observed accuracy")
        for row in bins:
            axis.annotate(f"n={row['count']}", (row["confidence"], row["accuracy"]), xytext=(3, 6), textcoords="offset points", fontsize=7)
        axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Mean model score", ylabel="Fraction correct", title=f"{title}\nECE={measures['ece']:.3f}; NLL={measures['nll']:.3f}")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="upper left")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, help="Override the manifest's relative data root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--mode", choices=("full", "dual", "cropped"), default="full")
    parser.add_argument("--seg-model", choices=("unet", "resunet", "attention_unet"), default="resunet")
    parser.add_argument("--checkpoint", type=Path, help="Override the selected classifier's checkpoint")
    parser.add_argument("--full-checkpoint", type=Path, help="Override the full-image gate/reference checkpoint")
    parser.add_argument("--seg-checkpoint", type=Path)
    parser.add_argument("--temperature", type=Path, help="Apply an already fitted temperature JSON")
    parser.add_argument("--fit-temperature", type=Path, help="Fit using --split val and save a temperature JSON")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/metrics/classification_evaluation.json")
    args = parser.parse_args(argv)
    if args.fit_temperature and args.split != "val":
        parser.error("Temperature fitting is allowed only with --split val; never fit on the test set")
    if args.temperature and args.fit_temperature:
        parser.error("Choose --temperature or --fit-temperature")

    split = load_classification_manifest(args.manifest, args.data_dir)
    from app.inference import PipelineRuntime

    classification = {}
    if args.full_checkpoint:
        classification["full"] = args.full_checkpoint
    if args.checkpoint:
        classification[args.mode] = args.checkpoint
    segmentation = {args.seg_model: args.seg_checkpoint} if args.seg_checkpoint else None
    runtime = PipelineRuntime(
        classification_checkpoints=classification or None,
        segmentation_checkpoints=segmentation, calibration_path="",
    )
    checkpoint = runtime._resolve_cls_path(args.mode)
    if checkpoint is None:
        raise FileNotFoundError(f"No {args.mode} classifier checkpoint is installed")
    checkpoint_hash = file_sha256(checkpoint)
    metadata = load_metadata(checkpoint, args.mode)
    seg_hash = None
    reference_hash = None
    if args.mode != "full":
        seg_path = runtime._resolve_seg_path(args.seg_model)
        if seg_path is None:
            raise FileNotFoundError(f"No {args.seg_model} segmentation checkpoint is installed")
        seg_hash = file_sha256(seg_path)
    if args.mode == "dual" and metadata.get("crop_contract") == "legacy_gate_v1":
        reference_path = runtime._resolve_full_cls()
        if reference_path is None:
            raise FileNotFoundError("Dual inference needs a full-image reference checkpoint")
        reference_hash = file_sha256(reference_path)
    training_split_verified = bool(metadata.get("split_fingerprint"))
    if training_split_verified and metadata["split_fingerprint"] != split.fingerprint:
        raise ValueError("Checkpoint was trained with a different frozen split manifest")
    if not training_split_verified:
        print("Checkpoint has no training split metadata; overlap with this split cannot be verified.")
        if args.fit_temperature:
            parser.error("Temperature fitting requires checkpoint metadata tied to this frozen split; retrain first")

    logits, labels, rows = [], [], []
    total_seconds = 0.0
    for index, (image_path, label) in enumerate(split.samples[args.split], 1):
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        start = time.perf_counter()
        result = runtime.predict(rgb, args.seg_model, args.mode)
        seconds = time.perf_counter() - start
        total_seconds += seconds
        logits.append(result["logits"])
        labels.append(label)
        rows.append({"path": Path(image_path).relative_to(split.root).as_posix(), "label": CLASS_NAMES[label], "seconds": seconds})
        if index % 25 == 0:
            print(f"Evaluated {index}/{len(split.samples[args.split])} images")

    logits_array = np.asarray(logits, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    temperature = 1.0
    seg_name = None if args.mode == "full" else args.seg_model
    if args.fit_temperature:
        temperature = fit_temperature(logits_array, label_array)
        save_temperature_artifact(
            args.fit_temperature, temperature=temperature, checkpoint_sha256=checkpoint_hash,
            split_fingerprint=split.fingerprint, mode=args.mode, segmentation_model=seg_name,
            segmentation_sha256=seg_hash, reference_full_sha256=reference_hash,
        )
    elif args.temperature:
        artifact = load_temperature_artifact(
            args.temperature, checkpoint_sha256=checkpoint_hash, mode=args.mode,
            segmentation_model=seg_name, split_fingerprint=split.fingerprint,
            segmentation_sha256=seg_hash, reference_full_sha256=reference_hash,
        )
        temperature = artifact["temperature"]

    probabilities = temperature_probabilities(logits_array, temperature)
    predictions = probabilities.argmax(axis=1)
    before = calibration_metrics(logits_array, label_array)
    after = calibration_metrics(logits_array, label_array, temperature)
    result = {
        "split": args.split, "mode": args.mode, "segmentation_model": seg_name,
        "class_names": CLASS_NAMES, "count": len(labels), "temperature": temperature,
        "checkpoint_sha256": checkpoint_hash, "split_fingerprint": split.fingerprint,
        "segmentation_sha256": seg_hash, "reference_full_sha256": reference_hash,
        "dataset_fingerprint": split.dataset_fingerprint, "training_split_verified": training_split_verified,
        "accuracy": float(accuracy_score(label_array, predictions)),
        "macro_f1": float(f1_score(label_array, predictions, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0)),
        "per_class_recall": dict(zip(CLASS_NAMES, recall_score(label_array, predictions, labels=range(len(CLASS_NAMES)), average=None, zero_division=0).tolist())),
        "confusion_matrix": confusion_matrix(label_array, predictions, labels=range(len(CLASS_NAMES))).tolist(),
        "inference_seconds": total_seconds, "seconds_per_image": total_seconds / len(labels),
        "calibration_before": {key: value for key, value in before.items() if key != "bins"},
        "calibration_after": {key: value for key, value in after.items() if key != "bins"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    reliability_path = args.output.with_suffix(".reliability.csv")
    with reliability_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stage", "lower", "upper", "count", "accuracy", "confidence"))
        writer.writeheader()
        for stage, measures in (("before", before), ("after", after)):
            writer.writerows({"stage": stage, **row} for row in measures["bins"])
    with args.output.with_suffix(".predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label", "prediction", "score", "seconds"))
        writer.writeheader()
        for row, prediction, scores in zip(rows, predictions, probabilities):
            writer.writerow({**row, "prediction": CLASS_NAMES[prediction], "score": float(scores[prediction])})
    plot_path = args.output.with_suffix(".reliability.png")
    _write_reliability_plot(plot_path, before, after)
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation: {args.output}; reliability: {reliability_path}, {plot_path}")


if __name__ == "__main__":
    main()
