"""Classification checkpoint paths and the preprocessing contract saved with weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.constants import CLASS_NAMES


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_path(root: Path, mode: str) -> Path:
    if mode not in {"full", "cropped", "dual"}:
        raise ValueError(f"Unknown classifier mode: {mode}")
    return root / "models" / "classification" / f"best_resnet50_{mode}.pth"


def metadata_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".metadata.json")


def save_metadata(path: str | Path, metadata: dict) -> dict:
    payload = dict(metadata, schema_version=1, checkpoint_sha256=sha256_file(path))
    metadata_path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_metadata(path: str | Path, mode: str, *, required: bool = False) -> dict:
    sidecar = metadata_path(path)
    if not sidecar.exists():
        if required:
            raise ValueError(f"Missing {sidecar.name}; retrain with checkpoint metadata.")
        return {"crop_contract": "legacy_gate_v1" if mode == "dual" else "full_image"}
    data = json.loads(sidecar.read_text())
    expected_arch = "resnet50_dual" if mode == "dual" else "resnet50"
    if (data.get("schema_version") != 1 or data.get("mode") != mode
            or data.get("architecture") != expected_arch or data.get("class_names") != CLASS_NAMES
            or data.get("normalization") != "grayscale_imagenet_v1"
            or data.get("input_size") != 224
            or data.get("crop_contract") != ("full_image" if mode == "full" else "roi_v1")):
        raise ValueError(f"Unsupported classifier metadata in {sidecar.name}")
    if data.get("checkpoint_sha256") != sha256_file(path):
        raise ValueError(f"Checkpoint checksum does not match {sidecar.name}")
    if not isinstance(data.get("split_fingerprint"), str) or not data["split_fingerprint"]:
        raise ValueError("Checkpoint metadata needs a split fingerprint")
    if mode != "full":
        seg = data.get("segmentation", {})
        if seg.get("model") not in {"unet", "resunet", "attention_unet"} or not isinstance(seg.get("sha256"), str):
            raise ValueError("Checkpoint metadata needs its segmentation model and checksum")
    return data
