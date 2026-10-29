"""Validation-only temperature scaling and reliability measurements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import CLASS_NAMES


def _validated(logits, labels=None):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty logits with one column per class")
    if labels is None:
        return values
    targets = np.asarray(labels)
    if targets.shape != (len(values),) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Expected one integer label per image")
    if (targets < 0).any() or (targets >= len(CLASS_NAMES)).any():
        raise ValueError("Label is outside the class order")
    return values, targets.astype(np.int64)


def temperature_probabilities(logits, temperature: float = 1.0) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values = _validated(logits) / temperature
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels) -> float:
    values, targets = _validated(logits, labels)
    tensor = torch.as_tensor(values)
    target = torch.as_tensor(targets)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 100)
        loss = F.cross_entropy(tensor / temperature, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 100))
    before = F.cross_entropy(tensor, target)
    after = F.cross_entropy(tensor / temperature, target)
    return temperature if after <= before else 1.0


def calibration_metrics(logits, labels, temperature: float = 1.0, n_bins: int = 10) -> dict:
    values, targets = _validated(logits, labels)
    if n_bins < 1:
        raise ValueError("Reliability plot needs at least one bin")
    probabilities = temperature_probabilities(values, temperature)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    bins, ece = [], 0.0
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        selected = (confidence >= lower) & ((confidence < upper) if index < n_bins - 1 else (confidence <= upper))
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean_score = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(targets) * abs(accuracy - mean_score)
        bins.append({"lower": lower, "upper": upper, "count": count, "accuracy": accuracy, "confidence": mean_score})
    nll = float(F.cross_entropy(torch.as_tensor(values / temperature), torch.as_tensor(targets)))
    return {"ece": float(ece), "nll": nll, "bins": bins}


def save_temperature_artifact(
    path: str | Path, *, temperature: float, checkpoint_sha256: str,
    split_fingerprint: str, mode: str, segmentation_model: str | None = None,
    segmentation_sha256: str | None = None, reference_full_sha256: str | None = None,
) -> dict:
    artifact = {
        "version": 1, "temperature": float(temperature), "checkpoint_sha256": checkpoint_sha256,
        "split_fingerprint": split_fingerprint, "fitted_split": "val", "class_names": CLASS_NAMES,
        "mode": mode, "segmentation_model": segmentation_model,
        "segmentation_sha256": segmentation_sha256, "reference_full_sha256": reference_full_sha256,
    }
    _validate_artifact(artifact)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def _validate_artifact(artifact: dict) -> None:
    if artifact.get("version") != 1 or artifact.get("fitted_split") != "val" or artifact.get("class_names") != CLASS_NAMES:
        raise ValueError("Temperature artifact must use this class order and validation-only fitting")
    temperature = artifact.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature artifact has an invalid temperature")
    keys = ["checkpoint_sha256", "split_fingerprint"]
    if artifact.get("mode") in {"dual", "cropped"}:
        keys.append("segmentation_sha256")
    if artifact.get("mode") == "dual" and artifact.get("reference_full_sha256") is not None:
        keys.append("reference_full_sha256")
    for key in keys:
        value = artifact.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Temperature artifact has an invalid {key}")
    if artifact.get("mode") not in {"full", "dual", "cropped"}:
        raise ValueError("Temperature artifact has an invalid classifier mode")
    if artifact["mode"] == "full":
        if artifact.get("segmentation_model") is not None or artifact.get("segmentation_sha256") is not None:
            raise ValueError("Full-image calibration must not depend on a segmenter")
    elif artifact.get("segmentation_model") not in {"unet", "resunet", "attention_unet"}:
        raise ValueError("Cropped/dual calibration must name its segmenter")
    if artifact["mode"] != "dual" and artifact.get("reference_full_sha256") is not None:
        raise ValueError("Only dual calibration uses a full-image gate fingerprint")


def load_temperature_artifact(
    path: str | Path, *, checkpoint_sha256: str | None = None,
    mode: str | None = None, segmentation_model: str | None = None,
    split_fingerprint: str | None = None,
    segmentation_sha256: str | None = None, reference_full_sha256: str | None = None,
) -> dict:
    artifact = json.loads(Path(path).read_text())
    _validate_artifact(artifact)
    for key, expected in (("checkpoint_sha256", checkpoint_sha256), ("mode", mode), ("split_fingerprint", split_fingerprint)):
        if expected is not None and artifact[key] != expected:
            raise ValueError(f"Temperature artifact does not match {key}")
    if mode is not None and artifact["segmentation_model"] != (None if mode == "full" else segmentation_model):
        raise ValueError("Temperature artifact does not match segmentation model")
    if mode is not None:
        expected_seg = None if mode == "full" else segmentation_sha256
        expected_reference = reference_full_sha256 if mode == "dual" else None
        if artifact.get("segmentation_sha256") != expected_seg:
            raise ValueError("Temperature artifact does not match segmentation checkpoint")
        if artifact.get("reference_full_sha256") != expected_reference:
            raise ValueError("Temperature artifact does not match full-image gate checkpoint")
    return artifact
