"""Single-image inference for the local frontend."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.classification import build_resnet50, build_resnet50_dual_input, get_val_transforms
from src.constants import CLASS_NAMES
from src.checkpoints import classification_path, load_metadata, sha256_file
from src.calibration import load_temperature_artifact, temperature_probabilities
from src.explainability import (
    compute_stage_consistency_iou,
    generate_gradcam,
    generate_template_report,
    overlay_gradcam,
)
from src.roi import crop_tumor_region
from src.segmentation import build_segmentation_model, predict_mask
from src.utils import get_device, load_config, project_root

logger = logging.getLogger(__name__)

ROOT = project_root()
MODELS_ROOT = (ROOT / "models").resolve()
MAX_IMAGE_PIXELS = 25_000_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

SEG_CHECKPOINTS = {
    "unet": ROOT / "models" / "segmentation" / "best_unet_real_mask.pth",
    "resunet": ROOT / "models" / "segmentation" / "best_resunet.pth",
    "attention_unet": ROOT / "models" / "segmentation" / "best_attention_unet.pth",
}
SEG_FALLBACKS = {
    "unet": ROOT / "models" / "segmentation" / "unet_full_checkpoint.pth",
    "resunet": ROOT / "models" / "segmentation" / "resunet_full_checkpoint.pth",
}
CLS_DUAL = ROOT / "models" / "classification" / "best_resnet50_dual_input.pth"
CLS_FULL = ROOT / "models" / "classification" / "pure_baseline" / "best_resnet50_pure_classification.pth"
CLS_FULL_ALT = ROOT / "models" / "classification" / "pure_baseline" / "best_resnet50.pth"
SAMPLE_ROOT = ROOT / "data" / "raw" / "Classification" / "Testing"
SAMPLE_LABELS = {
    "glioma": "Glioma", "meningioma": "Meningioma",
    "pituitary": "Pituitary", "notumor": "No tumor",
}

SAMPLE_SPECS = [
    {
        "id": "glioma",
        "label": "Glioma",
        "path": ROOT / "data" / "raw" / "Classification" / "Testing" / "glioma" / "enh_Te-glTr_0000.jpg",
    },
    {
        "id": "meningioma",
        "label": "Meningioma",
        "path": ROOT / "data" / "raw" / "Classification" / "Testing" / "meningioma" / "enh_Te-me_0016.jpg",
    },
    {
        "id": "pituitary",
        "label": "Pituitary",
        "path": ROOT / "data" / "raw" / "Classification" / "Testing" / "pituitary" / "enh_Te-piTr_0000.jpg",
    },
    {
        "id": "notumor",
        "label": "No tumor",
        "path": ROOT / "data" / "raw" / "Classification" / "Testing" / "notumor" / "enh_Te-noTr_0006.jpg",
    },
]


def _is_under_models(path: Path) -> bool:
    resolved = path.resolve()
    try:
        resolved.relative_to(MODELS_ROOT)
        return True
    except ValueError:
        return False


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
    for key in ("state_dict", "model_state_dict", "model"):
        val = obj.get(key)
        if isinstance(val, dict) and val and any(torch.is_tensor(v) for v in val.values()):
            obj = val
            break
    if any(k.startswith("module.") for k in obj):
        obj = {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}
    return obj


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not _is_under_models(path):
        raise ValueError(f"Refusing to load checkpoint outside models/: {path}")
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        logger.warning("weights_only load failed for %s; retrying unsafe pickle load", path.name)
        obj = torch.load(path, map_location=device, weights_only=False)
    return _extract_state_dict(obj)


def _load_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    state = _load_checkpoint(path, device)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"Checkpoint {path.name} is missing {len(missing)} parameter(s): {preview}"
        )
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        logger.info("Ignoring %s unexpected keys in %s", len(unexpected), path.name)


def _encode_image(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        image = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    color = np.zeros_like(rgb)
    color[..., 1] = 210
    color[..., 2] = 230
    alpha = (mask > 0).astype(np.float32) * 0.42
    alpha = alpha[..., None]
    blended = (rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_dual_batch(
    crop_tensor: torch.Tensor,
    orig_tensor: torch.Tensor,
    gate_idx: int,
) -> Tuple[torch.Tensor, str]:
    """Build the dual input using the checkpoint's deployable crop contract."""
    if not 0 <= gate_idx < len(CLASS_NAMES):
        raise ValueError(f"Invalid gate class index: {gate_idx}")
    gate_is_notumor = CLASS_NAMES[gate_idx] == "notumor"
    first_view = orig_tensor if gate_is_notumor else crop_tensor
    source = "full_image_gate" if gate_is_notumor else "segmentation_roi"
    return torch.cat([first_view, orig_tensor], dim=0).unsqueeze(0), source


def _fallback_gradcam(
    model: nn.Module,
    batch: torch.Tensor,
    target_layer: nn.Module,
    pred_class: int,
) -> np.ndarray:
    activations = []
    gradients = []

    def on_fwd(_module, _inp, out):
        activations.append(out)

    def on_bwd(_module, _gin, gout):
        gradients.append(gout[0])

    h1 = target_layer.register_forward_hook(on_fwd)
    h2 = target_layer.register_full_backward_hook(on_bwd)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        logits[0, pred_class].backward()
        act = activations[0].detach()
        grad = gradients[0].detach()
    finally:
        h1.remove()
        h2.remove()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=batch.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


class PipelineRuntime:
    def __init__(self, *, classification_checkpoints=None, segmentation_checkpoints=None, calibration_path=None) -> None:
        self.cfg = load_config()
        self.device = get_device()
        self._lock = threading.Lock()
        self._seg_cache: Dict[str, nn.Module] = {}
        self._cls_cache: Dict[str, nn.Module] = {}
        self._cls_metadata = {}
        self.classification_checkpoints = classification_checkpoints or {}
        self.segmentation_checkpoints = segmentation_checkpoints or {}
        self.calibration_path = calibration_path if calibration_path is not None else os.environ.get("MRI_CALIBRATION")
        self._calibration_cache = {}
        self._seg_hashes = {}
        self.input_size = int(self.cfg.get("classification", {}).get("input_size", 224))
        self.val_tf = get_val_transforms(self.input_size)
        self._sample_index = self._discover_samples()

    def _discover_samples(self) -> Dict[str, Dict[str, Any]]:
        """Index local test images with opaque IDs, never client-supplied paths."""
        root = SAMPLE_ROOT.resolve()
        samples = {}
        for class_name in CLASS_NAMES:
            folder = root / class_name
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    continue
                if not path.is_file() or not path.resolve().is_relative_to(root):
                    continue
                relative = path.relative_to(root).as_posix()
                sample_id = "local_" + hashlib.sha256(relative.encode()).hexdigest()[:24]
                samples[sample_id] = {
                    "id": sample_id,
                    "class_name": class_name,
                    "label": SAMPLE_LABELS[class_name],
                    "filename": path.name,
                    "path": path,
                }
        return samples

    def list_samples(
        self, class_name: str | None = None, search: str = "",
        page: int = 1, page_size: int = 8,
    ) -> Dict[str, Any]:
        """Return a bounded gallery page without exposing filesystem paths."""
        if class_name is not None and class_name not in CLASS_NAMES:
            raise ValueError("Unknown dataset class")
        if page < 1 or not 1 <= page_size <= 24:
            raise ValueError("Invalid gallery pagination")
        query = search.strip().casefold()
        items = [
            item for item in self._sample_index.values()
            if (class_name is None or item["class_name"] == class_name)
            and query in item["filename"].casefold()
        ]
        total = len(items)
        pages = (total + page_size - 1) // page_size
        page = min(page, max(1, pages))
        start = (page - 1) * page_size
        return {
            "items": [
                {key: value for key, value in item.items() if key != "path"}
                for item in items[start:start + page_size]
            ],
            "total": total, "page": page, "page_size": page_size, "pages": pages,
        }

    def status(self) -> Dict[str, Any]:
        samples = []
        for spec in SAMPLE_SPECS:
            samples.append(
                {"id": spec["id"], "label": spec["label"], "available": spec["path"].exists()}
            )
        return {
            "device": str(self.device),
            "classes": CLASS_NAMES,
            "checkpoints": {
                "segmentation": {
                    name: self._resolve_seg_path(name) is not None
                    for name in ("unet", "resunet", "attention_unet")
                },
                "classification": {
                    "dual": self._resolve_cls_path("dual") is not None and (
                        self._resolve_cls_path("dual") != CLS_DUAL or self._resolve_full_cls() is not None
                    ),
                    "full": self._resolve_full_cls() is not None,
                },
            },
            "samples": samples,
        }

    def sample_path(self, sample_id: str) -> Path:
        for spec in SAMPLE_SPECS:
            if spec["id"] == sample_id:
                if not spec["path"].exists():
                    raise FileNotFoundError(f"Sample not found on disk: {spec['id']}")
                return spec["path"]
        item = self._sample_index.get(sample_id)
        if item is not None:
            path = item["path"].resolve()
            if not path.is_relative_to(SAMPLE_ROOT.resolve()) or not path.is_file():
                raise FileNotFoundError("Local sample is no longer available")
            return path
        raise KeyError(f"Unknown sample id: {sample_id}")

    def _resolve_seg_path(self, name: str) -> Optional[Path]:
        override = getattr(self, "segmentation_checkpoints", {}).get(name)
        if override is not None:
            return Path(override) if Path(override).is_file() else None
        primary = SEG_CHECKPOINTS.get(name)
        if primary is not None and primary.exists():
            return primary
        fallback = SEG_FALLBACKS.get(name)
        if fallback is not None and fallback.exists():
            return fallback
        return None

    def _resolve_full_cls(self) -> Optional[Path]:
        override = getattr(self, "classification_checkpoints", {}).get("full")
        if override is not None:
            return Path(override) if Path(override).is_file() else None
        trained = classification_path(ROOT, "full")
        if trained.is_file():
            return trained
        if CLS_FULL.exists():
            return CLS_FULL
        if CLS_FULL_ALT.exists():
            return CLS_FULL_ALT
        return None

    def _resolve_cls_path(self, mode: str) -> Optional[Path]:
        if mode not in {"full", "dual", "cropped"}:
            raise ValueError(f"Unknown classifier mode: {mode}")
        if mode == "full":
            return self._resolve_full_cls()
        override = getattr(self, "classification_checkpoints", {}).get(mode)
        if override is not None:
            return Path(override) if Path(override).is_file() else None
        trained = classification_path(ROOT, mode)
        if trained.is_file():
            return trained
        return CLS_DUAL if mode == "dual" and CLS_DUAL.exists() else None

    def _load_seg(self, name: str) -> nn.Module:
        if name in self._seg_cache:
            return self._seg_cache[name]
        path = self._resolve_seg_path(name)
        if path is None:
            raise FileNotFoundError(f"No segmentation checkpoint for '{name}'")
        model = build_segmentation_model(name).to(self.device)
        _load_weights(model, path, self.device)
        model.eval()
        self._seg_cache[name] = model
        return model

    def _load_cls(self, mode: str) -> nn.Module:
        if mode in self._cls_cache:
            return self._cls_cache[mode]
        if mode == "dual":
            path = self._resolve_cls_path(mode)
            if path is None:
                raise FileNotFoundError("Dual-input checkpoint missing")
            model = build_resnet50_dual_input(len(CLASS_NAMES), pretrained=False)
        elif mode in {"full", "cropped"}:
            path = self._resolve_cls_path(mode)
            if path is None:
                raise FileNotFoundError("Full-image ResNet50 checkpoint missing")
            model = build_resnet50(len(CLASS_NAMES), pretrained=False)
        else:
            raise ValueError(f"Unknown classifier mode: {mode}")
        model = model.to(self.device)
        metadata = load_metadata(path, mode, required=path == classification_path(ROOT, mode))
        if "input_size" in metadata and metadata["input_size"] != self.input_size:
            raise ValueError("Checkpoint input size does not match the configured input size")
        _load_weights(model, path, self.device)
        model.eval()
        self._cls_cache[mode] = model
        if not hasattr(self, "_cls_metadata"):
            self._cls_metadata = {}
        self._cls_metadata[mode] = metadata
        return model

    def predict(self, image: Image.Image, seg_model_name="resunet", classifier_mode="full") -> Dict[str, Any]:
        """Use the deployed prediction path without report or attention generation."""
        return self.analyze(image, "evaluation", seg_model_name, classifier_mode, False, explain=False)

    def analyze(
        self,
        image: Image.Image,
        filename: str,
        seg_model_name: str = "resunet",
        classifier_mode: str = "dual",
        generate_report: bool = True,
        *,
        explain: bool = True,
    ) -> Dict[str, Any]:
        gray = np.array(image.convert("L"))
        if gray.size == 0:
            raise ValueError("Empty image")
        if gray.shape[0] * gray.shape[1] > MAX_IMAGE_PIXELS:
            raise ValueError("Image is too large")

        if seg_model_name not in SEG_CHECKPOINTS:
            raise ValueError(f"Unknown segmentation model: {seg_model_name}")
        if classifier_mode not in {"dual", "full", "cropped"}:
            raise ValueError(f"Unknown classifier mode: {classifier_mode}")

        notes = []

        with self._lock:
            seg_model = self._load_seg(seg_model_name)
            mask = predict_mask(seg_model, gray, self.device)
            crop, _cropped_mask = crop_tumor_region(
                gray, mask, margin=20, output_size=(self.input_size, self.input_size)
            )
            orig_rgb = Image.fromarray(gray).convert("RGB")
            orig_tensor = self.val_tf(orig_rgb)
            cls_model = self._load_cls(classifier_mode)
            metadata = getattr(self, "_cls_metadata", {}).get(classifier_mode, {})
            contract = metadata.get("crop_contract", "legacy_gate_v1" if classifier_mode == "dual" else "full_image")
            expected_seg = metadata.get("segmentation")
            if expected_seg:
                seg_path = self._resolve_seg_path(seg_model_name)
                hashes = getattr(self, "_seg_hashes", {})
                if seg_path not in hashes:
                    hashes[seg_path] = sha256_file(seg_path)
                self._seg_hashes = hashes
                if expected_seg["model"] != seg_model_name or expected_seg["sha256"] != hashes[seg_path]:
                    raise ValueError("Select the segmentation checkpoint used to train this classifier")

            # The legacy dual checkpoint was trained with a full-size first view for
            # no-tumor images, but an ROI first view for tumor images. A full-image
            # classifier therefore acts as a deployable no-tumor gate before the
            # dual classifier is called. This reproduces the training contract
            # without using the unknown ground-truth label at inference time.
            orig_batch = orig_tensor.unsqueeze(0).to(self.device)
            full_model = None
            gate_idx = gate_confidence = None
            needs_gate = contract == "legacy_gate_v1"
            if classifier_mode == "full" or needs_gate or (explain and self._resolve_full_cls() is not None):
                full_model = cls_model if classifier_mode == "full" else self._load_cls("full")
                with torch.inference_mode():
                    full_logits = full_model(orig_batch)
                    full_probs = torch.softmax(full_logits, dim=1)[0]
                gate_idx = int(full_probs.argmax())
                gate_confidence = float(full_probs[gate_idx].detach().cpu())
            if gate_idx is not None and CLASS_NAMES[gate_idx] == "notumor" and bool(mask.any()):
                notes.append(
                    "The segmenter is a tumor-localization model; its mask is not treated as lesion evidence when the full-image gate predicts no tumor."
                )

            if classifier_mode in {"dual", "cropped"}:
                crop_tensor = self.val_tf(Image.fromarray(crop).convert("RGB"))
                if classifier_mode == "cropped":
                    batch, crop_source = crop_tensor.unsqueeze(0), "segmentation_roi"
                elif contract == "legacy_gate_v1":
                    batch, crop_source = _build_dual_batch(crop_tensor, orig_tensor, gate_idx)
                else:
                    batch, crop_source = torch.cat([crop_tensor, orig_tensor], dim=0).unsqueeze(0), "segmentation_roi"
                if crop_source == "full_image_gate":
                    crop = np.array(orig_rgb.resize((self.input_size, self.input_size)))
                batch = batch.to(self.device)
                if contract == "legacy_gate_v1":
                    notes.append("Legacy dual weights use the full-image no-tumor gate to match their training contract.")
                with torch.inference_mode():
                    logits = cls_model(batch)
            else:
                crop_source = "not_used"
                logits = full_logits
            raw_logits = logits[0].detach().cpu().numpy()
            temperature = 1.0
            calibration_path = getattr(self, "calibration_path", None)
            if calibration_path:
                if not metadata.get("split_fingerprint"):
                    raise ValueError("Calibration requires checkpoint training split metadata")
                cache = self._calibration_cache
                key = (classifier_mode, seg_model_name)
                if key not in cache:
                    cache[key] = load_temperature_artifact(
                        calibration_path, checkpoint_sha256=sha256_file(self._resolve_cls_path(classifier_mode)),
                        mode=classifier_mode, segmentation_model=seg_model_name if classifier_mode != "full" else None,
                        segmentation_sha256=sha256_file(self._resolve_seg_path(seg_model_name)) if classifier_mode != "full" else None,
                        reference_full_sha256=sha256_file(self._resolve_full_cls()) if classifier_mode == "dual" and needs_gate else None,
                        split_fingerprint=metadata.get("split_fingerprint"),
                    )
                temperature = cache[key]["temperature"]
            probs = np.asarray(temperature_probabilities(raw_logits[None, :], temperature))[0]
            score_type = "temperature_scaled_softmax" if calibration_path else "uncalibrated_softmax"
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])

            # Early-fusion dual Grad-CAM mixes two different coordinate systems
            # (ROI and original). Explain the spatially valid full-image view instead.
            explanation_model = full_model
            explanation_batch = orig_batch
            explanation_method = (
                "full-image reference Grad-CAM" if classifier_mode != "full" else "Grad-CAM"
            )
            heatmap = None
            gradcam_overlay = None
            try:
                try:
                    heatmap = generate_gradcam(
                        explanation_model,
                        explanation_batch,
                        explanation_model.layer4,
                        self.device,
                        pred_class=pred_idx,
                    ) if explain and explanation_model is not None else None
                except Exception as exc:
                    logger.info("pytorch-grad-cam failed (%s); using fallback", exc)
                    heatmap = _fallback_gradcam(
                        explanation_model,
                        explanation_batch,
                        explanation_model.layer4,
                        pred_idx,
                    )
                if heatmap is not None:
                    gradcam_overlay = overlay_gradcam(gray, heatmap, alpha=0.45)
            except Exception as exc:
                logger.warning("Grad-CAM unavailable: %s", exc)
                notes.append(f"Grad-CAM unavailable ({exc.__class__.__name__}).")
            if explain and explanation_model is None:
                notes.append("Reference Grad-CAM unavailable: no full-image checkpoint installed.")

            stage_iou = None
            if heatmap is not None and CLASS_NAMES[pred_idx] != "notumor":
                stage_iou = compute_stage_consistency_iou(mask, heatmap)
            report = generate_template_report(
                pred_idx,
                confidence,
                heatmap if heatmap is not None else np.zeros(
                    (self.input_size, self.input_size), dtype=np.float32
                ),
                mask_fraction=float((mask > 0).mean()), explanation_method=explanation_method,
                score_type=score_type,
            ) if generate_report else None

        tumor_px = int((mask > 0).sum())
        tumor_frac = float(tumor_px / mask.size) if mask.size else 0.0
        return {
            "filename": filename,
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "logits": raw_logits.tolist(),
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            },
            "segmentation_model": seg_model_name,
            "classifier_mode": classifier_mode,
            "classifier_gate": {
                "prediction": CLASS_NAMES[gate_idx],
                "score": gate_confidence,
            } if needs_gate and gate_idx is not None else None,
            "model_details": {
                "segmentation": {
                    "architecture": type(seg_model).__name__,
                    "checkpoint": self._resolve_seg_path(seg_model_name).name,
                },
                "classification": {
                    "architecture": "ResNet50 dual-input" if classifier_mode == "dual" else "ResNet50",
                    "checkpoint": (
                        self._resolve_cls_path(classifier_mode).name
                    ),
                },
            },
            "device": str(self.device),
            "mask_pixels": tumor_px,
            "mask_fraction": tumor_frac,
            "stage_iou": stage_iou,
            "stage_iou_label": "segmentation / full-image attention overlap",
            "crop_source": crop_source,
            "explanation_method": explanation_method,
            "score_type": score_type,
            "temperature": float(temperature),
            "preprocessing_contract": contract,
            "report": report,
            "report_generated": report is not None,
            "notes": notes,
            "images": {
                "original": _encode_image(gray),
                "segmentation": _encode_image(_overlay_mask(gray, mask)),
                "crop": _encode_image(crop),
                "gradcam": _encode_image(gradcam_overlay) if gradcam_overlay is not None else None,
            },
        }
