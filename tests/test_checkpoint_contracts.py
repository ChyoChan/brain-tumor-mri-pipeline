import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import app.inference as inference
from src.checkpoints import load_metadata, save_metadata
from src.constants import CLASS_NAMES
from src.calibration import save_temperature_artifact


def test_metadata_is_bound_to_weights_and_class_order(tmp_path):
    path = tmp_path / "model.pth"
    path.write_bytes(b"test weights")
    metadata = dict(mode="full", architecture="resnet50", class_names=CLASS_NAMES,
                    input_size=224, normalization="grayscale_imagenet_v1", crop_contract="full_image",
                    split_fingerprint="a" * 64)
    save_metadata(path, metadata)
    assert load_metadata(path, "full")["crop_contract"] == "full_image"
    path.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checksum"):
        load_metadata(path, "full")
    save_metadata(path, dict(metadata, class_names=list(reversed(CLASS_NAMES))))
    with pytest.raises(ValueError, match="metadata"):
        load_metadata(path, "full")


@pytest.mark.parametrize("contract,first_view", [("legacy_gate_v1", 0), ("roi_v1", 150)])
def test_new_dual_contract_does_not_use_legacy_gate(monkeypatch, contract, first_view):
    runtime = object.__new__(inference.PipelineRuntime)
    runtime.device = torch.device("cpu")
    runtime._lock = threading.Lock()
    runtime.input_size = 8
    runtime.val_tf = lambda image: torch.full((3, 8, 8), float(np.array(image)[0, 0, 0]))
    runtime._cls_metadata = {"dual": {"crop_contract": contract}}
    runtime._resolve_cls_path = lambda mode: Path(f"{mode}.pth")
    runtime._resolve_seg_path = lambda name: Path("seg.pth")
    runtime._resolve_full_cls = lambda: Path("full.pth")
    runtime._load_seg = lambda name: torch.nn.Identity()

    class Full(torch.nn.Module):
        layer4 = torch.nn.Identity()

        def forward(self, batch):
            return torch.tensor([[0., 0., 0., 10.]])

    class Dual(Full):
        def forward(self, batch):
            assert batch.shape == (1, 6, 8, 8)
            assert torch.all(batch[:, :3] == first_view)
            return super().forward(batch)

    runtime._load_cls = lambda mode: Dual() if mode == "dual" else Full()
    monkeypatch.setattr(inference, "predict_mask", lambda *args: np.ones((8, 8), dtype=np.uint8))
    monkeypatch.setattr(inference, "crop_tumor_region", lambda *args, **kwargs: (np.full((8, 8), 150, dtype=np.uint8), None))

    def unexpected_image_export(*args, **kwargs):
        pytest.fail("Batch prediction must skip image rendering and encoding")

    monkeypatch.setattr(inference, "_encode_image", unexpected_image_export)
    monkeypatch.setattr(inference, "_overlay_mask", unexpected_image_export)
    result = runtime.predict(Image.new("RGB", (8, 8)), "unet", "dual")
    assert result["preprocessing_contract"] == contract
    assert result["crop_source"] == ("full_image_gate" if contract == "legacy_gate_v1" else "segmentation_roi")
    assert result["report"] is None
    assert result["images"] == dict.fromkeys(("original", "segmentation", "crop", "gradcam"))


def test_runtime_applies_only_matching_validation_calibration(monkeypatch, tmp_path):
    runtime = object.__new__(inference.PipelineRuntime)
    runtime.device = torch.device("cpu")
    runtime._lock = threading.Lock()
    runtime.input_size = 8
    runtime.val_tf = lambda image: torch.zeros(3, 8, 8)
    runtime._cls_metadata = {"full": {"split_fingerprint": "a" * 64}}
    runtime._calibration_cache = {}
    weights = tmp_path / "full.pth"
    weights.write_bytes(b"weights")
    runtime._resolve_cls_path = lambda mode: weights
    runtime._resolve_full_cls = lambda: weights
    runtime._resolve_seg_path = lambda name: Path("seg.pth")
    runtime._load_seg = lambda name: torch.nn.Identity()

    class Classifier(torch.nn.Module):
        layer4 = torch.nn.Identity()

        def forward(self, batch):
            return torch.tensor([[3., 0., 0., 0.]])

    runtime._load_cls = lambda mode: Classifier()
    monkeypatch.setattr(inference, "predict_mask", lambda *args: np.zeros((8, 8), dtype=np.uint8))
    image = Image.new("RGB", (8, 8))
    raw = runtime.predict(image)
    runtime.calibration_path = tmp_path / "temperature.json"
    save_temperature_artifact(
        runtime.calibration_path, temperature=2, checkpoint_sha256=inference.sha256_file(weights),
        split_fingerprint="a" * 64, mode="full",
    )
    calibrated = runtime.predict(image)
    assert calibrated["confidence"] < raw["confidence"]
    assert calibrated["prediction"] == raw["prediction"]
    assert calibrated["score_type"] == "temperature_scaled_softmax"
    runtime._calibration_cache.clear()
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        runtime.predict(image)
