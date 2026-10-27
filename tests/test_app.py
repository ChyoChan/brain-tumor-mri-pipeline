from io import BytesIO

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from app.server import MAX_UPLOAD_BYTES, app, get_runtime
from app.inference import PipelineRuntime, _build_dual_batch
from src.constants import CLASS_TO_IDX
from src.segmentation import predict_mask


class FakeRuntime:
    def status(self):
        return {
            "device": "cpu",
            "classes": ["glioma", "meningioma", "pituitary", "notumor"],
            "checkpoints": {
                "segmentation": {"unet": True, "resunet": True, "attention_unet": False},
                "classification": {"dual": True, "full": True},
            },
            "samples": [{"id": "glioma", "label": "Glioma", "available": False}],
        }

    def sample_path(self, sample_id: str):
        raise KeyError(sample_id)

    def analyze(self, image, filename, seg_model_name, classifier_mode, generate_report=True):
        return {
            "filename": filename,
            "prediction": "glioma",
            "confidence": 0.91,
            "probabilities": {
                "glioma": 0.91,
                "meningioma": 0.09,
                "pituitary": 0.0,
                "notumor": 0.0,
            },
            "segmentation_model": seg_model_name,
            "classifier_mode": classifier_mode,
            "device": "cpu",
            "mask_pixels": 12,
            "mask_fraction": 0.1,
            "stage_iou": 0.2,
            "report": "Findings: test" if generate_report else None,
            "report_generated": generate_report,
            "notes": [],
            "images": {
                "original": "YQ==",
                "segmentation": "YQ==",
                "crop": "YQ==",
                "gradcam": None,
            },
        }


@pytest.fixture()
def client():
    app.dependency_overrides[get_runtime] = lambda: FakeRuntime()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("L", (8, 8), color=40).save(buf, format="PNG")
    return buf.getvalue()


def test_health_has_no_absolute_paths(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["checkpoints"]["segmentation"]["resunet"] is True
    assert "/Users/" not in res.text
    assert "/home/" not in res.text


def test_unknown_sample_404(client):
    res = client.get("/api/samples/not-a-sample")
    assert res.status_code == 404


def test_analyze_rejects_bad_seg_model(client):
    res = client.post(
        "/api/analyze",
        data={"seg_model": "transformer", "classifier_mode": "dual", "sample_id": "glioma"},
    )
    assert res.status_code == 400


def test_analyze_rejects_bad_classifier_mode(client):
    res = client.post(
        "/api/analyze",
        data={"seg_model": "resunet", "classifier_mode": "typo", "sample_id": "glioma"},
    )
    assert res.status_code == 400


def test_analyze_requires_input(client):
    res = client.post("/api/analyze", data={"seg_model": "resunet", "classifier_mode": "dual"})
    assert res.status_code == 400


def test_analyze_upload(client):
    res = client.post(
        "/api/analyze",
        data={"seg_model": "resunet", "classifier_mode": "dual"},
        files={"file": ("slice.png", _png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "glioma"
    assert body["segmentation_model"] == "resunet"


def test_analyze_rejects_oversize(client, monkeypatch):
    import app.server as server

    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 16)
    res = client.post(
        "/api/analyze",
        data={"seg_model": "unet", "classifier_mode": "full"},
        files={"file": ("slice.png", _png_bytes(), "image/png")},
    )
    assert res.status_code == 413


def test_dual_batch_uses_full_image_when_gate_predicts_no_tumor():
    crop = torch.full((3, 2, 2), 1.0)
    original = torch.full((3, 2, 2), 2.0)

    batch, source = _build_dual_batch(crop, original, CLASS_TO_IDX["notumor"])

    assert source == "full_image_gate"
    assert batch.shape == (1, 6, 2, 2)
    assert torch.equal(batch[0, :3], original)
    assert torch.equal(batch[0, 3:], original)


def test_dual_batch_uses_roi_when_gate_predicts_tumor():
    crop = torch.full((3, 2, 2), 1.0)
    original = torch.full((3, 2, 2), 2.0)

    batch, source = _build_dual_batch(crop, original, CLASS_TO_IDX["glioma"])

    assert source == "segmentation_roi"
    assert torch.equal(batch[0, :3], crop)
    assert torch.equal(batch[0, 3:], original)


def test_classifier_loader_routes_only_known_modes(monkeypatch, tmp_path):
    import app.inference as inference

    class TinyModel(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, 1, 1)

    dual_path = tmp_path / "dual.pth"
    full_path = tmp_path / "full.pth"
    dual_path.touch()
    full_path.touch()
    calls = []
    monkeypatch.setattr(inference, "CLS_DUAL", dual_path)
    monkeypatch.setattr(inference, "build_resnet50_dual_input", lambda *_args, **_kwargs: TinyModel(6))
    monkeypatch.setattr(inference, "build_resnet50", lambda *_args, **_kwargs: TinyModel(3))
    monkeypatch.setattr(inference, "_load_weights", lambda model, path, device: calls.append(path))

    runtime = object.__new__(PipelineRuntime)
    runtime.device = torch.device("cpu")
    runtime._cls_cache = {}
    runtime._resolve_full_cls = lambda: full_path

    assert runtime._load_cls("dual").conv1.in_channels == 6
    assert runtime._load_cls("full").conv1.in_channels == 3
    assert calls == [dual_path, full_path]
    with pytest.raises(ValueError, match="Unknown classifier mode"):
        runtime._load_cls("typo")


def test_segmentation_loader_routes_each_frontend_option(monkeypatch, tmp_path):
    import app.inference as inference

    class TinySegmentationModel(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.model_name = model_name
            self.weight = nn.Parameter(torch.ones(1))

    paths = {}
    for name in ("unet", "resunet", "attention_unet"):
        paths[name] = tmp_path / f"{name}.pth"
        paths[name].touch()
    built = []
    monkeypatch.setattr(
        inference,
        "build_segmentation_model",
        lambda name: built.append(name) or TinySegmentationModel(name),
    )
    monkeypatch.setattr(inference, "_load_weights", lambda *_args, **_kwargs: None)

    runtime = object.__new__(PipelineRuntime)
    runtime.device = torch.device("cpu")
    runtime._seg_cache = {}
    runtime._resolve_seg_path = lambda name: paths.get(name)

    for name in ("unet", "resunet", "attention_unet"):
        assert runtime._load_seg(name).model_name == name
    assert built == ["unet", "resunet", "attention_unet"]


def test_predict_mask_does_not_invent_a_lesion_for_empty_prediction():
    class EmptySegmentationModel(nn.Module):
        def forward(self, batch):
            return torch.full(
                (batch.shape[0], 1, batch.shape[2], batch.shape[3]),
                -100.0,
                device=batch.device,
            )

    mask = predict_mask(
        EmptySegmentationModel(),
        np.zeros((32, 32), dtype=np.uint8),
        torch.device("cpu"),
    )

    assert mask.shape == (32, 32)
    assert mask.sum() == 0
