import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

import app.inference as inference
from app.inference import PipelineRuntime
from app.server import app, get_runtime


def png_bytes():
    stream = BytesIO()
    Image.new("RGB", (8, 8)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def options_client():
    class ReportRuntime:
        def analyze(self, image, filename, seg_model_name, classifier_mode, generate_report=True):
            return {
                "report": "Findings: test" if generate_report else None,
                "report_generated": generate_report,
            }

    app.dependency_overrides[get_runtime] = lambda: ReportRuntime()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("flag,expected", [(None, True), ("true", True), ("false", False)])
def test_report_flag_reaches_runtime(options_client, flag, expected):
    form = {"seg_model": "unet", "classifier_mode": "full"}
    if flag is not None:
        form["generate_report"] = flag
    response = options_client.post(
        "/api/analyze", data=form, files={"file": ("scan.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["report_generated"] is expected
    assert (result["report"] is not None) is expected


def test_invalid_report_flag_is_rejected(options_client):
    response = options_client.post(
        "/api/analyze", data={"generate_report": "maybe"},
        files={"file": ("scan.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 400


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    root = tmp_path / "Testing"
    for class_name in inference.CLASS_NAMES:
        folder = root / class_name
        folder.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (8, 8)).save(folder / f"scan_{index:02}.png")
    (root / "glioma" / "notes.txt").touch()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8)).save(outside)
    (root / "glioma" / "outside.png").symlink_to(outside)
    monkeypatch.setattr(inference, "SAMPLE_ROOT", root)
    runtime = object.__new__(PipelineRuntime)
    runtime._sample_index = runtime._discover_samples()
    return runtime


def test_catalog_indexes_more_than_four_images_without_exposing_paths(catalog):
    result = catalog.list_samples()
    assert result["total"] == 12
    assert len(result["items"]) == 8
    assert result["pages"] == 2
    assert all(set(item) == {"id", "filename", "label", "class_name"} for item in result["items"])
    assert all(item["id"].startswith("local_") for item in result["items"])


def test_catalog_filter_search_and_pagination(catalog):
    result = catalog.list_samples("pituitary", "SCAN_01", page_size=2)
    assert result["total"] == 1
    assert result["items"][0]["filename"] == "scan_01.png"
    assert result["items"][0]["class_name"] == "pituitary"
    first = catalog.list_samples(page=1)
    second = catalog.list_samples(page=2)
    assert len(second["items"]) == 4
    assert not ({item["id"] for item in first["items"]} & {item["id"] for item in second["items"]})
    assert catalog.list_samples(page=999)["page"] == 2
    assert catalog.list_samples(search="missing")["total"] == 0


def test_catalog_sample_resolution_is_confined_to_dataset(catalog, monkeypatch, tmp_path):
    item = catalog.list_samples()["items"][0]
    path = catalog.sample_path(item["id"])
    assert path.is_file()
    with pytest.raises(KeyError):
        catalog.sample_path("../../outside.png")
    # Also protect against an indexed image being replaced with an external symlink.
    path.unlink()
    outside = tmp_path / "secret.png"
    Image.new("RGB", (8, 8)).save(outside)
    path.symlink_to(outside)
    with pytest.raises(FileNotFoundError):
        catalog.sample_path(item["id"])


def test_gallery_endpoints_serve_selected_images(catalog):
    app.dependency_overrides[get_runtime] = lambda: catalog
    try:
        with TestClient(app) as client:
            response = client.get("/api/samples?class_name=meningioma&page_size=2")
            assert response.status_code == 200
            result = response.json()
            assert result["total"] == 3
            image = client.get(f"/api/samples/{result['items'][0]['id']}")
            assert image.status_code == 200
            assert image.headers["content-type"] == "image/png"
            for query in ("page=0", "page_size=25", "class_name=unknown", "search=" + "x" * 121):
                assert client.get(f"/api/samples?{query}").status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_disabling_report_skips_generator_but_preserves_model_results(monkeypatch):
    runtime = object.__new__(PipelineRuntime)
    runtime.device = torch.device("cpu")
    runtime._lock = threading.Lock()
    runtime.input_size = 8
    runtime.val_tf = lambda image: torch.zeros(3, 8, 8)
    calls = {"segment": 0, "classify": 0, "attention": 0, "report": 0}

    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Identity()

        def forward(self, batch):
            calls["classify"] += 1
            return torch.tensor([[3.0, 0.0, 0.0, 0.0]])

    def mask(*args):
        calls["segment"] += 1
        return np.zeros((8, 8), dtype=np.uint8)

    def attention(*args, **kwargs):
        calls["attention"] += 1
        return np.ones((8, 8), dtype=np.float32)

    def report(*args):
        calls["report"] += 1
        return "Findings: template"

    runtime._load_seg = lambda name: nn.Identity()
    runtime._load_cls = lambda mode: TinyClassifier()
    runtime._resolve_seg_path = lambda name: Path("seg.pth")
    runtime._resolve_full_cls = lambda: Path("full.pth")
    monkeypatch.setattr(inference, "predict_mask", mask)
    monkeypatch.setattr(inference, "generate_gradcam", attention)
    monkeypatch.setattr(inference, "generate_template_report", report)
    image = Image.new("RGB", (8, 8))
    without = runtime.analyze(image, "scan.png", "unet", "full", False)
    assert calls == {"segment": 1, "classify": 1, "attention": 1, "report": 0}
    assert without["report"] is None
    assert without["report_generated"] is False
    with_report = runtime.analyze(image, "scan.png", "unet", "full", True)
    assert calls["report"] == 1
    assert with_report["report"] == "Findings: template"
    for field in ("prediction", "probabilities", "images"):
        assert without[field] == with_report[field]
