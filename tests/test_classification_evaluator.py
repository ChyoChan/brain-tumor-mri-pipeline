import json
from types import SimpleNamespace

import pytest
from PIL import Image

import app.inference as inference
from scripts import eval_classification
from src.constants import CLASS_NAMES
from src.data_splits import create_classification_manifest


@pytest.mark.parametrize("mode", ["full", "dual", "cropped"])
def test_evaluator_uses_raw_deployed_predictions(tmp_path, monkeypatch, mode):
    root = tmp_path / "data"
    for label, class_name in enumerate(CLASS_NAMES):
        folder = root / "Training" / class_name
        folder.mkdir(parents=True)
        for index in range(5):
            Image.new("RGB", (4, 4), (label * 50, index * 40, 5)).save(folder / f"{index}.png")
    manifest = tmp_path / "split.json"
    frozen = create_classification_manifest(root, manifest)
    checkpoint = tmp_path / "full.pth"
    checkpoint.write_bytes(b"legacy weights")
    segmenter = tmp_path / "seg.pth"
    segmenter.write_bytes(b"segmentation weights")
    calls = []

    class FakeRuntime:
        def __init__(self, **kwargs):
            assert kwargs["calibration_path"] == ""

        def _resolve_cls_path(self, mode):
            return checkpoint

        def _resolve_seg_path(self, model):
            return segmenter

        def _resolve_full_cls(self):
            pytest.fail("ROI classifiers do not need a reference classifier during prediction")

        def predict(self, image, seg_model_name, classifier_mode):
            calls.append((seg_model_name, classifier_mode))
            return {"logits": [4.0, 0.0, 0.0, 0.0]}

    monkeypatch.setattr(inference, "PipelineRuntime", FakeRuntime)
    if mode != "full":
        monkeypatch.setattr(eval_classification, "load_metadata", lambda *args: {"crop_contract": "roi_v1", "split_fingerprint": frozen["fingerprint"]})
    output = tmp_path / "evaluation.json"
    eval_classification.main(["--manifest", str(manifest), "--mode", mode, "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["training_split_verified"] is (mode != "full")
    assert result["count"] == 4
    assert len(calls) == 4
    assert all(call == ("resunet", mode) for call in calls)
    assert output.with_suffix(".reliability.csv").is_file()
    assert output.with_suffix(".reliability.png").is_file()
    assert output.with_suffix(".predictions.csv").is_file()


def test_evaluator_refuses_temperature_fitting_on_test_set(tmp_path):
    with pytest.raises(SystemExit) as error:
        eval_classification.main(["--manifest", "unused.json", "--split", "test", "--fit-temperature", str(tmp_path / "temperature.json")])
    assert error.value.code == 2
    assert not (tmp_path / "temperature.json").exists()


def test_evaluator_refuses_fitting_legacy_weights_without_audited_split(tmp_path, monkeypatch):
    checkpoint = tmp_path / "legacy.pth"
    checkpoint.write_bytes(b"legacy weights")
    monkeypatch.setattr(eval_classification, "load_classification_manifest", lambda *args: SimpleNamespace(fingerprint="a" * 64))

    class FakeRuntime:
        def __init__(self, **kwargs):
            pass

        def _resolve_cls_path(self, mode):
            return checkpoint

    monkeypatch.setattr(inference, "PipelineRuntime", FakeRuntime)
    with pytest.raises(SystemExit) as error:
        eval_classification.main(["--manifest", "unused.json", "--split", "val", "--fit-temperature", str(tmp_path / "temperature.json")])
    assert error.value.code == 2
    assert not (tmp_path / "temperature.json").exists()
