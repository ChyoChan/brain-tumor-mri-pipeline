import numpy as np
import pytest

from src.calibration import (
    calibration_metrics, fit_temperature, load_temperature_artifact,
    save_temperature_artifact, temperature_probabilities,
)


def test_temperature_reduces_nll_without_changing_class_predictions():
    logits = np.array([[12.0, 0.0, 0.0, 0.0]] * 20)
    labels = np.array([0] * 10 + [1] * 10)
    temperature = fit_temperature(logits, labels)
    assert temperature > 1
    assert calibration_metrics(logits, labels, temperature)["nll"] < calibration_metrics(logits, labels)["nll"]
    assert np.array_equal(temperature_probabilities(logits).argmax(1), temperature_probabilities(logits, temperature).argmax(1))


def test_reliability_bins_account_for_all_examples():
    logits = [[0.0] * 4, [1000.0, 0.0, 0.0, 0.0]]
    metrics = calibration_metrics(logits, np.array([0, 1]))
    assert sum(row["count"] for row in metrics["bins"]) == 2
    assert 0 <= metrics["ece"] <= 1
    assert np.isfinite(metrics["nll"])


def test_calibration_artifact_is_bound_to_model_split_and_preprocessing(tmp_path):
    path = tmp_path / "temperature.json"
    save_temperature_artifact(
        path, temperature=2.0, checkpoint_sha256="a" * 64, split_fingerprint="b" * 64,
        mode="dual", segmentation_model="resunet", segmentation_sha256="c" * 64,
        reference_full_sha256="d" * 64,
    )
    expected = dict(checkpoint_sha256="a" * 64, split_fingerprint="b" * 64, mode="dual", segmentation_model="resunet", segmentation_sha256="c" * 64, reference_full_sha256="d" * 64)
    assert load_temperature_artifact(path, **expected)["temperature"] == 2.0
    for key in ("checkpoint_sha256", "split_fingerprint", "segmentation_sha256", "reference_full_sha256"):
        with pytest.raises(ValueError, match="does not match"):
            load_temperature_artifact(path, **{**expected, key: "f" * 64})
    with pytest.raises(ValueError, match="segmentation model"):
        load_temperature_artifact(path, **{**expected, "segmentation_model": "unet"})


def test_invalid_calibration_values_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        temperature_probabilities([[1.0, 0.0, 0.0, 0.0]], -1)
    with pytest.raises(ValueError, match="segmentation_sha256"):
        save_temperature_artifact(tmp_path / "bad.json", temperature=1, checkpoint_sha256="a" * 64, split_fingerprint="b" * 64, mode="cropped", segmentation_model="unet")


def test_new_dual_calibration_does_not_require_a_reference_classifier(tmp_path):
    path = tmp_path / "new-dual.json"
    parameters = dict(checkpoint_sha256="a" * 64, split_fingerprint="b" * 64, mode="dual", segmentation_model="resunet", segmentation_sha256="c" * 64)
    save_temperature_artifact(path, temperature=2, **parameters)
    assert load_temperature_artifact(path, **parameters)["reference_full_sha256"] is None
    with pytest.raises(ValueError, match="full-image gate"):
        load_temperature_artifact(path, **parameters, reference_full_sha256="d" * 64)
