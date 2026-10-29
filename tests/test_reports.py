import numpy as np
import pytest
from PIL import Image

from src.constants import CLASS_NAMES
from src.explainability import (
    CLIPReportGenerator,
    _describe_focus_region,
    generate_template_report,
    summarize_report_text,
)


@pytest.mark.parametrize("class_index", range(len(CLASS_NAMES)))
def test_template_reports_model_outputs_without_invented_findings(class_index):
    heatmap = np.zeros((9, 9), dtype=np.float32)
    heatmap[2, 6] = 1
    report = generate_template_report(class_index, 0.96, heatmap, mask_fraction=0.125)

    assert f"Classifier predicted class: {CLASS_NAMES[class_index]}" in report
    assert "96.0% (uncalibrated softmax score)" in report
    assert "not a calibrated probability" in report
    assert "12.50% of image pixels" in report
    assert "x=75.0%, y=25.0%" in report
    assert "Attention does not confirm a tumor" in report
    for unsupported_claim in (
        "infiltrative", "dural-based", "sellar", "frontal lobe", "temporal lobe",
        "neurosurgical", "histopathologic", "enhancing lesion", "no suspicious",
        "no radiologically evident", "very high", "recommendation:",
    ):
        assert unsupported_claim not in report.lower()


def test_legacy_three_argument_call_and_dual_reference_explanation():
    legacy_report = generate_template_report(0, 0.8, None)
    assert "Segmentation coverage:\nNot provided." in legacy_report
    assert "unavailable (no spatial attention map)" in legacy_report
    dual_report = generate_template_report(
        0, 0.8, None, explanation_method="full-image reference Grad-CAM",
    )
    assert "from the full-image reference classifier" in dual_report
    assert "does not directly explain the selected crop-based classifier" in dual_report


@pytest.mark.parametrize(
    "heatmap",
    [None, np.empty((0, 0)), np.zeros((4, 4)), np.ones((4, 4)), np.full((4, 4), np.nan)],
)
def test_missing_or_flat_attention_does_not_produce_a_location(heatmap):
    assert "unavailable" in _describe_focus_region(heatmap, "pituitary")


def test_focus_uses_image_coordinates_independently_of_predicted_class():
    heatmap = np.array([[1.0, 0.0], [0.0, 0.0]])
    descriptions = {_describe_focus_region(heatmap, class_name) for class_name in CLASS_NAMES}
    assert descriptions == {"image coordinates x=0.0%, y=0.0% measured from the top-left corner"}


@pytest.mark.parametrize("class_index", [-1, len(CLASS_NAMES)])
def test_invalid_prediction_is_not_silently_reported_as_no_tumor(class_index):
    with pytest.raises(ValueError, match="Unknown predicted class"):
        generate_template_report(class_index, 0.8, None)


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError, match="Model score"):
        generate_template_report(0, score, None)


@pytest.mark.parametrize("coverage", [-0.1, 1.1, float("nan")])
def test_invalid_coverage_is_rejected(coverage):
    with pytest.raises(ValueError, match="Mask fraction"):
        generate_template_report(0, 0.8, None, mask_fraction=coverage)


def test_unknown_score_type_does_not_claim_calibration():
    report = generate_template_report(0, 0.8, None, score_type="experimental_score")
    assert "score type: experimental_score" in report
    assert "Calibration is not established" in report


def test_experimental_generator_fallback_uses_the_same_factual_report(monkeypatch):
    generator = CLIPReportGenerator(device="cpu")
    monkeypatch.setattr(generator, "_generate_structured_report", lambda **kwargs: None)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0] = 1
    report = generator.generate_report(
        Image.new("RGB", (4, 4)), 3, 0.95, segmentation_mask=mask,
    )
    assert "Classifier predicted class: notumor" in report
    assert "25.00% of image pixels" in report
    assert "Report source: template_fallback" in report
    assert "no suspicious" not in report.lower()
    assert "no evidence of a brain tumor" not in report.lower()
    assert "clinical correlation" not in report.lower()


def test_summary_does_not_add_anatomical_causes_or_diagnostic_certainty():
    report = generate_template_report(0, 0.99, None)
    summary = summarize_report_text(
        report + "\n\nModel diagnostics:\nStage consistency IoU: 0.100",
        predicted_class=0, confidence=0.99, segmentation_mask=np.zeros((4, 4)),
        true_label="meningioma",
    )
    assert "Classifier prediction: glioma" in summary
    assert "Dataset class: meningioma" in summary
    assert "Segmentation foreground pixels: 0 of 16" in summary
    assert "uncalibrated model score: 99.0%" in summary.lower()
    assert "edema" not in summary.lower()
    assert "certainty" not in summary.lower()
    assert "recommended follow-up" not in summary.lower()


def test_experimental_prompts_use_recorded_outputs_without_anatomical_heuristics():
    generator = CLIPReportGenerator(device="cpu")
    focus = "image coordinates x=25.0%, y=50.0% measured from the top-left corner"
    prompt = generator._build_strong_prompt(0, 0.8, focus)
    assert prompt == generator._build_medgemma_instruction(0, 0.8, focus)
    assert "Uncalibrated softmax score: 80.0%" in prompt
    assert "Do not infer anatomy" in prompt
    assert "Do not provide treatment or follow-up advice" in prompt
    assert "no suspicious intracranial mass is identified" not in prompt
