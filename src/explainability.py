"""Stage 3 explainability utilities.

This module supports hybrid report generation with MedGemma-first decoding
and deterministic fallback for class-consistent outputs.
"""

import logging
import re
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.constants import CLASS_NAMES
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Grad-CAM (unchanged)
# ------------------------------------------------------------------ #
def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    device: torch.device,
    pred_class: Optional[int] = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for the input image tensor."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)] if pred_class is not None else None
    grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)
    return grayscale_cam[0]


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original MRI image."""
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    h, w = original_image.shape[:2]
    heatmap_color = cv2.resize(heatmap_color, (w, h))
    overlay = (alpha * heatmap_color + (1 - alpha) * original_image).astype(np.uint8)
    return overlay


# ------------------------------------------------------------------ #
# Stage-consistency and uncertainty utilities
# ------------------------------------------------------------------ #
def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize Grad-CAM heatmap into [0, 1] float32."""
    arr = np.asarray(heatmap, dtype=np.float32).squeeze()
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        arr = np.zeros((224, 224), dtype=np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    min_v = float(arr.min()) if arr.size else 0.0
    max_v = float(arr.max()) if arr.size else 0.0
    if max_v - min_v > 1e-8:
        arr = (arr - min_v) / (max_v - min_v)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr.astype(np.float32)


def _to_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert segmentation mask into a robust binary mask."""
    arr = np.asarray(mask, dtype=np.float32).squeeze()
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return np.zeros((224, 224), dtype=bool)

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if float(arr.max()) > 1.0:
        arr = arr / 255.0
    return arr >= threshold


def compute_stage_consistency_iou(
    segmentation_mask: Optional[np.ndarray],
    heatmap: Optional[np.ndarray],
    heatmap_threshold: float = 0.5,
) -> Optional[float]:
    """Compute IoU overlap between segmentation mask and Grad-CAM hotspot."""
    if segmentation_mask is None or heatmap is None:
        return None

    seg_bin = _to_binary_mask(segmentation_mask, threshold=0.5)
    cam = _normalize_heatmap(heatmap)
    if cam.shape != seg_bin.shape:
        cam = cv2.resize(
            cam,
            (seg_bin.shape[1], seg_bin.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    cam_bin = cam >= heatmap_threshold

    intersection = np.logical_and(seg_bin, cam_bin).sum()
    union = np.logical_or(seg_bin, cam_bin).sum()
    if union == 0:
        # Both model outputs are empty; this does not establish clinical absence.
        return 1.0
    return float(intersection / union)


def _build_stage_consistency_note(stage_iou: Optional[float], warning_threshold: float = 0.2) -> str:
    if stage_iou is None:
        return "Stage consistency IoU: unavailable."
    if stage_iou < warning_threshold:
        return (
            f"Stage consistency IoU: {stage_iou:.3f} "
            f"(below the configured overlap warning threshold of {warning_threshold:.3f})."
        )
    return f"Stage consistency IoU: {stage_iou:.3f} (model-output overlap, not segmentation accuracy)."


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def _sanitize_generation_text(text: str) -> str:
    """Remove common tokenizer artifacts from generated text."""
    cleaned = (text or "")
    # SentencePiece/wordpiece leftovers and model-specific placeholders.
    cleaned = cleaned.replace("▁", " ")
    cleaned = re.sub(r"<unused\d+>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*unused\d+\s*\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*/?\s*(?:s|pad|unk|bos|eos)\s*>", " ", cleaned, flags=re.IGNORECASE)
    # Keep newlines for section parsing but normalize spacing.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _clean_generated_section(text: str) -> str:
    """Normalize generated section text into concise, single-paragraph prose."""
    cleaned = _sanitize_generation_text(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("-:;,. *")
    return cleaned


def _extract_structured_sections(text: str):
    """Extract Findings/Impression/Recommendation sections from raw generation."""
    if not text:
        return None

    cleaned = (text or "").replace("\r\n", "\n").strip()
    # Remove common wrappers if model emits markdown fences.
    cleaned = cleaned.replace("```", "").strip()
    cleaned = cleaned.replace("**", "")
    cleaned = _sanitize_generation_text(cleaned)

    pattern = re.compile(
        r"findings\s*:\s*(?P<findings>.*?)\s*"
        r"impression\s*:\s*(?P<impression>.*?)\s*"
        r"recommendation\s*:\s*(?P<recommendation>.*)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(cleaned)
    if not match:
        return None

    findings = _clean_generated_section(match.group("findings"))
    impression = _clean_generated_section(match.group("impression"))
    recommendation = _clean_generated_section(match.group("recommendation"))

    if len(findings) < 20 or len(impression) < 12 or len(recommendation) < 12:
        return None
    return findings, impression, recommendation


def _format_structured_report(findings: str, impression: str, recommendation: str) -> str:
    return (
        f"Findings:\n{findings}\n\n"
        f"Impression:\n{impression}\n\n"
        f"Recommendation:\n{recommendation}"
    )


def _first_n_sentences(text: str, n: int = 1) -> str:
    """Return the first n sentences from text with whitespace normalized."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentences:
        return normalized
    return " ".join(sentences[: max(1, int(n))])


def _extract_stage_iou_value(text: str) -> Optional[float]:
    """Extract stage-consistency IoU value from a report diagnostics block."""
    match = re.search(r"stage consistency iou\s*:\s*([0-9]*\.?[0-9]+)", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _resolve_class_name(predicted_class: Optional[object]) -> Optional[str]:
    """Resolve class input (index or name) into canonical class string."""
    if predicted_class is None:
        return None
    if isinstance(predicted_class, int):
        if 0 <= predicted_class < len(CLASS_NAMES):
            return CLASS_NAMES[predicted_class]
        return None
    class_text = str(predicted_class).strip().lower()
    for class_name in CLASS_NAMES:
        if class_text == class_name:
            return class_name
    return class_text if class_text else None


def summarize_report_text(
    report_text: str,
    predicted_class: Optional[object] = None,
    confidence: Optional[float] = None,
    segmentation_mask: Optional[np.ndarray] = None,
    true_label: Optional[object] = None,
) -> str:
    """Summarize supplied text and model measurements without adding findings."""
    cleaned = _sanitize_generation_text(report_text or "")
    if not cleaned:
        return "Summary unavailable."

    stage_iou = _extract_stage_iou_value(cleaned)
    core_report = cleaned.split("Model diagnostics:")[0].strip()
    class_name = _resolve_class_name(predicted_class)
    true_class_name = _resolve_class_name(true_label)
    details = []
    if class_name:
        details.append(f"Classifier prediction: {class_name}.")
    if confidence is not None:
        details.append(f"Uncalibrated model score: {confidence:.1%}; this is not a calibrated probability.")
    if segmentation_mask is not None:
        mask = np.asarray(segmentation_mask)
        details.append(f"Segmentation foreground pixels: {np.count_nonzero(mask)} of {mask.size}.")
    if stage_iou is not None:
        details.append(_build_stage_consistency_note(stage_iou))
    if true_class_name and class_name:
        details.append(f"Dataset class: {true_class_name}; classifier prediction: {class_name}.")
    summary = core_report if core_report.startswith("Prediction:") else _first_n_sentences(core_report, n=10)
    return "\n\n".join([summary, " ".join(details)]).strip()


def _report_is_consistent_with_prediction(findings: str, impression: str, class_name: str) -> bool:
    """Reject obvious class drift in generated reports."""
    text = f"{findings} {impression}".lower()
    tumor_classes = ["glioma", "meningioma", "pituitary"]

    if class_name == "notumor":
        return not any(term in text for term in tumor_classes)

    other_classes = [c for c in tumor_classes if c != class_name]
    if any(term in text for term in other_classes):
        return False
    return True


def _build_aligned_template_report(
    class_name: str, confidence: float, focus_region: str, *,
    mask_fraction: Optional[float] = None,
    explanation_method: str = "Grad-CAM",
    score_type: str = "uncalibrated_softmax",
) -> str:
    """Describe model outputs without deriving clinical findings from a label."""
    if class_name not in CLASS_NAMES:
        raise ValueError(f"Unknown predicted class: {class_name}")
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Model score must be finite and between 0 and 1")
    coverage = "Not provided."
    if mask_fraction is not None:
        if not np.isfinite(mask_fraction) or not 0 <= mask_fraction <= 1:
            raise ValueError("Mask fraction must be finite and between 0 and 1")
        coverage = f"{mask_fraction:.2%} of image pixels were marked by the segmenter; this is not a tumor volume."
    score = (
        f"{confidence:.1%} (uncalibrated softmax score). This is not a calibrated probability of disease."
        if score_type == "uncalibrated_softmax"
        else f"{confidence:.1%} (score type: {score_type}). Calibration is not established by this report."
    )
    explanation = f"{explanation_method} describes classifier attention. Spatial focus: {focus_region}."
    if explanation_method == "full-image reference Grad-CAM":
        explanation += " This attention is from the full-image reference classifier and does not directly explain the selected crop-based classifier."
    return (
        f"Prediction:\nClassifier predicted class: {class_name}.\n\n"
        f"Model score:\n{score}\n\n"
        f"Segmentation coverage:\n{coverage}\n\n"
        f"Explanation:\n{explanation}\n\n"
        "Limitations:\nAttention does not confirm a tumor. Model outputs do not establish a diagnosis, "
        "and image coordinates do not identify anatomical structures."
    )


def _is_medgemma_model_name(model_name: str) -> bool:
    return "medgemma" in (model_name or "").lower()


def _describe_focus_region(
    heatmap: np.ndarray,
    class_name: Optional[str] = None,
) -> str:
    """Describe image coordinates; class_name is retained for notebook callers."""
    if heatmap is None or getattr(heatmap, "size", 0) == 0:
        return "unavailable (no spatial attention map)"

    arr = np.asarray(heatmap, dtype=np.float32).squeeze()
    if arr.ndim != 2 or arr.size == 0:
        return "unavailable (no spatial attention map)"
    arr = _normalize_heatmap(arr)
    if not arr.any():
        return "unavailable (no varying attention signal)"

    h, w = arr.shape
    mask = arr >= 0.5
    ys, xs = np.where(mask)
    weights = arr[ys, xs]

    cy = float(np.average(ys / max(h - 1, 1), weights=weights))
    cx = float(np.average(xs / max(w - 1, 1), weights=weights))

    return f"image coordinates x={cx:.1%}, y={cy:.1%} measured from the top-left corner"


def generate_template_report(
    predicted_class: int,
    confidence: float,
    heatmap: Optional[np.ndarray],
    *,
    mask_fraction: Optional[float] = None,
    explanation_method: str = "Grad-CAM",
    score_type: str = "uncalibrated_softmax",
) -> str:
    """Report model measurements while keeping the legacy three-argument call."""
    if not 0 <= predicted_class < len(CLASS_NAMES):
        raise ValueError(f"Unknown predicted class index: {predicted_class}")
    return _build_aligned_template_report(
        CLASS_NAMES[predicted_class], confidence, _describe_focus_region(heatmap),
        mask_fraction=mask_fraction, explanation_method=explanation_method, score_type=score_type,
    )


# ------------------------------------------------------------------ #
# CLIPReportGenerator - hybrid generation with deterministic fallback
# ------------------------------------------------------------------ #
class CLIPReportGenerator:
    """Hybrid report generator with constrained decoding and safe fallback."""

    def __init__(
        self,
        device: str = "auto",
        decoder_model_name: str = "google/medgemma-1.5-4b-it",
        model_name: str = None,          # compatibility with old notebook calls
        max_new_tokens: int = 220,
        repetition_penalty: float = 1.3,
    ):
        if model_name is not None:
            decoder_model_name = model_name

        self.device = device
        self.decoder_model_name = decoder_model_name
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self._decoder = None
        self._tokenizer = None
        self._processor = None
        self._generation_device = None
        self._generation_unavailable = False
        self.last_report_source = "unknown"
        self.last_generation_error = ""

    def _uses_medgemma(self) -> bool:
        return _is_medgemma_model_name(self.decoder_model_name)

    def reset_generation_state(self, unload_model: bool = False) -> None:
        """Clear failure latch so generation can retry after env/auth changes."""
        self._generation_unavailable = False
        self.last_generation_error = ""
        self.last_report_source = "unknown"
        if unload_model:
            self._decoder = None
            self._tokenizer = None
            self._processor = None

    def _get_device(self) -> torch.device:
        """Auto-select the best available device."""
        device_str = str(self.device).strip().lower() if self.device is not None else "auto"
        if not device_str:
            device_str = "auto"

        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            return torch.device(device_str)
        except Exception:
            logger.warning("Invalid device string '%s'; falling back to auto device.", self.device)
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

    def load_model(self):
        """Load report decoder (MedGemma multimodal or causal text model)."""
        device = self._get_device()
        self._generation_device = device
        self.device = str(device)

        print(f"Loading {self.decoder_model_name} on {device}...")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if self._uses_medgemma():
            from transformers import AutoModelForImageTextToText, AutoProcessor

            torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(self.decoder_model_name)
            self._decoder = AutoModelForImageTextToText.from_pretrained(
                self.decoder_model_name,
                torch_dtype=torch_dtype,
            ).to(device).eval()
            self._tokenizer = getattr(self._processor, "tokenizer", None)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.decoder_model_name)
            self._decoder = AutoModelForCausalLM.from_pretrained(self.decoder_model_name).to(device).eval()
            self._processor = None

        if self._tokenizer is not None and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._generation_unavailable = False
        self.last_generation_error = ""
        print(f"CLIPReportGenerator loaded successfully (using {self.decoder_model_name})")

    def _build_strong_prompt(self, predicted_class: int, confidence: float, focus_region: str) -> str:
        class_name = CLASS_NAMES[predicted_class]
        return f"""Summarize only the recorded outputs of an experimental MRI classification pipeline.
Do not infer anatomy, tissue morphology, malignancy, disease absence, or clinical findings from a class score or attention map.
Image coordinates describe positions in the displayed image; they do not identify anatomical structures.
Do not provide treatment or follow-up advice.

Output exactly these three sections:
Findings:
[State the classifier prediction and the image-coordinate attention information.]
Impression:
[Quote the uncalibrated softmax score and explain that it is not a calibrated disease probability.]
Recommendation:
[State that these model outputs do not establish clinical findings.]

Recorded model outputs:
- Classifier prediction: {class_name}
- Uncalibrated softmax score: {confidence:.1%}
- Classifier attention focus: {focus_region}
"""

    def _build_medgemma_instruction(self, predicted_class: int, confidence: float, focus_region: str) -> str:
        return self._build_strong_prompt(predicted_class, confidence, focus_region)

    def _generate_text_with_causal_lm(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        decoder = self._decoder
        if tokenizer is None or decoder is None:
            raise RuntimeError("Causal LM is not loaded.")

        encoded = tokenizer(prompt, return_tensors="pt")
        device = self._generation_device or self._get_device()
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            generated_ids = decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=max(1.0, float(self.repetition_penalty)),
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0, input_ids.shape[-1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if generated_text:
            return generated_text
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def _generate_text_with_medgemma(self, image: Image.Image, instruction: str) -> str:
        processor = self._processor
        decoder = self._decoder
        if processor is None or decoder is None:
            raise RuntimeError("MedGemma model/processor is not loaded.")

        image_rgb = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = self._generation_device or self._get_device()
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(model_device)

        input_len = inputs["input_ids"].shape[-1]
        tokenizer = self._tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer is not None else None
        eos_token_id = tokenizer.eos_token_id if tokenizer is not None else None
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": max(1.0, float(self.repetition_penalty)),
        }
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        with torch.inference_mode():
            generated_ids = decoder.generate(**inputs, **generation_kwargs)

        generated_tokens = generated_ids[0][input_len:]
        if tokenizer is not None:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            if generated_text:
                return generated_text
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        generated_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()
        if generated_text:
            return generated_text
        return processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    def _generate_structured_report(
        self,
        image: Image.Image,
        predicted_class: int,
        class_name: str,
        confidence: float,
        focus_region: str,
    ) -> Optional[str]:
        """Run constrained decoding and validate generated report structure."""
        try:
            if self._generation_unavailable:
                # Allow auto-retry when no usable model is loaded (common after fixing
                # HF auth/dependency issues during an active notebook session).
                if self._decoder is None:
                    self._generation_unavailable = False
                else:
                    return None

            if self._uses_medgemma():
                model_missing = self._decoder is None or self._processor is None
            else:
                model_missing = self._decoder is None or self._tokenizer is None
            if model_missing:
                self.load_model()

            if self._decoder is None:
                return None

            if self._uses_medgemma():
                instruction = self._build_medgemma_instruction(predicted_class, confidence, focus_region)
                generated_text = self._generate_text_with_medgemma(image, instruction)
            else:
                prompt = self._build_strong_prompt(predicted_class, confidence, focus_region)
                generated_text = self._generate_text_with_causal_lm(prompt)

            sections = _extract_structured_sections(generated_text)
            if sections is None:
                self.last_generation_error = "Generated text missing required report sections."
                logger.warning("Generated report missing required sections; using template fallback.")
                return None

            findings, impression, recommendation = sections
            if not _report_is_consistent_with_prediction(findings, impression, class_name):
                self.last_generation_error = "Generated text failed class-consistency validation."
                logger.warning(
                    "Generated report inconsistent with predicted class (%s); using template fallback.",
                    class_name,
                )
                return None

            self.last_generation_error = ""
            return _format_structured_report(findings, impression, recommendation)
        except Exception as exc:
            # Avoid repeated heavyweight load/generate failures on every sample.
            self._generation_unavailable = True
            self.last_generation_error = str(exc)
            logger.warning("Generative report failed; using template fallback (%s).", exc)
            return None

    def generate_report(
        self,
        image: Image.Image,
        predicted_class: int,
        confidence: float,
        heatmap: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None,
        stage_iou: Optional[float] = None,
        iou_warning_threshold: float = 0.2,
    ) -> str:
        """Generate a class-grounded report with generative-first fallback logic."""
        if heatmap is None:
            heatmap = np.zeros((224, 224), dtype=np.float32)

        if not 0 <= predicted_class < len(CLASS_NAMES):
            raise ValueError(f"Unknown predicted class index: {predicted_class}")
        safe_class = predicted_class
        class_name = CLASS_NAMES[safe_class]
        focus_region = _describe_focus_region(heatmap, class_name)
        report = self._generate_structured_report(
            image=image,
            predicted_class=safe_class,
            class_name=class_name,
            confidence=confidence,
            focus_region=focus_region,
        )
        report_source = "generated"
        if report is None:
            mask_fraction = None
            if segmentation_mask is not None and np.asarray(segmentation_mask).size:
                mask = np.asarray(segmentation_mask)
                mask_fraction = float(np.count_nonzero(mask) / mask.size)
            report = _build_aligned_template_report(
                class_name, confidence, focus_region, mask_fraction=mask_fraction,
            )
            report_source = "template_fallback"
        self.last_report_source = report_source

        computed_stage_iou = stage_iou
        if computed_stage_iou is None and segmentation_mask is not None:
            computed_stage_iou = compute_stage_consistency_iou(segmentation_mask, heatmap)
        consistency_note = _build_stage_consistency_note(
            computed_stage_iou,
            warning_threshold=iou_warning_threshold,
        )
        fallback_reason_line = ""
        if report_source != "generated":
            reason = self.last_generation_error or "Unknown generation fallback reason."
            reason = _sanitize_generation_text(reason)
            if len(reason) > 220:
                reason = f"{reason[:217]}..."
            fallback_reason_line = f"Fallback reason: {reason}\n"
        report = (
            f"{report}\n\nModel diagnostics:\n"
            f"Report source: {report_source}\n"
            f"Decoder model: {self.decoder_model_name}\n"
            f"{fallback_reason_line}"
            f"{consistency_note}"
        )

        iou_msg = "N/A" if computed_stage_iou is None else f"{computed_stage_iou:.3f}"
        print(
            f"Model-output report generated successfully "
            f"({class_name}, {confidence:.1%}, source={report_source}, stage_iou={iou_msg})"
        )
        return report
