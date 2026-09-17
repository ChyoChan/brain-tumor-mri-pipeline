"""
Visualization utilities: segmentation overlays, Grad-CAM, confusion matrices,
training curves, and qualitative side-by-side panels.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

from src.constants import CLASS_NAMES


def plot_segmentation_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side: original | GT overlay | predicted overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(gt_mask, cmap="Reds", alpha=0.4)
    axes[1].set_title("Ground-Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(pred_mask, cmap="Blues", alpha=0.4)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and metric curves from training history."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for metric in ("accuracy", "precision", "recall", "f1"):
        if metric in history:
            axes[1].plot(epochs, history[metric], label=metric.capitalize())
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Classification Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_qualitative_panel(
    original_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gradcam_overlay: np.ndarray,
    report_text: str,
    uncertainty_map: Optional[np.ndarray] = None,
    include_uncertainty: bool = True,
    predicted_class: str = "",
    confidence: float = 0.0,
    save_path: Optional[str] = None,
    context_img: Optional[np.ndarray] = None,
    right_panel_title: str = "VLM Summary",
    gt_panel_title: str = "Ground-Truth Mask",
    pred_panel_title: str = "Predicted Mask",
    panel_linewidth: float = 2.4,
    right_panel_width: float = 3.0,
    text_wrap_width: int = 68,
    report_fontsize: float = 10.2,
    header_fontsize: float = 11.2,
) -> None:
    """Qualitative panel display.

    Default panels:
    Original | GT Mask | Predicted Mask | Grad-CAM | Uncertainty | VLM Summary

    If include_uncertainty=False:
    Original | GT Mask | Predicted Mask | Grad-CAM | VLM Summary
    """
    def _show_mri(ax, image: np.ndarray) -> None:
        arr = np.asarray(image)
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray")
        else:
            ax.imshow(arr)

    def _draw_panel_border(ax) -> None:
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                linewidth=panel_linewidth,
                edgecolor="#202020",
            )
        )

    context_arr = np.asarray(context_img) if context_img is not None else np.asarray(original_img)

    def _safe_overlay(mask_like: np.ndarray, ref_shape: tuple) -> np.ndarray:
        arr = np.asarray(mask_like)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        arr = arr.squeeze()
        # Align orientation/shape if mismatched
        if arr.shape != ref_shape:
            if arr.T.shape == ref_shape:
                arr = arr.T
            else:
                from PIL import Image as PILImage
                h, w = ref_shape
                arr = np.array(
                    PILImage.fromarray(arr.astype(np.float32), mode="F").resize((w, h), resample=PILImage.BILINEAR),
                    dtype=np.float32,
                )
        # Normalize to [0,1] then binarize for cleaner overlay
        arr = arr.astype(np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = (arr > 0.5).astype(np.float32)
        return arr

    if include_uncertainty:
        fig = plt.figure(figsize=(31, 6.4))
        gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, right_panel_width])
    else:
        fig = plt.figure(figsize=(28, 6.4))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, right_panel_width])

    ax0 = fig.add_subplot(gs[0])
    _show_mri(ax0, context_arr)
    ax0.set_title("Original MRI", fontsize=11)
    ax0.axis("off")
    _draw_panel_border(ax0)

    # Guard against legacy notebook label overrides like "Reference mask (Otsu, approx.)"
    gt_title = gt_panel_title
    if isinstance(gt_title, str) and "reference" in gt_title.lower():
        gt_title = "Ground-Truth Mask"

    gt_overlay = _safe_overlay(gt_mask, np.asarray(original_img).shape[:2])

    ax1 = fig.add_subplot(gs[1])
    _show_mri(ax1, original_img)
    ax1.imshow(gt_overlay, cmap="Reds", alpha=0.45)
    ax1.set_title(gt_title, fontsize=11)
    ax1.axis("off")
    _draw_panel_border(ax1)

    pred_overlay = _safe_overlay(pred_mask, np.asarray(original_img).shape[:2])

    ax2 = fig.add_subplot(gs[2])
    _show_mri(ax2, original_img)
    ax2.imshow(pred_overlay, cmap="Blues", alpha=0.45)
    ax2.set_title(pred_panel_title, fontsize=11)
    ax2.axis("off")
    _draw_panel_border(ax2)

    # Always use ORIGINAL MRI as the Grad-CAM background.
    ax3 = fig.add_subplot(gs[3])
    _show_mri(ax3, original_img)
    grad_arr = np.asarray(gradcam_overlay)
    ref_shape = np.asarray(original_img).shape[:2]
    if grad_arr.ndim == 3 and grad_arr.shape[-1] in (3, 4):
        if grad_arr.shape[:2] != ref_shape:
            from PIL import Image as PILImage
            h, w = ref_shape
            grad_arr = np.array(
                PILImage.fromarray(grad_arr.astype(np.uint8)).resize((w, h), resample=PILImage.BILINEAR)
            )
        ax3.imshow(grad_arr, alpha=0.65)
    else:
        grad_map = _safe_overlay(grad_arr, ref_shape)
        ax3.imshow(grad_map, cmap="jet", alpha=0.55)
    ax3.set_title("Grad-CAM", fontsize=11)
    ax3.axis("off")
    _draw_panel_border(ax3)

    if include_uncertainty:
        ax4 = fig.add_subplot(gs[4])
        if uncertainty_map is None:
            uncertainty_map = np.zeros_like(pred_mask, dtype=np.float32)
        uncertainty_arr = np.asarray(uncertainty_map, dtype=np.float32).squeeze()
        if uncertainty_arr.ndim == 3:
            uncertainty_arr = uncertainty_arr.mean(axis=-1)
        ax4.imshow(uncertainty_arr, cmap="magma", vmin=0.0, vmax=1.0)
        ax4.set_title("Uncertainty Map", fontsize=11)
        ax4.axis("off")
        _draw_panel_border(ax4)
        report_ax = fig.add_subplot(gs[5])
    else:
        report_ax = fig.add_subplot(gs[4])

    report_ax.axis("off")
    _draw_panel_border(report_ax)
    header = f"Prediction: {predicted_class}  (conf: {confidence:.1%})"
    width_scale = max(1.0, float(right_panel_width) / 2.4)
    effective_wrap_width = max(40, min(160, int(text_wrap_width * width_scale)))
    wrapped = _wrap_text(report_text, width=effective_wrap_width)
    report_ax.text(
        0.05, 0.95, header, transform=report_ax.transAxes,
        fontsize=header_fontsize, fontweight="bold", verticalalignment="top",
    )
    report_ax.text(
        0.04, 0.84, wrapped, transform=report_ax.transAxes,
        fontsize=report_fontsize, verticalalignment="top", wrap=True, linespacing=1.48,
        family="serif",
    )
    report_ax.set_title(right_panel_title, fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(
    labels: List[int],
    class_names: List[str] = CLASS_NAMES,
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
) -> None:
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, counts, color=sns.color_palette("Set2", len(class_names)))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def _wrap_text(text: str, width: int = 60) -> str:
    import textwrap
    wrapped_lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            wrapped_lines.append("")
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(wrapped if wrapped else [""])
    return "\n".join(wrapped_lines)
