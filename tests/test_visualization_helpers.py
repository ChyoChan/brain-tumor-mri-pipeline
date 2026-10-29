import builtins

import numpy as np
import pytest

pytest.importorskip("seaborn")

from src.visualization import _wrap_text, refine_panel_mask


def test_display_mask_removes_speckles_fills_holes_and_keeps_largest_region():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:22, 10:22] = 255
    mask[15, 15] = 0
    mask[29:34, 29:34] = 255
    mask[4, 4] = 255

    expected = np.zeros_like(mask)
    expected[9:23, 9:23] = 1
    result = refine_panel_mask(mask)

    assert result.dtype == np.uint8
    assert np.array_equal(result, expected)


def test_display_mask_falls_back_to_binary_pixels_without_opencv(monkeypatch):
    original_import = builtins.__import__

    def import_without_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("OpenCV unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cv2)
    mask = np.array([[0, 255], [-1, 0]], dtype=np.int16)
    assert np.array_equal(refine_panel_mask(mask), [[0, 1], [0, 0]])


def test_report_wrapping_preserves_paragraphs_and_unbroken_words():
    text = "  Alpha beta gamma   \n\nvery-long-unbroken-word"
    assert _wrap_text(text, width=10) == "Alpha beta\ngamma\n\nvery-long-unbroken-word"
