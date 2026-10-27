import numpy as np
import pytest
from PIL import Image

pytest.importorskip("albumentations", reason="Segmentation training dependencies are optional in the frontend environment")

from src.preprocessing import generate_classification_patches


def test_patch_generation_does_not_use_ground_truth_to_choose_crop(monkeypatch, tmp_path):
    pixels = np.tile(np.arange(16, dtype=np.uint8), (16, 1)) * 12
    image_path = tmp_path / "scan.png"
    Image.fromarray(pixels).save(image_path)
    mask = np.zeros_like(pixels)
    mask[4:12, 6:10] = 1
    monkeypatch.setattr("src.segmentation.predict_mask", lambda *args: mask)

    patches = generate_classification_patches(
        [(str(image_path), 0), (str(image_path), 3)],
        seg_model=object(), device="cpu", output_dir=str(tmp_path / "patches"),
        output_size=(8, 8), margin=0,
    )
    with Image.open(patches[0][0]) as image:
        tumor_crop = np.array(image)
    with Image.open(patches[1][0]) as image:
        no_tumor_crop = np.array(image)
    assert np.array_equal(tumor_crop, no_tumor_crop)
    assert tumor_crop.shape == (8, 8)
    assert not np.array_equal(tumor_crop, np.array(Image.fromarray(pixels).resize((8, 8))))
