import numpy as np
import pytest
from PIL import Image

from src.preprocessing import generate_classification_patches, get_seg_train_transforms, get_seg_val_transforms
from src.roi import crop_tumor_region


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


@pytest.mark.parametrize("with_mask", [False, True])
def test_jpeg_sources_produce_lossless_patches_matching_inference(monkeypatch, tmp_path, with_mask):
    pixels = np.random.default_rng(5).integers(0, 256, size=(24, 24), dtype=np.uint8)
    source = tmp_path / "scan.jpg"
    Image.fromarray(pixels).save(source, quality=83)
    with Image.open(source) as image:
        decoded = np.array(image.convert("L"))
    mask = np.zeros_like(decoded)
    if with_mask:
        mask[6:18, 7:16] = 1
    monkeypatch.setattr("src.segmentation.predict_mask", lambda *args: mask)
    patches = generate_classification_patches(
        [(str(source), 0)], seg_model=object(), device="cpu",
        output_dir=str(tmp_path / "patches"), output_size=(16, 16), margin=0,
    )
    if with_mask:
        expected, _ = crop_tumor_region(decoded, mask, margin=0, output_size=(16, 16))
    else:
        expected = np.array(Image.fromarray(decoded).resize((16, 16)))
    with Image.open(patches[0][0]) as patch:
        assert patch.format == "PNG"
        assert np.array_equal(np.array(patch), expected)
    assert patches[0][0].endswith("scan.jpg.png")


@pytest.mark.parametrize("training", [False, True])
def test_segmentation_transforms_match_served_input_range(training):
    pytest.importorskip("albumentations")
    transform = get_seg_train_transforms() if training else get_seg_val_transforms()
    pixels = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    tensor = transform(image=pixels, mask=np.zeros_like(pixels))["image"]
    assert tensor.shape == (1, 256, 256)
    assert tensor.min() >= 0 and tensor.max() <= 1
    if not training:
        assert np.allclose(tensor.numpy()[0], pixels.astype(np.float32) / 255)
