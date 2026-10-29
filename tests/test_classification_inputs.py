import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from src.classification import DualInputBrainTumorDataset, build_original_lookup


def test_lossless_patch_name_resolves_to_original_image(tmp_path):
    source = tmp_path / "scan.jpg"
    Image.new("RGB", (8, 8), (100, 150, 200)).save(source)
    patch = tmp_path / "scan.jpg.png"
    Image.new("RGB", (8, 8), (20, 40, 60)).save(patch)
    lookup = build_original_lookup([(str(source), 0)])
    dataset = DualInputBrainTumorDataset(
        [(str(patch), 0)], lookup, patch_transform=transforms.ToTensor(),
        original_transform=transforms.ToTensor(),
    )
    tensor, label = dataset[0]
    with Image.open(source) as image:
        original = transforms.ToTensor()(image.convert("RGB"))
    assert lookup[(0, "scan.jpg.png")] == str(source)
    assert label == 0
    assert np.array_equal(tensor[3:].numpy(), original.numpy())
    assert not np.array_equal(tensor[:3].numpy(), tensor[3:].numpy())


def test_original_lookup_rejects_patch_alias_collisions(tmp_path):
    with pytest.raises(ValueError, match="Ambiguous"):
        build_original_lookup([(str(tmp_path / "scan.jpg"), 0), (str(tmp_path / "scan.jpg.png"), 0)])
