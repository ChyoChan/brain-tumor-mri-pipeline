import csv
from pathlib import Path

import pytest
from PIL import Image

from src.constants import CLASS_NAMES
from src.data_splits import create_classification_manifest, load_classification_manifest


@pytest.fixture
def classification_images(tmp_path):
    root = tmp_path / "data"
    for label, class_name in enumerate(CLASS_NAMES):
        folder = root / "Training" / class_name
        folder.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (5, 5), (label * 50, index * 20, 10)).save(folder / f"{index:02}.png")
    return root


def test_frozen_split_groups_duplicate_bytes_and_patients(classification_images, tmp_path):
    root = classification_images
    original = root / "Training/glioma/00.png"
    duplicate = root / "Training/glioma/copy.png"
    duplicate.write_bytes(original.read_bytes())
    patient_csv = tmp_path / "patients.csv"
    with patient_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "patient_id"))
        for path in sorted(root.rglob("*.png")):
            relative = path.relative_to(root).as_posix()
            patient = "shared-patient" if relative in {"Training/glioma/00.png", "Training/glioma/01.png"} else relative
            writer.writerow((relative, patient))
    output = tmp_path / "split.json"
    manifest = create_classification_manifest(root, output, groups_csv=patient_csv)
    second = create_classification_manifest(root, tmp_path / "second.json", groups_csv=patient_csv)
    assert manifest["fingerprint"] == second["fingerprint"]
    shared = [record for record in manifest["samples"] if record["path"] in {"Training/glioma/00.png", "Training/glioma/01.png", "Training/glioma/copy.png"}]
    assert len({record["split"] for record in shared}) == 1
    assert len({record["group"] for record in shared}) == 1
    assert manifest["audit"]["duplicate_files"] == 1
    assert manifest["audit"]["grouping"] == "patient+exact_bytes"
    frozen = load_classification_manifest(output)
    assert sum(map(len, frozen.samples.values())) == 41
    assert all(not Path(record["path"]).is_absolute() for record in manifest["samples"])
    assert all({label for _, label in samples} == set(range(4)) for samples in frozen.samples.values())
    with pytest.raises(FileExistsError):
        create_classification_manifest(root, output)


@pytest.mark.parametrize("change", ["modify", "remove", "add"])
def test_manifest_rejects_dataset_changes(classification_images, tmp_path, change):
    output = tmp_path / "split.json"
    create_classification_manifest(classification_images, output)
    image = classification_images / "Training/glioma/00.png"
    if change == "modify":
        Image.new("RGB", (5, 5), "white").save(image)
    elif change == "remove":
        image.unlink()
    else:
        Image.new("RGB", (5, 5), "white").save(image.with_name("new.png"))
    with pytest.raises(ValueError, match="Dataset files changed"):
        load_classification_manifest(output)


def test_conflicting_duplicate_labels_are_rejected(classification_images, tmp_path):
    source = classification_images / "Training/glioma/00.png"
    target = classification_images / "Training/meningioma/conflict.png"
    target.write_bytes(source.read_bytes())
    output = tmp_path / "split.json"
    with pytest.raises(ValueError, match="conflicting class labels"):
        create_classification_manifest(classification_images, output)
    assert not output.exists()


def test_incomplete_patient_csv_is_rejected(classification_images, tmp_path):
    patient_csv = tmp_path / "patients.csv"
    patient_csv.write_text("path,patient_id\nTraining/glioma/00.png,one\n")
    with pytest.raises(ValueError, match="Patient CSV is incomplete"):
        create_classification_manifest(classification_images, tmp_path / "split.json", groups_csv=patient_csv)


def test_dataset_directory_aliases_do_not_double_count_images(classification_images, tmp_path):
    root = classification_images
    canonical = root / "Classification"
    canonical.mkdir()
    (root / "Training").rename(canonical / "Training")
    alias = root / "classification"
    if not alias.exists():  # On macOS, the differently cased name may already refer to the same directory.
        alias.symlink_to(canonical, target_is_directory=True)
    manifest = create_classification_manifest(root, tmp_path / "split.json")
    assert manifest["audit"]["images"] == 40
    assert all(record["path"].startswith("Classification/") for record in manifest["samples"])
