"""Frozen classification splits with content and optional patient grouping."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.constants import CLASS_NAMES, FOLDER_ALIASES
from src.checkpoints import sha256_file as file_sha256

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _discover_records(root: Path) -> list[dict]:
    records = {}
    search_roots = [root] + [root / name for name in ("Classification", "classification") if (root / name).is_dir()]
    seen_roots = set()
    for search_root in search_roots:
        identity = (search_root.stat().st_dev, search_root.stat().st_ino)
        if identity in seen_roots:
            continue
        seen_roots.add(identity)
        folders = [search_root / name for name in ("Training", "Testing") if (search_root / name).is_dir()]
        for folder in folders or [search_root]:
            for label, class_name in enumerate(CLASS_NAMES):
                aliases = {alias.casefold() for alias in FOLDER_ALIASES[class_name]}
                for class_dir in sorted(folder.iterdir()):
                    if not class_dir.is_dir() or class_dir.name.casefold() not in aliases:
                        continue
                    for path in sorted(class_dir.rglob("*")):
                        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                            continue
                        resolved = path.resolve()
                        if not resolved.is_relative_to(root):
                            raise ValueError(f"Dataset image resolves outside data directory: {path.name}")
                        relative = path.relative_to(root).as_posix()
                        records[relative] = {
                            "path": relative, "label": label, "sha256": file_sha256(path),
                        }
    if not records:
        raise ValueError(f"No classification images found under {root}")
    return [records[key] for key in sorted(records)]


def _group_records(records: list[dict], groups_csv: str | Path | None) -> None:
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(right)] = find(left)

    by_hash = {}
    for index, record in enumerate(records):
        previous = by_hash.setdefault(record["sha256"], index)
        if records[previous]["label"] != record["label"]:
            raise ValueError(
                "Identical image content has conflicting class labels: "
                f"{records[previous]['path']} and {record['path']}"
            )
        union(previous, index)

    if groups_csv:
        by_path = {record["path"]: index for index, record in enumerate(records)}
        patients, assigned = {}, {}
        with Path(groups_csv).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "patient_id"}.issubset(reader.fieldnames or []):
                raise ValueError("Patient CSV must contain path and patient_id columns")
            for row in reader:
                relative, patient = row["path"].strip(), row["patient_id"].strip()
                if relative not in by_path or not patient:
                    raise ValueError("Patient CSV contains an unknown image path or empty patient_id")
                if relative in assigned and assigned[relative] != patient:
                    raise ValueError("Patient CSV assigns one image to different patients")
                assigned[relative] = patient
                index = by_path[relative]
                union(patients.setdefault(patient, index), index)
        missing = set(by_path) - set(assigned)
        if missing:
            raise ValueError(f"Patient CSV is incomplete: {len(missing)} dataset image(s) have no patient_id")

    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    for members in groups.values():
        group = _fingerprint([record["path"] for record in members])
        for record in members:
            record["group"] = group


def _assign_splits(records: list[dict], ratios: tuple[float, float, float], seed: int) -> None:
    groups = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    strata = defaultdict(list)
    for group, members in groups.items():
        # A patient can contain different labels; keep the whole group together.
        counts = Counter(record["label"] for record in members)
        label = min(counts, key=lambda value: (-counts[value], value))
        strata[label].append(group)
    rng = np.random.default_rng(seed)
    for group_ids in strata.values():
        if len(group_ids) < 3:
            raise ValueError("Each class stratum needs at least three independent groups for train/val/test")
        order = list(rng.permutation(sorted(group_ids)))
        counts = np.maximum(1, np.floor(np.asarray(ratios) * len(order)).astype(int))
        while counts.sum() > len(order):
            candidates = [index for index in range(3) if counts[index] > 1]
            counts[max(candidates, key=lambda index: counts[index] - ratios[index] * len(order))] -= 1
        while counts.sum() < len(order):
            counts[max(range(3), key=lambda index: ratios[index] * len(order) - counts[index])] += 1
        offset = 0
        for split, count in zip(SPLITS, counts):
            for group in order[offset:offset + count]:
                for record in groups[group]:
                    record["split"] = split
            offset += count


def create_classification_manifest(
    data_dir: str | Path, output_path: str | Path, *, seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), groups_csv: str | Path | None = None,
) -> dict:
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive split ratios must sum to one")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Split manifest already exists: {output}")
    root = Path(data_dir).resolve()
    records = _discover_records(root)
    dataset_fingerprint = _fingerprint(records)
    _group_records(records, groups_csv)
    _assign_splits(records, ratios, seed)
    duplicate_counts = Counter(record["sha256"] for record in records)
    audit = {
        "images": len(records), "groups": len({record["group"] for record in records}),
        "duplicate_groups": sum(count > 1 for count in duplicate_counts.values()),
        "duplicate_files": sum(count - 1 for count in duplicate_counts.values()),
        "grouping": "patient+exact_bytes" if groups_csv else "exact_bytes_only",
        "notes": [
            "Duplicate files counts extra byte-identical copies; re-encoded and near-duplicate images are not detected.",
            "Patient isolation is enforced only when a complete patient CSV is supplied.",
        ],
    }
    body = {
        "version": 1, "class_names": CLASS_NAMES, "seed": seed, "ratios": ratios,
        "dataset_fingerprint": dataset_fingerprint, "audit": audit, "samples": records,
    }
    manifest = {**body, "fingerprint": _fingerprint(body), "data_root": os.path.relpath(root, output.parent)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


@dataclass(frozen=True)
class ClassificationSplit:
    root: Path
    samples: dict[str, list[tuple[str, int]]]
    fingerprint: str
    dataset_fingerprint: str
    seed: int


def load_classification_manifest(path: str | Path, data_dir: str | Path | None = None) -> ClassificationSplit:
    source = Path(path).resolve()
    manifest = json.loads(source.read_text())
    if manifest.get("version") != 1 or manifest.get("class_names") != CLASS_NAMES:
        raise ValueError("Split manifest version or class order is incompatible")
    body = {key: value for key, value in manifest.items() if key not in {"fingerprint", "data_root"}}
    if _fingerprint(body) != manifest.get("fingerprint"):
        raise ValueError("Split manifest fingerprint does not match its contents")
    root = Path(data_dir).resolve() if data_dir else (source.parent / manifest["data_root"]).resolve()
    records = manifest["samples"]
    actual = _discover_records(root)
    if _fingerprint(actual) != manifest["dataset_fingerprint"]:
        raise ValueError("Dataset files changed, are missing, or do not match the frozen manifest")
    expected = [{key: record[key] for key in ("path", "label", "sha256")} for record in records]
    if expected != actual:
        raise ValueError("Manifest images or class labels do not match the dataset")
    samples = {split: [] for split in SPLITS}
    hashes, groups = {}, {}
    seen = set()
    for record in records:
        relative = Path(record["path"])
        image = (root / relative).resolve()
        label, split = record["label"], record["split"]
        if relative.is_absolute() or not image.is_relative_to(root) or record["path"] in seen:
            raise ValueError("Manifest contains an unsafe or repeated image path")
        if type(label) is not int or not 0 <= label < len(CLASS_NAMES) or split not in SPLITS:
            raise ValueError("Manifest contains an invalid label or split")
        for mapping, key in ((hashes, record["sha256"]), (groups, record["group"])):
            if mapping.setdefault(key, split) != split:
                raise ValueError("Duplicate image or patient group crosses split boundaries")
        seen.add(record["path"])
        samples[split].append((str(image), label))
    if not all(samples.values()):
        raise ValueError("Frozen manifest must contain nonempty train, val, and test splits")
    return ClassificationSplit(root, samples, manifest["fingerprint"], manifest["dataset_fingerprint"], manifest["seed"])
