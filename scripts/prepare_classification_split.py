#!/usr/bin/env python3
"""Freeze classification image splits and duplicate/patient groups."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_splits import create_classification_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/classification_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--groups-csv", type=Path, help="Optional CSV with path (relative to data-dir),patient_id columns")
    args = parser.parse_args(argv)
    manifest = create_classification_manifest(
        args.data_dir, args.output, seed=args.seed, ratios=tuple(args.ratios), groups_csv=args.groups_csv,
    )
    counts = Counter(record["split"] for record in manifest["samples"])
    print(f"Frozen split: {args.output}")
    print(f"Images: {dict(counts)}; independent groups: {len({record['group'] for record in manifest['samples']})}")
    print(f"Fingerprint: {manifest['fingerprint']}")
    print(json.dumps(manifest["audit"], indent=2))
    print("Groups keep byte-identical images together. Patient/source groups require --groups-csv.")


if __name__ == "__main__":
    main()
