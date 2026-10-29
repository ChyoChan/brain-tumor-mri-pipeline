#!/usr/bin/env python3
"""Verify local checkpoint hashes, or download published HTTPS checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, root: Path) -> list[tuple[dict, Path]]:
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or not isinstance(data.get("checkpoints"), list) or not data["checkpoints"]):
        raise ValueError("Unsupported checkpoint manifest")
    root = root.resolve()
    models = (root / "models").resolve()
    if not models.is_relative_to(root):
        raise ValueError("The models directory must stay inside the project")
    entries = []
    seen = set()
    for item in data["checkpoints"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Each checkpoint needs a relative model path")
        relative = PurePosixPath(item["path"])
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or relative.parts[0] != "models" or relative.suffix != ".pth"):
            raise ValueError("Checkpoint targets must be .pth files under models/")
        target = root.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(models) or target in seen:
            raise ValueError("Checkpoint target escapes models/ or is repeated")
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError("Each checkpoint needs a SHA-256 checksum")
        if type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError("Each checkpoint needs a positive byte size")
        if item.get("url") is not None:
            if not isinstance(item["url"], str):
                raise ValueError("Checkpoint download URLs must use HTTPS")
            url = urlsplit(item["url"])
            if url.scheme != "https" or not url.netloc:
                raise ValueError("Checkpoint download URLs must use HTTPS")
        seen.add(target)
        entries.append((item, target))
    return entries


def ensure_checkpoint(item: dict, target: Path, verify_only: bool = False) -> None:
    if target.is_file() and target.stat().st_size == item["bytes"] and sha256_file(target) == item["sha256"]:
        return
    if verify_only:
        raise ValueError(f"Missing or mismatched checkpoint: {item['path']}")
    if not item.get("url"):
        raise ValueError(f"No published URL for {item['path']}; obtain this weight from the project owner")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(item["url"], timeout=30) as response:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("Checkpoint download redirected outside HTTPS")
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                digest, received = hashlib.sha256(), 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(chunk)
                    if received > item["bytes"]:
                        raise ValueError("Checkpoint download exceeds its declared size")
                    digest.update(chunk)
                    handle.write(chunk)
                if received != item["bytes"] or digest.hexdigest() != item["sha256"]:
                    raise ValueError("Checkpoint checksum or size mismatch")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/checkpoints.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        entries = load_manifest(args.manifest, ROOT)
        failed = False
        for item, target in entries:
            try:
                ensure_checkpoint(item, target, args.verify_only)
                print(f"Verified {item['path']}")
            except (OSError, ValueError) as error:
                failed = True
                print(f"ERROR: {error}")
        return int(failed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
