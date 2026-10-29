from io import BytesIO
import hashlib
import json

import pytest

from scripts import download_checkpoints as downloads


def checkpoint(payload=b"verified model"):
    return {
        "id": "example", "path": "models/classification/example.pth",
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://example.org/example.pth",
    }


def response(payload):
    stream = BytesIO(payload)
    stream.geturl = lambda: "https://example.org/example.pth"
    return stream


def test_valid_existing_checkpoint_does_not_download(monkeypatch, tmp_path):
    target = tmp_path / "example.pth"
    target.write_bytes(b"verified model")
    monkeypatch.setattr(downloads, "urlopen", lambda *args, **kwargs: pytest.fail("network must not be used"))
    downloads.ensure_checkpoint(checkpoint(), target, verify_only=True)


def test_download_is_verified_before_replacing_existing_file(monkeypatch, tmp_path):
    target = tmp_path / "example.pth"
    target.write_bytes(b"old weight")
    monkeypatch.setattr(downloads, "urlopen", lambda *args, **kwargs: response(b"wrong"))
    with pytest.raises(ValueError, match="mismatch"):
        downloads.ensure_checkpoint(checkpoint(), target)
    assert target.read_bytes() == b"old weight"
    assert list(tmp_path.glob("*.part")) == []


def test_verified_download_is_saved(monkeypatch, tmp_path):
    target = tmp_path / "models" / "example.pth"
    monkeypatch.setattr(downloads, "urlopen", lambda *args, **kwargs: response(b"verified model"))
    downloads.ensure_checkpoint(checkpoint(), target)
    assert target.read_bytes() == b"verified model"


def test_missing_url_is_reported_without_network_or_partial_file(monkeypatch, tmp_path):
    item = checkpoint()
    item["url"] = None
    monkeypatch.setattr(downloads, "urlopen", lambda *args, **kwargs: pytest.fail("network must not be used"))
    with pytest.raises(ValueError, match="No published URL"):
        downloads.ensure_checkpoint(item, tmp_path / "models" / "example.pth")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", ["../example.pth", "/tmp/example.pth", "models/../../example.pth", "models/example.txt"])
def test_manifest_rejects_unsafe_targets(tmp_path, path):
    item = checkpoint()
    item["path"] = path
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "checkpoints": [item]}))
    with pytest.raises(ValueError, match="targets"):
        downloads.load_manifest(manifest, tmp_path)


def test_manifest_rejects_models_symlink_outside_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "models").symlink_to(tmp_path, target_is_directory=True)
    manifest = project / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "checkpoints": [checkpoint()]}))
    with pytest.raises(ValueError, match="inside the project"):
        downloads.load_manifest(manifest, project)


def test_manifest_rejects_insecure_download_url(tmp_path):
    item = checkpoint()
    item["url"] = "http://example.org/example.pth"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "checkpoints": [item]}))
    with pytest.raises(ValueError, match="HTTPS"):
        downloads.load_manifest(manifest, tmp_path)


@pytest.mark.parametrize("data", [[], {}, {"schema_version": 1, "checkpoints": []}, {"schema_version": 1, "checkpoints": [None]}])
def test_manifest_rejects_malformed_data(tmp_path, data):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        downloads.load_manifest(manifest, tmp_path)
