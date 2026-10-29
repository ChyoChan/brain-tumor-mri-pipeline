# Reproducible setup and weights

Use Python 3.11. Install into a fresh environment rather than upgrading an
unrelated Conda environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt
python scripts/run_frontend.py
```

`requirements-runtime.txt` pins direct frontend dependencies. The core pins come
from the local inference environment; `grad-cam` is the correct distribution
name for `pytorch_grad_cam`. Grad-CAM declares `opencv-python`, while
Albumentations declares `opencv-python-headless`; training installs both at the
same pinned OpenCV version. The project uses image-processing APIs, not OpenCV
GUI windows. Minimal Linux systems may need their distribution's `libGL` runtime
package when installing the frontend environment alone.

For training, evaluation plots, and notebook utilities:

```bash
python -m pip install -r requirements-training.txt
```

For tests, add `requirements-dev.txt`; `requirements.txt` installs all three
groups. These are direct dependency pins, not a complete transitive lock file.
The pinned requirements resolve in a clean-install dry run on the development
Mac without modifying its existing environment. The training extras and fresh
Linux installation are checked by GitHub Actions;
the workflow does not download datasets or model weights. A workflow run is
required to establish whether the full environment resolves on GitHub's runner.
CUDA users should install matching PyTorch wheels from the official PyTorch
instructions instead of using the CPU-only CI installation command.

## Existing checkpoints

Model weights are not included in Git. `models/checkpoints.json` records the
size and SHA-256 of the five original checkpoints used by the frontend. It is a
download-integrity manifest, separate from each model's inference metadata.
To verify weights already on disk:

```bash
python scripts/download_checkpoints.py --verify-only
```

The manifest deliberately has `url: null` entries because there are no published
download URLs supplied with this repository. A fresh clone therefore cannot run
live inference until the owner provides the weights or the user trains models.
An absent or mismatched checkpoint is reported, and the app identifies missing
models in its health status.

To publish an approved checkpoint as the repository owner:

1. Confirm the file may be redistributed under its data/model license.
2. Upload that `.pth` file as a GitHub Release asset.
3. Copy the asset's HTTPS download URL into the matching manifest entry.
4. Confirm its size and checksum still match the manifest.

Users can then run:

```bash
python scripts/download_checkpoints.py
```

Downloads are streamed to a temporary file and checked against both size and
SHA-256 before an existing checkpoint is replaced. Weights without a published
URL produce a clear error. Do not put access tokens in manifest URLs.

Newly trained weights need their own published asset, hash entry, and model
metadata; they must not be substituted under an old checkpoint's checksum.

## Automated checks

```bash
python -m pip install -r requirements.txt
python -m pip check
PYTHONPATH=. python -m pytest tests -q
```

`.github/workflows/tests.yml` runs these checks on pushes and pull requests using
CPU PyTorch. Tests use small generated images and model doubles, so the private
dataset and full checkpoint files are unnecessary.
