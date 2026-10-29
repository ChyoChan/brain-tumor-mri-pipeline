# Brain Tumor MRI Analysis Pipeline

COMP4471 term project: a three-stage pipeline for brain-tumor MRI analysis.

**Raw MRI → segmentation (U-Net / ResUNet / Attention U-Net) → ROI crop → ResNet50 classification → Grad-CAM + report**

This folder is the GitHub-ready copy. Large datasets and `.pth` checkpoints stay on disk locally and are ignored by git.

## Layout

```text
configs/                 pipeline hyperparameters
src/                     training, models, data, reports
  constants.py           class names and folder aliases
  preprocessing.py       image/mask discovery, splits, ROI crop, patches
  segmentation.py        UNet, ResUNet, AttentionUNet, train/eval
  classification.py      ResNet50, dual-input dataset, train/eval
  explainability.py      Grad-CAM and VLM/template reports
  visualization.py       figures and qualitative panels
  utils.py               seed, device, config, meters
scripts/                 CLI entry points (no Colab required)
app/                     local FastAPI UI
notebooks/               experiment logs; import from src/
outputs/metrics/         saved scores (checked in)
data/ and models/        local only, not committed
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt
```

Put the Kaggle MRI dataset under `data/raw/` (classification images plus paired segmentation masks). Trained weights go under `models/`.

Use `requirements-training.txt` for training/evaluation and `requirements-dev.txt`
for tests. `requirements.txt` includes all groups. Direct dependency versions are
pinned, including the correct `grad-cam` package name. See [setup and checkpoint
instructions](docs/setup.md) for environment details, CPU/CUDA notes, and
checksum-verified weights. Original weight download URLs are not yet published;
obtain them from the owner or train models before running inference.


## Local frontend

A localhost workstation UI for single-slice inference:

Quick start with the existing `rag` Conda environment:

```bash
./scripts/start_frontend.command
```

On macOS, double-click `scripts/start_frontend.command` in Finder to host the
frontend locally, and `scripts/stop_frontend.command` to close the server.
Keep the start terminal open while using the app. The stop launcher frees the
port; it does not close browser tabs or delete project files.
Both launchers locate the project automatically and work from any directory.
It does not install dependencies; the selected environment must already have
the packages from `requirements.txt`. To use a different Conda environment:

```bash
MRI_CONDA_ENV=myenv ./scripts/start_frontend.command
```

Or start with an already activated Python environment:

```bash
python -m pip install -r requirements-runtime.txt
python scripts/run_frontend.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Port 8000 is left available
for other services. Stop with **Ctrl+C** in the server terminal, or run:

```bash
./scripts/stop_frontend.command
```

An activated Python environment can also run `python scripts/stop_frontend.py`.
The stop script only signals this project's
frontend on the selected port, and refuses to terminate unrelated services.
For a custom port, use matching start/stop arguments:

```bash
./scripts/start_frontend.command --port 8766
./scripts/stop_frontend.command --port 8766
```

Keep trained weights under `models/` (gitignored). If `data/raw/Classification/Testing` is present, the UI also offers one sample per class.

Use **Browse local images** to select other images from that folder, filter by
dataset class, search filenames, and page through results. Folder labels are
dataset annotations, not predictions, and the Testing folder is not necessarily
the same as the project's deterministic held-out evaluation split. Newly added
local images appear after restarting the server. You can also upload any
supported PNG, JPG, or TIFF image (up to 20 MB).

Uncheck **Generate report** before analyzing to skip the template text summary.
Segmentation, classification, and Grad-CAM still run normally. The API accepts
`generate_report=false` in the analysis form and returns `report: null` and
`report_generated: false`; existing callers that omit it still receive a report.

The full-image classifier is the recommended default. The existing dual-input
checkpoint used label-dependent preprocessing during training: no-tumor cases
received the full image in both input views. The frontend now reproduces that
contract without ground-truth labels by running the full-image ResNet50 as a
no-tumor gate before the dual-input ResNet50. A future retrained dual checkpoint
should use the corrected, label-independent patch generation in
`src/preprocessing.py`.

Displayed percentages are softmax model scores, not calibrated clinical
probabilities. The included demo images are members of the deterministic held-out
split (seed 42), rather than training examples.

Verified locally on the 703-image held-out split:

- full-image checkpoint: 99.15% accuracy
- gated dual-input pipeline: 98.86% accuracy
- ungated dual-input pipeline (old frontend behavior): 86.34% accuracy

## Train and evaluate

From the project root:

```bash
python scripts/train_segmentation.py --model unet
python scripts/train_segmentation.py --model resunet
python scripts/train_segmentation.py --model attention_unet
python scripts/train_classification.py --mode dual --manifest data/processed/classification_split_v1.json --seg-checkpoint models/segmentation/best_unet_real_mask.pth
python scripts/eval_segmentation.py --checkpoint models/segmentation/best_unet.pth
```

Notebooks under `notebooks/` still run against the same `src/` modules. Prefer the scripts for a clean rerun.

## Frozen classification evaluation and calibration

Create a split once before training. This groups byte-identical files together;
it checks for conflicting duplicate labels and detects later dataset changes:

```bash
python scripts/prepare_classification_split.py --data-dir data/raw --output data/processed/classification_split_v1.json
python scripts/train_classification.py --mode full --manifest data/processed/classification_split_v1.json
python scripts/eval_classification.py --manifest data/processed/classification_split_v1.json --mode full --output outputs/metrics/full_evaluation.json
```

Add `--groups-csv patients.csv` when creating the split to keep patients together.
The CSV needs `path,patient_id` columns, paths relative to `--data-dir`, and a row
for every image. Without it, the audit guarantees exact-file duplicate grouping,
not patient isolation or detection of re-encoded/augmented copies. Existing split
files are never overwritten. New splits can change the old reported scores.

Evaluation supports `--mode full`, `--mode cropped`, and `--mode dual`. Use the
same manifest for all comparisons and pass `--seg-model`/`--seg-checkpoint` matching
the classifier's recorded training segmenter. Metrics include accuracy, macro
F1, per-class recall, confusion matrix, end-to-end prediction time, ECE, and NLL.
Per-image predictions and reliability CSV/PNG files are written beside the JSON.

New training writes `best_resnet50_<mode>.pth` and a `.metadata.json` sidecar
containing class order, preprocessing, split fingerprints and weight hashes. The
app prefers new full/dual checkpoints automatically. Sidecars must stay with their
weights. Legacy dual weights retain their no-tumor gate; new `roi_v1` weights use
the label-independent crop path. New training also converts both views to
grayscale RGB, matching the app. The UI's available segmentation choices are
validated against a new checkpoint's recorded segmenter and checksum.

Fit temperature scaling on validation data after training, then evaluate once on
test data. Fitting is refused for test data or legacy weights without a verifiable
training split:

```bash
python scripts/eval_classification.py --manifest data/processed/classification_split_v1.json --split val --mode full --fit-temperature models/classification/full_temperature.json --output outputs/metrics/full_validation.json
python scripts/eval_classification.py --manifest data/processed/classification_split_v1.json --split test --mode full --temperature models/classification/full_temperature.json --output outputs/metrics/full_calibrated_test.json
MRI_CALIBRATION=models/classification/full_temperature.json ./scripts/start_frontend.command
```

Calibration artifacts are checked against the selected model, split, class order,
and any segmenter/gate weights. An incompatible artifact is rejected. With no
artifact configured, the app reports uncalibrated softmax scores. Temperature
scaling does not establish clinical reliability or change the predicted class.
Restart the frontend after replacing weights, metadata, or calibration files;
loaded models and calibration settings are cached for the server session.

The optional UI report is now a factual model summary: prediction, score type,
segmentation pixel coverage and attention coordinates. It makes no anatomical
or diagnostic findings from the heatmap.

The main notebook uses the shared, label-independent patch generator and stores
new lossless PNG crops in `data/processed/patches_resunet_roi_v1/`. Existing saved
metrics describe the original experiments; retraining with the corrected crops
can produce different results.

Future segmentation training and validation also use the deployed `[0, 1]`
grayscale input range. Earlier shared training transforms used `[-1, 1]`, which
did not match `predict_mask`. Existing weights and their serving path are
unchanged; saved segmentation metrics are historical results, and comparisons
with retrained segmenters must be rerun.

## What is not in git

- `data/raw` and `data/processed` (MRI and generated patches)
- `models/**/*.pth`
- `outputs/figures` and Grad-CAM dumps
- `Report/` and `Final_Report/` (proposal, milestone, paper)

Keep `outputs/metrics/*.json` and `vlm_eval/*.csv`; those are the reported numbers.

## Course materials

Assignments, lectures, and labs live in the parent `COMP4471/` folder and should not be pushed with this project.
