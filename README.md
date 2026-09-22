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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the Kaggle MRI dataset under `data/raw/` (classification images plus paired segmentation masks). Trained weights go under `models/`.


## Local frontend

A localhost workstation UI for single-slice inference:

```bash
pip install -r requirements.txt
python scripts/run_frontend.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Stop it with `python scripts/stop_frontend.py`. Keep trained weights under `models/` (gitignored). If `data/raw/Classification/Testing` is present, the UI also offers one sample per class.

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
python scripts/train_classification.py --mode dual
python scripts/eval_segmentation.py --checkpoint models/segmentation/best_unet.pth
```

Notebooks under `notebooks/` still run against the same `src/` modules. Prefer the scripts for a clean rerun.

## What is not in git

- `data/raw` and `data/processed` (MRI and generated patches)
- `models/**/*.pth`
- `outputs/figures` and Grad-CAM dumps
- `Report/` and `Final_Report/` (proposal, milestone, paper)

Keep `outputs/metrics/*.json` and `vlm_eval/*.csv`; those are the reported numbers.

## Course materials

Assignments, lectures, and labs live in the parent `COMP4471/` folder and should not be pushed with this project.
