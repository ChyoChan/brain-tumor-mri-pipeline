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
notebooks/               experiment logs; import from src/
outputs/metrics/         saved scores (checked in)
report/ and Final_Report/  proposal, milestone, paper
data/ and models/        local only, not committed
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the Kaggle MRI dataset under `data/raw/` (classification images plus paired segmentation masks). Trained weights go under `models/`.

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
- LaTeX aux files

Keep `outputs/metrics/*.json` and `vlm_eval/*.csv`; those are the reported numbers.

## Course materials

Assignments, lectures, and labs live in the parent `COMP4471/` folder and should not be pushed with this project.
