Experiment notebooks. Training and models live in `src/`; use `scripts/` to rerun without Colab.

The main notebook imports shared patch generation, the dual-input dataset, and
the ResNet50 builder. New patches are stored in
`data/processed/patches_resunet_roi_v1/` so reruns do not reuse older crops.
Each crop is a lossless PNG named after the original filename plus `.png`
(for example, `scan.jpg.png`), preserving the same crop pixels used during
inference. Shared lookup helpers map these names back to the original image.
Retraining can therefore produce different results from the archived metrics
and checkpoints.
