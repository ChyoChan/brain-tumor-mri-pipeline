Experiment notebooks. Training and models live in `src/`; use `scripts/` to rerun without Colab.

The main notebook imports shared patch generation, the dual-input dataset, and
the ResNet50 builder. New patches are stored in
`data/processed/patches_resunet_label_independent/` so reruns do not reuse older
label-dependent crops. Retraining can therefore produce different results from
the archived metrics and checkpoints.
