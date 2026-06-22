# MRD Multimodal Prediction

Patient-level binary MRD prediction from four registered 3D CT sequences and
clinical tabular variables. The primary model uses a MedicalNet-pretrained 3D
ResNet-18, a compact clinical MLP, and feature-wise gated fusion.

## Data layout

```text
data/raw/
├── clinical_data.csv
└── <patient_id>/
    ├── sequence_1.nii.gz
    ├── sequence_2.nii.gz
    ├── sequence_3.nii.gz
    └── sequence_4.nii.gz
```

Edit column names, sequence names, and preprocessing dimensions in
`configs/default.yaml`.

## Workflow

```bash
pip install -r requirements.txt

# Validate IDs, labels, sequence completeness/registration, and create
# a stratified 15% test holdout plus five CV folds.
python scripts/preprocess.py --config configs/default.yaml

# Optional deterministic CT cache.
python scripts/preprocess.py --config configs/default.yaml --cache-ct

# Train one fold or all five.
python scripts/train.py --fold 0
python scripts/train.py --all-folds

# Mandatory modality ablations.
python scripts/train.py --all-folds --variant clinical_only
python scripts/train.py --all-folds --variant ct_only

# Pool OOF predictions, select the Youden threshold, and ensemble test scores.
python scripts/evaluate.py \
  --checkpoint-dir experiments/gated_fusion \
  --aggregate-oof --ensemble-test
```

Each fold stores its effective configuration, fitted clinical processor,
best checkpoint, and OOF predictions. Evaluation reads these artifacts and
never refits preprocessing on validation or test patients.

## Tests

```bash
python -m pytest
python -m compileall src scripts
```

Tests use synthetic data and do not require private patient records.

