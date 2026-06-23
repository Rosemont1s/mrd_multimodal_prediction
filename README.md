# MRD Multimodal Prediction

Patient-level prediction of the first valid postoperative MRD result in
stage II–III, pMMR/MSS colorectal cancer after R0 surgery. This is the
baseline subproject of a broader study of DFS, WES, and longitudinal ctDNA.

Every eligible development patient must receive the MRD assay. Model output is
not used to decide who gets tested while the model is being developed.

## Data layout

```text
data/raw/
├── tables/
│   ├── cohort.csv
│   ├── preoperative.csv
│   ├── pathology.csv
│   ├── mrd.csv
│   ├── ct_manifest.csv
│   ├── longitudinal_ctdna.csv
│   ├── wes.csv
│   └── follow_up.csv
└── clinical_data.csv  # generated only from the audited matched cohort
```

`ct_manifest.csv` links each patient and confirmed CT phase to a de-identified
model-ready NIfTI file and retains acquisition/QC metadata. Original DICOM
locations can be recorded separately in `dicom_source_path`.
If fewer than four phases are consistently available, reduce
`data.ct_sequences` and set `ct_extractor.in_channels` to the same count rather
than synthesizing missing channels.

The bundled workbooks are outdated and must be treated as schema references
only. When a current export with the same sheet/column structure is available,
convert it into canonical, one-row-per-patient baseline columns:

```bash
python scripts/prepare_clinical.py \
  --workbook path/to/current_baseline_export.xlsx \
  --labels-csv path/to/current_mrd_labels.csv \
  --output data/raw/clinical_data.csv
```

This conversion reads only baseline demographics, preoperative laboratory and
treatment data, surgery, and pathology. It excludes names, dates of birth,
telephone numbers, addresses, recurrence, survival, and follow-up treatment.
The WES workbooks remain separate long-form molecular data and are not passed
directly into the clinical MLP.

The source export stores T category, N category, M category, and overall stage
in a combined pathology/clinical sheet. The adapter maps them to postoperative
pathological fields, but this interpretation must be confirmed against the
current data dictionary. The legacy CEA unit is also unusual and must be
verified before combining centers or time periods.

The full linked-table contract is documented in
[`docs/dataset_contract.md`](docs/dataset_contract.md).

## Workflow

```bash
# Create and activate the reproducible Conda environment.
conda env create -f environment.yml
conda activate mrd-multimodal

# Verify the interpreter and GPU-enabled PyTorch installation.
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Create empty linked-table templates.
python scripts/audit_dataset.py --init-templates data/raw/tables

# Populate the templates, confirm the assay/window and true CT phase names in
# configs/default.yaml, then build clinical_data.csv and readiness reports.
python scripts/audit_dataset.py --config configs/default.yaml

# Validate model-ready CT geometry and create retrospective CV folds while
# freezing the prospective cohort as the temporal test set.
python scripts/preprocess.py --config configs/default.yaml

# Optional deterministic CT cache.
python scripts/preprocess.py --config configs/default.yaml --cache-ct

# Train one fold or all five.
python scripts/train.py --fold 0
python scripts/train.py --all-folds

# Required comparators and modality ablations.
python scripts/train.py --all-folds --variant clinical_only \
  --clinical-profile stage_only
python scripts/train.py --all-folds --variant clinical_only \
  --clinical-profile clinical_pathology
python scripts/train.py --all-folds --variant ct_only
python scripts/train.py --all-folds --variant gated_fusion \
  --clinical-profile clinical_pathology

# Fusion ablation: unrestricted concatenation instead of convex gating.
python scripts/train.py --all-folds --variant gated_fusion \
  --clinical-profile clinical_pathology \
  --override fusion.method=concat

# Prespecified broad-window sensitivity analysis. Do not select the window
# from prospective-cohort performance.
python scripts/train.py --all-folds --variant ct_only \
  --override ct_preprocessing.intensity_min=-1024 \
             ct_preprocessing.intensity_max=3071

# Pool OOF predictions, select a high-sensitivity triage threshold, and
# apply it unchanged to the prospective temporal cohort.
python scripts/evaluate.py \
  --checkpoint-dir experiments/gated_fusion_clinical_pathology \
  --aggregate-oof --ensemble-test
```

After dependency changes, update the existing environment with:

```bash
conda env update -f environment.yml --prune
```

The environment uses Python 3.11 and installs the CUDA 12.6 PyTorch wheels
inside the Conda environment. The NVIDIA driver supplies host-level GPU
support; a separate system CUDA toolkit is not required for the prebuilt
PyTorch binaries. CPU-only systems can remove the PyTorch CUDA index line from
`environment.yml` before creating the environment.

The audit refuses definitive training until:

- the MRD assay definition and first-draw window are finalized;
- every otherwise eligible patient has a valid first MRD result;
- stage II–III, R0, pMMR/MSS, treatment and date-order criteria pass;
- true CT phase identities are confirmed and no MRI-style names are used;
- both retrospective and prospective cohorts contain MRD-positive and
  MRD-negative patients.

Audit outputs are written to `data/processed/readiness_report.json` and
`data/processed/audit_issues.csv`. The binding feasibility count is:

```text
eligible ∩ CT-complete ∩ pathology-complete ∩ valid first MRD
```

Each fold stores its effective configuration, fitted clinical processor,
best checkpoint, and OOF predictions. Evaluation reads these artifacts and
never refits preprocessing on validation or test patients.

The CT branch applies one shared single-channel MedicalNet ResNet-18 to each
registered contrast phase and combines the resulting phase embeddings with
learned attention. This preserves the input contract of the pretrained model;
the phases are not averaged by a frozen multi-channel input convolution.
Training first fits the phase-attention, fusion, and classifier layers, then
unfreezes the final residual block and finally the complete fourth residual
stage at a lower learning rate. Preprocessing canonicalizes orientation,
crops the complete CT body foreground, and resizes the complete bounding box
instead of applying an unverified fixed center crop.

Before definitive training, visually audit transformed volumes and verify
tumor coverage for every patient using lesion masks or documented lesion
coordinates where available. Shape and affine agreement alone do not establish
successful inter-phase registration.

The default operating threshold targets 95% sensitivity on pooled out-of-fold
predictions. Test-set reporting includes sensitivity, specificity, PPV, NPV,
the proportion of MRD tests avoided, MRD-positive miss rate, Brier score,
calibration bins, decision-curve data, and configured subgroup results. The
threshold is never selected on the prospective cohort.

## Tests

```bash
python -m pytest
python -m compileall src scripts
```

Tests use synthetic data and do not require private patient records.
