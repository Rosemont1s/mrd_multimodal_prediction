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
pip install -r requirements.txt

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

# Pool OOF predictions, select a high-sensitivity triage threshold, and
# apply it unchanged to the prospective temporal cohort.
python scripts/evaluate.py \
  --checkpoint-dir experiments/gated_fusion_clinical_pathology \
  --aggregate-oof --ensemble-test
```

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
