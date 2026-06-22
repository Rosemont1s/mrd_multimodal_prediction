# Baseline MRD Dataset Contract

## Endpoint and analytical population

The endpoint is the first postoperative MRD assay that has both:

- a result of `positive` or `negative`; and
- an assay-QC status of `pass`, `passed`, `valid`, or `acceptable`.

`invalid` and `indeterminate` attempts are retained and counted, never
converted to MRD-negative. The analytical cohort is restricted to explicitly
included patients who are age 18–75, ECOG 0–1, stage II–III, pMMR/MSS, without
distant metastasis, after curative-intent R0 surgery, with no neoadjuvant
treatment, no visible residual disease, and no systemic treatment before the
first valid MRD draw.

## Linked source tables

Generate exact CSV headers with:

```bash
python scripts/audit_dataset.py --init-templates data/raw/tables
```

- `cohort.csv`: screening disposition, retrospective/prospective period,
  age, sex, and ECOG. Excluded patients require one exclusion reason.
- `preoperative.csv`: clinical TNM/site, CEA, CA19-9, dates, units, reference
  limits, laboratory, comorbidities, and neoadjuvant-treatment status.
- `pathology.csv`: surgery, R0 status, pathological TNM/stage, histology,
  tumor size, node counts, invasion/deposits, obstruction/perforation,
  MMR/MSI, HER2, BRAF, Ki-67, and residual/metastatic-disease checks.
- `mrd.csv`: one row per postoperative assay attempt, including platform,
  version, positivity rule, draw date, plasma volume, QC and treatment timing.
- `ct_manifest.csv`: one row per patient and CT phase with de-identified
  source/model paths, acquisition metadata, and geometry/registration QC.
- `longitudinal_ctdna.csv`, `wes.csv`, and `follow_up.csv`: retained for the
  parent study and excluded from baseline MRD predictors.

Direct identifiers such as names, medical-record numbers, birth dates,
telephone numbers and addresses are forbidden in these analytical tables.

## Timing and consistency rules

- Preoperative CT and laboratory measurements must precede surgery.
- The first valid MRD draw must follow surgery and fall inside the finalized
  protocol window.
- Positive lymph nodes cannot exceed examined lymph nodes.
- Stage II requires N0/M0; stage III requires node-positive/M0 disease.
- Derived pMMR cannot conflict with loss of MLH1, PMS2, MSH2 or MSH6.
- CT phase sets must exactly match `data.ct_sequences`, use modality `CT`, and
  must not use MRI names such as T1, T2, T1C or FLAIR.
- CT phase identity, de-identification, geometry and registration must pass
  before a patient enters the matched cohort.

## Leakage boundary

The baseline model may use only preoperative CT, preoperative clinical/lab
data, surgery and postoperative pathology available before the first MRD
result. Later ctDNA results, recurrence, survival, surveillance imaging and
post-MRD treatment remain in parent-study tables and cannot enter the clinical
feature allowlist.

## Validation contract

`cohort_period=retrospective` supplies patient-level development folds.
`cohort_period=prospective` is the untouched temporal test cohort. Imputation,
encoding, scaling, model selection and the high-sensitivity operating
threshold are fitted using retrospective data only.
