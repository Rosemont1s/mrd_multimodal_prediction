"""Linked-table assembly and readiness auditing for the MRD subproject."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_COLUMNS = [
    "age",
    "sex",
    "ecog",
    "cea_preoperative",
    "ca199_preoperative",
    "clinical_t",
    "clinical_n",
    "clinical_m",
    "tumor_location",
    "tumor_side",
    "pathological_t",
    "pathological_n",
    "pathological_m",
    "pathological_stage",
    "tumor_size_pathological_mm",
    "histological_type",
    "histological_grade",
    "lymphovascular_invasion",
    "perineural_invasion",
    "tumor_deposits",
    "bowel_obstruction",
    "tumor_perforation",
    "examined_lymph_nodes",
    "positive_lymph_nodes",
    "surgery_procedure",
    "resection_margin",
    "her2_status",
    "braf_status",
    "ki67_percent",
]

QC_COLUMNS = [
    "mmr_status",
    "msi_status",
    "mlh1_status",
    "pms2_status",
    "msh2_status",
    "msh6_status",
    "assay_platform",
    "assay_version",
    "positivity_rule",
    "plasma_volume_ml",
    "assay_qc_status",
    "mrd_attempt_count",
    "invalid_mrd_attempt_count",
    "scanner_manufacturer",
    "scanner_model",
    "ct_phase_count",
]

TABLE_SCHEMAS: Dict[str, list[str]] = {
    "cohort": [
        "patient_id",
        "cohort_period",
        "screened",
        "inclusion_status",
        "exclusion_reason",
        "age",
        "sex",
        "ecog",
    ],
    "preoperative": [
        "patient_id",
        "clinical_t",
        "clinical_n",
        "clinical_m",
        "tumor_location",
        "tumor_side",
        "cea_preoperative",
        "cea_unit",
        "cea_date",
        "cea_reference_upper",
        "ca199_preoperative",
        "ca199_unit",
        "ca199_date",
        "ca199_reference_upper",
        "laboratory_id",
        "relevant_comorbidities",
        "neoadjuvant_treatment",
    ],
    "pathology": [
        "patient_id",
        "primary_crc_adenocarcinoma",
        "surgery_date",
        "surgery_procedure",
        "curative_intent",
        "resection_margin",
        "pathological_t",
        "pathological_n",
        "pathological_m",
        "pathological_stage",
        "tumor_size_pathological_mm",
        "histological_type",
        "histological_grade",
        "positive_lymph_nodes",
        "examined_lymph_nodes",
        "lymphovascular_invasion",
        "perineural_invasion",
        "tumor_deposits",
        "bowel_obstruction",
        "tumor_perforation",
        "mlh1_status",
        "pms2_status",
        "msh2_status",
        "msh6_status",
        "mmr_status",
        "msi_status",
        "her2_status",
        "her2_method",
        "braf_status",
        "braf_method",
        "braf_variant",
        "ki67_percent",
        "distant_metastasis",
        "visible_residual_disease",
    ],
    "mrd": [
        "patient_id",
        "blood_draw_date",
        "mrd_result",
        "assay_platform",
        "assay_version",
        "positivity_rule",
        "plasma_volume_ml",
        "assay_qc_status",
        "systemic_treatment_before_draw",
    ],
    "ct_manifest": [
        "patient_id",
        "study_date",
        "phase_name",
        "modality",
        "dicom_source_path",
        "image_path",
        "anatomical_coverage",
        "scanner_manufacturer",
        "scanner_model",
        "tube_voltage_kvp",
        "tube_current_ma",
        "slice_thickness_mm",
        "spacing_x_mm",
        "spacing_y_mm",
        "reconstruction_kernel",
        "contrast_timing_seconds",
        "contrast_dose_ml",
        "deidentified",
        "geometry_qc",
        "registration_qc",
        "tumor_annotation_path",
    ],
    "longitudinal_ctdna": [
        "patient_id",
        "blood_draw_date",
        "ctdna_result",
        "ctdna_value",
        "assay_platform",
        "assay_qc_status",
    ],
    "wes": [
        "patient_id",
        "tumor_sample_id",
        "normal_sample_id",
        "sequencing_qc_status",
        "tumor_mutational_burden",
        "variant_file",
        "pathway_feature_file",
    ],
    "follow_up": [
        "patient_id",
        "last_follow_up_date",
        "recurrence_status",
        "recurrence_date",
        "death_status",
        "death_date",
        "dfs_days",
        "os_days",
    ],
}

REQUIRED_TABLES = {"cohort", "preoperative", "pathology", "mrd", "ct_manifest"}

PHI_COLUMNS = {
    "name",
    "patient_name",
    "姓名",
    "phone",
    "phone_number",
    "联系电话",
    "address",
    "联系地址",
    "medical_record_number",
    "病历号",
    "date_of_birth",
    "出生日期",
    "patient_initials",
    "患者缩写",
}

LEAKAGE_COLUMNS = {
    "recurrence",
    "recurrence_status",
    "recurrence_date",
    "death_status",
    "death_date",
    "dfs_days",
    "os_days",
    "overall_survival",
    "disease_free_survival",
    "later_ctdna_result",
    "ctdna_trajectory",
    "adjuvant_treatment_after_mrd",
}

VALID_MRD_RESULTS = {"positive", "negative", "invalid", "indeterminate"}
VALID_QC_PASS = {"pass", "passed", "valid", "acceptable"}


@dataclass
class AuditResult:
    """Cohort assembly result and machine-readable readiness report."""

    analytical_cohort: pd.DataFrame
    report: Dict[str, Any]
    issues: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ready(self) -> bool:
        return not self.report["blockers"]


def write_table_templates(output_dir: str | Path) -> None:
    """Write empty CSV templates for every linked source table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, columns in TABLE_SCHEMAS.items():
        pd.DataFrame(columns=columns).to_csv(output / f"{name}.csv", index=False)


def _normalize_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _normalized_text(value: Any) -> str:
    return str(value).strip().lower()


def _is_yes(value: Any) -> bool:
    return _normalized_text(value) in {"yes", "y", "true", "1", "是", "有"}


def _is_no(value: Any) -> bool:
    return _normalized_text(value) in {"no", "n", "false", "0", "否", "无"}


def _is_stage_ii_or_iii(value: Any) -> bool:
    return _stage_group(value) in {2, 3}


def _stage_group(value: Any) -> int | None:
    text = _normalized_text(value).replace("stage", "").strip()
    if text.startswith(("iii", "Ⅲ", "3")):
        return 3
    if text.startswith(("ii", "Ⅱ", "2")):
        return 2
    return None


def _is_r0(value: Any) -> bool:
    return _normalized_text(value).replace(" ", "") in {
        "r0",
        "negative",
        "阴性",
    }


def _is_pmmr(value: Any) -> bool:
    return _normalized_text(value).replace("-", "") in {
        "pmmr",
        "proficient",
        "intact",
    }


def _is_mss(value: Any) -> bool:
    return _normalized_text(value).replace("-", "") in {"mss", "stable"}


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _load_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing linked table '{name}': {path}")
    frame = pd.read_csv(path, dtype={"patient_id": str})
    missing = sorted(set(TABLE_SCHEMAS[name]) - set(frame.columns))
    if missing:
        raise ValueError(f"Table '{name}' is missing columns: {missing}")
    phi_found = sorted(set(frame.columns).intersection(PHI_COLUMNS))
    if phi_found:
        raise ValueError(
            f"Table '{name}' contains direct identifiers: {phi_found}"
        )
    frame["patient_id"] = frame["patient_id"].map(_normalize_id)
    if frame["patient_id"].eq("").any():
        raise ValueError(f"Table '{name}' contains blank patient IDs.")
    if name != "mrd" and name != "ct_manifest":
        if frame["patient_id"].duplicated().any():
            duplicates = frame.loc[
                frame["patient_id"].duplicated(), "patient_id"
            ].tolist()
            raise ValueError(
                f"Table '{name}' contains duplicate patients: {duplicates[:10]}"
            )
    return frame


def load_linked_tables(data_cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load configured source tables using the canonical schemas."""
    configured = data_cfg.get("cohort_tables", {})
    missing = sorted(REQUIRED_TABLES - set(configured))
    if missing:
        raise ValueError(f"data.cohort_tables is missing entries: {missing}")
    missing_files = [
        name
        for name in REQUIRED_TABLES
        if not Path(configured[name]).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Required linked-table files do not exist: {sorted(missing_files)}"
        )
    return {
        name: _load_table(Path(path), name)
        for name, path in configured.items()
        if name in TABLE_SCHEMAS and Path(path).exists()
    }


def _select_first_valid_mrd(
    mrd: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = mrd.copy()
    frame["mrd_result"] = frame["mrd_result"].map(_normalized_text)
    frame["assay_qc_status"] = frame["assay_qc_status"].map(_normalized_text)
    invalid_results = ~frame["mrd_result"].isin(VALID_MRD_RESULTS)
    if invalid_results.any():
        values = sorted(frame.loc[invalid_results, "mrd_result"].unique())
        raise ValueError(f"Unknown MRD result values: {values}")
    frame["blood_draw_date_parsed"] = _parse_date(frame["blood_draw_date"])
    frame = frame.sort_values(["patient_id", "blood_draw_date_parsed"])
    valid = frame[
        frame["mrd_result"].isin({"positive", "negative"})
        & frame["assay_qc_status"].isin(VALID_QC_PASS)
    ].copy()
    duplicate_valid_draw = valid.duplicated(
        ["patient_id", "blood_draw_date_parsed"], keep=False
    )
    if duplicate_valid_draw.any():
        patients = sorted(
            valid.loc[duplicate_valid_draw, "patient_id"].unique()
        )
        raise ValueError(
            "Patients have multiple valid MRD rows on the same draw date: "
            f"{patients[:10]}"
        )
    first_valid = valid.drop_duplicates("patient_id", keep="first")
    first_attempt = frame.drop_duplicates("patient_id", keep="first")
    attempts = frame.groupby("patient_id").size().rename("mrd_attempt_count")
    invalid_before_valid = (
        frame.assign(
            invalid_attempt=~(
                frame["mrd_result"].isin({"positive", "negative"})
                & frame["assay_qc_status"].isin(VALID_QC_PASS)
            )
        )
        .groupby("patient_id")["invalid_attempt"]
        .sum()
        .rename("invalid_mrd_attempt_count")
    )
    first_valid = first_valid.merge(
        attempts, on="patient_id", how="left"
    ).merge(invalid_before_valid, on="patient_id", how="left")
    return first_valid, first_attempt


def _patient_issue(
    issues: list[dict[str, str]],
    patient_ids: Iterable[str],
    code: str,
    severity: str,
    message: str,
) -> None:
    for patient_id in patient_ids:
        issues.append(
            {
                "patient_id": str(patient_id),
                "code": code,
                "severity": severity,
                "message": message,
            }
        )


def _audit_ct(
    ct_manifest: pd.DataFrame,
    expected_phases: list[str],
    phases_confirmed: bool,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    ct = ct_manifest.copy()
    ct["study_date_parsed"] = _parse_date(ct["study_date"])
    ct["phase_name"] = ct["phase_name"].astype(str).str.strip()
    ct["modality"] = ct["modality"].map(_normalized_text)
    duplicate = ct.duplicated(["patient_id", "phase_name"], keep=False)
    _patient_issue(
        issues,
        ct.loc[duplicate, "patient_id"].unique(),
        "duplicate_ct_phase",
        "blocker",
        "Patient has duplicate rows for the same CT phase.",
    )
    non_ct = ct["modality"].ne("ct")
    _patient_issue(
        issues,
        ct.loc[non_ct, "patient_id"].unique(),
        "non_ct_modality",
        "blocker",
        "Imaging manifest contains a non-CT modality.",
    )
    mri_names = {"t1", "t2", "t1c", "t2flair", "t2-flair", "flair"}
    mislabeled = ct["phase_name"].str.lower().isin(mri_names)
    _patient_issue(
        issues,
        ct.loc[mislabeled, "patient_id"].unique(),
        "mri_sequence_name",
        "blocker",
        "CT phase uses an MRI sequence name.",
    )
    if expected_phases:
        phase_sets = ct.groupby("patient_id")["phase_name"].agg(set)
        incomplete = phase_sets[
            phase_sets.map(lambda values: set(expected_phases) != values)
        ].index
        _patient_issue(
            issues,
            incomplete,
            "incomplete_ct_phases",
            "blocker",
            "CT phase set does not match the prespecified phase list.",
        )
    if not phases_confirmed:
        _patient_issue(
            issues,
            ct["patient_id"].unique(),
            "ct_phases_not_confirmed",
            "blocker",
            "CT phase identities have not been confirmed from source metadata.",
        )
    for column, code in (
        ("deidentified", "ct_not_deidentified"),
        ("geometry_qc", "ct_geometry_qc_failed"),
        ("registration_qc", "ct_registration_qc_failed"),
    ):
        failed = ~ct[column].map(_is_yes)
        _patient_issue(
            issues,
            ct.loc[failed, "patient_id"].unique(),
            code,
            "blocker",
            f"CT manifest failed required {column} check.",
        )
    missing_path = ~ct["image_path"].map(lambda value: Path(str(value)).exists())
    _patient_issue(
        issues,
        ct.loc[missing_path, "patient_id"].unique(),
        "missing_ct_path",
        "blocker",
        "Configured CT image path does not exist.",
    )
    return ct


def audit_and_build_cohort(cfg: Dict[str, Any]) -> AuditResult:
    """Audit linked tables and build the first-postoperative-MRD model table."""
    data_cfg = cfg["data"]
    tables = load_linked_tables(data_cfg)
    issues: list[dict[str, str]] = []

    cohort = tables["cohort"].copy()
    preoperative = tables["preoperative"].copy()
    pathology = tables["pathology"].copy()
    first_mrd, first_attempt = _select_first_valid_mrd(tables["mrd"])
    ct = _audit_ct(
        tables["ct_manifest"],
        list(data_cfg.get("ct_sequences", [])),
        bool(data_cfg.get("ct_phases_confirmed", False)),
        issues,
    )

    merged = cohort.merge(
        preoperative, on="patient_id", how="left", validate="one_to_one"
    ).merge(pathology, on="patient_id", how="left", validate="one_to_one")
    merged = merged.merge(
        first_mrd, on="patient_id", how="left", validate="one_to_one"
    )

    ct_summary = (
        ct.groupby("patient_id")
        .agg(
            ct_study_date=("study_date_parsed", "min"),
            ct_phase_count=("phase_name", "nunique"),
            scanner_manufacturer=("scanner_manufacturer", "first"),
            scanner_model=("scanner_model", "first"),
        )
        .reset_index()
    )
    merged = merged.merge(
        ct_summary, on="patient_id", how="left", validate="one_to_one"
    )

    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    merged["ecog"] = pd.to_numeric(merged["ecog"], errors="coerce")
    merged["positive_lymph_nodes"] = pd.to_numeric(
        merged["positive_lymph_nodes"], errors="coerce"
    )
    merged["examined_lymph_nodes"] = pd.to_numeric(
        merged["examined_lymph_nodes"], errors="coerce"
    )
    merged["surgery_date_parsed"] = _parse_date(merged["surgery_date"])
    merged["cea_date_parsed"] = _parse_date(merged["cea_date"])
    merged["ca199_date_parsed"] = _parse_date(merged["ca199_date"])
    merged["blood_draw_date_parsed"] = _parse_date(merged["blood_draw_date"])
    merged["mrd_status"] = merged["mrd_result"].map(
        {"negative": 0, "positive": 1}
    )
    merged["surgery_to_mrd_days"] = (
        merged["blood_draw_date_parsed"] - merged["surgery_date_parsed"]
    ).dt.days
    merged["ct_to_surgery_days"] = (
        merged["surgery_date_parsed"] - merged["ct_study_date"]
    ).dt.days

    protocol_checks = {
        "explicitly_included": merged["inclusion_status"].map(
            _normalized_text
        ).eq("included"),
        "age_18_75": merged["age"].between(18, 75, inclusive="both"),
        "ecog_0_1": merged["ecog"].isin([0, 1]),
        "primary_crc": merged["primary_crc_adenocarcinoma"].map(_is_yes),
        "stage_ii_iii": merged["pathological_stage"].map(_is_stage_ii_or_iii),
        "curative_intent": merged["curative_intent"].map(_is_yes),
        "r0_resection": merged["resection_margin"].map(_is_r0),
        "pmmr": merged["mmr_status"].map(_is_pmmr),
        "mss": merged["msi_status"].map(_is_mss),
        "no_distant_metastasis": merged["distant_metastasis"].map(_is_no),
        "no_visible_residual": merged["visible_residual_disease"].map(_is_no),
        "no_neoadjuvant": merged["neoadjuvant_treatment"].map(_is_no),
    }
    eligible = pd.Series(True, index=merged.index)
    for name, passed in protocol_checks.items():
        failed = ~passed.fillna(False)
        eligible &= ~failed
        _patient_issue(
            issues,
            merged.loc[failed, "patient_id"],
            name,
            "exclusion",
            f"Patient failed cohort criterion: {name}.",
        )

    protocol_eligible = eligible.copy()
    missing_valid_mrd = protocol_eligible & ~merged["mrd_status"].isin([0, 1])
    _patient_issue(
        issues,
        merged.loc[missing_valid_mrd, "patient_id"],
        "missing_valid_first_mrd",
        "blocker",
        "Protocol-eligible patient has no valid first postoperative MRD result.",
    )
    eligible &= ~missing_valid_mrd
    treatment_before_mrd = ~merged[
        "systemic_treatment_before_draw"
    ].map(_is_no).fillna(False)
    eligible &= ~treatment_before_mrd
    _patient_issue(
        issues,
        merged.loc[treatment_before_mrd, "patient_id"],
        "treatment_before_first_mrd",
        "exclusion",
        "Systemic treatment occurred before the first valid MRD draw.",
    )

    chronology_checks = {
        "ct_before_surgery": merged["ct_study_date"].le(
            merged["surgery_date_parsed"]
        ),
        "cea_before_surgery": merged["cea_date_parsed"].le(
            merged["surgery_date_parsed"]
        ),
        "ca199_before_surgery": merged["ca199_date_parsed"].le(
            merged["surgery_date_parsed"]
        ),
        "mrd_after_surgery": merged["blood_draw_date_parsed"].gt(
            merged["surgery_date_parsed"]
        ),
    }
    for name, passed in chronology_checks.items():
        failed = ~passed.fillna(False)
        eligible &= ~failed
        _patient_issue(
            issues,
            merged.loc[failed, "patient_id"],
            name,
            "blocker",
            f"Patient failed date-order check: {name}.",
        )

    expected_units = data_cfg.get(
        "expected_lab_units",
        {"cea": "ng/mL", "ca199": "U/mL"},
    )
    for column, key in (("cea_unit", "cea"), ("ca199_unit", "ca199")):
        expected = str(expected_units[key]).strip().lower()
        unit_mismatch = merged[column].astype(str).str.strip().str.lower().ne(
            expected
        )
        eligible &= ~unit_mismatch
        _patient_issue(
            issues,
            merged.loc[unit_mismatch, "patient_id"],
            f"{key}_unit_mismatch",
            "blocker",
            f"{key.upper()} unit does not match prespecified unit {expected}.",
        )

    node_error = (
        merged["positive_lymph_nodes"].notna()
        & merged["examined_lymph_nodes"].notna()
        & (
            merged["positive_lymph_nodes"]
            > merged["examined_lymph_nodes"]
        )
    )
    eligible &= ~node_error
    _patient_issue(
        issues,
        merged.loc[node_error, "patient_id"],
        "invalid_lymph_node_counts",
        "blocker",
        "Positive lymph nodes exceed examined lymph nodes.",
    )

    stage_group = merged["pathological_stage"].map(_stage_group)
    node_text = merged["pathological_n"].map(_normalized_text).str.replace(
        r"^[pc]", "", regex=True
    )
    metastasis_text = merged["pathological_m"].map(
        _normalized_text
    ).str.replace(r"^[pc]", "", regex=True)
    stage_ii = stage_group.eq(2)
    stage_iii = stage_group.eq(3)
    stage_discordance = (
        (stage_ii & ~node_text.str.startswith("n0"))
        | (stage_iii & node_text.str.startswith("n0"))
        | (~metastasis_text.str.startswith("m0"))
    )
    eligible &= ~stage_discordance
    _patient_issue(
        issues,
        merged.loc[stage_discordance, "patient_id"],
        "stage_tnm_discordance",
        "blocker",
        "Pathological stage is inconsistent with N/M components.",
    )

    mmr_proteins = [
        "mlh1_status",
        "pms2_status",
        "msh2_status",
        "msh6_status",
    ]
    protein_loss = merged[mmr_proteins].apply(
        lambda column: column.map(_normalized_text).isin(
            {"negative", "loss", "lost", "-", "阴性"}
        )
    )
    mmr_discordance = merged["mmr_status"].map(_is_pmmr) & protein_loss.any(axis=1)
    eligible &= ~mmr_discordance
    _patient_issue(
        issues,
        merged.loc[mmr_discordance, "patient_id"],
        "mmr_ihc_discordance",
        "blocker",
        "Derived pMMR status conflicts with loss of an MMR protein.",
    )

    endpoint_cfg = data_cfg.get("mrd_endpoint", {})
    if endpoint_cfg.get("blood_draw_window_finalized"):
        minimum = endpoint_cfg.get("blood_draw_min_days")
        maximum = endpoint_cfg.get("blood_draw_max_days")
        in_window = pd.Series(True, index=merged.index)
        if minimum is not None:
            in_window &= merged["surgery_to_mrd_days"].ge(int(minimum))
        if maximum is not None:
            in_window &= merged["surgery_to_mrd_days"].le(int(maximum))
        outside_window = ~in_window.fillna(False)
        eligible &= ~outside_window
        _patient_issue(
            issues,
            merged.loc[outside_window, "patient_id"],
            "mrd_draw_outside_window",
            "blocker",
            "First valid MRD draw is outside the prespecified window.",
        )

    explicit_included = merged["inclusion_status"].map(_normalized_text).eq(
        "included"
    )
    included_with_reason = explicit_included & merged[
        "exclusion_reason"
    ].notna()
    _patient_issue(
        issues,
        merged.loc[included_with_reason, "patient_id"],
        "included_with_exclusion_reason",
        "warning",
        "Included patient has a non-empty exclusion reason.",
    )
    excluded_without_reason = ~explicit_included & (
        merged["exclusion_reason"].isna()
        | merged["exclusion_reason"].astype(str).str.strip().eq("")
    )
    _patient_issue(
        issues,
        merged.loc[excluded_without_reason, "patient_id"],
        "excluded_without_reason",
        "warning",
        "Screened but excluded patient lacks an explicit exclusion reason.",
    )

    ct_blocked_patients = {
        issue["patient_id"]
        for issue in issues
        if issue["severity"] == "blocker"
        and issue["code"].startswith(
            (
                "ct_",
                "duplicate_ct",
                "incomplete_ct",
                "missing_ct",
                "non_ct",
                "mri_",
            )
        )
    }
    eligible &= ~merged["patient_id"].isin(ct_blocked_patients)

    analytical = merged.loc[eligible].copy()
    analytical["cohort_period"] = analytical["cohort_period"].map(
        _normalized_text
    )
    output_columns = [
        "patient_id",
        "cohort_period",
        "mrd_status",
        "surgery_date",
        "blood_draw_date",
        "surgery_to_mrd_days",
        "ct_study_date",
        "ct_to_surgery_days",
        *MODEL_COLUMNS,
        *QC_COLUMNS,
    ]
    output_columns = list(dict.fromkeys(output_columns))
    analytical = analytical[output_columns].sort_values("patient_id")

    missingness = {
        column: float(analytical[column].isna().mean())
        for column in MODEL_COLUMNS
        if column in analytical
    }
    missingness_by_mrd_class = {
        str(label): {
            column: float(group[column].isna().mean())
            for column in MODEL_COLUMNS
            if column in group
        }
        for label, group in analytical.groupby("mrd_status")
    }
    missingness_by_period = {
        str(period): {
            column: float(group[column].isna().mean())
            for column in MODEL_COLUMNS
            if column in group
        }
        for period, group in analytical.groupby("cohort_period")
    }
    period_counts = (
        analytical.groupby(["cohort_period", "mrd_status"])
        .size()
        .rename("patients")
        .reset_index()
        .to_dict(orient="records")
    )
    issue_frame = pd.DataFrame(
        issues,
        columns=["patient_id", "code", "severity", "message"],
    )
    global_blockers = []
    if not data_cfg.get("mrd_endpoint", {}).get("assay_definition_finalized"):
        global_blockers.append("MRD assay and positivity rule are not finalized.")
    else:
        assay_definitions = first_mrd[
            ["assay_platform", "assay_version", "positivity_rule"]
        ].drop_duplicates()
        if len(assay_definitions) != 1:
            global_blockers.append(
                "Valid first MRD results use multiple assay definitions."
            )
    if not data_cfg.get("mrd_endpoint", {}).get("blood_draw_window_finalized"):
        global_blockers.append(
            "First postoperative blood-draw window is not finalized."
        )
    if analytical.empty:
        global_blockers.append("No patient passes the complete linked-data audit.")
    if missing_valid_mrd.any():
        global_blockers.append(
            "Not every otherwise eligible patient has a valid first MRD result."
        )
    development_value = data_cfg.get("cross_validation", {}).get(
        "development_value", "retrospective"
    )
    test_value = data_cfg.get("cross_validation", {}).get(
        "test_value", "prospective"
    )
    for value, label in (
        (development_value, "development"),
        (test_value, "temporal validation"),
    ):
        subset = analytical[analytical["cohort_period"].eq(value)]
        if subset.empty:
            global_blockers.append(f"No patients are available for {label}: {value}.")
        elif set(subset["mrd_status"]) != {0, 1}:
            global_blockers.append(
                f"The {label} cohort does not contain both MRD classes."
            )

    patient_blockers = (
        issue_frame.loc[issue_frame["severity"].eq("blocker"), "code"]
        .value_counts()
        .to_dict()
        if not issue_frame.empty
        else {}
    )
    report = {
        "ready_for_definitive_training": not global_blockers,
        "screened_patients": int(len(cohort)),
        "patients_with_valid_first_mrd": int(first_mrd["patient_id"].nunique()),
        "matched_analytical_patients": int(len(analytical)),
        "mrd_positive_patients": int(analytical["mrd_status"].sum()),
        "mrd_prevalence": (
            float(analytical["mrd_status"].mean())
            if len(analytical)
            else None
        ),
        "period_class_counts": period_counts,
        "stage_counts": (
            analytical["pathological_stage"]
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
        ),
        "tumor_site_counts": (
            analytical["tumor_side"]
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
        ),
        "scanner_counts": (
            analytical[["scanner_manufacturer", "scanner_model"]]
            .astype(str)
            .value_counts(dropna=False)
            .rename("patients")
            .reset_index()
            .to_dict(orient="records")
        ),
        "missingness": missingness,
        "missingness_by_mrd_class": missingness_by_mrd_class,
        "missingness_by_cohort_period": missingness_by_period,
        "patient_blocker_counts": patient_blockers,
        "blockers": global_blockers,
        "warnings": (
            issue_frame.loc[issue_frame["severity"].eq("warning"), "code"]
            .value_counts()
            .to_dict()
            if not issue_frame.empty
            else {}
        ),
        "first_attempt_invalid_or_indeterminate": int(
            first_attempt["mrd_result"].isin({"invalid", "indeterminate"}).sum()
        ),
        "binding_constraint": (
            "eligible ∩ CT-complete ∩ pathology-complete ∩ valid first MRD"
        ),
    }
    return AuditResult(analytical, report, issue_frame)


def save_audit_result(
    result: AuditResult,
    clinical_output: str | Path,
    report_output: str | Path,
    issues_output: str | Path,
) -> None:
    """Persist the model table, readiness JSON, and patient-level issue log."""
    clinical_path = Path(clinical_output)
    report_path = Path(report_output)
    issues_path = Path(issues_output)
    clinical_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    result.analytical_cohort.to_csv(clinical_path, index=False)
    report_path.write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result.issues.to_csv(issues_path, index=False)
    logger.info("Saved analytical cohort to %s", clinical_path)
