"""Build a canonical clinical table from the cohort workbook export."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PATIENT_ID_SOURCE = "患者编号"

SHEET_COLUMN_MAP: Dict[str, Dict[str, str]] = {
    "基本信息": {
        PATIENT_ID_SOURCE: "patient_id",
        "性别": "sex",
        "年龄": "age",
    },
    "术前检验": {
        PATIENT_ID_SOURCE: "patient_id",
        "癌胚抗原(CEA)": "cea_preoperative",
        "癌胚抗原(CEA)(单位)": "cea_preoperative_unit",
        "CA19-9(糖链抗原19-9)": "ca199_preoperative",
        "CA19-9(糖链抗原19-9)(单位)": "ca199_preoperative_unit",
    },
    "入组前治疗情况": {
        PATIENT_ID_SOURCE: "patient_id",
        "是否新辅/诱导": "neoadjuvant_treatment",
        "新辅/诱导药物方案": "neoadjuvant_regimen",
        "新辅/诱导周期": "neoadjuvant_cycles",
        "既往是否有放疗": "preoperative_radiotherapy",
    },
    "手术情况": {
        PATIENT_ID_SOURCE: "patient_id",
        "手术时间": "surgery_date",
        "手术名称（原发灶手术）": "surgery_procedure",
        "手术名称（转移灶手术）": "metastatic_surgery_procedure",
        "手术R0切除情况（原发灶手术）": "resection_margin",
    },
    "患者病理与临床信息": {
        PATIENT_ID_SOURCE: "patient_id",
        "T分期": "pathological_t",
        "N分期": "pathological_n",
        "M分期": "pathological_m",
        "分期": "pathological_stage",
        "临床诊断": "clinical_diagnosis",
        "原发部位": "tumor_location",
        "位置": "tumor_side",
        "组织学分型": "histological_type",
        "分化程度": "histological_grade",
        "原发肿瘤直径/mm": "tumor_dimensions_pathological",
        "MLH1": "mlh1_status",
        "PMS2": "pms2_status",
        "MSH2": "msh2_status",
        "MSH6": "msh6_status",
        "HER2-G": "her2_status",
        "BRAF": "braf_status",
        "Ki-67": "ki67_percent",
        "淋巴结浸润 （LV_invasion）": "lymphovascular_invasion",
        "淋巴结总量": "lymph_node_summary",
        "神经浸润 （N_invasion）": "perineural_invasion",
        "MMR缺陷": "mmr_status",
        "MSI status": "msi_status",
        "KPS评分": "kps_score",
    },
}

MODEL_FEATURE_COLUMNS = [
    "age",
    "sex",
    "cea_preoperative",
    "ca199_preoperative",
    "neoadjuvant_treatment",
    "preoperative_radiotherapy",
    "pathological_t",
    "pathological_n",
    "pathological_m",
    "pathological_stage",
    "tumor_location",
    "tumor_side",
    "tumor_size_pathological_mm",
    "histological_type",
    "histological_grade",
    "lymphovascular_invasion",
    "perineural_invasion",
    "resection_margin",
    "examined_lymph_nodes",
    "positive_lymph_nodes",
    "surgery_procedure",
    "metastatic_surgery",
    "mlh1_status",
    "pms2_status",
    "msh2_status",
    "msh6_status",
    "her2_status",
    "braf_status",
    "ki67_percent",
]

_MISSING_STRINGS = {"", "ND", "NA", "N/A", "NULL", "NONE", "未知", "未检测"}


def _clean_missing(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    return np.nan if text.upper() in _MISSING_STRINGS else text


def _numeric(value: Any) -> float:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def _maximum_dimension_mm(value: Any) -> float:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan
    dimensions = [
        float(number)
        for number in re.findall(r"\d+(?:\.\d+)?", str(value))
    ]
    return max(dimensions) if dimensions else np.nan


def _lymph_node_counts(value: Any) -> tuple[float, float]:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan, np.nan
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if len(numbers) < 2:
        return np.nan, np.nan
    return float(numbers[0]), float(numbers[1])


def _binary_chinese(value: Any) -> Any:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan
    normalized = str(value).strip()
    mapping = {"是": "yes", "否": "no", "有": "yes", "无": "no"}
    return mapping.get(normalized, normalized)


def _ihc_status(value: Any) -> Any:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan
    normalized = str(value).strip().replace("＋", "+").replace("－", "-")
    if normalized in {"+", "阳性"}:
        return "positive"
    if normalized in {"-", "阴性"}:
        return "negative"
    return normalized


def _sex(value: Any) -> Any:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan
    return {"男": "male", "女": "female"}.get(str(value).strip(), value)


def _primary_surgery(value: Any) -> Any:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan
    return re.split(r"[,，]|其他#", str(value).strip(), maxsplit=1)[0].strip()


def _metastatic_surgery(value: Any) -> Any:
    value = _clean_missing(value)
    if pd.isna(value):
        return np.nan
    return "no" if str(value).strip() in {"无", "否"} else "yes"


def _patient_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _read_sheet(workbook: Path, sheet_name: str) -> pd.DataFrame:
    mapping = SHEET_COLUMN_MAP[sheet_name]
    frame = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        usecols=lambda column: str(column).strip() in mapping,
        dtype=object,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = sorted(set(mapping) - set(frame.columns))
    if missing:
        raise ValueError(f"Sheet '{sheet_name}' is missing columns: {missing}")
    frame = frame.rename(columns=mapping)
    frame["patient_id"] = frame["patient_id"].map(_patient_id)
    frame = frame[frame["patient_id"].ne("") & frame["patient_id"].ne("nan")]
    if frame["patient_id"].duplicated().any():
        duplicates = frame.loc[
            frame["patient_id"].duplicated(), "patient_id"
        ].tolist()
        raise ValueError(
            f"Sheet '{sheet_name}' contains duplicate patient IDs: {duplicates[:10]}"
        )
    return frame


def build_baseline_clinical_table(workbook_path: str | Path) -> pd.DataFrame:
    """Merge baseline sheets and normalize spreadsheet-specific encodings.

    Personally identifying fields and all follow-up/outcome sheets are
    intentionally excluded.
    """
    workbook = Path(workbook_path)
    if not workbook.exists():
        raise FileNotFoundError(f"Clinical workbook not found: {workbook}")

    frames = {
        sheet_name: _read_sheet(workbook, sheet_name)
        for sheet_name in SHEET_COLUMN_MAP
    }
    cea_units = sorted(
        {
            str(value).strip()
            for value in frames["术前检验"]["cea_preoperative_unit"].dropna()
        }
    )
    if cea_units and cea_units != ["ng/mL"]:
        logger.warning(
            "CEA units require source verification before cross-cohort use: %s",
            cea_units,
        )
    clinical = frames["基本信息"]
    for sheet_name in SHEET_COLUMN_MAP:
        if sheet_name == "基本信息":
            continue
        clinical = clinical.merge(
            frames[sheet_name], on="patient_id", how="left", validate="one_to_one"
        )

    for column in ("age", "cea_preoperative", "ca199_preoperative", "kps_score"):
        clinical[column] = clinical[column].map(_numeric)
    clinical["neoadjuvant_cycles"] = clinical["neoadjuvant_cycles"].map(_numeric)
    clinical["ki67_percent"] = clinical["ki67_percent"].map(_numeric)
    clinical["tumor_size_pathological_mm"] = clinical[
        "tumor_dimensions_pathological"
    ].map(_maximum_dimension_mm)

    node_counts = clinical["lymph_node_summary"].map(_lymph_node_counts)
    clinical["positive_lymph_nodes"] = node_counts.map(lambda pair: pair[0])
    clinical["examined_lymph_nodes"] = node_counts.map(lambda pair: pair[1])

    clinical["sex"] = clinical["sex"].map(_sex)
    clinical["surgery_procedure"] = clinical["surgery_procedure"].map(
        _primary_surgery
    )
    clinical["metastatic_surgery"] = clinical[
        "metastatic_surgery_procedure"
    ].map(_metastatic_surgery)
    for column in (
        "neoadjuvant_treatment",
        "preoperative_radiotherapy",
        "lymphovascular_invasion",
        "perineural_invasion",
    ):
        clinical[column] = clinical[column].map(_binary_chinese)
    for column in (
        "mlh1_status",
        "pms2_status",
        "msh2_status",
        "msh6_status",
        "braf_status",
    ):
        clinical[column] = clinical[column].map(_ihc_status)

    clinical["surgery_date"] = pd.to_datetime(
        clinical["surgery_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    ordered = ["patient_id", *MODEL_FEATURE_COLUMNS]
    audit_columns = [
        column
        for column in clinical.columns
        if column not in ordered
    ]
    return clinical[ordered + audit_columns].sort_values(
        "patient_id"
    ).reset_index(drop=True)


def attach_mrd_labels(
    clinical: pd.DataFrame,
    labels: pd.DataFrame,
    label_id_column: str = "patient_id",
    label_column: str = "mrd_status",
) -> pd.DataFrame:
    """Attach independently sourced binary MRD labels to a baseline table."""
    required = {label_id_column, label_column}
    if not required.issubset(labels.columns):
        missing = sorted(required - set(labels))
        raise ValueError(f"Label table is missing columns: {missing}")
    label_table = labels[[label_id_column, label_column]].copy()
    label_table = label_table.rename(columns={label_id_column: "patient_id"})
    label_table["patient_id"] = label_table["patient_id"].map(_patient_id)
    if label_table["patient_id"].duplicated().any():
        raise ValueError("MRD label table contains duplicate patient IDs.")
    label_table[label_column] = pd.to_numeric(
        label_table[label_column], errors="coerce"
    )
    if (
        label_table[label_column].isna().any()
        or not set(label_table[label_column].unique()).issubset({0, 1})
    ):
        raise ValueError(f"'{label_column}' must contain only binary 0/1 labels.")
    merged = clinical.merge(
        label_table, on="patient_id", how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("No patient IDs overlap between baseline and MRD labels.")
    merged[label_column] = merged[label_column].astype(int)
    return merged
