#!/usr/bin/env python3
"""Prepare a de-identified patient-level sheet from raw MRD CSV exports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPORT_FILES = {
    "demographics": "基线数据.00-0 基线-姓名、电话、性别、身份证_gbk.csv",
    "clinical_t": "基线数据.01-1-T_分期小结_gbk.csv",
    "clinical_n": "基线数据.01-2-N_分期小结_gbk.csv",
    "pathological_tnm": "基线数据.01-3-ptnm_分期小结_gbk.csv",
    "surgery": "基线数据.05-1-手术名称_手术_gbk.csv",
    "preop_last_blood": "基线数据.11-2-术前末次抽血_gbk.csv",
    "first_blood": "基线数据.11-1-首诊抽血_gbk.csv",
    "bowel_ct": "基线数据.02-2-肠CT_CT检查_gbk.csv",
    "lung_ct": "基线数据.02-5-肺CT_CT检查_gbk.csv",
    "preop_last_exam": "基线数据.11-4-术前末次检查_gbk.csv",
    "first_exam": "基线数据.11-3-首次检查_gbk.csv",
    "surgical_pathology": "基线数据.11-6-手术病理_gbk.csv",
}

SOURCE_INPATIENT_COLUMN = "00-0 基线-姓名、电话、性别、身份证_inpatient_no_id"

MODEL_COLUMNS = [
    "age",
    "sex",
    "ecog",
    "cea_preoperative",
    "ca199_preoperative",
    "clinical_t",
    "clinical_n",
    "clinical_m",
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
    "tumor_deposits",
    "bowel_obstruction",
    "tumor_perforation",
    "resection_margin",
    "examined_lymph_nodes",
    "positive_lymph_nodes",
    "surgery_procedure",
    "her2_status",
    "braf_status",
    "ki67_percent",
]

RAW_SHEET_NON_QC_COLUMNS = {
    "age",
    "ecog",
}

MISSING_STRINGS = {"", "NA", "N/A", "NULL", "NONE", "NAN", "未知", "未检测"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/raw/mrd_data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--excel-output",
        default="mrd_data_consolidated.xlsx",
        help="Excel workbook name written under --output-dir.",
    )
    return parser.parse_args()


def read_export(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / EXPORT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing export for {name}: {path}")
    frame = pd.read_csv(path, encoding="gbk", header=2, dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].map(clean_value)
    frame["patient_id"] = frame["patient_sn"].map(normalize_id)
    frame = frame[frame["patient_id"].notna() & frame["patient_id"].ne("")]
    return frame


def clean_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return pd.NA if text.upper() in MISSING_STRINGS else text


def normalize_id(value: Any) -> str:
    value = clean_value(value)
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_number(value: Any) -> float:
    value = clean_value(value)
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def iso_date(value: Any) -> Any:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NA
    return parsed.strftime("%Y-%m-%d")


def iso_datetime(value: Any) -> Any:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NA
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def first_nonmissing(values: pd.Series) -> Any:
    values = values.dropna()
    return values.iloc[0] if len(values) else pd.NA


def last_nonmissing(values: pd.Series) -> Any:
    values = values.dropna()
    return values.iloc[-1] if len(values) else pd.NA


def distinct_count(values: pd.Series) -> int:
    values = values.dropna().astype(str).str.strip()
    return int(values[values.ne("")].nunique())


def one_row_by_patient(
    frame: pd.DataFrame,
    columns: list[str],
    prefer: str = "last",
    date_column: str | None = None,
) -> pd.DataFrame:
    work = frame[["patient_id", *columns]].copy()
    if date_column:
        work["_parsed_date"] = pd.to_datetime(work[date_column], errors="coerce")
        work = work.sort_values(["patient_id", "_parsed_date"], na_position="last")
    aggregations = {
        column: (last_nonmissing if prefer == "last" else first_nonmissing)
        for column in columns
    }
    return work.groupby("patient_id", as_index=False).agg(aggregations)


def add_distinct_count(
    output: pd.DataFrame,
    source: pd.DataFrame,
    value_column: str,
    count_column: str,
) -> pd.DataFrame:
    counts = (
        source.groupby("patient_id")[value_column]
        .agg(distinct_count)
        .rename(count_column)
        .reset_index()
    )
    return output.merge(counts, on="patient_id", how="left")


def extract_marker(row: pd.Series, marker: str) -> tuple[Any, Any]:
    marker_upper = marker.upper()
    for item_column in row.index:
        if "item_name" not in item_column:
            continue
        item = row.get(item_column)
        if pd.isna(item) or marker_upper not in str(item).upper():
            continue
        prefix, suffix = item_column.split("_item_name", maxsplit=1)
        time_column = f"{prefix}_test_time{suffix}"
        result_column = f"{prefix}_test_result{suffix}"
        return row.get(result_column, pd.NA), row.get(time_column, pd.NA)
    return pd.NA, pd.NA


def extract_blood_markers(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        cea, cea_date = extract_marker(row, "CEA")
        ca199, ca199_date = extract_marker(row, "CA199")
        rows.append(
            {
                "patient_id": row["patient_id"],
                f"cea_{source_name}": to_number(cea),
                f"cea_date_{source_name}": iso_date(cea_date),
                f"ca199_{source_name}": to_number(ca199),
                f"ca199_date_{source_name}": iso_date(ca199_date),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("patient_id", keep="last")


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return (
        str(value)
        .replace("\u3000", " ")
        .replace("（", "(")
        .replace("）", ")")
        .replace("×", "x")
        .replace("；", ";")
        .strip()
    )


def section_value(text: str, label: str) -> str | None:
    match = re.search(rf"{label}[:：]\s*([^【\n。;]+)", text)
    return match.group(1).strip() if match else None


def yes_no_from_text(text: str, label: str) -> Any:
    value = section_value(text, label)
    if not value:
        return pd.NA
    if "未见" in value or "无" in value or "否" in value:
        return "no"
    if "可见" in value or "有" in value or "是" in value:
        return "yes"
    return pd.NA


def parse_size_mm(text: str) -> float:
    value = section_value(text, "最大径")
    if not value:
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)\s*(cm|mm)?", value, flags=re.I)
    if not match:
        return np.nan
    size = float(match.group(1))
    unit = (match.group(2) or "cm").lower()
    return size if unit == "mm" else size * 10.0


def parse_tnm(text: str) -> tuple[Any, Any, Any]:
    match = re.search(
        r"((?:y)?pT[0-4xX][a-c]?)(N[0-2xX][a-c]?)(M[0-1xX][a-c]?)?",
        text,
        flags=re.I,
    )
    if not match:
        return pd.NA, pd.NA, pd.NA
    pathological_t = match.group(1)
    pathological_n = f"p{match.group(2)}"
    pathological_m = f"p{match.group(3)}" if match.group(3) else pd.NA
    return pathological_t, pathological_n, pathological_m


def derive_stage(pathological_t: Any, pathological_n: Any, pathological_m: Any) -> Any:
    t_text = normalize_text(pathological_t).lower().replace("yp", "p")
    n_text = normalize_text(pathological_n).lower().replace("yp", "p")
    m_text = normalize_text(pathological_m).lower().replace("yp", "p")
    if m_text.startswith("pm1"):
        return "IV"
    if re.match(r"pn[12]", n_text):
        return "III"
    if n_text.startswith("pn0"):
        if re.match(r"pt[34]", t_text):
            return "II"
        if re.match(r"pt[12]", t_text):
            return "I"
    return pd.NA


def parse_grade(text: str) -> Any:
    for grade in ("中至低分化", "中低分化", "低分化", "中分化", "高分化", "低级别", "高级别"):
        if grade in text:
            return grade
    return pd.NA


def parse_ihc(text: str, marker: str) -> Any:
    match = re.search(rf"{marker}(?:-G)?\s*\(([^)]+)\)", text, flags=re.I)
    return match.group(1).strip() if match else pd.NA


def normalize_ihc_status(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().replace("＋", "+").replace("－", "-")
    if "+" in text or "阳" in text:
        return "positive"
    if "-" in text or "阴" in text:
        return "negative"
    return text


def parse_ki67(text: str) -> float:
    match = re.search(r"Ki-?67[^。;]*?(\d+(?:\.\d+)?)\s*%", text, flags=re.I)
    return float(match.group(1)) if match else np.nan


def parse_lymph_nodes(text: str) -> tuple[float, float]:
    match = re.search(r"转移个数/总个数[:：]\s*(\d+)\s*/\s*(\d+)", text)
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def parse_margin(text: str) -> Any:
    margin_match = re.search(r"【切缘】(.+?)(?:【|$)", text)
    margin_text = margin_match.group(1) if margin_match else text
    if "切缘" not in margin_text:
        return pd.NA
    if "阳性" in margin_text or "累及" in margin_text and "无浸润" not in margin_text:
        return "positive"
    if "未见癌" in margin_text or "无浸润" in margin_text or "阴性" in margin_text:
        return "negative"
    return pd.NA


def parse_location(text: str, surgery_name: Any) -> tuple[Any, Any]:
    value = section_value(text, "肿瘤部位")
    combined = f"{value or ''} {normalize_text(surgery_name)} {text[:120]}"
    location_patterns = [
        ("直肠", "rectum", "rectum"),
        ("乙状", "sigmoid_colon", "left_colon"),
        ("降结肠", "descending_colon", "left_colon"),
        ("左半结肠", "left_colon", "left_colon"),
        ("横结肠", "transverse_colon", "transverse_colon"),
        ("升结肠", "ascending_colon", "right_colon"),
        ("盲肠", "cecum", "right_colon"),
        ("右半结肠", "right_colon", "right_colon"),
    ]
    for pattern, location, side in location_patterns:
        if pattern in combined:
            return location, side
    return pd.NA, pd.NA


def parse_surgery_flag(text: str, label: str) -> Any:
    if not text or label not in text:
        return pd.NA
    match = re.search(rf"{label}[^。;，,]*[:：]?\s*([^。;，,]*)", text)
    value = match.group(1) if match else ""
    if "无" in value or "否" in value:
        return "no"
    if "有" in value or value.strip():
        return "yes"
    return pd.NA


def parse_pathology_row(row: pd.Series) -> dict[str, Any]:
    text = normalize_text(row.get("imaging_conclusion"))
    gross = normalize_text(row.get("gross_finding"))
    surgery_description = normalize_text(row.get("surgery_description"))
    surgical_name = row.get("surgery_procedure")
    pathological_t, pathological_n, pathological_m = parse_tnm(text)
    positive_nodes, examined_nodes = parse_lymph_nodes(text)
    location, side = parse_location(text or gross, surgical_name)
    histology = section_value(text, "组织学类型")
    if histology:
        histology = re.split(r"[;；]", histology)[0].strip()
    return {
        "pathology_exam_date": iso_date(row.get("exam_date")),
        "pathological_t": pathological_t,
        "pathological_n": pathological_n,
        "pathological_m": pathological_m,
        "pathological_stage": derive_stage(pathological_t, pathological_n, pathological_m),
        "tumor_location": location,
        "tumor_side": side,
        "tumor_size_pathological_mm": parse_size_mm(text),
        "histological_type": histology or pd.NA,
        "histological_grade": parse_grade(text),
        "lymphovascular_invasion": yes_no_from_text(text, "脉管癌栓"),
        "perineural_invasion": yes_no_from_text(text, "神经侵犯"),
        "tumor_deposits": yes_no_from_text(text, "是否发现癌结节"),
        "bowel_obstruction": parse_surgery_flag(surgery_description, "肠梗阻"),
        "tumor_perforation": parse_surgery_flag(surgery_description, "肠穿孔"),
        "resection_margin": parse_margin(text),
        "positive_lymph_nodes": positive_nodes,
        "examined_lymph_nodes": examined_nodes,
        "mlh1_status": normalize_ihc_status(parse_ihc(text, "MLH1")),
        "pms2_status": normalize_ihc_status(parse_ihc(text, "PMS2")),
        "msh2_status": normalize_ihc_status(parse_ihc(text, "MSH2")),
        "msh6_status": normalize_ihc_status(parse_ihc(text, "MSH6")),
        "her2_status": parse_ihc(text, "HER2"),
        "braf_status": normalize_ihc_status(parse_ihc(text, "BRAF")),
        "ki67_percent": parse_ki67(text),
        "mmr_status": section_value(text, "MMR状态") or pd.NA,
        "pathology_text_parsed": bool(text),
    }


def build_source_inventory(exports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in exports.items():
        rows.append(
            {
                "source": name,
                "filename": EXPORT_FILES[name],
                "rows": int(len(frame)),
                "patients": int(frame["patient_id"].nunique()),
                "duplicate_patient_rows": int(frame["patient_id"].duplicated().sum()),
                "columns": int(len(frame.columns)),
            }
        )
    return pd.DataFrame(rows)


def build_surgery_summary(exports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    surgery = exports["surgery"].copy()
    surgery["_surgery_datetime"] = pd.to_datetime(
        surgery["surgery_start_datetime"], errors="coerce"
    )
    surgery = surgery.sort_values(["patient_id", "_surgery_datetime"])
    summary = (
        surgery.groupby("patient_id", as_index=False)
        .agg(
            surgery_start_datetime=("surgery_start_datetime", first_nonmissing),
            surgery_date=("surgery_start_datetime", first_nonmissing),
            surgery_procedure=("surgical_name", first_nonmissing),
            surgery_description=("description", first_nonmissing),
        )
    )
    summary["surgery_start_datetime"] = summary["surgery_start_datetime"].map(
        iso_datetime
    )
    summary["surgery_date"] = summary["surgery_date"].map(iso_date)
    return summary


def _append_ct_record(
    records: list[dict[str, Any]],
    row: pd.Series,
    source: str,
    ct_type: str,
    visit_column: str,
    date_column: str,
    exam_id_column: str,
) -> None:
    ct_datetime = row.get(date_column)
    exam_id = row.get(exam_id_column)
    visit_sn = row.get(visit_column)
    if pd.isna(ct_datetime) and pd.isna(exam_id) and pd.isna(visit_sn):
        return
    records.append(
        {
            "patient_id": row["patient_id"],
            "inpatient_no_id": row.get(SOURCE_INPATIENT_COLUMN, pd.NA),
            "ct_source": source,
            "ct_type": ct_type,
            "ct_visit_sn": visit_sn,
            "ct_exam_datetime": iso_datetime(ct_datetime),
            "ct_exam_date": iso_date(ct_datetime),
            "ct_exam_id": exam_id,
        }
    )


def build_ct_records(exports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in exports["bowel_ct"].iterrows():
        _append_ct_record(
            records,
            row,
            "bowel_ct",
            "bowel_ct",
            "vsn_CT",
            "exam_date",
            "exam_id",
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "inpatient_no_id",
                "ct_source",
                "ct_type",
                "ct_visit_sn",
                "ct_exam_datetime",
                "ct_exam_date",
                "ct_exam_id",
            ]
        )
    frame = pd.DataFrame(records)
    frame = frame[frame["ct_exam_datetime"].notna() | frame["ct_exam_id"].notna()]
    frame = frame.drop_duplicates(
        ["patient_id", "ct_exam_datetime", "ct_exam_id", "ct_visit_sn"],
        keep="first",
    )
    frame = frame.sort_values(["patient_id", "ct_exam_datetime", "ct_exam_id"])
    return frame.reset_index(drop=True)


def build_ct_lookup(
    exports: dict[str, pd.DataFrame],
    surgery_summary: pd.DataFrame,
    ct_records: pd.DataFrame,
) -> pd.DataFrame:
    demographics = exports["demographics"][
        ["patient_id", "inpatient_no_id", SOURCE_INPATIENT_COLUMN]
    ].drop_duplicates("patient_id", keep="last")
    demographics["inpatient_no_id"] = demographics["inpatient_no_id"].combine_first(
        demographics[SOURCE_INPATIENT_COLUMN]
    )
    demographics = demographics.drop(columns=[SOURCE_INPATIENT_COLUMN])
    lookup = demographics.merge(
        surgery_summary[["patient_id", "surgery_start_datetime", "surgery_date"]],
        on="patient_id",
        how="left",
    )
    all_counts = (
        ct_records.groupby("patient_id")
        .size()
        .rename("all_ct_records_count")
        .reset_index()
    )
    lookup = lookup.merge(all_counts, on="patient_id", how="left")
    if ct_records.empty:
        lookup["ct_records_before_surgery_count"] = 0
        lookup["ct_exam_datetime_last_before_surgery"] = pd.NA
        lookup["ct_exam_date_last_before_surgery"] = pd.NA
        lookup["ct_exam_id_last_before_surgery"] = pd.NA
        lookup["ct_visit_sn_last_before_surgery"] = pd.NA
        lookup["ct_source_last_before_surgery"] = pd.NA
        lookup["ct_type_last_before_surgery"] = pd.NA
        lookup["ct_days_before_surgery"] = pd.NA
        return lookup

    timed = ct_records.merge(
        surgery_summary[["patient_id", "surgery_start_datetime"]],
        on="patient_id",
        how="left",
    )
    timed["_ct_datetime"] = pd.to_datetime(
        timed["ct_exam_datetime"], errors="coerce"
    )
    timed["_surgery_datetime"] = pd.to_datetime(
        timed["surgery_start_datetime"], errors="coerce"
    )
    before_surgery = timed[
        timed["_ct_datetime"].notna()
        & timed["_surgery_datetime"].notna()
        & timed["_ct_datetime"].lt(timed["_surgery_datetime"])
    ].copy()
    before_counts = (
        before_surgery.groupby("patient_id")
        .size()
        .rename("ct_records_before_surgery_count")
        .reset_index()
    )
    lookup = lookup.merge(before_counts, on="patient_id", how="left")
    if before_surgery.empty:
        last_ct = pd.DataFrame(columns=["patient_id"])
    else:
        before_surgery = before_surgery.sort_values(
            ["patient_id", "_ct_datetime", "ct_exam_id"]
        )
        last_ct = before_surgery.drop_duplicates("patient_id", keep="last").copy()
        last_ct["ct_days_before_surgery"] = (
            (
                last_ct["_surgery_datetime"] - last_ct["_ct_datetime"]
            ).dt.total_seconds()
            / 86400.0
        ).round(3)
        last_ct = last_ct[
            [
                "patient_id",
                "ct_exam_datetime",
                "ct_exam_date",
                "ct_exam_id",
                "ct_visit_sn",
                "ct_source",
                "ct_type",
                "ct_days_before_surgery",
            ]
        ].rename(
            columns={
                "ct_exam_datetime": "ct_exam_datetime_last_before_surgery",
                "ct_exam_date": "ct_exam_date_last_before_surgery",
                "ct_exam_id": "ct_exam_id_last_before_surgery",
                "ct_visit_sn": "ct_visit_sn_last_before_surgery",
                "ct_source": "ct_source_last_before_surgery",
                "ct_type": "ct_type_last_before_surgery",
            }
        )
    lookup = lookup.merge(last_ct, on="patient_id", how="left")
    for column in (
        "ct_exam_datetime_last_before_surgery",
        "ct_exam_date_last_before_surgery",
        "ct_exam_id_last_before_surgery",
        "ct_visit_sn_last_before_surgery",
        "ct_source_last_before_surgery",
        "ct_type_last_before_surgery",
        "ct_days_before_surgery",
    ):
        if column not in lookup:
            lookup[column] = pd.NA
    lookup["all_ct_records_count"] = lookup["all_ct_records_count"].fillna(0).astype(int)
    lookup["ct_records_before_surgery_count"] = (
        lookup["ct_records_before_surgery_count"].fillna(0).astype(int)
    )
    return lookup.sort_values("patient_id").reset_index(drop=True)


def add_issue(
    issues: list[dict[str, Any]],
    patient_id: Any,
    code: str,
    severity: str,
    message: str,
    column: str | None = None,
) -> None:
    issues.append(
        {
            "patient_id": patient_id,
            "code": code,
            "severity": severity,
            "column": column,
            "message": message,
        }
    )


def apply_qc(sheet: pd.DataFrame, inventory: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if sheet["patient_id"].duplicated().any():
        for patient_id in sheet.loc[sheet["patient_id"].duplicated(), "patient_id"]:
            add_issue(issues, patient_id, "duplicate_patient_id", "blocker", "Duplicate output patient.")

    invalid_sex = ~sheet["sex"].isin(["male", "female"]) & sheet["sex"].notna()
    for patient_id in sheet.loc[invalid_sex, "patient_id"]:
        add_issue(issues, patient_id, "invalid_sex", "warning", "Sex is not male/female.", "sex")

    node_error = (
        sheet["positive_lymph_nodes"].notna()
        & sheet["examined_lymph_nodes"].notna()
        & (sheet["positive_lymph_nodes"] > sheet["examined_lymph_nodes"])
    )
    for patient_id in sheet.loc[node_error, "patient_id"]:
        add_issue(
            issues,
            patient_id,
            "invalid_lymph_node_counts",
            "blocker",
            "Positive lymph nodes exceed examined lymph nodes.",
            "positive_lymph_nodes",
        )

    stage_discordance = []
    for _, row in sheet.iterrows():
        stage = normalize_text(row["pathological_stage"]).upper()
        n_text = normalize_text(row["pathological_n"]).upper()
        m_text = normalize_text(row["pathological_m"]).upper()
        if stage == "II" and not n_text.startswith("PN0"):
            stage_discordance.append(row["patient_id"])
        elif stage == "III" and n_text.startswith("PN0"):
            stage_discordance.append(row["patient_id"])
        elif stage in {"I", "II", "III"} and "PM1" in m_text:
            stage_discordance.append(row["patient_id"])
    for patient_id in stage_discordance:
        add_issue(
            issues,
            patient_id,
            "stage_tnm_discordance",
            "blocker",
            "Derived pathological stage conflicts with TNM components.",
            "pathological_stage",
        )

    for column in MODEL_COLUMNS:
        if column in RAW_SHEET_NON_QC_COLUMNS:
            continue
        if column not in sheet:
            continue
        missing = sheet[column].isna()
        for patient_id in sheet.loc[missing, "patient_id"]:
            add_issue(
                issues,
                patient_id,
                "missing_model_feature",
                "warning",
                f"Model feature '{column}' is missing.",
                column,
            )

    for column in ("clinical_t_distinct_count", "clinical_n_distinct_count", "pathological_tnm_distinct_count"):
        if column not in sheet:
            continue
        conflicts = pd.to_numeric(sheet[column], errors="coerce").fillna(0).gt(1)
        for patient_id in sheet.loc[conflicts, "patient_id"]:
            add_issue(
                issues,
                patient_id,
                "conflicting_source_values",
                "warning",
                f"Source export contains multiple distinct values for {column}.",
                column,
            )

    for patient_id in sheet["patient_id"]:
        add_issue(
            issues,
            patient_id,
            "missing_mrd_label",
            "blocker",
            "No MRD result column keyed by patient_sn was found in the new exports.",
            "mrd_status",
        )

    issue_frame = pd.DataFrame(
        issues,
        columns=["patient_id", "code", "severity", "column", "message"],
    )
    missingness = {
        column: float(sheet[column].isna().mean())
        for column in MODEL_COLUMNS
        if column in sheet
    }
    issue_counts = (
        issue_frame.groupby(["severity", "code"]).size().reset_index(name="count")
        if not issue_frame.empty
        else pd.DataFrame(columns=["severity", "code", "count"])
    )
    summary = {
        "patients": int(len(sheet)),
        "model_features": int(len(MODEL_COLUMNS)),
        "qc_scope": (
            "Raw-source consolidation QC only. Protocol eligibility criteria "
            "such as age range and ECOG are not applied here."
        ),
        "non_qc_columns": sorted(RAW_SHEET_NON_QC_COLUMNS),
        "features_complete_for_all_patients": [
            column for column, rate in missingness.items() if rate == 0.0
        ],
        "patients_with_preop_last_blood": int(sheet["has_preop_last_blood"].sum()),
        "patients_with_surgical_pathology": int(sheet["has_surgical_pathology"].sum()),
        "patients_with_preop_last_exam": int(sheet["has_preop_last_exam"].sum()),
        "patients_with_ct_before_surgery": int(
            sheet["ct_exam_date_last_before_surgery"].notna().sum()
        ),
        "mrd_label_available": False,
        "direct_identifier_columns_in_output": sorted(
            set(sheet.columns)
            & {
                "name",
                "telephone",
                "contact_name",
                "contact_telephone",
                "id_card_no",
                "inpatient_no_id",
            }
        ),
        "source_inventory": inventory.to_dict(orient="records"),
        "missingness": missingness,
        "issue_counts": issue_counts.to_dict(orient="records"),
        "blockers": [
            "MRD labels are not linked to patient_sn.",
            "CT image files/phase manifest are not available in these CSV exports.",
        ],
    }
    return issue_frame, summary


def build_sheet(
    exports: dict[str, pd.DataFrame],
    surgery_summary: pd.DataFrame,
    ct_lookup: pd.DataFrame,
) -> pd.DataFrame:
    demographics = exports["demographics"]
    sheet = demographics[
        [
            "patient_id",
            "group_name",
            "gender",
            "age",
        ]
    ].drop_duplicates("patient_id", keep="last")
    sheet = sheet.rename(columns={"group_name": "source_group_name"})
    sheet["sex"] = sheet["gender"].map({"男": "male", "女": "female"})
    sheet["age"] = sheet["age"].map(to_number)
    sheet = sheet.drop(columns=["gender"])
    sheet["mrd_status"] = pd.NA
    sheet["cohort_period"] = pd.NA
    sheet["ecog"] = pd.NA
    sheet["clinical_m"] = pd.NA

    clinical_t = one_row_by_patient(exports["clinical_t"], ["c_t_stage"])
    clinical_t = clinical_t.rename(columns={"c_t_stage": "clinical_t"})
    clinical_t = add_distinct_count(
        clinical_t, exports["clinical_t"], "c_t_stage", "clinical_t_distinct_count"
    )
    sheet = sheet.merge(clinical_t, on="patient_id", how="left")

    clinical_n = one_row_by_patient(exports["clinical_n"], ["c_n_stage"])
    clinical_n = clinical_n.rename(columns={"c_n_stage": "clinical_n"})
    clinical_n = add_distinct_count(
        clinical_n, exports["clinical_n"], "c_n_stage", "clinical_n_distinct_count"
    )
    sheet = sheet.merge(clinical_n, on="patient_id", how="left")

    ptnm = one_row_by_patient(
        exports["pathological_tnm"],
        ["p_t_stage", "p_n_stage", "p_m_stage"],
    ).rename(
        columns={
            "p_t_stage": "pathological_t_from_stage_summary",
            "p_n_stage": "pathological_n_from_stage_summary",
            "p_m_stage": "pathological_m_from_stage_summary",
        }
    )
    tnm_combo = exports["pathological_tnm"].assign(
        tnm_combo=lambda frame: (
            frame["p_t_stage"].fillna("")
            + "|"
            + frame["p_n_stage"].fillna("")
            + "|"
            + frame["p_m_stage"].fillna("")
        )
    )
    ptnm = add_distinct_count(
        ptnm, tnm_combo, "tnm_combo", "pathological_tnm_distinct_count"
    )
    sheet = sheet.merge(ptnm, on="patient_id", how="left")

    sheet = sheet.merge(surgery_summary, on="patient_id", how="left")

    last_blood = extract_blood_markers(exports["preop_last_blood"], "last")
    first_blood = extract_blood_markers(exports["first_blood"], "first")
    sheet = sheet.merge(last_blood, on="patient_id", how="left").merge(
        first_blood, on="patient_id", how="left"
    )
    sheet["cea_preoperative"] = sheet["cea_last"].combine_first(sheet["cea_first"])
    sheet["cea_date"] = sheet["cea_date_last"].combine_first(sheet["cea_date_first"])
    sheet["ca199_preoperative"] = sheet["ca199_last"].combine_first(sheet["ca199_first"])
    sheet["ca199_date"] = sheet["ca199_date_last"].combine_first(sheet["ca199_date_first"])
    sheet["cea_source"] = np.where(sheet["cea_last"].notna(), "preop_last_blood", "first_blood")
    sheet["ca199_source"] = np.where(sheet["ca199_last"].notna(), "preop_last_blood", "first_blood")

    exam = exports["preop_last_exam"].copy()
    exam_summary = pd.DataFrame(
        {
            "patient_id": exam["patient_id"],
            "mri_exam_date": exam["exam_date"],
            "colonoscopy_date": exam["exam_date.3"],
        }
    ).drop_duplicates("patient_id", keep="last")
    for column in ("mri_exam_date", "colonoscopy_date"):
        exam_summary[column] = exam_summary[column].map(iso_date)
    sheet = sheet.merge(exam_summary, on="patient_id", how="left")
    sheet = sheet.merge(
        ct_lookup[
            [
                "patient_id",
                "ct_exam_date_last_before_surgery",
                "ct_exam_datetime_last_before_surgery",
                "ct_records_before_surgery_count",
                "all_ct_records_count",
            ]
        ],
        on="patient_id",
        how="left",
    )
    sheet["ct_exam_date"] = sheet["ct_exam_date_last_before_surgery"]

    pathology = exports["surgical_pathology"].merge(
        surgery_summary[["patient_id", "surgery_procedure", "surgery_description"]],
        on="patient_id",
        how="left",
    )
    parsed_pathology = pd.DataFrame(
        [
            {"patient_id": row["patient_id"], **parse_pathology_row(row)}
            for _, row in pathology.iterrows()
        ]
    )
    sheet = sheet.merge(parsed_pathology, on="patient_id", how="left")

    for target, fallback in (
        ("pathological_t", "pathological_t_from_stage_summary"),
        ("pathological_n", "pathological_n_from_stage_summary"),
        ("pathological_m", "pathological_m_from_stage_summary"),
    ):
        sheet[target] = sheet[target].combine_first(sheet[fallback])
    sheet["pathological_stage"] = sheet.apply(
        lambda row: derive_stage(
            row["pathological_t"],
            row["pathological_n"],
            row["pathological_m"],
        ),
        axis=1,
    )

    sheet["has_preop_last_blood"] = sheet["patient_id"].isin(
        exports["preop_last_blood"]["patient_id"]
    )
    sheet["has_surgical_pathology"] = sheet["patient_id"].isin(
        exports["surgical_pathology"]["patient_id"]
    )
    sheet["has_preop_last_exam"] = sheet["patient_id"].isin(
        exports["preop_last_exam"]["patient_id"]
    )
    sheet["ready_for_manual_mrd_linkage"] = (
        sheet["has_preop_last_blood"]
        & sheet["has_surgical_pathology"]
        & sheet["has_preop_last_exam"]
    )

    output_columns = [
        "patient_id",
        "source_group_name",
        "cohort_period",
        "mrd_status",
        "ready_for_manual_mrd_linkage",
        "surgery_date",
        "surgery_start_datetime",
        "pathology_exam_date",
        "ct_exam_date",
        "ct_exam_date_last_before_surgery",
        "ct_exam_datetime_last_before_surgery",
        "ct_records_before_surgery_count",
        "all_ct_records_count",
        "mri_exam_date",
        "colonoscopy_date",
        *MODEL_COLUMNS,
        "cea_date",
        "cea_source",
        "ca199_date",
        "ca199_source",
        "mlh1_status",
        "pms2_status",
        "msh2_status",
        "msh6_status",
        "mmr_status",
        "pathology_text_parsed",
        "has_preop_last_blood",
        "has_surgical_pathology",
        "has_preop_last_exam",
        "clinical_t_distinct_count",
        "clinical_n_distinct_count",
        "pathological_tnm_distinct_count",
    ]
    output_columns = list(dict.fromkeys(output_columns))
    return sheet[output_columns].sort_values("patient_id").reset_index(drop=True)


def write_outputs(
    sheet: pd.DataFrame,
    issues: pd.DataFrame,
    inventory: pd.DataFrame,
    ct_lookup: pd.DataFrame,
    ct_records: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    excel_output: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / "mrd_data_model_sheet.csv"
    issues_path = output_dir / "mrd_data_qc_issues.csv"
    inventory_path = output_dir / "mrd_data_source_inventory.csv"
    ct_lookup_path = output_dir / "mrd_ct_lookup.csv"
    ct_records_path = output_dir / "mrd_ct_records.csv"
    summary_path = output_dir / "mrd_data_qc_summary.json"
    excel_path = output_dir / excel_output

    sheet.to_csv(sheet_path, index=False)
    issues.to_csv(issues_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    ct_lookup.to_csv(ct_lookup_path, index=False)
    ct_records.to_csv(ct_records_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    qc_summary = pd.DataFrame(
        [
            {"metric": key, "value": json.dumps(value, ensure_ascii=False)}
            if isinstance(value, (dict, list))
            else {"metric": key, "value": value}
            for key, value in summary.items()
        ]
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="model_sheet", index=False)
        qc_summary.to_excel(writer, sheet_name="qc_summary", index=False)
        issues.to_excel(writer, sheet_name="qc_issues", index=False)
        inventory.to_excel(writer, sheet_name="source_inventory", index=False)
        ct_lookup.to_excel(writer, sheet_name="ct_lookup", index=False)
        ct_records.to_excel(writer, sheet_name="ct_records", index=False)


def write_stage_ct_subset_outputs(sheet: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Write stage II/III CT-record subsets and return their sample counts."""
    subset_csv = output_dir / "mrd_stage_ii_iii_ct_cohort.csv"
    subset_excel = output_dir / "mrd_stage_ii_iii_ct_cohort.xlsx"
    subset_summary = output_dir / "mrd_stage_ii_iii_ct_sample_size.json"

    work = sheet.copy()
    work["all_ct_records_count"] = pd.to_numeric(
        work["all_ct_records_count"], errors="coerce"
    ).fillna(0).astype(int)
    work["ct_records_before_surgery_count"] = pd.to_numeric(
        work["ct_records_before_surgery_count"], errors="coerce"
    ).fillna(0).astype(int)
    stage = work["pathological_stage"].astype(str).str.upper().str.strip()
    stage_ii_iii = work[stage.isin(["II", "III"])].copy()
    any_ct = stage_ii_iii[stage_ii_iii["all_ct_records_count"].gt(0)].copy()
    preop_ct = stage_ii_iii[
        stage_ii_iii["ct_records_before_surgery_count"].gt(0)
    ].copy()

    any_ct["has_any_ct_record"] = True
    any_ct["has_preop_ct_record"] = any_ct[
        "ct_records_before_surgery_count"
    ].gt(0)
    preop_ct["has_any_ct_record"] = True
    preop_ct["has_preop_ct_record"] = True

    any_ct.to_csv(subset_csv, index=False)
    with pd.ExcelWriter(subset_excel, engine="openpyxl") as writer:
        any_ct.to_excel(writer, sheet_name="stage_ii_iii_any_ct", index=False)
        preop_ct.to_excel(writer, sheet_name="stage_ii_iii_preop_ct", index=False)
        stage_ii_iii.to_excel(writer, sheet_name="stage_ii_iii_all", index=False)

    summary = {
        "total_patients": int(len(work)),
        "pathological_stage_counts_all": work["pathological_stage"]
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict(),
        "stage_ii_iii_patients": int(len(stage_ii_iii)),
        "stage_ii_iii_counts": stage_ii_iii["pathological_stage"]
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict(),
        "stage_ii_iii_with_any_ct_record": int(len(any_ct)),
        "stage_ii_iii_with_any_ct_by_stage": any_ct["pathological_stage"]
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict(),
        "stage_ii_iii_with_preop_ct_record": int(len(preop_ct)),
        "stage_ii_iii_with_preop_ct_by_stage": preop_ct["pathological_stage"]
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict(),
        "stage_ii_iii_with_surgical_pathology": int(
            stage_ii_iii["has_surgical_pathology"].sum()
        ),
        "stage_ii_iii_with_preop_last_blood": int(
            stage_ii_iii["has_preop_last_blood"].sum()
        ),
        "stage_ii_iii_ready_for_manual_mrd_linkage": int(
            stage_ii_iii["ready_for_manual_mrd_linkage"].sum()
        ),
        "main_output_csv": str(subset_csv),
        "main_output_xlsx": str(subset_excel),
        "note": (
            "The main CSV keeps pathological stage II/III patients with any CT "
            "record. For preoperative CT modelling, use the stricter preop CT "
            "count or the stage_ii_iii_preop_ct Excel sheet."
        ),
    }
    subset_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    exports = {name: read_export(input_dir, name) for name in EXPORT_FILES}
    inventory = build_source_inventory(exports)
    surgery_summary = build_surgery_summary(exports)
    ct_records = build_ct_records(exports)
    ct_lookup = build_ct_lookup(exports, surgery_summary, ct_records)
    sheet = build_sheet(exports, surgery_summary, ct_lookup)
    issues, summary = apply_qc(sheet, inventory)
    write_outputs(
        sheet,
        issues,
        inventory,
        ct_lookup,
        ct_records,
        summary,
        output_dir,
        args.excel_output,
    )
    subset_summary = write_stage_ct_subset_outputs(sheet, output_dir)
    print(
        f"Wrote {len(sheet)} patients, {len(ct_records)} CT records, "
        f"{len(issues)} QC issues, "
        f"and {len(inventory)} source inventory rows to {output_dir}. "
        f"Stage II/III with any CT: "
        f"{subset_summary['stage_ii_iii_with_any_ct_record']}; "
        f"with preoperative CT: "
        f"{subset_summary['stage_ii_iii_with_preop_ct_record']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"prepare_mrd_data_sheet failed: {exc}") from exc
