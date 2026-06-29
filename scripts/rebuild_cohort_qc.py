#!/usr/bin/env python3
"""Rebuild cohort-level QC statistics directly from raw MRD exports."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


RAW_EXPORTS = {
    "demographics": "基线数据.00-0 基线-姓名、电话、性别、身份证_gbk.csv",
    "diagnosis_all": "基线数据.00-1-诊断名称_全部诊断_gbk.csv",
    "followup_start": "基线数据.00-4-随访起点时间_gbk.csv",
    "pathological_tnm": "基线数据.01-3-ptnm_分期小结_gbk.csv",
    "bowel_ct": "基线数据.02-2-肠CT_CT检查_gbk.csv",
    "outpatient": "基线数据.03-1-门诊_gbk.csv",
    "inpatient": "基线数据.03-2-住院_gbk.csv",
    "last_visit": "基线数据.04-2-末次就诊_gbk.csv",
    "surgery": "基线数据.05-1-手术名称_手术_gbk.csv",
    "all_pathology": "基线数据.05-2-病理_全部病理_gbk.csv",
    "first_exam": "基线数据.11-3-首次检查_gbk.csv",
    "preop_last_blood": "基线数据.11-2-术前末次抽血_gbk.csv",
    "preop_last_exam": "基线数据.11-4-术前末次检查_gbk.csv",
    "surgical_pathology": "基线数据.11-6-手术病理_gbk.csv",
}

SUPPLEMENT_WORKBOOK = "data/raw/RSTL2021012-91数据导出20230804-yqr.xlsx"

MISSING_STRINGS = {"", "NA", "N/A", "NULL", "NONE", "NAN", "未知", "未检测"}
SOURCE_INPATIENT_COLUMN = "00-0 基线-姓名、电话、性别、身份证_inpatient_no_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/mrd_data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--supplement-workbook", default=SUPPLEMENT_WORKBOOK)
    return parser.parse_args()


def clean_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return pd.NA if text.upper() in MISSING_STRINGS else text


def normalize_id(value: Any) -> str:
    value = clean_value(value)
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_record_key(value: Any) -> str:
    value = normalize_id(value)
    if not value:
        return ""
    value = re.sub(r"\.0$", "", value)
    value = value.strip()
    stripped = value.lstrip("0")
    return stripped or "0"


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return (
        str(value)
        .replace("\u3000", " ")
        .replace("（", "(")
        .replace("）", ")")
        .replace("＋", "+")
        .replace("－", "-")
        .strip()
    )


def read_export(raw_dir: Path, name: str) -> pd.DataFrame:
    path = raw_dir / RAW_EXPORTS[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw export for {name}: {path}")
    frame = pd.read_csv(path, encoding="gbk", header=2, dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].map(clean_value)
    frame["patient_id"] = frame["patient_sn"].map(normalize_id)
    frame = frame[frame["patient_id"].ne("")]
    return frame.reset_index(drop=True)


def read_supplement_sheet(workbook: Path, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet_name, dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].map(clean_value)
    if "病历号" in frame.columns:
        frame["record_key"] = frame["病历号"].map(normalize_record_key)
        frame = frame[frame["record_key"].ne("")]
    return frame.reset_index(drop=True)


def normalize_supplement_tnm(value: Any, prefix: str) -> Any:
    text = normalize_text(value)
    if not text:
        return pd.NA
    if text.lower().startswith(prefix.lower()):
        return f"p{text[1:]}" if text[0].lower() != "p" else text
    return f"p{text}"


def normalize_supplement_mmr(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return "unknown"
    lowered = text.lower()
    if "pmmr" in lowered or "mss" in lowered:
        return "pMMR"
    if "dmmr" in lowered or "msi" in lowered:
        return "dMMR"
    return "unknown"


def direct_mmr_status_from_text(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return "unknown"
    lowered = text.lower()
    has_pmmr = (
        re.search(r"\bp\s*mmr\b", lowered) is not None
        or re.search(r"\bmss\b", lowered) is not None
        or "微卫星稳定" in text
    )
    has_dmmr = (
        re.search(r"\bd\s*mmr\b", lowered) is not None
        or re.search(r"\bmsi\b", lowered) is not None
        or "微卫星不稳定" in text
    )
    if has_dmmr:
        return "dMMR"
    if has_pmmr:
        return "pMMR"
    return "unknown"


def normalize_r0_status(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return "unknown"
    if text == "R0" or "R0" in text:
        return "negative"
    if "R1" in text or "R2" in text or "阳性" in text:
        return "positive"
    return "unknown"


def normalize_supplement_treatment(row: pd.Series) -> tuple[str, str, bool]:
    flag = normalize_text(row.get("是否新辅/诱导"))
    regimen = normalize_text(row.get("新辅/诱导药物方案"))
    if flag == "是":
        return "neoadjuvant_or_induction_present", "exclude_preoperative_treatment", True
    if flag == "否":
        return (
            "no_neoadjuvant_or_induction_documented",
            "no_explicit_preoperative_treatment",
            False,
        )
    if "新辅" in regimen or "诱导" in regimen:
        return "neoadjuvant_or_induction_present", "exclude_preoperative_treatment", True
    return "unknown", "unknown", False


def build_patient_record_map(demographics: pd.DataFrame) -> pd.DataFrame:
    mapping = demographics[["patient_id", "inpatient_no_id"]].drop_duplicates(
        "patient_id", keep="last"
    )
    mapping["record_key"] = mapping["inpatient_no_id"].map(normalize_record_key)
    mapping = mapping[mapping["record_key"].ne("")]
    return mapping[["patient_id", "record_key"]]


def build_supplement_summary(
    workbook: Path, demographics: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = build_patient_record_map(demographics)
    if not workbook.exists():
        empty = pd.DataFrame({"patient_id": demographics["patient_id"].unique()})
        empty["supplement_linked"] = False
        return empty, {"available": False, "path": str(workbook), "linked_patients": 0}

    basic = read_supplement_sheet(workbook, "基本信息")
    pathology = read_supplement_sheet(workbook, "患者病理与临床信息")
    surgery = read_supplement_sheet(workbook, "手术情况")
    treatment = read_supplement_sheet(workbook, "入组前治疗情况")
    basic_code_map = basic[["患者编号", "record_key"]].dropna().drop_duplicates(
        "患者编号", keep="last"
    )
    for frame in (pathology, surgery, treatment):
        if "record_key" not in frame.columns:
            frame["record_key"] = pd.NA
        missing_record = frame["record_key"].isna()
        if missing_record.any() and "患者编号" in frame.columns:
            filled = frame.loc[missing_record, ["患者编号"]].merge(
                basic_code_map, on="患者编号", how="left"
            )
            frame.loc[missing_record, "record_key"] = filled["record_key"].values
        frame.dropna(subset=["record_key"], inplace=True)

    linked_basic = mapping.merge(
        basic[["record_key", "患者编号", "入组状态"]].drop_duplicates("record_key"),
        on="record_key",
        how="left",
    )
    linked_pathology = mapping.merge(
        pathology.drop_duplicates("record_key", keep="last"),
        on="record_key",
        how="left",
    )
    linked_surgery = mapping.merge(
        surgery.drop_duplicates("record_key", keep="last"),
        on="record_key",
        how="left",
    )
    linked_treatment = mapping.merge(
        treatment.drop_duplicates("record_key", keep="last"),
        on="record_key",
        how="left",
    )

    rows = []
    for _, base in mapping.iterrows():
        patient_id = base["patient_id"]
        path_row = linked_pathology[linked_pathology["patient_id"].eq(patient_id)]
        surg_row = linked_surgery[linked_surgery["patient_id"].eq(patient_id)]
        treat_row = linked_treatment[linked_treatment["patient_id"].eq(patient_id)]
        basic_row = linked_basic[linked_basic["patient_id"].eq(patient_id)]

        path = path_row.iloc[0] if not path_row.empty else pd.Series(dtype=object)
        surg = surg_row.iloc[0] if not surg_row.empty else pd.Series(dtype=object)
        treat = treat_row.iloc[0] if not treat_row.empty else pd.Series(dtype=object)
        basic_one = basic_row.iloc[0] if not basic_row.empty else pd.Series(dtype=object)

        supp_t = normalize_supplement_tnm(path.get("T分期"), "T")
        supp_n = normalize_supplement_tnm(path.get("N分期"), "N")
        supp_m = normalize_supplement_tnm(path.get("M分期"), "M")
        supp_stage = stage_from_text(path.get("分期"))
        if pd.isna(supp_stage):
            supp_stage = derive_pathological_stage(supp_t, supp_n, supp_m)
        supp_mmr = normalize_supplement_mmr(path.get("MMR缺陷"))
        if supp_mmr == "unknown":
            parsed = {
                "mlh1_status": normalize_ihc(path.get("MLH1")),
                "pms2_status": normalize_ihc(path.get("PMS2")),
                "msh2_status": normalize_ihc(path.get("MSH2")),
                "msh6_status": normalize_ihc(path.get("MSH6")),
            }
            supp_mmr = derive_mmr_status(pd.Series(parsed))
        treatment_status, chemo_status, chemo_exclude = normalize_supplement_treatment(
            treat
        )
        rows.append(
            {
                "patient_id": patient_id,
                "supplement_linked": pd.notna(basic_one.get("患者编号")),
                "supplement_patient_code_available": pd.notna(
                    basic_one.get("患者编号")
                ),
                "supplement_enrollment_status": basic_one.get("入组状态", pd.NA),
                "supplement_pathological_t": supp_t,
                "supplement_pathological_n": supp_n,
                "supplement_pathological_m": supp_m,
                "supplement_pathological_stage": supp_stage,
                "supplement_mmr_status": supp_mmr,
                "supplement_resection_margin_status": normalize_r0_status(
                    surg.get("手术R0切除情况（原发灶手术）")
                ),
                "supplement_preoperative_treatment_status": treatment_status,
                "supplement_preoperative_treatment_exclusion_status": chemo_status,
                "supplement_exclude_preoperative_treatment": chemo_exclude,
            }
        )
    summary = {
        "available": True,
        "path": str(workbook),
        "raw_rows_basic": int(len(basic)),
        "raw_rows_pathology": int(len(pathology)),
        "raw_rows_surgery": int(len(surgery)),
        "raw_rows_treatment": int(len(treatment)),
        "linked_patients": int(pd.DataFrame(rows)["supplement_linked"].sum()),
        "matched_by": "raw inpatient_no_id stripped of leading zeros to supplement 病历号",
    }
    return pd.DataFrame(rows), summary


def first_nonmissing(values: pd.Series) -> Any:
    values = values.dropna()
    return values.iloc[0] if len(values) else pd.NA


def last_nonmissing(values: pd.Series) -> Any:
    values = values.dropna()
    return values.iloc[-1] if len(values) else pd.NA


def distinct_nonmissing_count(values: pd.Series) -> int:
    values = values.dropna().astype(str).str.strip()
    return int(values[values.ne("")].nunique())


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


def one_row_by_patient(
    frame: pd.DataFrame,
    columns: Iterable[str],
    prefer: str = "last",
    date_column: str | None = None,
) -> pd.DataFrame:
    columns = list(columns)
    work = frame[["patient_id", *columns]].copy()
    if date_column:
        work["_parsed_date"] = pd.to_datetime(work[date_column], errors="coerce")
        work = work.sort_values(["patient_id", "_parsed_date"], na_position="last")
    aggregation = last_nonmissing if prefer == "last" else first_nonmissing
    return work.groupby("patient_id", as_index=False).agg(
        {column: aggregation for column in columns}
    )


def normalize_tnm(value: Any) -> str:
    return normalize_text(value).lower().replace("yp", "p")


def derive_pathological_stage(t_value: Any, n_value: Any, m_value: Any) -> Any:
    t_text = normalize_tnm(t_value)
    n_text = normalize_tnm(n_value)
    m_text = normalize_tnm(m_value)
    if re.match(r"pm1", m_text):
        return "IV"
    if re.match(r"pn[12]", n_text):
        return "III"
    if n_text.startswith("pn0"):
        if re.match(r"pt[34]", t_text):
            return "II"
        if re.match(r"pt[12]", t_text):
            return "I"
    return pd.NA


def t_category_rank(value: Any) -> int:
    text = normalize_tnm(value)
    match = re.search(r"pt([0-4])", text)
    return int(match.group(1)) if match else -1


def n_category_rank(value: Any) -> int:
    text = normalize_tnm(value)
    match = re.search(r"pn([0-2])", text)
    return int(match.group(1)) if match else -1


def m_category_rank(value: Any) -> int:
    text = normalize_tnm(value)
    match = re.search(r"pm([0-1])", text)
    return int(match.group(1)) if match else -1


def stage_rank(value: Any) -> int:
    text = normalize_text(value).upper()
    return {"I": 1, "II": 2, "III": 3, "IV": 4}.get(text, 0)


def stage_from_text(value: Any) -> Any:
    text = normalize_text(value).upper()
    if not text:
        return pd.NA
    if "Ⅳ" in text or "IV" in text:
        return "IV"
    if "Ⅲ" in text or "III" in text:
        return "III"
    if "Ⅱ" in text or "II" in text:
        return "II"
    if "Ⅰ" in text or re.search(r"\bI[A-C]?\b", text):
        return "I"
    return pd.NA


def parse_tnm_from_text(text: str) -> tuple[Any, Any, Any]:
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


def build_stage_summary(pathological_tnm: pd.DataFrame) -> pd.DataFrame:
    work = pathological_tnm[
        ["patient_id", "p_t_stage", "p_n_stage", "p_m_stage"]
    ].copy()
    work["pathological_stage_candidate"] = work.apply(
        lambda row: derive_pathological_stage(
            row["p_t_stage"], row["p_n_stage"], row["p_m_stage"]
        ),
        axis=1,
    )
    work["_stage_rank"] = work["pathological_stage_candidate"].map(stage_rank)
    work["_m_rank"] = work["p_m_stage"].map(m_category_rank)
    work["_n_rank"] = work["p_n_stage"].map(n_category_rank)
    work["_t_rank"] = work["p_t_stage"].map(t_category_rank)
    work["_source_order"] = np.arange(len(work))
    summary = (
        work.sort_values(
            [
                "patient_id",
                "_stage_rank",
                "_m_rank",
                "_n_rank",
                "_t_rank",
                "_source_order",
            ]
        )
        .drop_duplicates("patient_id", keep="last")
        .rename(
        columns={
            "p_t_stage": "pathological_t",
            "p_n_stage": "pathological_n",
            "p_m_stage": "pathological_m",
            "pathological_stage_candidate": "pathological_stage",
        }
        )
    )
    summary = summary[
        ["patient_id", "pathological_t", "pathological_n", "pathological_m", "pathological_stage"]
    ]
    combo = pathological_tnm.assign(
        tnm_combo=lambda frame: (
            frame["p_t_stage"].fillna("")
            + "|"
            + frame["p_n_stage"].fillna("")
            + "|"
            + frame["p_m_stage"].fillna("")
        )
    )
    tnm_counts = (
        combo.groupby("patient_id")["tnm_combo"]
        .agg(distinct_nonmissing_count)
        .rename("pathological_tnm_distinct_count")
        .reset_index()
    )
    summary = summary.merge(tnm_counts, on="patient_id", how="left")
    summary["stage_ii_iii"] = summary["pathological_stage"].isin(["II", "III"])
    summary["stage_tnm_conflict"] = summary[
        "pathological_tnm_distinct_count"
    ].fillna(0).gt(1)
    return summary


def build_surgery_summary(surgery: pd.DataFrame) -> pd.DataFrame:
    work = surgery.copy()
    work["_surgery_datetime"] = pd.to_datetime(
        work["surgery_start_datetime"], errors="coerce"
    )
    work = work.sort_values(["patient_id", "_surgery_datetime"], na_position="last")
    summary = (
        work.groupby("patient_id", as_index=False)
        .agg(
            surgery_start_datetime=("surgery_start_datetime", first_nonmissing),
            surgery_date=("surgery_start_datetime", first_nonmissing),
            surgery_procedure=("surgical_name", first_nonmissing),
        )
    )
    summary["surgery_start_datetime"] = summary["surgery_start_datetime"].map(
        iso_datetime
    )
    summary["surgery_date"] = summary["surgery_date"].map(iso_date)
    return summary


def _ct_records_from_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    records = frame[
        [
            "patient_id",
            SOURCE_INPATIENT_COLUMN,
            "vsn_CT",
            "exam_date",
            "exam_id",
        ]
    ].copy()
    records = records.rename(
        columns={
            SOURCE_INPATIENT_COLUMN: "source_inpatient_no_id",
            "vsn_CT": "ct_visit_sn",
            "exam_date": "ct_exam_datetime",
            "exam_id": "ct_exam_id",
        }
    )
    records["ct_source"] = source
    records["ct_exam_datetime"] = records["ct_exam_datetime"].map(iso_datetime)
    records["ct_exam_date"] = records["ct_exam_datetime"].map(iso_date)
    records = records[
        records["ct_exam_datetime"].notna() | records["ct_exam_id"].notna()
    ]
    return records


def build_ct_summary(
    bowel_ct: pd.DataFrame,
    surgery_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = _ct_records_from_source(bowel_ct, "bowel_ct")
    records = records.drop_duplicates(
        ["patient_id", "ct_exam_datetime", "ct_exam_id", "ct_visit_sn"],
        keep="first",
    )
    records = records.sort_values(["patient_id", "ct_exam_datetime", "ct_exam_id"])

    timed = records.merge(
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
    timed["is_preoperative_ct"] = (
        timed["_ct_datetime"].notna()
        & timed["_surgery_datetime"].notna()
        & timed["_ct_datetime"].lt(timed["_surgery_datetime"])
    )

    any_counts = records.groupby("patient_id").size().rename("all_ct_records_count")
    preop_counts = (
        timed[timed["is_preoperative_ct"]]
        .groupby("patient_id")
        .size()
        .rename("preop_ct_records_count")
    )
    latest_preop = (
        timed[timed["is_preoperative_ct"]]
        .sort_values(["patient_id", "_ct_datetime"])
        .drop_duplicates("patient_id", keep="last")
    )
    latest_preop = latest_preop[
        [
            "patient_id",
            "ct_exam_datetime",
            "ct_exam_date",
            "ct_exam_id",
            "ct_visit_sn",
            "ct_source",
        ]
    ].rename(
        columns={
            "ct_exam_datetime": "last_preop_ct_datetime",
            "ct_exam_date": "last_preop_ct_date",
            "ct_exam_id": "last_preop_ct_exam_id",
            "ct_visit_sn": "last_preop_ct_visit_sn",
            "ct_source": "last_preop_ct_source",
        }
    )
    summary = pd.DataFrame({"patient_id": records["patient_id"].drop_duplicates()})
    summary = summary.merge(any_counts, on="patient_id", how="left")
    summary = summary.merge(preop_counts, on="patient_id", how="left")
    summary = summary.merge(latest_preop, on="patient_id", how="left")
    summary["all_ct_records_count"] = (
        summary["all_ct_records_count"].fillna(0).astype(int)
    )
    summary["preop_ct_records_count"] = (
        summary["preop_ct_records_count"].fillna(0).astype(int)
    )
    summary["has_any_ct_record"] = summary["all_ct_records_count"].gt(0)
    summary["has_preop_ct_record"] = summary["preop_ct_records_count"].gt(0)
    return summary, timed.drop(columns=["_ct_datetime", "_surgery_datetime"])


def section_value(text: str, label: str) -> str | None:
    match = re.search(rf"{label}[:：]\s*(.*?)(?=\s+[\\u4e00-\\u9fa5A-Za-z0-9_-]+[:：]|【|$)", text)
    if not match:
        return None
    value = match.group(1).strip()
    value = re.split(r"\s{2,}|。|；|;", value)[0].strip()
    return value or None


def parse_ihc(text: str, marker: str) -> Any:
    match = re.search(rf"{marker}(?:-G)?\s*\(([^)]+)\)", text, flags=re.I)
    return match.group(1).strip() if match else pd.NA


def normalize_ihc(value: Any) -> Any:
    text = normalize_text(value)
    if not text:
        return pd.NA
    if "+" in text or "阳" in text:
        return "positive"
    if "-" in text or "阴" in text:
        return "negative"
    return text


def derive_mmr_status(row: pd.Series) -> str:
    statuses = [
        row.get("mlh1_status"),
        row.get("pms2_status"),
        row.get("msh2_status"),
        row.get("msh6_status"),
    ]
    statuses = [status if isinstance(status, str) else "unknown" for status in statuses]
    if all(status == "positive" for status in statuses):
        return "pMMR"
    if any(status == "negative" for status in statuses):
        return "dMMR"
    return "unknown"


def parse_margin_status(text: str) -> str:
    margin_match = re.search(r"【切缘】(.+?)(?:【|$)", text)
    margin_text = margin_match.group(1) if margin_match else text
    if "切缘" not in margin_text:
        return "unknown"
    if ("阳性" in margin_text or "累及" in margin_text) and "无浸润" not in margin_text:
        return "positive"
    if "未见癌" in margin_text or "无浸润" in margin_text or "阴性" in margin_text:
        return "negative"
    return "unknown"


def parse_preoperative_treatment(text: str) -> tuple[str, str, bool]:
    value = section_value(text, "术前辅助治疗")
    if value is None:
        return "unknown", "unknown", False
    value = normalize_text(value)
    if value.startswith("无") or "未行" in value or "未予" in value:
        return "none", "no_explicit_preoperative_treatment", False
    if "新辅助" in value or "诱导" in value:
        return "neoadjuvant_or_induction_present", "exclude_preoperative_treatment", True
    if "术前辅助" in value or "术前治疗" in value or "化疗" in value:
        return "preoperative_treatment_present", "exclude_preoperative_treatment", True
    if value.startswith("有") or "治疗" in value or "化疗" in value:
        return "preoperative_treatment_present", "exclude_preoperative_treatment", True
    return "unknown", "unknown", False


def parse_pathology_summary(surgical_pathology: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = surgical_pathology.copy()
    work["_exam_date"] = pd.to_datetime(work["exam_date"], errors="coerce")
    work = work.sort_values(["patient_id", "_exam_date"], na_position="last")
    work = work.drop_duplicates("patient_id", keep="last")
    for _, row in work.iterrows():
        text = normalize_text(row.get("imaging_conclusion"))
        treatment, chemo, chemo_excluded = parse_preoperative_treatment(text)
        pathological_t, pathological_n, pathological_m = parse_tnm_from_text(text)
        parsed = {
            "patient_id": row["patient_id"],
            "pathology_exam_date": iso_date(row.get("exam_date")),
            "has_surgical_pathology": True,
            "pathological_t_from_pathology": pathological_t,
            "pathological_n_from_pathology": pathological_n,
            "pathological_m_from_pathology": pathological_m,
            "pathological_stage_from_pathology": derive_pathological_stage(
                pathological_t, pathological_n, pathological_m
            ),
            "preoperative_treatment_status": treatment,
            "preoperative_treatment_exclusion_status": chemo,
            "exclude_preoperative_treatment": chemo_excluded,
            "resection_margin_status": parse_margin_status(text),
            "mlh1_status": normalize_ihc(parse_ihc(text, "MLH1")),
            "pms2_status": normalize_ihc(parse_ihc(text, "PMS2")),
            "msh2_status": normalize_ihc(parse_ihc(text, "MSH2")),
            "msh6_status": normalize_ihc(parse_ihc(text, "MSH6")),
        }
        parsed["mmr_status"] = derive_mmr_status(pd.Series(parsed))
        rows.append(parsed)
    return pd.DataFrame(rows)


def mmr_status_from_text(value: Any) -> str:
    direct = direct_mmr_status_from_text(value)
    if direct != "unknown":
        return direct
    text = normalize_text(value)
    if not text:
        return "unknown"
    parsed = {
        "mlh1_status": normalize_ihc(parse_ihc(text, "MLH1")),
        "pms2_status": normalize_ihc(parse_ihc(text, "PMS2")),
        "msh2_status": normalize_ihc(parse_ihc(text, "MSH2")),
        "msh6_status": normalize_ihc(parse_ihc(text, "MSH6")),
    }
    return derive_mmr_status(pd.Series(parsed))


def available_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def exam_text_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("exam_finding") or column.startswith("exam_conclusion")
    ]


def build_raw_mmr_summary(exports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source_definitions = [
        (
            "surgical_pathology_text",
            "surgical_pathology",
            ["imaging_conclusion", "imaging_finding", "gross_finding"],
            "exam_date",
            1,
        ),
        (
            "all_pathology_text",
            "all_pathology",
            ["imaging_conclusion", "imaging_finding", "gross_finding"],
            "exam_date",
            2,
        ),
        ("diagnosis_text", "diagnosis_all", ["disease_name"], "diagnosis_date", 3),
        ("followup_start_text", "followup_start", ["disease_name"], "diagnosis_date", 4),
        (
            "inpatient_history_text",
            "inpatient",
            ["present_illness_history"],
            "admission_date",
            5,
        ),
        (
            "outpatient_history_text",
            "outpatient",
            ["chief_complain", "present_illness_history"],
            "date_of_note",
            6,
        ),
        (
            "last_visit_history_text",
            "last_visit",
            [
                "chief_complain",
                "present_illness_history",
                "chief_complain.1",
                "present_illness_history.1",
            ],
            "date_of_note",
            7,
        ),
        ("first_exam_text", "first_exam", None, None, 8),
        ("preop_last_exam_text", "preop_last_exam", None, None, 9),
    ]
    records: list[dict[str, Any]] = []
    source_order = 0
    for source_name, export_key, configured_columns, date_column, priority in source_definitions:
        frame = exports.get(export_key)
        if frame is None or frame.empty:
            continue
        columns = (
            exam_text_columns(frame)
            if configured_columns is None
            else available_columns(frame, configured_columns)
        )
        if not columns:
            continue
        for _, row in frame.iterrows():
            text = " ".join(
                normalize_text(row.get(column)) for column in columns if normalize_text(row.get(column))
            )
            status = mmr_status_from_text(text)
            if status == "unknown":
                continue
            records.append(
                {
                    "patient_id": row["patient_id"],
                    "raw_mmr_status": status,
                    "raw_mmr_status_source": source_name,
                    "raw_mmr_evidence_date": iso_date(row.get(date_column))
                    if date_column
                    else pd.NA,
                    "_priority": priority,
                    "_source_order": source_order,
                }
            )
            source_order += 1
    columns = [
        "patient_id",
        "raw_mmr_status",
        "raw_mmr_status_source",
        "raw_mmr_evidence_date",
        "raw_mmr_status_conflict",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    evidence = pd.DataFrame(records)
    evidence["_date"] = pd.to_datetime(evidence["raw_mmr_evidence_date"], errors="coerce")
    rows = []
    for patient_id, group in evidence.groupby("patient_id"):
        conflict = group["raw_mmr_status"].nunique() > 1
        top_priority = group["_priority"].min()
        top = group[group["_priority"].eq(top_priority)].copy()
        top_conflict = top["raw_mmr_status"].nunique() > 1
        if top_conflict and top["raw_mmr_status"].eq("dMMR").any():
            chosen = top[top["raw_mmr_status"].eq("dMMR")]
        else:
            chosen = top
        chosen = chosen.sort_values(["_date", "_source_order"], na_position="first")
        selected = chosen.iloc[-1]
        rows.append(
            {
                "patient_id": patient_id,
                "raw_mmr_status": selected["raw_mmr_status"],
                "raw_mmr_status_source": selected["raw_mmr_status_source"],
                "raw_mmr_evidence_date": selected["raw_mmr_evidence_date"],
                "raw_mmr_status_conflict": bool(conflict),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_patient_level(
    exports: dict[str, pd.DataFrame],
    supplement: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    demographics = exports["demographics"][
        ["patient_id", "group_name", "gender", "age"]
    ].drop_duplicates("patient_id", keep="last")
    demographics = demographics.rename(columns={"group_name": "source_group_name"})
    demographics["sex"] = demographics["gender"].map({"男": "male", "女": "female"})
    demographics["age"] = pd.to_numeric(demographics["age"], errors="coerce")
    demographics = demographics.drop(columns=["gender"])

    stage = build_stage_summary(exports["pathological_tnm"])
    surgery = build_surgery_summary(exports["surgery"])
    ct_summary, ct_records = build_ct_summary(
        exports["bowel_ct"], surgery
    )
    pathology = parse_pathology_summary(exports["surgical_pathology"])
    raw_mmr = build_raw_mmr_summary(exports)

    patient = demographics.merge(stage, on="patient_id", how="left")
    patient = patient.merge(surgery, on="patient_id", how="left")
    patient = patient.merge(ct_summary, on="patient_id", how="left")
    patient = patient.merge(pathology, on="patient_id", how="left")
    patient = patient.merge(raw_mmr, on="patient_id", how="left")
    patient = patient.merge(supplement, on="patient_id", how="left")
    stage_from_pathology = patient["pathological_stage_from_pathology"].notna()
    stage_from_ptnm = patient["pathological_stage"].notna()
    stage_from_supplement = (
        ~stage_from_pathology
        & ~stage_from_ptnm
        & patient["supplement_pathological_stage"].notna()
    )
    for target, source in [
        ("pathological_t", "pathological_t_from_pathology"),
        ("pathological_n", "pathological_n_from_pathology"),
        ("pathological_m", "pathological_m_from_pathology"),
        ("pathological_stage", "pathological_stage_from_pathology"),
    ]:
        patient[target] = patient[source].combine_first(patient[target])
    for target, source in [
        ("pathological_t", "supplement_pathological_t"),
        ("pathological_n", "supplement_pathological_n"),
        ("pathological_m", "supplement_pathological_m"),
        ("pathological_stage", "supplement_pathological_stage"),
    ]:
        patient[target] = patient[target].combine_first(patient[source])
    patient["pathological_stage_source"] = np.where(
        stage_from_pathology,
        "surgical_pathology_text",
        np.where(stage_from_supplement, "supplement_workbook", np.where(stage_from_ptnm, "ptnm_summary", "missing")),
    )
    patient["stage_ii_iii"] = patient["pathological_stage"].isin(["II", "III"])
    for column in [
        "all_ct_records_count",
        "preop_ct_records_count",
    ]:
        patient[column] = patient[column].fillna(0).astype(int)
    for column in [
        "stage_ii_iii",
        "stage_tnm_conflict",
        "has_any_ct_record",
        "has_preop_ct_record",
        "has_surgical_pathology",
        "exclude_preoperative_treatment",
    ]:
        patient[column] = (
            patient[column].fillna(False).infer_objects(copy=False).astype(bool)
        )
    patient["has_preop_last_blood"] = patient["patient_id"].isin(
        exports["preop_last_blood"]["patient_id"]
    )
    patient["has_preop_last_exam"] = patient["patient_id"].isin(
        exports["preop_last_exam"]["patient_id"]
    )
    patient["preoperative_treatment_status"] = patient[
        "preoperative_treatment_status"
    ].fillna("unknown")
    patient["preoperative_treatment_exclusion_status"] = patient[
        "preoperative_treatment_exclusion_status"
    ].fillna("unknown")
    patient["mmr_status"] = patient["mmr_status"].fillna("unknown")
    patient["resection_margin_status"] = patient[
        "resection_margin_status"
    ].fillna("unknown")
    patient["raw_mmr_status"] = patient["raw_mmr_status"].fillna("unknown")
    patient["raw_mmr_status_source"] = patient["raw_mmr_status_source"].fillna(
        "unknown"
    )
    patient["raw_mmr_status_conflict"] = (
        patient["raw_mmr_status_conflict"]
        .fillna(False)
        .infer_objects(copy=False)
        .astype(bool)
    )
    patient["supplement_mmr_status"] = patient["supplement_mmr_status"].fillna(
        "unknown"
    )
    patient["supplement_linked"] = patient["supplement_linked"].fillna(False).astype(bool)

    patient["mmr_status_source"] = np.where(
        patient["mmr_status"].isin(["pMMR", "dMMR"]),
        "surgical_pathology_text",
        "unknown",
    )
    fill_from_raw_mmr = (
        patient["mmr_status"].eq("unknown")
        & patient["raw_mmr_status"].isin(["pMMR", "dMMR"])
    )
    patient["mmr_status"] = np.where(
        fill_from_raw_mmr,
        patient["raw_mmr_status"],
        patient["mmr_status"],
    )
    patient["mmr_status_source"] = np.where(
        fill_from_raw_mmr,
        patient["raw_mmr_status_source"],
        patient["mmr_status_source"],
    )
    fill_from_supplement_mmr = (
        patient["mmr_status"].eq("unknown")
        & patient["supplement_mmr_status"].isin(["pMMR", "dMMR"])
    )
    patient["mmr_status"] = np.where(
        fill_from_supplement_mmr,
        patient["supplement_mmr_status"],
        patient["mmr_status"],
    )
    patient["mmr_status_source"] = np.where(
        fill_from_supplement_mmr,
        "baseline_supplement",
        patient["mmr_status_source"],
    )
    patient["resection_margin_status"] = np.where(
        patient["resection_margin_status"].eq("unknown")
        & patient["supplement_resection_margin_status"].notna()
        & ~patient["supplement_resection_margin_status"].eq("unknown"),
        patient["supplement_resection_margin_status"],
        patient["resection_margin_status"],
    )
    patient["preoperative_treatment_status"] = np.where(
        patient["preoperative_treatment_status"].eq("unknown")
        & patient["supplement_preoperative_treatment_status"].notna()
        & ~patient["supplement_preoperative_treatment_status"].eq("unknown"),
        patient["supplement_preoperative_treatment_status"],
        patient["preoperative_treatment_status"],
    )
    patient["preoperative_treatment_exclusion_status"] = np.where(
        patient["preoperative_treatment_exclusion_status"].eq("unknown")
        & patient["supplement_preoperative_treatment_exclusion_status"].notna()
        & ~patient["supplement_preoperative_treatment_exclusion_status"].eq("unknown"),
        patient["supplement_preoperative_treatment_exclusion_status"],
        patient["preoperative_treatment_exclusion_status"],
    )
    patient["exclude_preoperative_treatment"] = (
        patient["exclude_preoperative_treatment"]
        | patient["supplement_exclude_preoperative_treatment"].fillna(False).astype(bool)
    )

    patient["core_qc_pass_pre_mrd"] = (
        patient["stage_ii_iii"]
        & patient["has_preop_ct_record"]
        & ~patient["exclude_preoperative_treatment"]
    )
    patient["documented_plan_qc_pass_pre_mrd"] = (
        patient["core_qc_pass_pre_mrd"]
        & patient["mmr_status"].eq("pMMR")
        & patient["resection_margin_status"].eq("negative")
    )
    patient["mrd_status"] = pd.NA
    patient["mrd_label_status"] = "not_linked"

    output_columns = [
        "patient_id",
        "source_group_name",
        "age",
        "sex",
        "pathological_t",
        "pathological_n",
        "pathological_m",
        "pathological_stage",
        "pathological_stage_source",
        "stage_ii_iii",
        "stage_tnm_conflict",
        "supplement_linked",
        "surgery_date",
        "surgery_start_datetime",
        "has_any_ct_record",
        "all_ct_records_count",
        "has_preop_ct_record",
        "preop_ct_records_count",
        "last_preop_ct_date",
        "last_preop_ct_source",
        "has_surgical_pathology",
        "pathology_exam_date",
        "resection_margin_status",
        "mmr_status",
        "mmr_status_source",
        "raw_mmr_status",
        "raw_mmr_status_source",
        "raw_mmr_evidence_date",
        "raw_mmr_status_conflict",
        "supplement_mmr_status",
        "mlh1_status",
        "pms2_status",
        "msh2_status",
        "msh6_status",
        "preoperative_treatment_status",
        "preoperative_treatment_exclusion_status",
        "exclude_preoperative_treatment",
        "has_preop_last_blood",
        "has_preop_last_exam",
        "core_qc_pass_pre_mrd",
        "documented_plan_qc_pass_pre_mrd",
        "mrd_label_status",
        "mrd_status",
    ]
    return patient[output_columns].sort_values("patient_id").reset_index(drop=True), ct_records


def build_exclusion_reasons(patient: pd.DataFrame) -> pd.DataFrame:
    reasons = []
    for _, row in patient.iterrows():
        patient_id = row["patient_id"]
        if not row["stage_ii_iii"]:
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "pathological_stage_ii_iii",
                    "severity": "exclude_core",
                    "reason": "Not pathological stage II/III or stage unavailable.",
                }
            )
        if row["stage_tnm_conflict"]:
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "pathological_tnm_consistency",
                    "severity": "warning",
                    "reason": "Multiple distinct pathological TNM values in raw export.",
                }
            )
        if not row["has_preop_ct_record"]:
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "preoperative_ct_record",
                    "severity": "exclude_core",
                    "reason": "No CT record before surgery.",
                }
            )
        if row["exclude_preoperative_treatment"]:
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "preoperative_treatment",
                    "severity": "exclude_core",
                    "reason": "Explicit preoperative treatment documented.",
                }
            )
        if row["mmr_status"] == "dMMR":
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "pmmr_mss",
                    "severity": "exclude_documented_plan",
                    "reason": "dMMR documented.",
                }
            )
        elif row["mmr_status"] == "unknown":
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "pmmr_mss",
                    "severity": "missing_plan_field",
                    "reason": "MMR/MSI status unavailable after configured raw and supplement sources.",
                }
            )
        if row["resection_margin_status"] == "positive":
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "r0_resection",
                    "severity": "exclude_documented_plan",
                    "reason": "Positive margin documented.",
                }
            )
        elif row["resection_margin_status"] == "unknown":
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "r0_resection",
                    "severity": "missing_plan_field",
                    "reason": "Resection margin unavailable in parsed raw pathology.",
                }
            )
        if row["mrd_label_status"] == "not_linked":
            reasons.append(
                {
                    "patient_id": patient_id,
                    "criterion": "valid_first_mrd",
                    "severity": "missing_endpoint",
                    "reason": "MRD/ctDNA endpoint has not been linked yet.",
                }
            )
    return pd.DataFrame(
        reasons,
        columns=["patient_id", "criterion", "severity", "reason"],
    )


def count_row(step: str, frame: pd.DataFrame, note: str = "") -> dict[str, Any]:
    return {
        "step": step,
        "n": int(len(frame)),
        "stage_ii": int(frame["pathological_stage"].eq("II").sum())
        if "pathological_stage" in frame
        else np.nan,
        "stage_iii": int(frame["pathological_stage"].eq("III").sum())
        if "pathological_stage" in frame
        else np.nan,
        "note": note,
    }


def build_counts(patient: pd.DataFrame) -> pd.DataFrame:
    stage_ii_iii = patient[patient["stage_ii_iii"]]
    preop_ct = stage_ii_iii[stage_ii_iii["has_preop_ct_record"]]
    no_explicit_preop_treatment = preop_ct[
        ~preop_ct["exclude_preoperative_treatment"]
    ]
    with_pmmr = no_explicit_preop_treatment[
        no_explicit_preop_treatment["mmr_status"].eq("pMMR")
    ]
    with_r0 = with_pmmr[with_pmmr["resection_margin_status"].eq("negative")]
    with_raw_pathology = no_explicit_preop_treatment[
        no_explicit_preop_treatment["has_surgical_pathology"]
    ]
    with_preop_blood_exam = no_explicit_preop_treatment[
        no_explicit_preop_treatment["has_preop_last_blood"]
        & no_explicit_preop_treatment["has_preop_last_exam"]
    ]
    rows = [
        count_row("raw_candidate_patients", patient, "Starting patients from raw demographics export."),
        count_row("pathological_stage_ii_iii", stage_ii_iii, "Required pathological stage II/III."),
        count_row("stage_ii_iii_with_preoperative_ct", preop_ct, "Primary CT criterion confirmed by CT time before surgery."),
        count_row("no_explicit_preoperative_treatment", no_explicit_preop_treatment, "Unknown preoperative treatment status is not excluded; explicit neoadjuvant/induction/preoperative treatment is excluded."),
        count_row("plus_documented_pmmr", with_pmmr, "pMMR documented from configured data/raw MMR evidence or baseline supplement."),
        count_row("plus_documented_r0", with_r0, "R0/negative margin documented from raw pathology or baseline supplement."),
        count_row("raw_surgical_pathology_record_available", with_raw_pathology, "Raw surgical pathology completeness checkpoint, not required after baseline supplement."),
        count_row("plus_preop_blood_and_exam_records", with_preop_blood_exam, "Auxiliary completeness checkpoint for manual MRD linkage."),
    ]
    return pd.DataFrame(rows)


def build_summary(
    patient: pd.DataFrame,
    counts: pd.DataFrame,
    supplement_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qc_standard": {
            "raw_source": "data/raw/mrd_data plus baseline supplement workbook",
            "primary_ct_standard": "CT record before surgery",
            "primary_ct_sources": "bowel_ct raw CT export only",
            "supplement_source": supplement_summary,
            "stage_standard": (
                "pathological stage II/III derived from raw pTNM and surgical "
                "pathology text, supplemented from baseline workbook when missing"
            ),
            "mmr_standard": (
                "MMR status is first derived from raw surgical pathology IHC text; "
                "if unavailable, all configured data/raw diagnosis, pathology, "
                "history, and exam text sources are scanned; baseline supplement "
                "is used only when raw-derived MMR remains unknown."
            ),
            "multiple_ptnm_rule": (
                "When multiple distinct raw pTNM records exist for one patient, "
                "select the highest-stage record. Ties are broken by higher M, "
                "then higher N, then higher T category."
            ),
            "preoperative_treatment_standard": (
                "Exclude explicitly documented preoperative treatment, including "
                "neoadjuvant or induction treatment. Postoperative adjuvant treatment "
                "is allowed. Unknown preoperative treatment status is retained."
            ),
            "mrd_status": "not linked in this rebuild",
        },
        "counts": counts.to_dict(orient="records"),
        "pathological_stage_distribution": patient["pathological_stage"]
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict(),
        "preoperative_treatment_status_distribution": patient[
            "preoperative_treatment_status"
        ]
        .fillna("unknown")
        .value_counts(dropna=False)
        .to_dict(),
        "mmr_status_distribution": patient["mmr_status"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .to_dict(),
        "mmr_status_source_distribution": patient["mmr_status_source"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .to_dict(),
        "resection_margin_status_distribution": patient[
            "resection_margin_status"
        ]
        .fillna("unknown")
        .value_counts(dropna=False)
        .to_dict(),
        "core_pre_mrd_last_preop_ct_source_distribution": patient.loc[
            patient["core_qc_pass_pre_mrd"], "last_preop_ct_source"
        ]
        .fillna("missing")
        .value_counts(dropna=False)
        .to_dict(),
        "supplement_linked_distribution": patient["supplement_linked"]
        .value_counts(dropna=False)
        .to_dict(),
        "direct_identifier_columns_in_patient_output": [],
    }


def write_outputs(
    patient: pd.DataFrame,
    counts: pd.DataFrame,
    exclusions: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_path = output_dir / "cohort_patient_level.csv"
    counts_path = output_dir / "cohort_qc_counts.csv"
    exclusions_path = output_dir / "cohort_exclusion_reasons.csv"
    summary_path = output_dir / "cohort_qc_summary.json"
    report_path = output_dir / "cohort_qc_report.xlsx"

    patient.to_csv(patient_path, index=False)
    counts.to_csv(counts_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        counts.to_excel(writer, sheet_name="qc_counts", index=False)
        patient.to_excel(writer, sheet_name="patient_level", index=False)
        exclusions.to_excel(writer, sheet_name="exclusion_reasons", index=False)
        pd.DataFrame(
            [
                {"field": key, "value": json.dumps(value, ensure_ascii=False)}
                if isinstance(value, (dict, list))
                else {"field": key, "value": value}
                for key, value in summary.items()
            ]
        ).to_excel(writer, sheet_name="summary", index=False)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    supplement_workbook = Path(args.supplement_workbook)
    exports = {name: read_export(raw_dir, name) for name in RAW_EXPORTS}
    supplement, supplement_summary = build_supplement_summary(
        supplement_workbook, exports["demographics"]
    )
    patient, _ct_records = build_patient_level(exports, supplement)
    counts = build_counts(patient)
    exclusions = build_exclusion_reasons(patient)
    summary = build_summary(patient, counts, supplement_summary)
    write_outputs(patient, counts, exclusions, summary, output_dir)
    core_n = int(patient["core_qc_pass_pre_mrd"].sum())
    documented_n = int(patient["documented_plan_qc_pass_pre_mrd"].sum())
    print(
        f"Wrote cohort QC for {len(patient)} patients to {output_dir}. "
        f"Core pre-MRD QC: {core_n}; documented pMMR/R0 subset: {documented_n}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"rebuild_cohort_qc failed: {exc}") from exc
