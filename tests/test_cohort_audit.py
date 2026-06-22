import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.cohort_audit import (
    TABLE_SCHEMAS,
    audit_and_build_cohort,
    write_table_templates,
)


def _write_linked_tables(tmp_path: Path, missing_mrd_patient: str | None = None):
    table_dir = tmp_path / "tables"
    write_table_templates(table_dir)
    patients = [f"p{index}" for index in range(8)]
    periods = ["retrospective"] * 4 + ["prospective"] * 4
    pd.DataFrame(
        {
            "patient_id": patients,
            "cohort_period": periods,
            "screened": ["yes"] * 8,
            "inclusion_status": ["included"] * 8,
            "exclusion_reason": [pd.NA] * 8,
            "age": [50 + index for index in range(8)],
            "sex": ["female", "male"] * 4,
            "ecog": [0, 1] * 4,
        }
    ).to_csv(table_dir / "cohort.csv", index=False)
    pd.DataFrame(
        {
            "patient_id": patients,
            "clinical_t": ["cT3"] * 8,
            "clinical_n": ["cN0", "cN1"] * 4,
            "clinical_m": ["cM0"] * 8,
            "tumor_location": ["colon", "rectum"] * 4,
            "tumor_side": ["right", "rectal"] * 4,
            "cea_preoperative": [2.0 + index for index in range(8)],
            "cea_unit": ["ng/mL"] * 8,
            "cea_date": ["2025-01-01"] * 8,
            "cea_reference_upper": [5.0] * 8,
            "ca199_preoperative": [10 + index for index in range(8)],
            "ca199_unit": ["U/mL"] * 8,
            "ca199_date": ["2025-01-01"] * 8,
            "ca199_reference_upper": [37.0] * 8,
            "laboratory_id": ["lab-a"] * 8,
            "relevant_comorbidities": [pd.NA] * 8,
            "neoadjuvant_treatment": ["no"] * 8,
        }
    ).to_csv(table_dir / "preoperative.csv", index=False)
    stage = ["II", "III"] * 4
    pd.DataFrame(
        {
            "patient_id": patients,
            "primary_crc_adenocarcinoma": ["yes"] * 8,
            "surgery_date": ["2025-02-01"] * 8,
            "surgery_procedure": ["colectomy"] * 8,
            "curative_intent": ["yes"] * 8,
            "resection_margin": ["R0"] * 8,
            "pathological_t": ["pT3"] * 8,
            "pathological_n": ["pN0", "pN1"] * 4,
            "pathological_m": ["pM0"] * 8,
            "pathological_stage": stage,
            "tumor_size_pathological_mm": [35.0] * 8,
            "histological_type": ["adenocarcinoma"] * 8,
            "histological_grade": ["moderate"] * 8,
            "positive_lymph_nodes": [0, 2] * 4,
            "examined_lymph_nodes": [20] * 8,
            "lymphovascular_invasion": ["no", "yes"] * 4,
            "perineural_invasion": ["no"] * 8,
            "tumor_deposits": ["no"] * 8,
            "bowel_obstruction": ["no"] * 8,
            "tumor_perforation": ["no"] * 8,
            "mlh1_status": ["positive"] * 8,
            "pms2_status": ["positive"] * 8,
            "msh2_status": ["positive"] * 8,
            "msh6_status": ["positive"] * 8,
            "mmr_status": ["pMMR"] * 8,
            "msi_status": ["MSS"] * 8,
            "her2_status": ["0"] * 8,
            "her2_method": ["IHC"] * 8,
            "braf_status": ["negative"] * 8,
            "braf_method": ["PCR"] * 8,
            "braf_variant": ["V600E"] * 8,
            "ki67_percent": [50.0] * 8,
            "distant_metastasis": ["no"] * 8,
            "visible_residual_disease": ["no"] * 8,
        }
    ).to_csv(table_dir / "pathology.csv", index=False)

    mrd_patients = [
        patient for patient in patients if patient != missing_mrd_patient
    ]
    pd.DataFrame(
        {
            "patient_id": mrd_patients,
            "blood_draw_date": ["2025-03-01"] * len(mrd_patients),
            "mrd_result": [
                "negative" if int(patient[1:]) % 2 == 0 else "positive"
                for patient in mrd_patients
            ],
            "assay_platform": ["assay-a"] * len(mrd_patients),
            "assay_version": ["v1"] * len(mrd_patients),
            "positivity_rule": ["two-variant rule"] * len(mrd_patients),
            "plasma_volume_ml": [10.0] * len(mrd_patients),
            "assay_qc_status": ["pass"] * len(mrd_patients),
            "systemic_treatment_before_draw": ["no"] * len(mrd_patients),
        }
    ).to_csv(table_dir / "mrd.csv", index=False)

    ct_rows = []
    for patient in patients:
        for phase in ("nc", "arterial", "portal", "delayed"):
            image_path = tmp_path / "images" / patient / f"{phase}.nii.gz"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.touch()
            ct_rows.append(
                {
                    "patient_id": patient,
                    "study_date": "2025-01-15",
                    "phase_name": phase,
                    "modality": "CT",
                    "dicom_source_path": f"/deidentified/{patient}/{phase}",
                    "image_path": str(image_path),
                    "anatomical_coverage": "chest-abdomen-pelvis",
                    "scanner_manufacturer": "vendor",
                    "scanner_model": "model",
                    "tube_voltage_kvp": 120,
                    "tube_current_ma": 200,
                    "slice_thickness_mm": 1.0,
                    "spacing_x_mm": 0.8,
                    "spacing_y_mm": 0.8,
                    "reconstruction_kernel": "standard",
                    "contrast_timing_seconds": 60,
                    "contrast_dose_ml": 80,
                    "deidentified": "yes",
                    "geometry_qc": "yes",
                    "registration_qc": "yes",
                    "tumor_annotation_path": pd.NA,
                }
            )
    pd.DataFrame(ct_rows).to_csv(table_dir / "ct_manifest.csv", index=False)
    return table_dir


def _config(tmp_path: Path, table_dir: Path):
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["data"]["cohort_tables"] = {
        name: str(table_dir / f"{name}.csv")
        for name in TABLE_SCHEMAS
    }
    config["data"]["ct_sequences"] = ["nc", "arterial", "portal", "delayed"]
    config["data"]["ct_phases_confirmed"] = True
    config["data"]["mrd_endpoint"] = {
        "assay_definition_finalized": True,
        "blood_draw_window_finalized": True,
        "blood_draw_min_days": 7,
        "blood_draw_max_days": 60,
    }
    return config


def test_audit_builds_matched_temporal_cohort(tmp_path):
    table_dir = _write_linked_tables(tmp_path)
    result = audit_and_build_cohort(_config(tmp_path, table_dir))

    assert result.ready
    assert len(result.analytical_cohort) == 8
    assert result.report["mrd_positive_patients"] == 4
    assert set(result.analytical_cohort["cohort_period"]) == {
        "retrospective",
        "prospective",
    }
    assert result.report["binding_constraint"].startswith("eligible")
    json.dumps(result.report)


def test_audit_blocks_when_eligible_patient_lacks_valid_mrd(tmp_path):
    table_dir = _write_linked_tables(tmp_path, missing_mrd_patient="p0")
    result = audit_and_build_cohort(_config(tmp_path, table_dir))

    assert not result.ready
    assert len(result.analytical_cohort) == 7
    assert any(
        "Not every otherwise eligible" in item
        for item in result.report["blockers"]
    )


def test_template_writer_covers_linked_architecture(tmp_path):
    write_table_templates(tmp_path)

    assert {
        path.stem for path in tmp_path.glob("*.csv")
    } == set(TABLE_SCHEMAS)
