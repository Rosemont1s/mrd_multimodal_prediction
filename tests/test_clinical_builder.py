import numpy as np
import pandas as pd

from src.data.clinical_builder import (
    MODEL_FEATURE_COLUMNS,
    attach_mrd_labels,
    build_baseline_clinical_table,
)


def test_build_baseline_clinical_table_parses_workbook_encodings(tmp_path):
    workbook = tmp_path / "baseline.xlsx"
    common = {"患者编号": ["p1", "p2"]}
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(
            {**common, "性别": ["男", "女"], "年龄": ["61", "52"]}
        ).to_excel(writer, sheet_name="基本信息", index=False)
        pd.DataFrame(
            {
                **common,
                "癌胚抗原(CEA)": [3.2, "ND"],
                "癌胚抗原(CEA)(单位)": ["ng/mL", "ng/mL"],
                "CA19-9(糖链抗原19-9)": [12, 25],
                "CA19-9(糖链抗原19-9)(单位)": ["U/mL", "U/mL"],
            }
        ).to_excel(writer, sheet_name="术前检验", index=False)
        pd.DataFrame(
            {
                **common,
                "是否新辅/诱导": ["是", "否"],
                "新辅/诱导药物方案": ["FOLFOX", np.nan],
                "新辅/诱导周期": ["4程", np.nan],
                "既往是否有放疗": ["否", "是"],
            }
        ).to_excel(writer, sheet_name="入组前治疗情况", index=False)
        pd.DataFrame(
            {
                **common,
                "手术时间": ["2024-01-01", "2024-02-01"],
                "手术名称（原发灶手术）": ["右半结肠切除术", "直肠前切除术"],
                "手术名称（转移灶手术）": ["ND", "无"],
                "手术R0切除情况（原发灶手术）": ["R0", "R1"],
            }
        ).to_excel(writer, sheet_name="手术情况", index=False)
        pd.DataFrame(
            {
                **common,
                "T分期": ["T3", "T4a"],
                "N分期": ["N0", "N1"],
                "M分期": ["M0", "M0"],
                "分期": ["ⅡA", "ⅢB"],
                "临床诊断": ["结肠癌", "直肠癌"],
                "原发部位": ["升结肠", "直肠"],
                "位置": ["Right", "Rectal"],
                "组织学分型": ["腺癌", "腺癌"],
                "分化程度": ["中分化", "低分化"],
                "原发肿瘤直径/mm": ["20*30*40", "15*25*35"],
                "MLH1": ["＋", "－"],
                "PMS2": ["＋", "－"],
                "MSH2": ["＋", "＋"],
                "MSH6": ["＋", "＋"],
                "HER2-G": ["0", "2+"],
                "BRAF": ["－", "＋"],
                "Ki-67": ["70%+", "+，40%"],
                "淋巴结浸润 （LV_invasion）": ["无", "有"],
                "淋巴结总量": ["0/20", "2/18"],
                "神经浸润 （N_invasion）": ["无", "有"],
                "MMR缺陷": ["pMMR", "dMMR"],
                "MSI status": ["MSS", "ND"],
                "KPS评分": ["90", "ND"],
            }
        ).to_excel(writer, sheet_name="患者病理与临床信息", index=False)

    clinical = build_baseline_clinical_table(workbook)

    assert set(MODEL_FEATURE_COLUMNS).issubset(clinical.columns)
    assert clinical.loc[0, "sex"] == "male"
    assert clinical.loc[1, "sex"] == "female"
    assert clinical.loc[0, "tumor_size_pathological_mm"] == 40
    assert clinical.loc[1, "ki67_percent"] == 40
    assert clinical.loc[1, "positive_lymph_nodes"] == 2
    assert clinical.loc[1, "examined_lymph_nodes"] == 18
    assert clinical.loc[1, "mlh1_status"] == "negative"
    assert clinical.loc[1, "metastatic_surgery"] == "no"
    assert pd.isna(clinical.loc[1, "cea_preoperative"])


def test_attach_mrd_labels_requires_independent_binary_labels():
    clinical = pd.DataFrame({"patient_id": ["p1", "p2"], "age": [50, 60]})
    labels = pd.DataFrame({"subject": ["p1", "p2"], "mrd_status": [0, 1]})

    merged = attach_mrd_labels(
        clinical, labels, label_id_column="subject"
    )

    assert merged["mrd_status"].tolist() == [0, 1]
