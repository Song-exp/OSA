"""환자별 분석 자격표와 분석별 N. 각 열은 독립된 기준이며, 하나의 순차 제외 흐름이 아닙니다.
입력: phase1/patient_master.csv, eval/eval_qc.csv (랜드마크 QC), shape/<구조>/review/qc_review.csv, cbct_eda/cbct_eda_flags.csv (있으면)
결과: Result_py/phase1/cohort_eligibility.csv, cohort_n.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Result_py"
M0_VARS = ["AHI", "Age", "Male", "BMI", "ANB", "MPFH", "TDI", "PMCA", "PAL"]


def main():
    m = pd.read_csv(R / "phase1/patient_master.csv", index_col="Patient", encoding="utf-8-sig")
    qc = pd.read_csv(R / "eval/eval_qc.csv", index_col=0, encoding="utf-8-sig").qc_flag.fillna("")
    e = pd.DataFrame(index=m.index)
    e["Excel"] = True
    e["PSG_source_confirmed"] = m.Report_AHI_matches_Excel.astype(bool)
    e["Valid_Main"] = m.Valid_Main.astype(bool)                     # Class III·CBCT 불일치 제외
    e["Gap_le_1y"] = m.Valid_Gap1y.astype(bool)
    e["M0_complete"] = m[M0_VARS].notna().all(axis=1)
    e["CBCT_processed"] = e.index.isin(qc.index)                    # ALI + AMASSS 결과 있음
    e["LM_QC_pass"] = e.index.map(lambda p: p in qc.index and qc[p] == "")
    e["LM_Po_fail"] = e.index.map(lambda p: p in qc.index and bool(pd.Series([qc[p]]).str.contains(r"[RL]Po\b")[0]))
    for s in ("MAND", "CB"):
        f = R / f"shape/{s}/review/qc_review.csv"
        if f.exists():
            q = pd.read_csv(f, index_col=0, encoding="utf-8-sig")
            e[f"SSM_{s}_procrustes_main"] = e.index.isin(q.index[q.confirmed_fail_procrustes.isna()])
            e[f"SSM_{s}_frame_main"] = e.index.isin(q.index[q.confirmed_fail_frame.isna()])
            e[f"SSM_{s}_review"] = e.index.isin(q.index[q.review])
    f = R / "cbct_eda/cbct_eda_flags.csv"
    if f.exists():
        fl = pd.read_csv(f, index_col=0, encoding="utf-8-sig")
        e["EDA_visual_review"] = e.index.isin(fl.index)              # 육안 검수 대상 (제외 아님)
    e.to_csv(R / "phase1/cohort_eligibility.csv", encoding="utf-8-sig")

    base = e.Valid_Main & e.CBCT_processed
    sets = {
        "원본 엑셀": e.Excel,
        "PSG 리포트 출처 확인": e.PSG_source_confirmed,
        "주 분석 (Valid_Main)": e.Valid_Main,
        "CBCT 처리됨": e.CBCT_processed,
        "주 분석 ∩ CBCT 처리 (자동 계측 비교 대상)": base,
        "  ∩ 랜드마크 QC 통과": base & e.LM_QC_pass,
        "  ∩ PSG 출처 확인": base & e.PSG_source_confirmed,
        "  ∩ 검사 간격 1년 이내": base & e.Gap_le_1y,
        "  ∩ PSG 출처 확인 ∩ 간격 1년 이내": base & e.PSG_source_confirmed & e.Gap_le_1y,
        "  ∩ M0 변수 완전": base & e.M0_complete,
    }
    for s in ("MAND", "CB"):
        if f"SSM_{s}_procrustes_main" in e:
            sets[f"  ∩ SSM {s} 주 분석 (procrustes)"] = base & e[f"SSM_{s}_procrustes_main"]
            sets[f"  ∩ SSM {s} 주 분석 (frame)"] = base & e[f"SSM_{s}_frame_main"]
    n = pd.DataFrame({"Set": list(sets), "N": [int(v.sum()) for v in sets.values()],
                      "Excluded": [", ".join(base.index[base & ~v]) if k.startswith("  ") else "" for k, v in sets.items()]})
    n.to_csv(R / "phase1/cohort_n.csv", index=False, encoding="utf-8-sig")
    print(n.to_string(index=False))


if __name__ == "__main__":
    main()
