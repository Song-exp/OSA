"""연구실 확인 없이 우리 규칙으로 PSG 값 복원 (규칙: 실험기록 "V 단계 변경").
R1 리포트 값 / R2 엑셀 기록 오류 3명은 이번 리포트 AHI·Supine AHI로 / R3 엑셀 검사 선택 / R4 등록번호 일치 포함 /
R5 원신호 쟁점 포함 / R6 Class III·19세 미만 제외 / R7 간격 1년 초과는 표시만.
출력: Result_py/phase1/psg_restored_own.csv
"""
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1]/"Result_py"; OUT = R/"phase1/psg_restored_own.csv"
R2_FIX = {"P_65": "report.pdf", "P_67": "report.pdf", "P_69": "report.pdf"}


def main():
    v = pd.read_csv(R/"psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
    m = pd.read_csv(R/"phase1/patient_master.csv", encoding="utf-8-sig").set_index("Patient")
    rep = pd.read_csv(R/"patient_matching/reports.csv", encoding="utf-8-sig")
    rows = []
    for p in m.index:
        r = {"Patient": p, "AHI_excel": m.at[p, "AHI"], "AHI_SUP_excel": m.at[p, "AHI_SUP"], "ODI_excel": m.at[p, "ODI"]}
        if p in R2_FIX:                                                  # R2: 같은 행 TST·사건수와 맞는 이번 리포트 값
            g = rep[(rep.환자 == p) & rep.파일.str.endswith(R2_FIX[p])].iloc[0]
            r |= {"AHI_r": g.AHI, "AHI_SUP_r": m.at[p, "AHI_SUP"], "ODI_r": g.ODI, "rule": "R2 엑셀 기록 오류 → 이번 리포트 값"}
        elif p in v.index and pd.notna(v.at[p, "V_AHI"]):                # R1: 리포트로 검증된 값
            r |= {"AHI_r": v.at[p, "V_AHI"], "AHI_SUP_r": m.at[p, "AHI_SUP"], "ODI_r": v.at[p, "V_ODI"], "rule": "R1 리포트 일치"}
        else:
            r |= {"AHI_r": np.nan, "AHI_SUP_r": np.nan, "ODI_r": np.nan, "rule": "리포트 없음"}
        under19 = str(v.at[p, "Under19_AtExam"]) == "True" if p in v.index else False
        r |= {"ClassIII": m.at[p, "Sag"] == "III", "Under19": under19, "Gap_gt1y": abs(m.at[p, "GapDays"]) > 365 if pd.notna(m.at[p, "GapDays"]) else False,
              "Identity": m.at[p, "CBCT_Identity"]}
        r["Include_own"] = bool(pd.notna(r["AHI_r"]) and not r["ClassIII"] and not under19 and r["Identity"] in ("match", "id_only"))
        r["Include_own_c3"] = bool(pd.notna(r["AHI_r"]) and not under19 and r["Identity"] in ("match", "id_only"))   # Class III 포함 판(2026-11-16 사용자 요청)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("Patient")
    assert (t.loc[list(R2_FIX), "AHI_r"].round(1).tolist() == [35.4, 9.6, 13.4]), t.loc[list(R2_FIX), "AHI_r"]
    t.to_csv(OUT, encoding="utf-8-sig"); print("저장", OUT, "| 포함", int(t.Include_own.sum()), "/", len(t))
    print(t[t.rule != "R1 리포트 일치"][["AHI_excel", "AHI_r", "AHI_SUP_r", "rule", "Include_own"]].to_string())


if __name__ == "__main__":
    main()
