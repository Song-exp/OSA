"""M0를 엑셀 ANB·MP-FH로 한 번, 자동 계측 ANB·MP-FH로 한 번 돌려 같은 환자 집합에서 비교합니다.

환자 제외는 엑셀을 보지 않는 랜드마크 QC로 합니다 (엑셀과의 오차로 고르면 순환 논리).
  Po실패제외: QC에서 Po 관련 거리가 표시된 5명 제외
  QC전체제외: QC 표시 8명 제외 (Pog–Me 3명 포함)
  민감도: QC 제외 + ANB 오차 >3° 제외 (엑셀을 본 기준이므로 주 비교로 쓰지 않음)
TDI·PMCA·PAL·나이·성별·BMI는 양쪽 모두 엑셀 값입니다. 바뀌는 것은 ANB와 MP-FH 두 개뿐입니다.
결과: Result_py/eval/m0_auto_vs_excel.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from check_landmarks import read_points
from osa.ceph import measures
from phase1_eda import fit_m0

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/eval"


def main():
    m = pd.read_csv(ROOT / "Result_py/phase1/patient_master.csv", encoding="utf-8-sig")
    m = m[m.Valid_Main].set_index("Patient")
    qc = pd.read_csv(OUT / "eval_qc.csv", index_col=0, encoding="utf-8-sig").qc_flag.fillna("")
    auto = {}
    for pid in qc.index:
        mm = measures({k: r[["x", "y", "z"]].to_numpy(float) for k, r in read_points(int(pid[2:])).iterrows()})
        auto[pid] = {"ANB_auto": mm["ANB"], "MPFH_auto": mm["MPFH_GoMe"]}
    m = m.join(pd.DataFrame(auto).T, how="inner").join(qc)
    m["logAHI"] = np.log1p(m.AHI)
    po = m.qc_flag.str.contains(r"[RL]Po\b")
    anb_big = (m.ANB_auto - m.ANB).abs() > 3
    sets = {"Po실패제외": m[~po], "QC전체제외": m[m.qc_flag == ""], "민감도: QC+ANB오차>3 제외": m[(m.qc_flag == "") & ~anb_big]}
    res = []
    for name, d in sets.items():
        d = d.dropna(subset=["AHI", "Age", "Male", "BMI", "ANB", "MPFH", "TDI", "PMCA", "PAL"])   # 양쪽 같은 환자
        for src in ("엑셀", "자동"):
            x = d if src == "엑셀" else d.assign(ANB=d.ANB_auto, MPFH=d.MPFH_auto)
            for y in ("AHI", "logAHI"):
                res.append(fit_m0(x, y, f"{name} | {src}"))
    r = pd.concat(res).round(4)
    r.to_csv(OUT / "m0_auto_vs_excel.csv", index=False, encoding="utf-8-sig")
    k = r[r.Term.isin(["ANB", "MPFH"])]
    k = k.assign(B_CI=k.B.round(3).astype(str) + " [" + k.CI_low.round(3).astype(str) + ", " + k.CI_high.round(3).astype(str) + "]")
    print(k.pivot_table(index=["DV", "Analysis", "N", "AdjR2"], columns="Term", values=["B_CI", "P"], aggfunc="first")
          .to_string())
    print(f"\n주 분석 표본(83명 중 자동 계측 있는 환자): {len(m)}명, Po 실패 {po.sum()}명, QC 표시 {(m.qc_flag != '').sum()}명")


if __name__ == "__main__":
    main()
