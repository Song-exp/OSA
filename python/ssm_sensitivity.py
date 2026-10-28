"""SSM PC가 정합·분할 품질을 반영하는지 민감도 분석 (결과만 읽음, 대응점은 다시 계산하지 않음).
환자 집합을 바꿔 GPA+PCA를 다시 하고, 새 PC1~3이 정합 잔차·마스크 조각화와 얼마나 상관되는지, 원래 PC와 같은 축인지 봅니다.
  조각화: EDA S_<구조>_largest_frac < 0.95 (SSM 표면은 가장 큰 덩어리만 쓰므로, 떨어진 조각만큼 해부 구조가 빠짐)
  잔차 상위: 정합 잔차 평균 상위 10%
결과: Result_py/shape/<구조>/review/sensitivity.csv
사용: python ssm_sensitivity.py --struct CB
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssm_review import gpa_main, pca

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="CB")
    a = ap.parse_args()
    src = ROOT / "Result_py/shape" / a.struct
    z = np.load(src / "points.npz")
    ids = [f"P_{p}" for p in z["ids"]]
    q = pd.read_csv(src / "review/qc_review.csv", index_col=0, encoding="utf-8-sig").loc[ids]
    e = pd.read_csv(ROOT / "Result_py/cbct_eda/cbct_eda_patient.csv", index_col=0, encoding="utf-8-sig").reindex(ids)
    frac = e[f"S_{a.struct}_largest_frac"].to_numpy()
    fit = q.fit_mean_mm.to_numpy()
    frag = frac < 0.95
    top = fit > np.quantile(fit, 0.9)
    sets = {"전원": np.ones(len(ids), bool), "조각화 제외": ~frag, "잔차 상위10% 제외": ~top, "둘 다 제외": ~frag & ~top}
    _, base = pca(gpa_main(z["frame"], sets["전원"])[0].reshape(len(ids), -1), sets["전원"])
    rows = []
    for name, use in sets.items():
        X, _ = gpa_main(z["frame"], use)
        var, sc = pca(X.reshape(len(ids), -1), use)
        for k in range(3):
            s = sc[use, k]
            rows.append({"set": name, "n": int(use.sum()), "PC": f"PC{k + 1}", "var_ratio": var[k] / var.sum(),
                         "r_fit": np.corrcoef(s, fit[use])[0, 1], "r_largest_frac": np.corrcoef(s, frac[use])[0, 1],
                         "max_abs_r_with_orig_PC1to3": max(abs(np.corrcoef(s, base[use, j])[0, 1]) for j in range(3)),
                         "r_with_orig_PC1": np.corrcoef(s, base[use, 0])[0, 1]})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(src / "review/sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"{a.struct}: 조각화 {frag.sum()}명 {[i for i, f in zip(ids, frag) if f]}, 잔차 상위10% {top.sum()}명 {[i for i, f in zip(ids, top) if f]}")
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
