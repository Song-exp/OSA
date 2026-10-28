"""형태 PC와 기존 계측값의 관계 (AHI 등 결과 변수는 보지 않음).
입력: Result_py/shape/<구조>/review/pc_scores_procrustes.csv (main_analysis 환자만), phase1/patient_master.csv (엑셀 계측값)
결과: Result_py/shape/<구조>/review/pc_relation.csv, pc_relation.png
사용: python pc_relation.py --struct MAND
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
VARS = ["ANB", "MPFH", "Age", "Male", "BMI"]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="MAND")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    out = ROOT / "Result_py/shape" / a.struct / "review"
    sc = pd.read_csv(out / "pc_scores_procrustes.csv", index_col=0, encoding="utf-8-sig")
    sc = sc[sc.main_analysis]
    m = pd.read_csv(ROOT / "Result_py/phase1/patient_master.csv", index_col="Patient", encoding="utf-8-sig")[VARS]
    d = sc.join(m, how="left")
    pcs = [f"PC{i + 1}" for i in range(a.k)] + ["CentroidSize"]
    rows = []
    for p in pcs:
        for v in VARS + (["CentroidSize"] if p != "CentroidSize" else []):
            x = d[[p, v]].dropna()
            r, pr = pearsonr(x[p], x[v])
            rho, _ = spearmanr(x[p], x[v])
            rows.append({"PC": p, "Var": v, "n": len(x), "r": r, "p": pr, "rho": rho})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(out / "pc_relation.csv", index=False, encoding="utf-8-sig")

    piv = t.pivot(index="PC", columns="Var", values="r").reindex(index=pcs, columns=VARS + ["CentroidSize"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(piv, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    pv = t.pivot(index="PC", columns="Var", values="p").reindex_like(piv)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if pd.notna(piv.iat[i, j]):
                ax.text(j, i, f"{piv.iat[i, j]:.2f}" + ("*" if pv.iat[i, j] < 0.05 else ""), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"{a.struct} 형태 PC(procrustes) vs 기존 계측값, n={len(sc)}  (* p<0.05, 다중비교 미보정)")
    fig.tight_layout()
    fig.savefig(out / "pc_relation.png", dpi=150)
    print(piv.round(2).to_string())
    print("\np<0.05:\n" + t[t.p < 0.05].round(3).to_string(index=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
