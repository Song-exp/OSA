"""M0 계측값(엑셀)과 PSG 지표의 산점도. 주 분석 환자(Valid_Main)만.
결과: Result_py/phase1/fig_m0_scatter.png, m0_scatter_r.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
X = ["ANB", "MPFH", "TDI", "PMCA", "PAL", "BMI"]
Y = ["AHI", "ODI", "LSaO2"]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

m = pd.read_csv(ROOT / "Result_py/phase1/patient_master.csv", encoding="utf-8-sig")
m = m[m.Valid_Main]
fig, axes = plt.subplots(len(Y), len(X), figsize=(3 * len(X), 2.8 * len(Y)))
rows = []
for i, y in enumerate(Y):
    for j, x in enumerate(X):
        d = m[[x, y]].dropna()
        r, p = pearsonr(d[x], d[y])
        rho, prho = spearmanr(d[x], d[y])
        rows.append({"X": x, "Y": y, "n": len(d), "r": r, "p": p, "rho": rho, "p_rho": prho})
        ax = axes[i, j]
        ax.scatter(d[x], d[y], s=10, alpha=.6)
        ax.plot(*zip(*sorted(zip(d[x], np.polyval(np.polyfit(d[x], d[y], 1), d[x])))), "r-", lw=1)
        ax.set_title(f"r={r:.2f} ρ={rho:.2f} (n={len(d)})", fontsize=9)
        ax.set_xlabel(x if i == len(Y) - 1 else "")
        ax.set_ylabel(y if j == 0 else "")
fig.suptitle("주 분석 환자 · 엑셀 계측값 vs PSG 지표 (단순 상관, 보정 없음)")
fig.tight_layout()
fig.savefig(ROOT / "Result_py/phase1/fig_m0_scatter.png", dpi=130)
t = pd.DataFrame(rows).round(4)
t.to_csv(ROOT / "Result_py/phase1/m0_scatter_r.csv", index=False, encoding="utf-8-sig")
print(t.pivot(index="X", columns="Y", values="r").round(2).to_string())
print(t[t.p < 0.05].to_string(index=False))
