"""PSG 내부 Y 지표(AHI, RDI, ODI, Lowest SpO2) 상관 히트맵. 1차 잠정 분석.

값: Result_py/psg_verified.csv의 V_* (리포트로 출처가 연결된 84명, P_22 ODI 정정 6.1 반영).
P_65·P_67·P_69는 V_*가 비어 있어 자동으로 빠집니다. 먼저 python/psg_reconcile.py를 실행해 두어야 합니다.

결과: Result_py/psg_y_correlation/
  y_corr_pearson.csv, y_corr_spearman.csv   상관계수 행렬
  y_corr_pairs.csv                          지표 쌍별 n, Pearson r·p, Spearman ρ·p
  y_corr_heatmap.png                        Pearson / Spearman 히트맵
사용: python psg_y_correlation.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py" / "psg_y_correlation"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9})

Y = {"V_AHI": "AHI", "V_RDI": "RDI", "V_ODI": "ODI", "V_LSaO2": "Lowest SpO₂"}
# 발산형: 음(빨강) ← 회색 0 → 양(파랑)
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#b2432f", "#e8a597", "#f2f2f2", "#8fbbe9", "#0b3d78"])
INK = "#1f2328"


def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "Result_py/psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
    return d[list(Y)].dropna().rename(columns=Y)


def pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(df.columns, 2):
        r, pr = stats.pearsonr(df[a], df[b])
        rho, ps = stats.spearmanr(df[a], df[b])
        rows.append({"X": a, "Y": b, "N": len(df), "Pearson_r": round(r, 3), "Pearson_p": pr,
                     "Spearman_rho": round(rho, 3), "Spearman_p": ps})
    return pd.DataFrame(rows)


def heatmap(ax, m: pd.DataFrame, title: str):
    k = len(m)
    mask = np.triu(np.ones((k, k), bool), 1)      # 대각선 위는 같은 값이라 비움
    im = ax.imshow(np.ma.masked_array(m.values, mask), cmap=CMAP, vmin=-1, vmax=1)
    for i in range(k):
        for j in range(i + 1):
            v = m.values[i, j]
            ax.text(j, i, "1" if i == j else f"{v:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(k), m.columns)
    ax.set_yticks(range(k), m.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    pear, spear = df.corr(), df.corr("spearman")
    pear.round(3).to_csv(OUT / "y_corr_pearson.csv", encoding="utf-8-sig")
    spear.round(3).to_csv(OUT / "y_corr_spearman.csv", encoding="utf-8-sig")
    p = pairs(df)
    p.to_csv(OUT / "y_corr_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"n={len(df)}")
    print(p.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    heatmap(axes[0], pear, "Pearson r (값 그대로의 직선 관계)")
    im = heatmap(axes[1], spear, "Spearman ρ (순위 기준, 극단값 영향 적음)")
    fig.colorbar(im, ax=axes, shrink=0.75, label="상관계수  (파랑 = 같이 커짐, 빨강 = 반대로 움직임)")
    fig.suptitle(f"PSG 내부 Y 지표 상관 · 리포트 출처가 연결된 {len(df)}명 (V_*, P_22 ODI 정정 반영) · 1차 잠정 분석",
                 fontsize=11, color=INK)
    fig.savefig(OUT / "y_corr_heatmap.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
