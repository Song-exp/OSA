"""형태 PC의 3D 변위 그림 (procrustes, 주 분석 환자로 만든 축). 결과만 읽음.
행 = PC, 열 = 옆·앞·아래. 평균 형태를 +2SD 변위 크기(mm)로 색칠하고, 화살표는 -2SD → +2SD 방향.
mm 환산은 평균 중심 크기를 곱한 값 (procrustes 형태는 크기 1로 맞춰져 있음).
결과: Result_py/shape/<구조>/review/pc_displacement.png, pc_displacement.csv (PC별 축 방향 평균 변위)
사용: python ssm_pc_displacement.py --struct MAND
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shape_ssm import pca
from ssm_review import gpa_main

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
VIEWS = ((1, 2, 0, "옆 (가로 = 앞뒤)"), (0, 2, 1, "앞 (가로 = 좌우)"), (0, 1, 2, "아래 (가로 = 좌우)"))
AXES = ("좌우", "앞뒤", "위아래")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="MAND")
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    src = ROOT / "Result_py/shape" / a.struct
    z = np.load(src / "points.npz")
    saved = pd.read_csv(src / "review/pc_scores_procrustes.csv", index_col=0, encoding="utf-8-sig")
    use = saved.main_analysis.to_numpy(bool)
    X, size = gpa_main(z["frame"], use)
    mean, vt, var, sc = pca(X, use)
    for c in range(a.k):                                   # ssm_review 결과와 같은 축·부호인지
        assert np.corrcoef(sc[:, c], saved[f"PC{c + 1}"])[0, 1] > 0.999, c
    M = X.shape[1]
    mm = size[use].mean()
    m3 = mean.reshape(M, 3) * mm
    sub = np.random.default_rng(0).choice(M, 250, replace=False)

    fig, ax = plt.subplots(a.k, 3, figsize=(14, 4.6 * a.k), squeeze=False)
    rows = []
    for c in range(a.k):
        d = 2 * np.sqrt(var[c]) * vt[c].reshape(M, 3) * mm      # 평균 → +2SD 변위 (mm)
        mag = np.linalg.norm(d, axis=1)
        rows.append({"PC": f"PC{c + 1}", "var_ratio": var[c] / var.sum(), "max_mm": mag.max(), "median_mm": np.median(mag),
                     **{f"mean_{n}_mm": d[:, i].mean() for i, n in enumerate(AXES)},
                     **{f"top10%_{n}_mm": d[mag > np.quantile(mag, 0.9), i].mean() for i, n in enumerate(AXES)}})
        for j, (h, v, depth, name) in enumerate(VIEWS):
            o = np.argsort(m3[:, depth])                         # 깊이 순으로 그려 겹침 완화
            s = ax[c, j].scatter(m3[o, h], m3[o, v], c=mag[o], s=3, cmap="viridis", vmin=0, vmax=mag.max())
            ax[c, j].quiver(m3[sub, h] - d[sub, h], m3[sub, v] - d[sub, v], 2 * d[sub, h], 2 * d[sub, v],
                            angles="xy", scale_units="xy", scale=1, width=0.003, color="red", alpha=0.7)
            ax[c, j].set_aspect("equal")
            ax[c, j].set_title(f"PC{c + 1} ({100 * var[c] / var.sum():.1f}%) · {name}", fontsize=10)
        fig.colorbar(s, ax=ax[c, :], shrink=0.8, label="+2SD 변위 크기 (mm)")
    fig.suptitle(f"{a.struct} procrustes PC 변위 (n={use.sum()}, 평균 중심 크기로 mm 환산). 빨간 화살표: -2SD → +2SD. "
                 "PC 점수 부호는 임의이며 해석은 잠정", fontsize=10)
    fig.savefig(src / "review/pc_displacement.png", dpi=90, bbox_inches="tight")
    t = pd.DataFrame(rows).round(3)
    t.to_csv(src / "review/pc_displacement.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))
    print("좌표축 확인용 평균 형태 범위(mm):", {n: np.ptp(m3[:, i]).round(1) for i, n in enumerate(AXES)})


if __name__ == "__main__":
    main()
