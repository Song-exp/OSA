"""형태 분석 4단계(PCA 전까지) 보고용 그림: 대응점 품질 분포와 대응점 예시. 결과만 읽음.
  qc_box.png      : 정합 잔차(평균·95백분위), 대응점 중복 없음 비율, 랜드마크 전달 오차. 빨간 점 = 검수 표시
  corr_example.png: 템플릿과 환자 3명(잔차 하위·중앙·상위)의 대응점. 같은 색 = 같은 번호의 점 (머리 좌표계, 옆에서 봄)
결과: Result_py/shape/<구조>/review/
사용: python ssm_qc_figs.py --struct MAND
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="MAND")
    a = ap.parse_args()
    src = ROOT / "Result_py/shape" / a.struct
    q = pd.read_csv(src / "review/qc_review.csv", index_col=0, encoding="utf-8-sig")
    z = np.load(src / "points.npz")
    te = [c for c in q.columns if c.startswith("te_") and c not in ("te_median_mm", "te_flag", "te_flag_items")]
    cols = ["fit_mean_mm", "fit_p95_mm", "unique_ratio", "te_median_mm"] + te
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 4))
    for axx, c in zip(ax, cols):
        x = q[c]
        flag = q.fit_flag if c.startswith("fit") else q.te_flag_items.fillna("").str.contains(c[3:]) if c in te else \
            q.te_flag_items.fillna("").str.contains("median") if c == "te_median_mm" else pd.Series(False, index=q.index)
        axx.boxplot(x, widths=0.5, showfliers=False)
        j = 1 + rng.uniform(-0.15, 0.15, len(x))
        axx.scatter(j[~flag], x[~flag], s=7, c="steelblue", alpha=0.6)
        axx.scatter(j[flag], x[flag], s=16, c="red")
        if c == "fit_mean_mm":
            axx.axhline(3 * x.median(), color="tomato", ls="--", lw=1)
        axx.set_xticks([])
        axx.set_title(f"{c}\n중앙값 {x.median():.2f}", fontsize=8)
    fig.suptitle(f"{a.struct} 대응점 품질 (n={len(q)}). 빨간 점 = 검수 표시 (제외 아님), 점선 = 정합 잔차 3배 기준. "
                 "전달 오차는 템플릿 점 간격 오차를 포함 → 환자 간 비교용", fontsize=9)
    fig.tight_layout()
    fig.savefig(src / "review/qc_box.png", dpi=90)
    plt.close(fig)

    ids = [f"P_{p}" for p in z["ids"]]
    tpl = f"P_{int(z['template_id'])}"
    o = q.fit_mean_mm.drop(tpl).sort_values()
    show = [tpl, o.index[2], o.index[len(o) // 2], o.index[-1]]
    col = z["frame"][ids.index(tpl)][:, 1]                       # 템플릿의 앞뒤 위치로 색 → 환자에서도 같은 번호면 같은 색
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for axx, p in zip(ax, show):
        P = z["frame"][ids.index(p)]
        axx.scatter(P[:, 1], P[:, 2], c=col, cmap="turbo", s=2)
        axx.set_aspect("equal")
        axx.set_title(f"{p}{' (템플릿)' if p == tpl else ''} 정합 잔차 {q.loc[p, 'fit_mean_mm']:.2f} mm", fontsize=9)
    fig.suptitle(f"{a.struct} 대응점 예시 (옆에서 봄, 머리 좌표계). 같은 색 = 모든 환자에서 같은 번호의 점", fontsize=10)
    fig.tight_layout()
    fig.savefig(src / "review/corr_example.png", dpi=90)
    plt.close(fig)
    print(f"{a.struct}: 저장 qc_box.png, corr_example.png ({show})")


if __name__ == "__main__":
    main()
