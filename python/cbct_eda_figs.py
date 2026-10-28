"""CBCT 기초 EDA(cbct_eda.py 결과)의 정량 그림과 요약표. 결과표만 읽음 (영상은 다시 읽지 않음).
  fig_box_metrics.png : 영상 품질·자세·분할 지표 박스플롯. 빨간 점선 = 검수 표시 기준 (중앙값 ± 3.5×MAD, 가장 큰 덩어리 비율은 0.95)
  fig_flag_items.png  : 검수 표시 항목별 환자 수
  fig_year.png        : 촬영 연도별 연조직 피크 HU (밝기 보정 차이)
  fig_uaw_vs_excel.png: AMASSS 상기도 마스크 부피 vs 엑셀 기도 계측값
  eda_touch.csv, eda_uaw_vs_excel.csv, eda_flag_items.csv
결과: Result_py/cbct_eda/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/cbct_eda"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
STRUCTS = ["MAND", "MAX", "UAW", "CB"]
METRICS = ["Q_noise_SD", "Q_soft_peak_HU", "Q_metal_vox", "Q_sat3071_frac", "Q_MAND_bone_p99_HU", "Q_metal_jaw_frac",
           "P_midplane_tilt_deg", "P_SN_pitch_deg", "P_palatal_pitch_deg", "P_gap_min_mm", "P_gap_incisor_mm"] + \
          [f"S_{s}_{k}" for k in ("vol_cc", "ncomp", "largest_frac") for s in STRUCTS]


def main():
    d = pd.read_csv(OUT / "cbct_eda_patient.csv", index_col=0, encoding="utf-8-sig")
    fl = pd.read_csv(OUT / "cbct_eda_flags.csv", encoding="utf-8-sig")

    # 1. 지표 박스플롯 + 검수 기준선
    ncol = 6
    fig, ax = plt.subplots(int(np.ceil(len(METRICS) / ncol)), ncol, figsize=(18, 3.3 * np.ceil(len(METRICS) / ncol)))
    rng = np.random.default_rng(0)
    for a, c in zip(ax.ravel(), METRICS):
        x = d[c].astype(float)
        flagged = d.index.isin(fl.Patient[fl.Item == c])
        a.boxplot(x.dropna(), widths=0.5, showfliers=False)
        j = 1 + rng.uniform(-0.15, 0.15, len(x))
        a.scatter(j[~flagged], x[~flagged], s=7, c="steelblue", alpha=0.6)
        a.scatter(j[flagged], x[flagged], s=14, c="red")
        if c.endswith("largest_frac"):
            a.axhline(0.95, color="tomato", ls="--", lw=1)
        else:
            mad = 1.4826 * (x - x.median()).abs().median()
            for y in (x.median() - 3.5 * mad, x.median() + 3.5 * mad) if mad > 0 else ():
                a.axhline(y, color="tomato", ls="--", lw=1)
        a.set_xticks([])
        a.set_title(f"{c} (표시 {flagged.sum()}명)", fontsize=8)
    for a in ax.ravel()[len(METRICS):]:
        a.axis("off")
    fig.suptitle("CBCT 기초 EDA 지표 (83명). 빨간 점 = 검수 표시 (제외 아님), 점선 = 표시 기준", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_box_metrics.png", dpi=85)
    plt.close(fig)

    # 2. 검수 항목별 환자 수
    it = fl.groupby("Item").Patient.nunique().sort_values()
    it.rename("n_patients").to_csv(OUT / "eda_flag_items.csv", encoding="utf-8-sig")
    fig, a = plt.subplots(figsize=(7, 0.35 * len(it) + 1))
    a.barh(it.index, it.values, color="gray")
    for i, v in enumerate(it.values):
        a.text(v + 0.1, i, str(v), va="center", fontsize=8)
    a.set_title(f"검수 표시 항목별 환자 수 (검수 대상 {d.review_items.notna().sum()}명, 한 환자가 여러 항목 가능)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_flag_items.png", dpi=90)
    plt.close(fig)

    # 3. 촬영 연도와 연조직 피크
    fig, a = plt.subplots(figsize=(7, 4))
    a.scatter(d.A_StudyYear + rng.uniform(-0.2, 0.2, len(d)), d.Q_soft_peak_HU, s=14)
    a.set_xlabel("촬영 연도")
    a.set_ylabel("연조직 피크 (HU)")
    a.set_title("촬영 연도별 연조직 피크 HU (2013~2015년이 높음 → 고정 HU 임계값은 연도 보정 필요)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_year.png", dpi=90)
    plt.close(fig)

    # 4. 구조별 영상 경계 접촉 (가장 큰 덩어리 기준)
    touch = pd.DataFrame({s: d[f"F_{s}_touch_main"].fillna("없음").value_counts() for s in STRUCTS}).fillna(0).astype(int)
    touch.to_csv(OUT / "eda_touch.csv", encoding="utf-8-sig")

    # 5. 상기도 마스크 부피 vs 엑셀 기도 계측 (Valid_Main 기준)
    m = pd.read_csv(ROOT / "Result_py/phase1/patient_master.csv", index_col="Patient", encoding="utf-8-sig")
    m = m[m.Valid_Main.astype(bool)]
    u = d.S_UAW_vol_cc.reindex(m.index)
    rows = []
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, v in zip(ax, ("PAV", "NAV", "PMCA", "PAL")):
        s = pd.concat([u, m[v]], axis=1).dropna()
        r, p = pearsonr(s.iloc[:, 0], s.iloc[:, 1])
        rows.append({"excel": v, "n": len(s), "pearson": round(r, 3), "p": round(p, 4), "spearman": round(spearmanr(s.iloc[:, 0], s.iloc[:, 1])[0], 3)})
        a.scatter(s.iloc[:, 0], s.iloc[:, 1], s=12)
        a.set_xlabel("AMASSS 상기도 부피 (cc)")
        a.set_ylabel(f"엑셀 {v}")
        a.set_title(f"{v}: r {r:.2f} (n={len(s)})", fontsize=10)
    fig.suptitle("상기도 마스크 부피 vs 엑셀 기도 계측 (마스크는 Roh 경계로 자르지 않은 전체 부피)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_uaw_vs_excel.png", dpi=90)
    plt.close(fig)
    uv = pd.DataFrame(rows)
    uv.to_csv(OUT / "eda_uaw_vs_excel.csv", index=False, encoding="utf-8-sig")
    print(it.to_string(), "\n", touch.to_string(), "\n", uv.to_string(index=False))


if __name__ == "__main__":
    main()
