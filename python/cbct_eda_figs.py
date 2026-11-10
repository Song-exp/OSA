"""CBCT 기초 EDA(cbct_eda.py 결과)의 정량 그림과 요약표. 결과표만 읽음 (영상은 다시 읽지 않음).
  분석 ①~⑥별 그림 (빨간 점 = 검수 표시, 제외 아님 / 점선 = 표시 기준)
  fig_1_A.png : ① 촬영 조건 (다수값과 다르면 표시: 장비·관전압·복셀·행렬. 촬영 범위는 로버스트 z)
  fig_2_Q.png : ② 영상 품질 (로버스트 z |z|>3.5 = 중앙값 ± 3.5×MAD)
  fig_3_P.png : ③ 자세·교합 (로버스트 z)
  fig_4_F.png : ④ 구조별 영상 경계 접촉 (가장 큰 덩어리 기준, 하악·상악만 표시 규칙)
  fig_5_S.png : ⑤ 분할 품질 (부피·덩어리 수는 로버스트 z, 가장 큰 덩어리 비율은 0.95 미만)
  fig_uaw_vs_excel.png: ⑥ AMASSS 상기도 마스크 부피 vs 엑셀 기도 계측값
  fig_flag_items.png  : 검수 표시 항목별 환자 수 / fig_year.png : 촬영 연도별 연조직 피크 HU
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
NAMES = {"MAND": "하악", "MAX": "상악", "UAW": "상기도", "CB": "두개저"}
# (열, 그림 제목). 제목의 설명은 docs/3단계_CBCT기초_EDA.md 3-0b와 같음
GROUPS = {
    "fig_1_A.png": ("① 촬영 조건 (A_)", [
        ("A_XRayTubeCurrent", "관전류 (mA)"), ("A_Exposure", "노출량 (mAs)"), ("A_voxel_mm", "복셀 크기 (mm)"),
        ("A_NFiles", "슬라이스 수"), ("A_FOV_x_mm", "촬영 범위 좌우 (mm)"), ("A_FOV_z_mm", "촬영 범위 위아래 (mm)"),
        ("A_StudyYear", "촬영 연도")]),
    "fig_2_Q.png": ("② 영상 품질 (Q_)", [
        ("Q_noise_SD", "잡음 SD (HU)"), ("Q_soft_peak_HU", "연조직 피크 (HU)"), ("Q_metal_vox", "금속 복셀 수 (≥3000 HU)"),
        ("Q_sat3071_frac", "포화 비율 (=3071 HU)"), ("Q_MAND_bone_p99_HU", "하악 뼈 99백분위 (HU)"),
        ("Q_metal_jaw_frac", "턱 부위 금속 비율")]),
    "fig_3_P.png": ("③ 자세·교합 (P_)", [
        ("P_midplane_tilt_deg", "정중면 좌우 기울기 (°)"), ("P_SN_pitch_deg", "S–N 앞뒤 기울기 (°)"),
        ("P_palatal_pitch_deg", "구개면 앞뒤 기울기 (°)"), ("P_gap_min_mm", "상·하악 최소 간격 (mm)"),
        ("P_gap_incisor_mm", "앞니 부위 상·하악 간격 (mm)")]),
    "fig_5_S.png": ("⑤ 분할 품질 (S_)", [(f"S_{s}_{k}", f"{NAMES[s]} {t}") for k, t in
                                     (("vol_cc", "부피 (cc)"), ("ncomp", "덩어리 수"), ("largest_frac", "가장 큰 덩어리 비율"))
                                     for s in STRUCTS]),
}
MODE_RULE = ("A_voxel_mm", "A_NFiles")          # 다수값 비교로 표시되는 항목 (cbct_eda.py: 복셀·행렬). 행렬 = 슬라이스 수로 대신 보여 줌


def box_group(d, fl, title, cols, path, ncol=None):
    ncol = ncol or min(len(cols), 4 if len(cols) % 4 == 0 else 6 if len(cols) > 6 else len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for a, (c, name) in zip(ax.ravel(), cols):
        x = d[c].astype(float)
        item = "A_matrix" if c == "A_NFiles" else c
        flagged = d.index.isin(fl.Patient[fl.Item == item])
        a.boxplot(x.dropna(), widths=0.5, showfliers=False)
        j = 1 + rng.uniform(-0.15, 0.15, len(x))
        a.scatter(j[~flagged], x[~flagged], s=8, c="steelblue", alpha=0.6)
        a.scatter(j[flagged], x[flagged], s=16, c="red")
        for p in d.index[flagged][:4]:
            a.annotate(p, (j[d.index.get_loc(p)] + 0.05, x[p]), fontsize=7, color="red")
        if c.endswith("largest_frac"):
            a.axhline(0.95, color="tomato", ls="--", lw=1)
            rule = "기준 < 0.95"
        elif c in MODE_RULE:
            rule = "기준: 다수값과 다름"
        elif c.startswith(("Q_", "P_", "S_", "A_FOV")):
            mad = 1.4826 * (x - x.median()).abs().median()
            for y in (x.median() - 3.5 * mad, x.median() + 3.5 * mad) if mad > 0 else ():
                a.axhline(y, color="tomato", ls="--", lw=1)
            rule = ("기준 |z|>3.5" if mad > 0 else
                    "반올림 전 값으로 |z|>3.5" if flagged.any() else "MAD 0 → z 계산 안 됨")   # 결과표는 소수점 셋째 자리 반올림
        else:
            rule = "표시 규칙 없음 (기술용)"
        a.set_xticks([])
        a.set_title(f"{name}\n{c} · 표시 {flagged.sum()}명 · {rule}", fontsize=8)
    for a in ax.ravel()[len(cols):]:
        a.axis("off")
    fig.suptitle(f"{title} — {len(d)}명. 빨간 점 = 검수 표시 (제외 아님), 점선 = 표시 기준", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    d = pd.read_csv(OUT / "cbct_eda_patient.csv", index_col=0, encoding="utf-8-sig")
    fl = pd.read_csv(OUT / "cbct_eda_flags.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(0)

    # 1. 분석 ①②③⑤ 박스플롯 (④는 4번, ⑥은 5번)
    const = {c: d[c].iloc[0] for c in ("A_Manufacturer", "A_ManufacturerModelName", "A_KVP", "A_ExposureTime") if d[c].nunique() == 1}
    for f, (title, cols) in GROUPS.items():
        if f == "fig_1_A.png":
            title += f" · 83명 모두 같음: {const['A_ManufacturerModelName']}, {const['A_KVP']:.0f} kVp, {const['A_ExposureTime'] / 1000:.0f} s"
        box_group(d, fl, title, cols, OUT / f, ncol=4 if f == "fig_5_S.png" else None)

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
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, (col, t) in zip(ax, (("_touch_main", "가장 큰 덩어리 기준 (표시 규칙에 씀)"), ("_touch", "모든 덩어리 기준 (참고, v1 방식)"))):
        tb = pd.DataFrame({NAMES[s]: d[f"F_{s}{col}"].astype(object).fillna("").apply(lambda v: v if v else "없음").value_counts()
                           for s in STRUCTS}).fillna(0).astype(int).T
        tb = tb[sorted(tb.columns, key=lambda c: (c != "없음", c))]           # 없음을 맨 아래, 방향별 색 고정 (두 패널 같게)
        pal = {"없음": "lightgray", "아래": "tab:red", "뒤": "tab:blue", "위": "tab:cyan"}
        tb.plot(kind="bar", stacked=True, ax=a, rot=0,
                color=[pal.get(c, plt.get_cmap("Set2")(i)) for i, c in enumerate(tb.columns)])
        a.set_title(t, fontsize=10)
        a.set_ylabel("환자 수")
        a.legend(title="닿는 방향", fontsize=8)
    fig.suptitle(f"④ 구조별 영상 경계 접촉 ({len(d)}명). 하악·상악이 닿으면 검수 표시 (0명). "
                 "상기도 아래(전원)·두개저는 코호트 특성으로 보고", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_4_F.png", dpi=90)
    plt.close(fig)

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
    fig.suptitle("⑥ 상기도 마스크 부피 vs 엑셀 기도 계측 (주 분석 환자, 마스크는 Roh 경계로 자르지 않은 전체 부피)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_uaw_vs_excel.png", dpi=90)
    plt.close(fig)
    uv = pd.DataFrame(rows)
    uv.to_csv(OUT / "eda_uaw_vs_excel.csv", index=False, encoding="utf-8-sig")
    print(it.to_string(), "\n", touch.to_string(), "\n", uv.to_string(index=False))


if __name__ == "__main__":
    main()
