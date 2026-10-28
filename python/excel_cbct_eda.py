"""엑셀의 CBCT·구강 관련 변수 기본 EDA (PSG·PCA·예측 모형 제외). 엑셀은 읽기만 함.
변수: CBCT 계측 27개 (Roh 11 + 추가 16), 임상 1개 (TMJ pain), 분류 2개 (Type, Group), 촬영 1개 (Slice 간격), 체격 3개
분포·상관은 주 분석 83명(Valid_Main), 결측은 87명과 83명 둘 다.
상관은 Spearman r과 쌍별 N만 냄 (전체 쌍 p값 없음). 관심 쌍만 따로 표.
결과: Result_py/excel_cbct_eda/ (cbct_excel_*.csv, fig_*.png). 환자는 P_n 번호만.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/excel_cbct_eda"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

# 이름: (엑셀 열, 묶음, 자료형, 단위, 정의 메모). 단위·정의는 엑셀에 없어서 "추정" 또는 "미확인"으로 표기
VARS = {
    "SliceSpacing": (3, "촬영", "연속", "mm", "엑셀 'CBCT Slice 간격'. DICOM 복셀 크기와 대조"),
    "Height": (7, "체격", "연속", "cm", ""), "Weight": (8, "체격", "연속", "kg", ""), "BMI": (9, "체격", "연속", "kg/m²", "키·몸무게와 대조"),
    "ANB": (22, "CBCT 골격", "연속", "°", "Roh"), "Nperp_A": (23, "CBCT 골격", "연속", "mm", "Roh"),
    "Nperp_Pog": (24, "CBCT 골격", "연속", "mm", "Roh"), "MPFH": (25, "CBCT 골격", "연속", "°", "Roh"),
    "NCWR": (26, "CBCT 골격", "연속", "%", "Roh"), "PHI": (27, "CBCT 골격", "연속", "%", "Roh"),
    "TDI": (28, "CBCT 골격", "연속", "mm", "Roh"), "NAV": (29, "CBCT 기도", "연속", "mm³", "Roh"),
    "PAV": (30, "CBCT 기도", "연속", "mm³", "Roh"), "PMCA": (31, "CBCT 기도", "연속", "mm²", "Roh"),
    "PAL": (32, "CBCT 기도", "연속", "mm", "Roh"),
    "OcclusalPlane": (33, "CBCT 골격", "연속(정수)", "° (추정)", "기준 평면 미확인"),
    "ChinDev": (34, "CBCT 골격", "이진", "0/1", "Asymmetry (Chin dev). 방향·기준값 미확인"),
    "TMJpain": (35, "임상", "이진", "0/1", "TMJ pain history (No 0, Yes 1). 영상 계측 아님"),
    "SeptalDev": (36, "CBCT 비상악", "이진", "0/1", "Septal deviation. 판정 기준 미확인"),
    "NasalFloorCant": (37, "CBCT 비상악", "이진", "0/1", "Nasal floor cant. 판정 기준 미확인"),
    "NasalObstruction": (38, "CBCT 비상악", "순서(0/1/2)", "0/1/2", "Nasal obstruction. 등급 의미·출처(영상/증상) 미확인"),
    "MaxSinusitis": (39, "CBCT 비상악", "이진", "0/1", "Maxillary sinusitis. 판정 기준 미확인"),
    "Type": (40, "분류", "범주(1~4)", "코드", "Type (Group분류). 정의 미확인 → 해석 금지, 빈도만"),
    "CondAx_R": (41, "CBCT 과두", "연속", "° (추정)", "Condylar angle Axial Rt"), "CondAx_L": (42, "CBCT 과두", "연속", "° (추정)", "Axial Lt"),
    "CondLat_R": (43, "CBCT 과두", "연속", "° (추정)", "Lat Rt"), "CondLat_L": (44, "CBCT 과두", "연속", "° (추정)", "Lat Lt"),
    "JS_Ant_R": (45, "CBCT 과두", "연속", "mm (추정)", "Joint space Rt Ant"), "JS_Sup_R": (46, "CBCT 과두", "연속", "mm (추정)", "Rt Sup"),
    "JS_Post_R": (47, "CBCT 과두", "연속", "mm (추정)", "Rt Post"), "JS_Ant_L": (48, "CBCT 과두", "연속", "mm (추정)", "Lt Ant"),
    "JS_Sup_L": (49, "CBCT 과두", "연속", "mm (추정)", "Lt Sup"), "JS_Post_L": (50, "CBCT 과두", "연속", "mm (추정)", "Lt Post"),
    "Group": (51, "분류", "범주(1/2)", "코드", "'CII HA : 1 CII HA (x): 2'. 정의 미확인 → 해석 금지, 빈도만"),
}
CBCT = [k for k, v in VARS.items() if v[1].startswith("CBCT")]
CONT = [k for k, v in VARS.items() if v[2].startswith("연속")]
PAIRS = [("CondAx_R", "CondAx_L"), ("CondLat_R", "CondLat_L"), ("JS_Ant_R", "JS_Ant_L"), ("JS_Sup_R", "JS_Sup_L"), ("JS_Post_R", "JS_Post_L")]
FOCUS = [("OcclusalPlane", "MPFH"), ("SeptalDev", "NasalObstruction"), ("NasalFloorCant", "ChinDev"),
         ("MaxSinusitis", "NasalObstruction"), ("SeptalDev", "NasalFloorCant")]


def med_iqr(x):
    x = x.dropna()
    return f"{x.median():.2f} [{x.quantile(.25):.2f}–{x.quantile(.75):.2f}]" if len(x) else ""


def load():
    x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None).iloc[3:]
    x = x[x[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    d = pd.DataFrame({k: pd.to_numeric(x[v[0]], errors="coerce").values for k, v in VARS.items()},
                     index="P_" + x[0].astype(int).astype(str))
    m = pd.read_csv(ROOT / "Result_py/phase1/patient_master.csv", index_col="Patient", encoding="utf-8-sig")
    e = pd.read_csv(ROOT / "Result_py/cbct_eda/cbct_eda_patient.csv", index_col=0, encoding="utf-8-sig")
    assert np.allclose(d.loc[m.index, "ANB"], m.ANB) and np.allclose(d.loc[m.index, "BMI"], m.BMI)   # phase1과 같은 열
    return d, m.Valid_Main.astype(bool).reindex(d.index).fillna(False), e


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, main_, e = load()
    d83 = d[main_]
    assert len(d) == 87 and len(d83) == 83

    # 1. 데이터 사전
    pd.DataFrame([{"var": k, "excel_col": v[0], "group": v[1], "dtype": v[2], "unit": v[3], "note": v[4],
                   "values": ", ".join(map(str, sorted(d[k].dropna().unique()))) if d[k].nunique() <= 5 else ""}
                  for k, v in VARS.items()]).to_csv(OUT / "cbct_excel_dictionary.csv", index=False, encoding="utf-8-sig")

    # 2. 결측 + NAV·PAV 결측 패턴
    pd.DataFrame({"missing_87": d.isna().sum(), "missing_83": d83.isna().sum(),
                  "pct_83": (100 * d83.isna().mean()).round(1)}).to_csv(OUT / "cbct_excel_missingness.csv", encoding="utf-8-sig")
    yr = e.A_StudyYear.reindex(d83.index)
    uaw = e.S_UAW_vol_cc.reindex(d83.index)
    pat = pd.crosstab(d83.NAV.isna().map({True: "NAV 결측", False: "NAV 있음"}), d83.PAV.isna().map({True: "PAV 결측", False: "PAV 있음"}))
    rows = []
    for tgt in ("NAV", "PAV"):
        miss = d83[tgt].isna()
        for v, s in [*[(c, d83[c]) for c in CONT if c not in ("NAV", "PAV", "SliceSpacing", "Height", "Weight")],
                     ("촬영 연도 (DICOM)", yr), ("AMASSS 상기도 부피 cc", uaw)]:
            a, b = s[~miss].dropna(), s[miss].dropna()
            rows.append({"target": tgt, "var": v, "measured": med_iqr(a), "missing": med_iqr(b), "n_meas": len(a), "n_miss": len(b),
                         "p_MW": round(mannwhitneyu(a, b).pvalue, 3) if len(a) and len(b) else np.nan})
    mp = pd.DataFrame(rows)
    with open(OUT / "cbct_excel_missing_pattern.csv", "w", encoding="utf-8-sig", newline="") as f:
        pat.to_csv(f)
        f.write("\n")
        mp.to_csv(f, index=False)
    combo = np.select([d83.NAV.notna() & d83.PAV.notna(), d83.NAV.notna()], ["NAV·PAV 둘 다 있음", "NAV만 있음"], "둘 다 결측")
    yr_tab = pd.crosstab(yr.rename("촬영 연도").astype("Int64"), pd.Series(combo, index=d83.index, name=""))

    # 3. 기술통계 (연속) + 빈도 (범주)
    rows = []
    for k in CONT:
        s = d83[k].dropna()
        q1, q3 = s.quantile([.25, .75])
        rz = (s - s.median()) / (1.4826 * (s - s.median()).abs().median() or np.nan)
        rows.append({"var": k, "group": VARS[k][1], "n": len(s), "mean": s.mean(), "sd": s.std(), "median": s.median(), "q1": q1, "q3": q3,
                     "min": s.min(), "max": s.max(), "skew": s.skew(),
                     "out_IQR1.5": int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum()), "out_robust_z3.5": int((rz.abs() > 3.5).sum())})
    desc = pd.DataFrame(rows).round(3)
    desc.to_csv(OUT / "cbct_excel_descriptive.csv", index=False, encoding="utf-8-sig")
    cat = [k for k in VARS if k not in CONT]
    freq = pd.DataFrame([{"var": k, "value": int(v), "n": int(n), "pct": round(100 * n / len(d83), 1)}
                         for k in cat for v, n in d83[k].value_counts().sort_index().items()])
    freq.to_csv(OUT / "cbct_excel_frequency.csv", index=False, encoding="utf-8-sig")

    # 4. 좌우 과두
    rows = []
    for r, l in PAIRS:
        diff = d83[r] - d83[l]
        rho = spearmanr(d83[r], d83[l])[0]
        a0, a1 = diff.abs()[d83.ChinDev == 0], diff.abs()[d83.ChinDev == 1]
        rows.append({"pair": f"{r}/{l}", "n": int(diff.notna().sum()), "median_R": d83[r].median(), "median_L": d83[l].median(),
                     "spearman_R_L": round(rho, 3), "diff_R-L": med_iqr(diff), "p_wilcoxon_R=L": round(wilcoxon(d83[r], d83[l]).pvalue, 3),
                     "abs_diff": med_iqr(diff.abs()), "abs_diff_ChinDev0": med_iqr(a0), "abs_diff_ChinDev1": med_iqr(a1),
                     "p_MW_ChinDev": round(mannwhitneyu(a0, a1).pvalue, 3),
                     "정의 일관성 단서": "R·L 양의 상관, 중앙값 비슷 → 같은 정의 가능성" if rho > 0.3 and abs(diff.median()) < 0.2 * abs(d83[r].median())
                     else "확인 필요"})
    bil = pd.DataFrame(rows)
    bil.round(3).to_csv(OUT / "cbct_excel_bilateral.csv", index=False, encoding="utf-8-sig")

    # 5. 상관 (Spearman, 쌍별 N) + 관심 쌍
    X = d83[CBCT]
    rho = X.corr(method="spearman")
    nn = X.notna().astype(int).T @ X.notna().astype(int)
    iu = np.triu_indices(len(CBCT), 1)
    corr = pd.DataFrame({"var1": rho.index[iu[0]], "var2": rho.columns[iu[1]], "spearman": rho.values[iu].round(3), "n": nn.values[iu]})
    corr = corr.reindex(corr.spearman.abs().sort_values(ascending=False).index)
    rows = []
    for a, b in FOCUS:
        s = d83[[a, b]].dropna()
        r_, p_ = spearmanr(s[a], s[b])
        tab = pd.crosstab(s[a], s[b]) if s[a].nunique() <= 3 and s[b].nunique() <= 3 else None
        rows.append({"var1": a, "var2": b, "n": len(s), "spearman": round(r_, 3), "p_spearman": round(p_, 3),
                     "p_fisher_2x2": round(fisher_exact(tab)[1], 3) if tab is not None and tab.shape == (2, 2) else np.nan,
                     "crosstab": "" if tab is None else "; ".join(f"{a}={i},{b}={j}: {tab.loc[i, j]}" for i in tab.index for j in tab.columns)})
    focus = pd.DataFrame(rows)
    with open(OUT / "cbct_excel_correlations.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("# 관심 쌍 (탐색용 p, 다중비교 보정 없음)\n")
        focus.to_csv(f, index=False)
        f.write("\n# 전체 쌍 (Spearman, 쌍별 N, |r| 큰 순)\n")
        corr.to_csv(f, index=False)

    # 6. QC: 불가능한 값 / 엑셀 내부·DICOM 불일치 / 극단값 (87명 전원, 극단값은 83명 분포 기준)
    qc = []
    add = lambda p, v, kind, why: qc.append({"patient": p, "var": v, "kind": kind, "detail": why})
    for k in [c for c in d if c.startswith("JS_")]:
        for p in d.index[d[k] <= 0]:
            add(p, k, "불가능", f"관절 공간 {d.loc[p, k]} ≤ 0")
    for k in ("NCWR", "PHI"):
        for p in d.index[(d[k] < 0) | (d[k] > 100)]:
            add(p, k, "불가능", f"비율 {d.loc[p, k]} 범위 밖")
    bmi = d.Weight / (d.Height / 100) ** 2
    for p in d.index[(bmi - d.BMI).abs() > 0.5]:
        add(p, "BMI", "불일치", f"엑셀 BMI {d.loc[p, 'BMI']:.1f} vs 키·몸무게 계산 {bmi[p]:.1f}")
    vox = e.A_voxel_mm.reindex(d.index)
    for p in d.index[vox.notna() & ((vox - d.SliceSpacing).abs() > 0.011)]:
        add(p, "SliceSpacing", "불일치", f"엑셀 {d.loc[p, 'SliceSpacing']} vs DICOM 복셀 {vox[p]}")
    for k in [c for c in CONT if c not in ("SliceSpacing", "Height", "Weight")]:
        s = d83[k].dropna()
        mad = 1.4826 * (s - s.median()).abs().median()
        rz = (d[k] - s.median()) / mad
        for p in d.index[rz.abs() > 3.5]:
            add(p, k, "극단", f"{d.loc[p, k]} (robust z {rz[p]:.1f}, 실제 환자일 수 있음)")
    q = pd.DataFrame(qc, columns=["patient", "var", "kind", "detail"])
    q.to_csv(OUT / "cbct_excel_qc.csv", index=False, encoding="utf-8-sig")

    # 그림 4개
    cont = [c for c in CONT if VARS[c][1].startswith("CBCT")]
    n = len(cont)
    fig, ax = plt.subplots(int(np.ceil(n / 6)), 6, figsize=(18, 2.6 * np.ceil(n / 6)))
    for a, k in zip(ax.ravel(), cont):
        a.hist(d83[k].dropna(), bins=20, color="steelblue")
        a.set_title(f"{k} ({VARS[k][3]}) n={d83[k].notna().sum()}", fontsize=9)
    for a in ax.ravel()[n:]:
        a.axis("off")
    fig.suptitle("엑셀 CBCT 연속 변수 분포 (주 분석 83명)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_distributions.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(1, 5, figsize=(18, 3.8))
    for a, (r, l) in zip(ax, PAIRS):
        for cd, col in ((0, "steelblue"), (1, "tomato")):
            s = d83[d83.ChinDev == cd]
            a.scatter(s[r], s[l], s=14, c=col, label=f"ChinDev={cd}")
        lo, hi = np.nanmin(d83[[r, l]].values), np.nanmax(d83[[r, l]].values)
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set_xlabel(r)
        a.set_ylabel(l)
        a.set_aspect("equal")
    ax[0].legend(fontsize=8)
    fig.suptitle("과두 좌우 비교 (점선 = 좌우 같음)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_bilateral.png", dpi=90)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(13, 11))
    im = a.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(CBCT)), CBCT, rotation=90, fontsize=8)
    a.set_yticks(range(len(CBCT)), CBCT, fontsize=8)
    for i in range(len(CBCT)):
        for j in range(len(CBCT)):
            if i != j and abs(rho.values[i, j]) >= 0.3:
                a.text(j, i, f"{rho.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, shrink=0.7, label="Spearman r")
    a.set_title("엑셀 CBCT 변수 간 Spearman 상관 (83명, |r|≥0.3만 숫자 표시)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_correlation.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    mc = d83.isna().sum()
    mc = mc[mc > 0]
    ax[0].bar(mc.index, mc.values, color="gray")
    ax[0].set_title("결측 수 (83명 중)")
    yr_tab.plot(kind="bar", stacked=True, ax=ax[1])
    ax[1].set_title("촬영 연도별 NAV·PAV 측정 여부 (83명)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_missingness.png", dpi=90)
    plt.close(fig)

    print(f"CBCT 계측 {len(CBCT)}개, 연속 {len(CONT)}개, 범주 {len(cat)}개")
    print("\n[결측 83명]\n" + d83.isna().sum()[lambda s: s > 0].to_string())
    print("\n[NAV×PAV]\n" + pat.to_string())
    print("\n[NAV·PAV 결측 vs 측정 (p<0.1)]\n" + mp[mp.p_MW < 0.1].to_string(index=False))
    print("\n[연도별]\n" + yr_tab.to_string())
    print("\n[기술통계]\n" + desc.drop(columns="group").to_string(index=False))
    print("\n[빈도]\n" + freq.to_string(index=False))
    print("\n[좌우]\n" + bil.drop(columns=["median_R", "median_L"]).to_string(index=False))
    print("\n[관심 쌍]\n" + focus.to_string(index=False))
    print("\n[상관 |r|≥0.4]\n" + corr[corr.spearman.abs() >= 0.4].to_string(index=False))
    print("\n[QC]\n" + q.kind.value_counts().to_string() + "\n" + q.to_string(index=False))


if __name__ == "__main__":
    main()
