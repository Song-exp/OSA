"""1단계(엑셀 EDA) 그림 전부. 대상은 87명 - P_39(폴더의 CBCT가 본인 영상 아님) = 86명. Class III 3명도 넣고 주황색으로 표시.
상관은 Spearman 하나로 통일. 기준선 M0 계수만 Roh 주 분석 83명(Valid_Main, Class III 제외) 모형 결과.
엑셀 CBCT 표(csv)는 excel_cbct_eda.py, M0는 phase1_eda.py가 만듦. 여기서는 그림만 다시 그림.
입력: 엑셀 (excel_cbct_eda.load), phase1/patient_master.csv, phase1/M0_baseline_results.csv, cbct_eda/cbct_eda_patient.csv
결과: Result_py/stage1_figs/ (fig_*.png, spearman_all.csv). 환자는 P_n 번호만.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from excel_cbct_eda import CBCT, CONT, PAIRS, ROH_VARS, ROOT, VARS, boxplots, load

OUT = ROOT / "Result_py/stage1_figs"
PSG = ["AHI", "RDI", "ODI", "LSaO2"]
C, C2, C3 = "steelblue", "tomato", "darkorange"   # 기본, 강조, Class III


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)


def bar(a, s, title):
    v = s.value_counts().sort_index()
    a.bar([str(k) for k in v.index], v.values, color=C)
    for i, n in enumerate(v.values):
        a.text(i, n, str(n), ha="center", va="bottom", fontsize=8)
    a.set_title(f"{title} (n={s.notna().sum()})", fontsize=9)


def scatter_rho(a, x, y, xl, yl, col):
    g = pd.concat([x, y], axis=1).dropna()
    rho, p = spearmanr(g.iloc[:, 0], g.iloc[:, 1])
    a.scatter(g.iloc[:, 0], g.iloc[:, 1], s=12, c=col.reindex(g.index), alpha=0.75)
    a.set_title(f"ρ={rho:.2f}, p={p:.3f} (n={len(g)})", fontsize=9)
    a.set_xlabel(xl)
    a.set_ylabel(yl)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, _, e = load()
    m = pd.read_csv(ROOT / "Result_py/phase1/patient_master.csv", index_col="Patient", encoding="utf-8-sig")
    keep = m.index[m.CBCT_Identity != "mismatch"]   # P_39만 빠짐. P_36·P_59(등록번호 일치)는 포함
    d, m = d.loc[keep], m.loc[keep]
    assert len(m) == 86
    c3 = list(m.index[m.Sag == "III"])
    col = pd.Series(np.where(m.Sag == "III", C3, C), index=m.index)
    plain = pd.Series(C, index=m.index)   # Class III 표시가 의미 없는 그림용
    N = f"{len(m)}명, 주황 = Class III {len(c3)}명"
    yr = e.A_StudyYear.reindex(m.index)

    # 인적·PSG·검사 간격 기술통계 (엑셀 CBCT 변수는 excel_cbct_eda의 cbct_excel_descriptive.csv)
    s = pd.concat([m[["Age", "BMI", "ESS"] + PSG], m.GapDays.abs().rename("AbsGapDays")], axis=1)
    t = s.describe().T.assign(skew=s.skew(), missing=s.isna().sum())
    t.loc["Male_n", "count"], t.loc["AbsGap_gt365_n", "count"] = m.Male.sum(), (s.AbsGapDays > 365).sum()
    for k, v in m.Severity.value_counts().items():
        t.loc[f"Severity {k}", "count"] = v
    t.round(2).to_csv(OUT / "stage1_summary.csv", encoding="utf-8-sig")

    # 1-0 결측: 변수별 결측 수, 촬영 연도별 NAV·PAV 측정 여부
    x = pd.concat([d, m[["ESS"]]], axis=1)
    mc = x.isna().sum()
    mc = mc[mc > 0]
    combo = pd.Series(np.select([d.NAV.notna() & d.PAV.notna(), d.NAV.notna()], ["NAV·PAV 둘 다 있음", "NAV만 있음"], "둘 다 결측"),
                      index=m.index, name="")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(mc.index, mc.values, color="gray")
    for i, n in enumerate(mc.values):
        ax[0].text(i, n, str(n), ha="center", va="bottom", fontsize=8)
    ax[0].set_title(f"결측 수 ({len(m)}명 중, 나머지 변수는 결측 0)", fontsize=10)
    pd.crosstab(yr.rename("촬영 연도").astype("Int64"), combo).plot(kind="bar", stacked=True, ax=ax[1])
    ax[1].set_title(f"촬영 연도별 NAV·PAV 측정 여부 (촬영 연도 있는 {yr.notna().sum()}명)", fontsize=10)
    ax[1].legend(fontsize=7)
    save(fig, "fig_1-0_missingness.png")

    # 1-3 NAV 측정군 vs 결측군. 위: 해부학 차이 없는 5개, 아래: 차이 나는 NCWR과 NCWR–연도의 계단 변화
    # PAV(측정군이 NAV 측정군의 일부)는 cbct_excel_missing_pattern.csv에만 둠
    miss = d.NAV.isna()
    fig, ax = plt.subplot_mosaic("ABCDE\nFFGGG", figsize=(15, 8))
    panels = {"A": ("AMASSS 상기도 부피 (cc)", e.S_UAW_vol_cc.reindex(m.index)), "B": ("ANB (°)", d.ANB),
              "C": ("MP-FH (°)", d.MPFH), "D": ("PMCA (mm²)", d.PMCA), "E": ("PAL (mm)", d.PAL), "F": ("NCWR (%)", d.NCWR)}
    for key, (k, s) in panels.items():
        a = ax[key]
        g = [s[~miss].dropna(), s[miss].dropna()]
        a.boxplot(g, widths=0.5, showfliers=False)
        for j, y in enumerate(g, 1):
            a.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(y)), y, s=10, c=C, alpha=0.7)
        p = mannwhitneyu(*g).pvalue
        a.set_xticks([1, 2], [f"NAV 측정\n(n={len(g[0])})", f"NAV 결측\n(n={len(g[1])})"], fontsize=8)
        a.set_title(f"{k}: p={p:.3f}", fontsize=10, color=C2 if p < 0.05 else "black")
    a = ax["G"]
    for flag, cc, lab in ((False, C, "NAV 측정"), (True, C2, "NAV 결측")):
        g = pd.concat([yr, d.NCWR], axis=1)[miss == flag].dropna()
        a.scatter(g.iloc[:, 0], g.iloc[:, 1], s=14, c=cc, alpha=0.75, label=f"{lab} (n={len(g)})")
    a.set_xlabel("촬영 연도 (DICOM)")
    a.set_ylabel("NCWR (%)")
    a.set_title("NCWR vs 촬영 연도", fontsize=10)
    a.legend(fontsize=8)
    fig.suptitle(f"1-3 NAV 측정군 vs 결측군 ({len(m)}명, Mann-Whitney, 탐색용)\n"
                 "위: 해부학 변수는 차이 없음 → 결측은 측정 시기 문제. 아래: NCWR만 측정 시기에 따라 갈림")
    save(fig, "fig_1-3_nav_pav_missing_compare.png")

    # 1-0 상관 (Spearman 하나로): 인적 3 + CBCT 25 + PSG 4, 쌍별 N
    X = pd.concat([m[["Age", "Male", "BMI"]], d[CBCT], m[PSG]], axis=1)
    rho = X.corr(method="spearman")
    n = X.notna().astype(int).T @ X.notna().astype(int)
    iu = np.triu_indices(len(rho), 1)
    pd.DataFrame({"var1": rho.index[iu[0]], "var2": rho.columns[iu[1]], "spearman": rho.values[iu].round(3),
                  "n": n.values[iu]}).to_csv(OUT / "spearman_all.csv", index=False, encoding="utf-8-sig")
    fig, a = plt.subplots(figsize=(14, 12))
    im = a.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(rho)), rho.columns, rotation=90, fontsize=8)
    a.set_yticks(range(len(rho)), [f"{c} (n={X[c].notna().sum()})" for c in rho.index], fontsize=8)
    for i, j in zip(*np.where((np.abs(rho.values) >= 0.3) & ~np.eye(len(rho), dtype=bool))):
        a.text(j, i, f"{rho.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
    for b in (3, 3 + len(CBCT)):   # 묶음 경계: 인적 | CBCT | PSG
        a.axhline(b - 0.5, color="k", lw=1)
        a.axvline(b - 0.5, color="k", lw=1)
    fig.colorbar(im, shrink=0.7, label="Spearman ρ")
    a.set_title(f"변수 간 Spearman 상관 ({len(m)}명, Class III 포함, 쌍별 결측 제외, |ρ|≥0.3만 숫자). 인적 | CBCT 25 | PSG")
    save(fig, "fig_1-0_spearman_all.png")

    # 1-1 인적·체격: 나이, 성별, BMI(극단 표시), 엑셀 BMI vs 키·몸무게 재계산
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    ax[0].hist(m.Age, bins=15, color=C, edgecolor="white")
    ax[0].set_title(f"나이 (세) n={m.Age.notna().sum()}", fontsize=9)
    bar(ax[1], m.Male.map({1: "남", 0: "여"}), "성별")
    s = m.BMI
    rz = (s - s.median()) / (1.4826 * (s - s.median()).abs().median())
    ax[2].boxplot(s, widths=0.5, showfliers=False)
    ax[2].scatter(1 + np.random.default_rng(0).uniform(-0.15, 0.15, len(s)), s, s=10, c=C)
    for p in s.index[rz.abs() > 3.5]:
        ax[2].annotate(p, (1.18, s[p]), fontsize=7, color=C2)
    ax[2].set_xticks([])
    ax[2].set_title("BMI (kg/m²), 이름 표시 = robust z>3.5", fontsize=9)
    calc = d.Weight / (d.Height / 100) ** 2
    ax[3].scatter(calc, d.BMI, s=12, c=C)
    lo, hi = calc.min(), calc.max()
    ax[3].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[3].set_xlabel("키·몸무게로 계산한 BMI")
    ax[3].set_ylabel("엑셀 BMI")
    ax[3].set_title(f"BMI 대조: |차이|>0.5 {int(((calc - d.BMI).abs() > 0.5).sum())}건 (최대 {(calc - d.BMI).abs().max():.2f})", fontsize=9)   # excel_cbct_eda QC와 같은 기준
    fig.suptitle(f"1-1 인적·체격 변수 ({len(m)}명)")
    save(fig, "fig_1-1_demographics.png")

    # 1-2·1-3 Roh 변수 박스플롯 (기준선), 1-4 추가 연속 변수 박스플롯
    boxplots(d, ROH_VARS, OUT / "fig_1-2_1-3_box_roh.png",
             f"Roh 2025 CBCT 변수 ({N}). 빨간 점선 = 논문 분류 기준 (ANB·MP-FH·TDI만 있음, -2.5~-2는 TSD 경계 구간)", hl=c3)
    added = [k for k in CONT if VARS[k][1].startswith("CBCT") and k not in ROH_VARS]
    boxplots(d, added, OUT / "fig_1-4_box_added.png", f"Roh 논문에 없는 추가 CBCT 연속 변수 ({len(m)}명). 기준치 없음")

    # 1-2 Roh 분류 빈도 + NCWR vs 촬영 연도
    tdi = pd.cut(m.TDI, [-np.inf, -2.5, -2, np.inf], right=False, labels=["TSD <-2.5", "경계", "non-TSD"])
    ver = pd.cut(m.MPFH, [-np.inf, 25, 30, np.inf], right=False, labels=["hypo <25", "normo", "hyper ≥30"])
    sag = pd.Categorical(m.Sag.map({"III": "Class III", "I": "Class I", "II": "Class II"}), ["Class III", "Class I", "Class II"])
    sag = pd.Series(sag, index=m.index)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    bar(ax[0], sag, "시상 (ANB)")
    bar(ax[1], ver, "수직 (MP-FH)")
    bar(ax[2], tdi, "횡방향 (TDI)")
    scatter_rho(ax[3], yr, d.NCWR, "촬영 연도 (DICOM)", "NCWR (%)", plain)
    fig.suptitle(f"1-2 Roh 기준 분류 빈도와 NCWR–촬영 연도 ({len(m)}명)")
    save(fig, "fig_1-2_roh_class_ncwr.png")

    # 1-4 과두 좌우 비교
    fig, ax = plt.subplots(1, 5, figsize=(18, 4.4))
    for a, (r, l) in zip(ax, PAIRS):
        for cd, cc in ((0, C), (1, C2)):
            g = d[d.ChinDev == cd]
            a.scatter(g[r], g[l], s=14, c=cc, label=f"ChinDev={cd}")
        lo, hi = np.nanmin(d[[r, l]].values), np.nanmax(d[[r, l]].values)
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set_xlabel(r)
        a.set_ylabel(l)
        a.set_aspect("equal")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"1-4 과두 좌우 비교 ({len(m)}명, 점선 = 좌우 같음)")
    save(fig, "fig_1-4_bilateral.png")

    # 1-4 본문에 나온 관심 쌍
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    scatter_rho(ax[0], d.OcclusalPlane, d.MPFH, "Occlusal plane (° 추정)", "MP-FH (°)", col)
    scatter_rho(ax[1], d.CondAx_R, d.ANB, "Condylar Axial R (°)", "ANB (°)", col)
    scatter_rho(ax[2], d.CondAx_L, d.ANB, "Condylar Axial L (°)", "ANB (°)", col)
    fig.suptitle(f"1-4 추가 변수 관심 쌍 (Spearman, {N})")
    save(fig, "fig_1-4_focus_pairs.png")

    # 범주 변수 빈도: 1-4 추가 CBCT 이진 3개 / 1-5 비계측 6개 (임상, 분류 코드, Slice 간격)
    for cats, name, title in (
            (["ChinDev", "SeptalDev", "NasalFloorCant"], "fig_1-4_binary.png", "1-4 추가 CBCT 이진 변수 빈도"),
            (["TMJpain", "NasalObstruction", "MaxSinusitis", "Type", "Group", "SliceSpacing"], "fig_1-5_noncbct.png",
             "1-5 영상 계측이 아닌 변수 빈도. Type·Group은 정의 미확인 → 빈도만")):
        fig, ax = plt.subplots(1, len(cats), figsize=(3.4 * len(cats), 3.6))
        for a, k in zip(ax, cats):
            bar(a, d[k], f"{k} [{VARS[k][1]}]")
        fig.suptitle(f"{title} ({len(m)}명)")
        save(fig, name)

    # 1-6 PSG 분포: AHI 중증도로 색을 나눈 누적 히스토그램 (중증도 구성은 범례 인원으로). 검사 간격은 문서 수치로만
    sev = ["정상 <5", "경증 5–15", "중등도 15–30", "중증 ≥30"]
    shades = plt.get_cmap("YlOrRd")(np.linspace(0.25, 0.9, len(sev)))   # 순서형이라 밝음 → 진함
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, k in zip(ax, PSG):
        bins = np.histogram_bin_edges(m[k].dropna(), bins=18)
        a.hist([m.loc[m.Severity == s, k].dropna() for s in sev], bins=bins, stacked=True, color=shades,
               edgecolor="white", label=[f"{s} ({(m.Severity == s).sum()}명)" for s in sev])
        a.set_title(f"{k} (n={m[k].notna().sum()})", fontsize=10)
    ax[0].legend(fontsize=8, title="AHI 중증도", title_fontsize=8)
    fig.suptitle(f"1-6 PSG 결과 변수 분포, 색 = AHI 중증도 ({len(m)}명, 정상 대조군 없음)")
    save(fig, "fig_1-6_psg.png")

    # 1-6 CBCT–PSG 검사 간격 (간격 1년 이내 민감도 분석의 근거)
    g = m.GapDays.abs().dropna() / 365.25
    fig, a = plt.subplots(figsize=(8, 3.6))
    a.hist(g, bins=np.arange(0, g.max() + 0.25, 0.25), color=C, edgecolor="white")
    a.axvline(1, color=C2, ls="--", lw=1)
    a.text(1.05, a.get_ylim()[1] * 0.9, f"1년 이내 {(g <= 1).sum()}명 | 1년 초과 {(g > 1).sum()}명", color=C2, fontsize=9)
    a.set_xlabel("|CBCT 촬영일 – PSG 검사일| (년, 3개월 구간)")
    a.set_ylabel("환자 수")
    a.set_title(f"1-6 CBCT–PSG 검사 간격 ({len(g)}명). 중앙값 {g.median() * 12:.1f}개월, 최대 {g.max():.1f}년", fontsize=10)
    save(fig, "fig_1-6_gap.png")

    # 1-6 Roh 재현: 그룹별 AHI, 그룹별 공변량 (나이·BMI·남성 비율). 성별·나이는 엑셀 값 그대로 (정정 전)
    groups = [("성별", m.Male.map({1: "남", 0: "여"})), ("시상", sag), ("수직", ver), ("횡방향", tdi)]
    levels = lambda gr: list(gr.cat.categories) if hasattr(gr, "cat") else sorted(gr.dropna().unique())

    def group_box(a, v, gr):
        lv = levels(gr)
        a.boxplot([v[gr == k] for k in lv], showfliers=False)
        for i, k in enumerate(lv, 1):
            y = v[gr == k]
            a.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(y)), y, s=8, c=col[y.index], alpha=0.7)
        a.set_xticks(range(1, len(lv) + 1), [f"{k}\n(n={(gr == k).sum()})" for k in lv], fontsize=8)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for a, (t, gr) in zip(ax, groups):
        group_box(a, m.AHI, gr)
        a.set_title(t, fontsize=10)
    ax[0].set_ylabel("AHI (/h)")
    fig.suptitle(f"1-6 분류별 AHI ({N}). 단순 비교, Roh 재현 검정은 Class III 제외 83명 (replication/ 표)")
    save(fig, "fig_1-6_replication_ahi.png")

    fig, ax = plt.subplots(3, 4, figsize=(16, 10), sharey="row")
    for j, (t, gr) in enumerate(groups):
        group_box(ax[0, j], m.Age, gr)
        group_box(ax[1, j], m.BMI, gr)
        ax[0, j].set_title(t, fontsize=10)
        a = ax[2, j]
        if t == "성별":
            a.axis("off")
            continue
        lv = levels(gr)
        pct = [100 * m.Male[gr == k].mean() for k in lv]
        a.bar(range(len(lv)), pct, color=C)
        for i, (k, p) in enumerate(zip(lv, pct)):
            a.text(i, p, f"{p:.0f}%\n({int(m.Male[gr == k].sum())}/{(gr == k).sum()})", ha="center", va="bottom", fontsize=8)
        a.set_xticks(range(len(lv)), lv, fontsize=8)
        a.set_ylim(0, 110)
    for i, lab in enumerate(["나이 (세)", "BMI (kg/m²)", "남성 비율 (%)"]):
        ax[i, 0 if i < 2 else 1].set_ylabel(lab)
    ax[2, 1].tick_params(labelleft=True)   # 성별 칸을 비워서 공유 y축 눈금이 사라지는 것 방지
    fig.suptitle(f"1-6 분류별 공변량 ({N}). 분류별 AHI 그림과 같이 읽음. 성별·나이는 엑셀 값 그대로 (정정 전)")
    save(fig, "fig_1-6_group_covariates.png")

    # 1-6 M0 변수 + 공변량(BMI, 나이) vs PSG 산점도 (단순 상관, 보정 없음)
    xs, ys = ["ANB", "MPFH", "TDI", "PMCA", "PAL", "BMI", "Age"], ["AHI", "ODI", "LSaO2"]
    fig, ax = plt.subplots(len(ys), len(xs), figsize=(3 * len(xs), 2.8 * len(ys)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            scatter_rho(ax[i, j], m[xv], m[yv], xv if i == len(ys) - 1 else "", yv if j == 0 else "", col)
    fig.suptitle(f"1-6 엑셀 계측값 vs PSG 지표 (단순 Spearman, 보정 없음, {N})")
    save(fig, "fig_1-6_m0_scatter.png")

    # 1-6 M0 표준화 계수 (Roh 주 분석 83명 모형 그대로, AHI / log AHI)
    r = pd.read_csv(ROOT / "Result_py/phase1/M0_baseline_results.csv", encoding="utf-8-sig")
    r = r[r.Analysis.str.startswith("전체 유효") & (r.Term != "const")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, (dv, t) in zip(ax, r.groupby("DV", sort=False)):
        k = t.B / t.Beta_std   # 원 단위 → 표준화 단위 환산
        y = np.arange(len(t))[::-1]
        a.errorbar(t.Beta_std, y, xerr=[(t.B - t.CI_low) / k, (t.CI_high - t.B) / k], fmt="o",
                   color=C, ecolor="gray", capsize=3)
        for yi, b, p in zip(y, t.Beta_std, t.P):
            a.text(b, yi + 0.2, f"p={p:.3f}", fontsize=7, ha="center", color=C2 if p < 0.05 else "black")
        a.axvline(0, color="k", lw=0.8)
        a.set_yticks(y, t.Term)
        a.set_title(f"{dv}: n={t.N.iloc[0]}, R²={t.R2.iloc[0]:.2f}, adj R²={t.AdjR2.iloc[0]:.2f}", fontsize=9)
        a.set_xlabel("표준화 β (95% CI)")
    fig.suptitle("1-6 기준선 M0 (Roh 주 분석 83명: Class III 3명·P_39 제외)")
    save(fig, "fig_1-6_m0_coef.png")
    print(f"그림 17개 → {OUT} (대상 {len(m)}명, Class III {c3})")


if __name__ == "__main__":
    main()
