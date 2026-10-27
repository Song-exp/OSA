"""Phase 1: 환자 통합표, 기초 통계, 결측, Roh 기준선 회귀(M0), 민감도 분석.

자료: 엑셀(전문가 측정·판독값) + 엑셀 값이 나온 PSG 리포트(검사일, 보조 지표) + CBCT 헤더(촬영일)
      + Result_py/quality_flags.csv (check_quality.py: CBCT 본인 여부, 원신호 검사 일치)
M0  : AHI ~ Age + Sex + BMI + ANB + MPFH + TDI + PMCA + PAL (OLS, 엑셀 값)
민감도: {전체 유효, CBCT–PSG 간격 ≤ 365일, 리포트와 엑셀 PSG 값이 다른 3명 제외} × {AHI, log(AHI+1)}
       1년은 비교용 기준이지 검증된 절단점이 아님. log 분석은 다른 종속변수이므로 따로 보고.
결과: Result_py/phase1/ (CSV, PNG). 검사일이 들어 있으므로 git에 올리지 않음.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from check_quality import dicom_series
from osa.report import parse_report, reports_in

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/phase1"
EXCEL_COLS = {"Age": 6, "BMI": 9, "Sex": 10, "TST": 11, "AHI": 12, "RDI": 13, "ODI": 14, "LSaO2": 15,
              "ANB": 22, "Nperp_A": 23, "Nperp_Pog": 24, "MPFH": 25, "NCWR": 26, "PHI": 27, "TDI": 28,
              "NAV": 29, "PAV": 30, "PMCA": 31, "PAL": 32}
REPORT_COLS = ["AHI_SUP", "AI", "HI", "CAI", "ArI", "ESS"]
M0_X = ["Age", "Male", "BMI", "ANB", "MPFH", "TDI", "PMCA", "PAL"]
STAT_VARS = ["Age", "BMI", "ANB", "Nperp_A", "Nperp_Pog", "MPFH", "TDI", "PHI", "NCWR", "PMCA", "PAL", "NAV", "PAV",
             "AHI", "RDI", "ODI", "LSaO2", "AHI_SUP", "AI", "HI", "CAI", "ArI", "ESS", "GapDays"]
GAP_MAX = 365


# ------------------------------------------------------------ A. 환자 통합표
def build_master() -> pd.DataFrame:
    x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None).iloc[3:]
    x = x[x[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    d = pd.DataFrame({"P": x[0].astype(int).values,
                      **{k: pd.to_numeric(x[c], errors="coerce").values for k, c in EXCEL_COLS.items()}})
    q = pd.read_csv(ROOT / "Result_py/quality_flags.csv").assign(P=lambda t: t.Patient.str[2:].astype(int))
    q = q.set_index("P")
    rows = []
    for p, tst, ahi in zip(d.P, d.TST, d.AHI):
        folder = next((ROOT / "Data/PSG").glob(f"P_{p}_*"))
        reps = [parse_report(f) for f in reports_in(folder)]
        k = next((i for i, r in enumerate(reps) if abs(r["TST"] - tst) <= 0.6), None)
        rep = reps[k] if k is not None else {}
        top = next(s for s in dicom_series(next((ROOT / "Data/CBCT").glob(f"P_{p}_*"))) if s["top"])
        psg_date = pd.Timestamp(rep["Study"]) if rep.get("Study") else pd.NaT
        cbct_date = pd.Timestamp(top["date"]) if top["date"] else pd.NaT
        rows.append({"P": p, "PSG_Format": "Embla(E)" if folder.name.endswith("_E") else "Nox(N)",
                     "N_Reports": len(reps), "PSG_ReportIndex": None if k is None else k + 1,
                     "PSG_Source": "엑셀 전문가 판독 (리포트 확인)" if k is not None else "엑셀 전문가 판독 (리포트 미확인)",
                     "Report_AHI_matches_Excel": bool(k is not None and abs(rep["AHI"] - ahi) <= 0.15),
                     "PSG_Date": psg_date, "CBCT_Date": cbct_date,
                     "GapDays": (cbct_date - psg_date).days if pd.notna(psg_date) and pd.notna(cbct_date) else np.nan,
                     **{c: rep.get(c, np.nan) for c in REPORT_COLS}})
    m = d.merge(pd.DataFrame(rows), on="P")
    m["Male"] = (m.Sex == 1).astype(int)
    m["SignalExam_is_ExcelExam"] = m.P.map(q.ExcelMatchesSignal)
    m["CBCT_Identity"] = m.P.map(q.CBCT_Identity)
    m["Sag"] = np.select([m.ANB >= 4, m.ANB > 0], ["II", "I"], "III")
    m["Severity"] = pd.cut(m.AHI, [-np.inf, 5, 15, 30, np.inf], right=False,
                           labels=["정상 <5", "경증 5–15", "중등도 15–30", "중증 ≥30"])
    m["GapGroup"] = pd.cut(m.GapDays.abs(), [-1, 30, 180, 365, 365 * 3, np.inf],
                           labels=["≤30일", "31–180일", "181–365일", "1–3년", ">3년"]).cat.add_categories("미확인")
    m.loc[m.GapDays.isna(), "GapGroup"] = "미확인"
    # 제외 규칙 (순서대로 첫 사유만 기록)
    reason = np.select([m.CBCT_Identity == "mismatch", m.Sag == "III"],
                       ["CBCT 본인 아님 (엑셀 CBCT 값 출처 불명)", "Class III (Roh 대상 아님)"], "")
    m["Exclude_Main"] = reason
    m["Valid_Main"] = reason == ""
    m["Valid_Gap1y"] = m.Valid_Main & (m.GapDays.abs() <= GAP_MAX)
    m["Note"] = np.where(m.CBCT_Identity == "id_only", "CBCT 생년월일 불일치(등록번호 일치) — 포함", "")
    m.insert(0, "Patient", "P_" + m.P.astype(str))
    return m.drop(columns="P")


# ------------------------------------------------------------ B. 기초 통계
def baseline(m: pd.DataFrame):
    v = m[m.Valid_Main]
    rows = []
    for c in STAT_VARS:
        s = v[c].astype(float)
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        z = (s - s.mean()) / s.std()
        rows.append({"Variable": c, "N": int(s.notna().sum()), "Missing": int(s.isna().sum()),
                     "Mean": s.mean(), "SD": s.std(), "Median": s.median(), "Q1": q1, "Q3": q3,
                     "Min": s.min(), "Max": s.max(), "Skew": s.skew(),
                     "Outliers_IQR1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
                     "Outliers_|z|>3": int((z.abs() > 3).sum())})
    stats = pd.DataFrame(rows).round(2)
    comp = []
    for col, label in [("Male", "성별(남=1)"), ("Severity", "AHI 중증도"), ("GapGroup", "CBCT–PSG 간격"),
                       ("Sag", "시상 분류"), ("PSG_Format", "PSG 장비"), ("SignalExam_is_ExcelExam", "원신호=엑셀 검사")]:
        vc = v[col].value_counts(sort=False)
        comp += [{"Group": label, "Category": str(k), "N": int(n), "Percent": round(100 * n / len(v), 1)}
                 for k, n in vc.items()]
    corr = v[["Age", "BMI", "ANB", "MPFH", "TDI", "PMCA", "PAL", "NAV", "PAV", "AHI", "RDI", "ODI", "LSaO2"]].corr()
    return stats, pd.DataFrame(comp), corr


def missingness(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in STAT_VARS + ["PSG_Date", "CBCT_Date"]:
        miss = m[c].isna()
        rows.append({"Variable": c, "Missing_All87": int(miss.sum()), "Missing_ValidMain": int(miss[m.Valid_Main].sum()),
                     "Missing_Pct_ValidMain": round(100 * miss[m.Valid_Main].mean(), 1),
                     "Source": "엑셀" if c in EXCEL_COLS else "리포트 PDF" if c in REPORT_COLS else "파생/헤더",
                     "Meaning": {"NAV": "참조 계측 결측 (구조 추출 가능 여부와 별개)",
                                 "PAV": "참조 계측 결측 (구조 추출 가능 여부와 별개)",
                                 "PSG_Date": "엑셀 값과 TST가 맞는 리포트 없음", "GapDays": "PSG 검사일 미확인"}.get(c, "")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ C·D. M0와 민감도 분석
def fit_m0(df: pd.DataFrame, y: str, label: str) -> pd.DataFrame:
    s = df[[y] + M0_X].dropna().astype(float)
    X = sm.add_constant(s[M0_X])
    f = sm.OLS(s[y], X).fit()
    z = (s - s.mean()) / s.std()
    beta = sm.OLS(z[y], z[M0_X]).fit().params
    ci = f.conf_int()
    return pd.DataFrame([{"Analysis": label, "DV": y, "N": int(f.nobs), "R2": f.rsquared, "AdjR2": f.rsquared_adj,
                          "Term": t, "B": f.params[t], "SE": f.bse[t], "CI_low": ci.loc[t, 0], "CI_high": ci.loc[t, 1],
                          "Beta_std": beta.get(t, np.nan), "P": f.pvalues[t],
                          "VIF": variance_inflation_factor(X.values, X.columns.get_loc(t)) if t != "const" else np.nan}
                         for t in X.columns])


def models(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(logAHI=np.log1p(m.AHI))
    sets = {"전체 유효": m[m.Valid_Main], f"간격 ≤{GAP_MAX}일": m[m.Valid_Gap1y],
            "리포트≠엑셀 PSG 제외": m[m.Valid_Main & m.Report_AHI_matches_Excel]}   # 같은 검사(TST 일치)인데 AHI·RDI·ODI·LSaO2가 다른 3명
    return pd.concat([fit_m0(df, y, f"{name} | {'AHI' if y == 'AHI' else 'log(AHI+1)'}")
                      for name, df in sets.items() for y in ("AHI", "logAHI")]).round(4)


# ------------------------------------------------------------ 그림
def figures(m: pd.DataFrame, corr: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    v = m[m.Valid_Main]
    ink, mark = "#3d3d3d", "#2f6db5"
    hist_vars = ["Age", "BMI", "ANB", "MPFH", "TDI", "PMCA", "PAL", "NAV", "PAV", "AHI", "RDI", "ODI", "LSaO2", "GapDays"]
    fig, ax = plt.subplots(4, 4, figsize=(13, 11))
    for a, c in zip(ax.flat, hist_vars + ["logAHI"]):
        s = np.log1p(v.AHI) if c == "logAHI" else v[c].dropna()
        a.hist(s, bins=18, color=mark, edgecolor="white", linewidth=1)
        a.set_title(f"{'log(AHI+1)' if c == 'logAHI' else c}  (n={s.notna().sum()})", fontsize=10, color=ink)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8, colors=ink)
    ax.flat[-1].axis("off")
    fig.suptitle(f"주요 변수 분포 (유효 환자 {len(v)}명)", color=ink)
    fig.tight_layout()
    fig.savefig(OUT / "fig_distributions.png", dpi=110)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(8.5, 7.5))
    im = a.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    a.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=9)
    a.set_yticks(range(len(corr)), corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            a.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=ink)
    fig.colorbar(im, shrink=.8, label="Pearson r")
    a.set_title("변수 간 상관 (pairwise)", color=ink)
    fig.tight_layout()
    fig.savefig(OUT / "fig_correlation.png", dpi=110)
    plt.close(fig)

    # 분석 가능 환자 흐름도
    n0 = len(m)
    ex = m.Exclude_Main.value_counts().drop("", errors="ignore")
    n1 = int(m.Valid_Main.sum())
    gap_unknown = int((m.Valid_Main & m.GapDays.isna()).sum())
    gap_over = int((m.Valid_Main & (m.GapDays.abs() > GAP_MAX)).sum())
    n2 = int(m.Valid_Gap1y.sum())
    steps = [(f"엑셀 등록 환자\nN = {n0}", None),
             (f"주 분석 (전체 유효)\nN = {n1}", "\n".join(f"제외: {k} {v}명" for k, v in ex.items())),
             (f"민감도 분석 (간격 ≤{GAP_MAX}일)\nN = {n2}", f"제외: PSG 검사일 미확인 {gap_unknown}명\n제외: CBCT–PSG 간격 >{GAP_MAX}일 {gap_over}명")]
    fig, a = plt.subplots(figsize=(8, 6))
    a.axis("off")
    for i, (txt, side) in enumerate(steps):
        y = 0.85 - i * 0.36
        a.text(0.3, y, txt, ha="center", va="center", fontsize=12, color=ink,
               bbox=dict(boxstyle="round,pad=0.6", fc="#eef3fa", ec=mark, lw=1.5))
        if i:
            a.annotate("", xy=(0.3, y + 0.09), xytext=(0.3, y + 0.27), arrowprops=dict(arrowstyle="->", color=ink, lw=1.5))
        if side:
            a.text(0.62, y + 0.18, side, ha="left", va="center", fontsize=10, color=ink,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b0b0b0"))
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    fig.savefig(OUT / "fig_flow.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = build_master()
    m.to_csv(OUT / "patient_master.csv", index=False, encoding="utf-8-sig")
    stats, comp, corr = baseline(m)
    stats.to_csv(OUT / "baseline_statistics.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(OUT / "baseline_composition.csv", index=False, encoding="utf-8-sig")
    corr.round(3).to_csv(OUT / "correlation_matrix.csv", encoding="utf-8-sig")
    missingness(m).to_csv(OUT / "missingness_report.csv", index=False, encoding="utf-8-sig")
    res = models(m)
    res.to_csv(OUT / "M0_baseline_results.csv", index=False, encoding="utf-8-sig")
    figures(m, corr)

    v = m[m.Valid_Main]
    print(f"전체 {len(m)} → 주 분석 {len(v)} → 간격 ≤{GAP_MAX}일 {int(m.Valid_Gap1y.sum())}")
    print("제외:", m.Exclude_Main[m.Exclude_Main != ""].value_counts().to_dict())
    print("리포트로 PSG 검사일 확인:", int(v.PSG_Date.notna().sum()), "/", len(v),
          "| 리포트 AHI = 엑셀 AHI:", int(v.Report_AHI_matches_Excel.sum()))
    print("원신호 검사 = 엑셀 검사:", int(v.SignalExam_is_ExcelExam.sum()), "/", len(v))
    print("CBCT–PSG 간격 분포:", v.GapGroup.value_counts(sort=False).to_dict())
    print("\n" + stats[stats.Variable.isin(["AHI", "ODI", "ANB", "TDI", "PMCA", "GapDays"])].to_string(index=False))
    summ = res[res.Term != "const"].pivot_table(index="Term", columns="Analysis", values="P").round(3)
    print("\nM0 계수 P값\n" + summ.to_string())
    print("\nR² / adj R² / N\n" + res.drop_duplicates("Analysis")[["Analysis", "N", "R2", "AdjR2"]].round(3).to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
