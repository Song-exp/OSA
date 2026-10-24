"""Roh et al. (2025, Eur J Orthod) 1차 재현: 엑셀 CBCT 수작업 측정값 + 같은 검사의 PSG 리포트 값.

대상: 엑셀 87명 − P_39(CBCT 본인 아님) − Class III(ANB ≤ 0)
PSG 값: 엑셀 TST와 일치하는 리포트 (= 엑셀 작성에 쓰인 검사)
분류 기준(논문): Class I 0<ANB<4, Class II ANB≥4 / normo 25≤MP-FH<30, hyper ≥30, hypo <25 /
          TSD TDI<-2.5, non-TSD TDI>-2 / CSP2D=II+hyper+nonTSD, CSP3D=II+hyper+TSD, nonCSP=I+normo+nonTSD
결과: Result_py/replication/*.csv, 요약은 화면 출력
"""
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from osa.report import parse_report, reports_in

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py" / "replication"
PSG = ["AHI", "AHI_SUP", "AI", "HI", "RDI", "ODI", "LSaO2", "ArI", "WASO", "AHI_REM", "AHI_NREM", "AHI_RN", "ESS"]
CBCT = ["ANB", "Nperp_A", "Nperp_Pog", "MPFH", "TDI", "PHI", "NCWR", "NAV", "PAV", "PMCA", "PAL"]
EXCEL_COLS = {"Age": 6, "BMI": 9, "Sex": 10, "TST": 11, "AHI": 12, "RDI": 13, "ODI": 14, "LSaO2": 15,
              "ANB": 22, "Nperp_A": 23, "Nperp_Pog": 24, "MPFH": 25, "NCWR": 26, "PHI": 27, "TDI": 28,
              "NAV": 29, "PAV": 30, "PMCA": 31, "PAL": 32}


def build() -> pd.DataFrame:
    x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None)
    r = x.iloc[3:]
    r = r[r[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    d = pd.DataFrame({"P": r[0].astype(int).values,
                      **{k: pd.to_numeric(r[c], errors="coerce").values for k, c in EXCEL_COLS.items()}})
    rows, check = [], []
    for p in d.P:
        folder = next((ROOT / "Data/PSG").glob(f"P_{p}_*"))
        reps = [parse_report(f) for f in reports_in(folder)]
        tst = d.loc[d.P == p, "TST"].item()
        rep = next((q for q in reps if abs(q["TST"] - tst) <= 0.6), None)
        if rep is None:
            rows.append({"P": p})
            continue
        rows.append({"P": p, **{k: rep[k] for k in ("AHI_SUP", "AI", "HI", "ArI", "WASO", "AHI_REM", "AHI_NREM", "ESS")}})
        e = d[d.P == p].iloc[0]
        check.append({k: abs(rep[k] - e[k]) for k in ("AHI", "RDI", "ODI", "LSaO2")})
    d = d.merge(pd.DataFrame(rows), on="P", how="left")
    d["AHI_RN"] = d.AHI_REM / d.AHI_NREM
    c = pd.DataFrame(check)
    print("리포트 vs 엑셀 일치 (|차이|≤0.15):", {k: f"{int((c[k] <= 0.15).sum())}/{c[k].notna().sum()}" for k in c})
    d["Male"] = d.Sex == 1
    d["Sag"] = np.select([d.ANB >= 4, (d.ANB > 0) & (d.ANB < 4)], ["II", "I"], "III")
    d["Ver"] = np.select([d.MPFH >= 30, d.MPFH >= 25], ["hyper", "normo"], "hypo")
    d["Trans"] = np.select([d.TDI < -2.5, d.TDI > -2], ["TSD", "nonTSD"], "boundary")
    d["Pheno"] = np.select([(d.Sag == "II") & (d.Ver == "hyper") & (d.Trans == "TSD"),
                            (d.Sag == "II") & (d.Ver == "hyper") & (d.Trans == "nonTSD"),
                            (d.Sag == "I") & (d.Ver == "normo") & (d.Trans == "nonTSD")],
                           ["CSP3D", "CSP2D", "nonCSP"], "other")
    n0 = len(d)
    d = d[(d.P != 39) & (d.Sag != "III")]
    print(f"대상: {len(d)}명 (엑셀 {n0}명 − P_39 − Class III {n0 - 1 - len(d)}명)")
    return d


ms = lambda s: f"{s.mean():.1f} ± {s.std():.1f} (n={s.notna().sum()})"
star = lambda p: "" if not np.isfinite(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def ttest(a, b):
    a, b = a.dropna(), b.dropna()
    return stats.ttest_ind(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan


def chi_male(groups):
    tab = pd.crosstab(groups.iloc[:, 0], groups.iloc[:, 1])
    return stats.chi2_contingency(tab)[1] if tab.shape == (2, 2) or min(tab.shape) > 1 else np.nan


def two_group(d, col, a, b, label):
    rows = [{"Variable": "Male %", a: f"{100 * d[d[col] == a].Male.mean():.1f}",
             b: f"{100 * d[d[col] == b].Male.mean():.1f}",
             "P": chi_male(d[d[col].isin([a, b])][[col, "Male"]])}]
    for v in ["Age", "BMI"] + PSG + CBCT:
        rows.append({"Variable": v, a: ms(d[d[col] == a][v]), b: ms(d[d[col] == b][v]),
                     "P": ttest(d[d[col] == a][v], d[d[col] == b][v])})
    t = pd.DataFrame(rows)
    t["Sig"] = t.P.map(star)
    t.to_csv(OUT / f"{label}.csv", index=False, encoding="utf-8-sig")
    return t


def table2(d):
    rows = []
    for v in ["Age", "BMI"] + PSG + CBCT:
        rows.append({"Variable": v, "Total": ms(d[v]), "Male": ms(d[d.Male][v]), "Female": ms(d[~d.Male][v]),
                     "P": ttest(d[d.Male][v], d[~d.Male][v])})
    t = pd.DataFrame(rows)
    t["Sig"] = t.P.map(star)
    t.to_csv(OUT / "table2_sex.csv", index=False, encoding="utf-8-sig")
    return t


def table3(d):
    preds, cov = ["ANB", "MPFH", "TDI", "PMCA", "PAL"], ["Male", "Age", "BMI"]
    rows = []
    for y in ["AHI", "AHI_SUP", "AI", "RDI", "ODI", "LSaO2"]:
        s = d[[y] + preds + cov].dropna().astype(float)
        X = sm.add_constant(s[preds + cov])
        m = sm.OLS(s[y], X).fit()
        z = (s - s.mean()) / s.std()
        beta = sm.OLS(z[y], z[preds + cov]).fit().params
        for i, v in enumerate(preds):
            rows.append({"DV": y, "IV": v, "B": m.params[v], "SE": m.bse[v], "beta": beta[v], "t": m.tvalues[v],
                         "P": m.pvalues[v], "VIF": variance_inflation_factor(X.values, X.columns.get_loc(v)),
                         "AdjR2": m.rsquared_adj, "n": int(m.nobs)})
    t = pd.DataFrame(rows).round(3)
    t["Sig"] = t.P.map(star)
    t.to_csv(OUT / "table3_regression.csv", index=False, encoding="utf-8-sig")
    return t


def table4_vertical(d):
    g = {k: d[d.Ver == k] for k in ("normo", "hypo", "hyper")}
    rows = []
    for v in ["Age", "BMI"] + PSG + CBCT:
        arr = [g[k][v].dropna() for k in g]
        p = stats.f_oneway(*arr).pvalue if all(len(a) > 1 for a in arr) else np.nan
        post = []
        for (i, a), (j, b) in combinations(enumerate(g), 2):
            pp = ttest(g[a][v], g[b][v]) * 3          # Bonferroni
            if pp < .05:
                post.append(f"{a}{'>' if g[a][v].mean() > g[b][v].mean() else '<'}{b}")
        rows.append({"Variable": v, **{k: ms(g[k][v]) for k in g}, "P": p, "PostHoc": ", ".join(post)})
    t = pd.DataFrame(rows)
    t["Sig"] = t.P.map(star)
    t.to_csv(OUT / "table4_vertical.csv", index=False, encoding="utf-8-sig")
    return t


def correlations(d):
    r = pd.DataFrame({(c, p): stats.pearsonr(*d[[c, p]].dropna().T.values) if d[[c, p]].dropna().shape[0] > 3
                      else (np.nan, np.nan) for c in CBCT for p in PSG}, index=["r", "P"]).T
    r.index.names = ["CBCT", "PSG"]
    r.round(3).to_csv(OUT / "supp_correlation.csv", encoding="utf-8-sig")
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build()
    print("\n그룹 인원:", {k: int(v) for k, v in d.Pheno.value_counts().items()},
          "| 시상:", d.Sag.value_counts().to_dict(), "| 수직:", d.Ver.value_counts().to_dict(),
          "| 횡:", d.Trans.value_counts().to_dict())
    show = lambda t, cols: print(t[cols].to_string(index=False))
    t2 = table2(d)
    print("\n[표2] 성별 비교"); show(t2, ["Variable", "Total", "Male", "Female", "P", "Sig"])
    t3 = table3(d)
    print("\n[표3] 다중회귀 (성별·나이·BMI 보정)"); show(t3, ["DV", "IV", "B", "beta", "P", "Sig", "VIF", "AdjR2", "n"])
    print("\n[표4-시상] Class I vs II"); show(two_group(d, "Sag", "I", "II", "table4_sagittal"), ["Variable", "I", "II", "P", "Sig"])
    print("\n[표4-수직]"); show(table4_vertical(d), ["Variable", "normo", "hypo", "hyper", "P", "Sig", "PostHoc"])
    print("\n[표4-횡] nonTSD vs TSD"); show(two_group(d, "Trans", "nonTSD", "TSD", "table4_transverse"), ["Variable", "nonTSD", "TSD", "P", "Sig"])
    d5 = d.assign(CSP=np.where(d.Pheno.isin(["CSP2D", "CSP3D"]), "CSP", np.where(d.Pheno == "nonCSP", "nonCSP", None)))
    print("\n[표5] nonCSP vs CSP"); show(two_group(d5, "CSP", "nonCSP", "CSP", "table5_csp"), ["Variable", "nonCSP", "CSP", "P", "Sig"])
    print("\n[표5] CSP2D vs CSP3D"); show(two_group(d, "Pheno", "CSP2D", "CSP3D", "table5_csp2d3d"), ["Variable", "CSP2D", "CSP3D", "P", "Sig"])
    correlations(d)
    d.drop(columns=["P"]).describe().T.round(2).to_csv(OUT / "descriptive.csv", encoding="utf-8-sig")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
