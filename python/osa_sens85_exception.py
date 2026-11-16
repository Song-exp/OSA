"""S1f: 85명 표본 H1에서 엑셀 ANB가 자동 ANB보다 강한 예외의 원인.
주 분석 59명과 추가 환자(85명 표본 − 59명)를 나눠 일치도·신호를 본다. 출력 Result_py/analysis/osa_experiments_v2/S1f_sens85_exception/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import osa_experiments as X

d = X.data(); fin = pd.read_csv(X.FINAL, encoding="utf-8-sig").set_index("Patient")
d["f_ANB"] = fin.ANB.where(fin.ANB_grade.isin(list("ABCD"))).reindex(d.index); d["ANB_grade"] = fin.ANB_grade.reindex(d.index)
s85 = d[d.restored.astype(bool)].dropna(subset=["logAHI"]); main = s85.index.isin(d.index[d.analysis_list.astype(bool)])
G = {"주 분석": s85[main], "추가 환자": s85[~main], "85명 전체": s85}
OUT = X.R_/"analysis/osa_experiments_v2/S1f_sens85_exception"; OUT.mkdir(parents=True, exist_ok=True)


def ols(s, y, v):
    u = s.dropna(subset=[y, v]); f = sm.OLS(u[y], sm.add_constant(u[X.B0+[v]].astype(float))).fit()
    return len(u), f.params[v]*u[v].std(), f.pvalues[v]


rows = []
for g, s in G.items():
    t = s.dropna(subset=["x_ANB", "m_ANB"]); e = (t.x_ANB-t.m_ANB).abs()
    r = {"group": g, "n": len(s), "AHI_median": np.expm1(s.logAHI).median(), "r_excel_auto": t.x_ANB.corr(t.m_ANB),
         "MAE_auto": e.mean(), "n_err_gt2": int((e > 2).sum()), "grade_E_or_missing": int(s.ANB_grade.isin(["E"]).sum()+s.ANB_grade.isna().sum())}
    for h, y in X.H.items():
        for lab, v in (("excel", "x_ANB"), ("auto", "m_ANB"), ("final", "f_ANB")):
            n, b, p = ols(s, y, v); r |= {f"{h}_{lab}_beta_sd": b, f"{h}_{lab}_p": p}
            r[f"{h}_{lab}_rho"] = spearmanr(s[v], s[y], nan_policy="omit")[0]
    rows.append(r)
t = pd.DataFrame(rows); t.round(4).to_csv(OUT/"groups.csv", index=False, encoding="utf-8-sig"); print(t.round(3).T.to_string())

# 85명에서 자동–엑셀 ANB 차이가 큰 환자를 빼면 격차가 줄어드는가
rows = []
for cut in (np.inf, 3, 2):
    s = s85[(s85.x_ANB-s85.m_ANB).abs() <= cut]
    for h, y in X.H.items():
        ex, au = X.compare(s, y, X.B0, ["x_ANB"]), X.compare(s, y, X.B0, ["m_ANB"])
        rows.append({"max_abs_diff": cut, "hyp": h, "n": ex["n"], "dR2_excel": ex["dR2"], "dR2_auto": au["dR2"]})
for h, y in X.H.items():                                                   # 추가 환자만
    s = G["추가 환자"]; ex, au = X.compare(s, y, X.B0, ["x_ANB"]), X.compare(s, y, X.B0, ["m_ANB"])
    rows.append({"max_abs_diff": "추가 환자만", "hyp": h, "n": ex["n"], "dR2_excel": ex.get("dR2"), "dR2_auto": au.get("dR2")})
t2 = pd.DataFrame(rows); t2.round(4).to_csv(OUT/"trim.csv", index=False, encoding="utf-8-sig"); print(t2.round(3).to_string(index=False))
