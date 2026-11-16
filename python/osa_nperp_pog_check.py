import sys
sys.path.insert(0, r"C:\Users\송정현\Documents\Projects\OSA\Project\python")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import osa_experiments as X

d = X.data()
f = pd.read_csv(X.FINAL, encoding="utf-8-sig").set_index("Patient")
for k in ("ANB", "Nperp-Pog", "MP-FH", "Nperp-A"):
    g = f[k].where(f[k+"_grade"].isin(list("ABCD")))
    d["f_"+k.replace("-", "")] = g.reindex(d.index)
for k in ("NperpPog", "MPFH", "NperpA", "ANB"):
    d["diff_"+k] = d["x_"+k] - d["m_"+k]          # 엑셀 − 자동
OUT = X.R_/"analysis/osa_experiments_v2/S1b_nperp_pog_check"; OUT.mkdir(parents=True, exist_ok=True)

# 1) 차이가 AHI와 관련 있나 + 차이끼리(FH 공유 오차) 상관
rows = []
for sn, s in X.samples(d).items():
    s = s.dropna(subset=["diff_NperpPog"])
    for y in ("logAHI", "logAHI_SUP"):
        ss = s.dropna(subset=[y]); A = sm.add_constant(ss[X.B0+["diff_NperpPog"]].astype(float))
        fit = sm.OLS(ss[y], A).fit()
        rows.append({"test": f"diff_NperpPog ~ {y}", "sample": sn, "n": len(ss), "spearman": spearmanr(ss.diff_NperpPog, ss[y])[0],
                     "p_spearman": spearmanr(ss.diff_NperpPog, ss[y])[1], "beta_B0adj_p": fit.pvalues["diff_NperpPog"],
                     "dR2cv": X.compare(ss, y, X.B0, ["diff_NperpPog"]).get("dR2")})
    for other in ("diff_MPFH", "diff_NperpA", "diff_ANB", "Pog_Ba_ant", "m_NperpPog", "x_NperpPog", "BMI"):
        t = s.dropna(subset=[other]); r, p = spearmanr(t.diff_NperpPog, t[other])
        rows.append({"test": f"diff_NperpPog vs {other}", "sample": sn, "n": len(t), "spearman": r, "p_spearman": p})
    rows.append({"test": "diff_NperpPog mean/sd (mm)", "sample": sn, "n": len(s), "spearman": s.diff_NperpPog.mean(), "p_spearman": s.diff_NperpPog.std()})
t1 = pd.DataFrame(rows); t1.round(4).to_csv(OUT/"diff_tests.csv", index=False, encoding="utf-8-sig")
print(t1.round(3).to_string(index=False))

# 2) 같은 환자에서 하악 후퇴 정보 비교: 엑셀 Nperp-Pog / 자동 / 최종 / Pog_Ba_ant, 단독과 ANB 위
rows = []
VARS = {"엑셀 Nperp-Pog": "x_NperpPog", "자동 Nperp-Pog": "m_NperpPog", "최종 Nperp-Pog": "f_NperpPog", "Pog_Ba_ant(3D)": "Pog_Ba_ant"}
for sn, s in X.samples(d).items():
    s = s.dropna(subset=list(VARS.values())+["x_ANB", "m_ANB", "f_ANB"])
    for h, y in X.H.items():
        for lab, v in VARS.items():
            rows.append({"sample": sn, "hyp": h, "model": f"B0 + {lab}", **X.compare(s, y, X.B0, [v])})
        for anb_lab, anb in (("엑셀 ANB", "x_ANB"), ("자동 ANB", "m_ANB"), ("최종 ANB", "f_ANB")):
            rows.append({"sample": sn, "hyp": h, "model": f"B0 + {anb_lab}", **X.compare(s, y, X.B0, [anb])})
        # 엑셀 조합 vs 자동 대체 조합 (B0 대비)
        for lab, add in (("엑셀 ANB + 엑셀 Nperp-Pog", ["x_ANB", "x_NperpPog"]), ("자동 ANB + 자동 Nperp-Pog", ["m_ANB", "m_NperpPog"]),
                         ("자동 ANB + Pog_Ba_ant", ["m_ANB", "Pog_Ba_ant"]), ("최종 ANB + 최종 Nperp-Pog", ["f_ANB", "f_NperpPog"]),
                         ("최종 ANB + Pog_Ba_ant", ["f_ANB", "Pog_Ba_ant"])):
            rows.append({"sample": sn, "hyp": h, "model": f"B0 + {lab}", **X.compare(s, y, X.B0, add)})
        # ANB 위 추가 이득
        for anb_lab, anb, v in (("엑셀 ANB", "x_ANB", "x_NperpPog"), ("자동 ANB", "m_ANB", "Pog_Ba_ant"), ("최종 ANB", "f_ANB", "Pog_Ba_ant")):
            rows.append({"sample": sn, "hyp": h, "model": f"{anb_lab} 위에 + {v}", **X.compare(s, y, X.B0, [v], ref_add=[anb])})
t2 = pd.DataFrame(rows); t2.round(4).to_csv(OUT/"replacement_compare.csv", index=False, encoding="utf-8-sig")
print(t2[["sample", "hyp", "model", "n", "dR2", "dR2_lo", "dR2_hi", "share_better", "dAdjR2", "F_p"]].round(3).to_string(index=False))
