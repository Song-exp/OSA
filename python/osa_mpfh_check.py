"""S1c: MP-FH 엑셀 신호 소실 원인과 FH 없는 수직 지표(SN-MP, PP-MP) 대체 확인.
osa_nperp_pog_check.py(S1b)와 같은 방식. 출력 Result_py/analysis/osa_experiments_v2/S1c_mpfh_check/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import osa_experiments as X


def ang(dx, dz):
    """정중면(앞, 위) 투영 방향각(도). 앞쪽 수평 = 0, 아래로 기울면 양수."""
    return np.degrees(np.arctan2(-dz, dx))


d = X.data()
f = pd.read_csv(X.FINAL, encoding="utf-8-sig").set_index("Patient")
for k in ("ANB", "MP-FH", "Nperp-A", "Nperp-Pog"):
    d["f_"+k.replace("-", "")] = f[k].where(f[k+"_grade"].isin(list("ABCD"))).reindex(d.index)
for k in ("MPFH", "ANB", "NperpA", "NperpPog"):
    d["diff_"+k] = d["x_"+k]-d["m_"+k]                     # 엑셀 − 자동
go_a, go_s = (d.c_RGo_ant+d.c_LGo_ant)/2, (d.c_RGo_sup+d.c_LGo_sup)/2
mp = ang(d.c_Me_ant-go_a, d.c_Me_sup-go_s)                  # 하악평면(Go→Me)의 기울기, 구개평면(ANS–PNS) 기준
d["PP_MP"] = mp                                             # 중간선 좌표계의 앞 축 = ANS−PNS
d["SN_MP"] = mp-ang(d.c_N_ant-d.c_S_ant, d.c_N_sup-d.c_S_sup)
OUT = X.R_/"analysis/osa_experiments_v2/S1c_mpfh_check"; OUT.mkdir(parents=True, exist_ok=True)

assert abs(ang(1, -1)-45) < 1e-9 and abs(ang(1, 0)) < 1e-9   # 아래로 45° = +45

rows = []
for sn, s in X.samples(d).items():
    s = s.dropna(subset=["diff_MPFH"])
    for y in ("logAHI", "logAHI_SUP"):
        ss = s.dropna(subset=[y]); fit = sm.OLS(ss[y], sm.add_constant(ss[X.B0+["diff_MPFH"]].astype(float))).fit()
        r, p = spearmanr(ss.diff_MPFH, ss[y])
        rows.append({"test": f"diff_MPFH ~ {y}", "sample": sn, "n": len(ss), "spearman": r, "p": p, "B0adj_p": fit.pvalues["diff_MPFH"],
                     "dR2cv": X.compare(ss, y, X.B0, ["diff_MPFH"]).get("dR2")})
    for o in ("diff_NperpA", "diff_NperpPog", "diff_ANB", "x_MPFH", "m_MPFH", "SN_MP", "PP_MP", "x_ANB"):
        t = s.dropna(subset=[o]); r, p = spearmanr(t.diff_MPFH, t[o])
        rows.append({"test": f"diff_MPFH vs {o}", "sample": sn, "n": len(t), "spearman": r, "p": p})
    for o in ("SN_MP", "PP_MP"):
        for v in ("x_MPFH", "m_MPFH", "x_ANB"):
            t = s.dropna(subset=[o, v]); r, p = spearmanr(t[o], t[v])
            rows.append({"test": f"{o} vs {v}", "sample": sn, "n": len(t), "spearman": r, "p": p})
    rows.append({"test": "diff_MPFH mean/sd (deg)", "sample": sn, "n": len(s), "spearman": s.diff_MPFH.mean(), "p": s.diff_MPFH.std()})
t1 = pd.DataFrame(rows); t1.round(4).to_csv(OUT/"diff_tests.csv", index=False, encoding="utf-8-sig")
print(t1.round(3).to_string(index=False))

rows = []
V = {"엑셀 MP-FH": "x_MPFH", "자동 MP-FH": "m_MPFH", "최종 MP-FH": "f_MPFH", "SN-MP": "SN_MP", "PP-MP": "PP_MP"}
for sn, s in X.samples(d).items():
    s = s.dropna(subset=list(V.values())+["x_ANB", "f_ANB", "Pog_Ba_ant"])
    for h, y in X.H.items():
        for lab, v in V.items():
            rows.append({"sample": sn, "hyp": h, "model": f"B0 + {lab}", **X.compare(s, y, X.B0, [v])})
        for lab, ref, v in (("엑셀 ANB 위 + 엑셀 MP-FH", ["x_ANB"], "x_MPFH"), ("엑셀 ANB·Nperp-Pog 위 + 엑셀 MP-FH", ["x_ANB", "x_NperpPog"], "x_MPFH"),
                            ("최종 ANB 위 + 최종 MP-FH", ["f_ANB"], "f_MPFH"), ("최종 ANB 위 + SN-MP", ["f_ANB"], "SN_MP"), ("최종 ANB 위 + PP-MP", ["f_ANB"], "PP_MP"),
                            ("최종 ANB·Pog_Ba 위 + SN-MP", ["f_ANB", "Pog_Ba_ant"], "SN_MP")):
            rows.append({"sample": sn, "hyp": h, "model": lab, **X.compare(s, y, X.B0, [v], ref_add=ref)})
        for lab, add in (("엑셀 ANB+Nperp-Pog+MP-FH", ["x_ANB", "x_NperpPog", "x_MPFH"]), ("최종 ANB+Pog_Ba+최종 MP-FH", ["f_ANB", "Pog_Ba_ant", "f_MPFH"]),
                         ("최종 ANB+Pog_Ba+SN-MP", ["f_ANB", "Pog_Ba_ant", "SN_MP"]), ("최종 ANB+Pog_Ba", ["f_ANB", "Pog_Ba_ant"])):
            rows.append({"sample": sn, "hyp": h, "model": f"B0 + {lab}", **X.compare(s, y, X.B0, add)})
t2 = pd.DataFrame(rows); t2.round(4).to_csv(OUT/"replacement_compare.csv", index=False, encoding="utf-8-sig")
print(t2[["sample", "hyp", "model", "n", "dR2", "share_better", "dAdjR2", "F_p", "coef_std"]].round(3).to_string(index=False))
