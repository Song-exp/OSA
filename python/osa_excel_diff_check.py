"""S1d: 골격 4개(ANB·Nperp-A·Nperp-Pog·MP-FH)의 엑셀 − 자동 차이와 대체 지표 예측력 — S1b·S1c 재계산.

S1b·S1c는 엑셀에 맞춘 환산값(aligned)으로 차이를 계산해 순환 문제가 있었다. 여기서는 엑셀을 쓰지 않은 자동값만 쓴다.
판(variant): final_ABCD(기본, yao4 FH, 등급 A~D) / final_AB / default_jung(aligned_metrics의 default, 환산 전).
엑셀 0값 의심은 osa_experiments.data()에서 결측 처리된다.
출력 Result_py/analysis/osa_experiments_v2/S1d_excel_diff/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import osa_experiments as X

K = ["ANB", "Nperp-A", "Nperp-Pog", "MP-FH"]; c = lambda k: k.replace("-", "")
OUT = X.R_/"analysis/osa_experiments_v2/S1d_excel_diff"; OUT.mkdir(parents=True, exist_ok=True)


def ang(dx, dz):
    """정중면(앞, 위) 투영 방향각(도). 앞쪽 수평 = 0, 아래로 기울면 양수."""
    return np.degrees(np.arctan2(-dz, dx))


assert abs(ang(1, -1)-45) < 1e-9 and abs(ang(1, 0)) < 1e-9

d = X.data()
go_a, go_s = (d.c_RGo_ant+d.c_LGo_ant)/2, (d.c_RGo_sup+d.c_LGo_sup)/2
d["PP_MP"] = ang(d.c_Me_ant-go_a, d.c_Me_sup-go_s)                       # 앞 축 = ANS−PNS(구개평면)
d["SN_MP"] = d.PP_MP-ang(d.c_N_ant-d.c_S_ant, d.c_N_sup-d.c_S_sup)
fin = pd.read_csv(X.FINAL, encoding="utf-8-sig").set_index("Patient")
al = pd.read_csv(X.bf.EXT/"alignment/aligned_metrics.csv", encoding="utf-8-sig").pivot(index="Patient", columns="metric", values="default")
VARIANTS = {}
for name, grades in (("final_ABCD", "ABCD"), ("final_AB", "AB")):
    VARIANTS[name] = {k: fin[k].where(fin[k+"_grade"].isin(list(grades))).reindex(d.index) for k in K}
VARIANTS["default_jung"] = {k: al[k].reindex(d.index) for k in K}

diff_rows, cross_rows, model_rows = [], [], []
for vn, vals in VARIANTS.items():
    for k in K: d["a_"+c(k)] = vals[k]; d["diff_"+c(k)] = d["x_"+c(k)]-d["a_"+c(k)]
    for sn, s in X.samples(d).items():
        # 1) 지표별 차이
        for k in K:
            x, a, df = "x_"+c(k), "a_"+c(k), "diff_"+c(k); t = s.dropna(subset=[x, a])
            row = {"variant": vn, "sample": sn, "metric": k, "n": len(t), "diff_mean": t[df].mean(), "diff_sd": t[df].std(),
                   "r_excel_auto": t[x].corr(t[a]), "slope_auto_on_excel": np.polyfit(t[x], t[a], 1)[0],
                   "sd_ratio_auto_excel": t[a].std()/t[x].std(), "rho_diff_excel": spearmanr(t[df], t[x])[0], "rho_diff_auto": spearmanr(t[df], t[a])[0]}
            for h, y in X.H.items():
                u = t.dropna(subset=[y]); fit = sm.OLS(u[y], sm.add_constant(u[X.B0+[df]].astype(float))).fit()
                ex, au = X.compare(u, y, X.B0, [x], ref_add=[a]), X.compare(u, y, X.B0, [a], ref_add=[x])
                row |= {f"{h}_diff_p": fit.pvalues[df], f"{h}_excel_over_auto_dR2": ex.get("dR2"), f"{h}_excel_over_auto_p": ex.get("F_p"),
                        f"{h}_auto_over_excel_dR2": au.get("dR2"), f"{h}_auto_over_excel_p": au.get("F_p")}
            diff_rows.append(row)
        # 2) 지표 사이 차이 상관 (같은 판 안에서만)
        for i, k1 in enumerate(K):
            for k2 in K[i+1:]:
                t = s.dropna(subset=["diff_"+c(k1), "diff_"+c(k2)]); r, p = spearmanr(t["diff_"+c(k1)], t["diff_"+c(k2)])
                cross_rows.append({"variant": vn, "sample": sn, "pair": f"{k1} ~ {k2}", "n": len(t), "rho": r, "p": p})
        # 3) 예측력 비교: 같은 환자 집합
        core = ["x_ANB", "x_NperpPog", "x_MPFH", "a_ANB", "a_NperpPog", "a_MPFH", "Pog_Ba_ant", "SN_MP", "PP_MP"]
        t = s.dropna(subset=core)
        for h, y in X.H.items():
            M = [("B0 + 엑셀 "+k, ["x_"+c(k)], []) for k in ("ANB", "Nperp-Pog", "MP-FH")] + \
                [("B0 + 자동 "+k, ["a_"+c(k)], []) for k in ("ANB", "Nperp-Pog", "MP-FH")] + \
                [("B0 + Pog_Ba_ant", ["Pog_Ba_ant"], []), ("B0 + SN-MP", ["SN_MP"], []), ("B0 + PP-MP", ["PP_MP"], []),
                 ("B0 + 엑셀 ANB+Nperp-Pog", ["x_ANB", "x_NperpPog"], []), ("B0 + 엑셀 ANB+Nperp-Pog+MP-FH", ["x_ANB", "x_NperpPog", "x_MPFH"], []),
                 ("B0 + 자동 ANB+Nperp-Pog", ["a_ANB", "a_NperpPog"], []), ("B0 + 자동 ANB+Pog_Ba", ["a_ANB", "Pog_Ba_ant"], []),
                 ("B0 + 자동 ANB+Pog_Ba+MP-FH", ["a_ANB", "Pog_Ba_ant", "a_MPFH"], []),
                 ("엑셀 ANB 위 + 엑셀 Nperp-Pog", ["x_NperpPog"], ["x_ANB"]), ("엑셀 ANB 위 + 엑셀 MP-FH", ["x_MPFH"], ["x_ANB"]),
                 ("자동 ANB 위 + Pog_Ba_ant", ["Pog_Ba_ant"], ["a_ANB"]), ("자동 ANB 위 + 자동 MP-FH", ["a_MPFH"], ["a_ANB"])]
            for lab, add, ref in M:
                model_rows.append({"variant": vn, "sample": sn, "hyp": h, "set": "공통", "model": lab, **X.compare(t, y, X.B0, add, ref_add=ref)})
            u = s.dropna(subset=["x_NperpA", "a_NperpA"])                       # Nperp-A는 엑셀 0값 제외로 표본이 따로
            for lab, add in (("B0 + 엑셀 Nperp-A", ["x_NperpA"]), ("B0 + 자동 Nperp-A", ["a_NperpA"])):
                model_rows.append({"variant": vn, "sample": sn, "hyp": h, "set": "Nperp-A", "model": lab, **X.compare(u, y, X.B0, add)})

for name, rows in (("diff_by_metric", diff_rows), ("diff_cross", cross_rows), ("model_compare", model_rows)):
    t = pd.DataFrame(rows); t.round(4).to_csv(OUT/f"{name}.csv", index=False, encoding="utf-8-sig")
    show = t if name != "model_compare" else t[["variant", "sample", "hyp", "set", "model", "n", "dR2", "share_better", "F_p"]]
    print(f"\n== {name}"); print(show.round(3).to_string(index=False))
