"""B 단계: 골격 배치·크기·비율 분해 (사전 등록: 실험기록 "B 단계 사전 등록").
B1 Pog_Ba = Pog_PNS_ant + PNS_Ba_depth 분해 / B2 ANB = SNA − SNB 분해 / B3 길이·비율·각도 블록.
좌표: 정중 좌표계(PNS 원점, 좌·앞·위), build_features.coords. Co = AMASSS 과두 최상점 좌우 평균, Gn = Pog·Me 중점.
출력: Result_py/analysis/osa_experiments_v2_final/B_proportions/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
import osa_experiments as X
import build_features as bf

OUT = X.R_/"analysis/osa_experiments_v2_final/B_proportions"; OUT.mkdir(parents=True, exist_ok=True)
LENGTHS = ["S_N", "S_Ba", "ANS_PNS", "Co_Gn", "Co_A", "Go_Me", "Co_Go", "N_Me"]
RATIOS = ["NSBa", "gonial", "Jarabak", "McN_diff", "ramus_body", "palate_cb", "mand_cb"]


def sag_angle(v1, v2):
    """정중 시상면(앞, 위) 투영 두 벡터 사이 각(도)."""
    a, b = v1[1:], v2[1:]; return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b), -1, 1))))


def geometry(q):
    L = lambda a, b: float(np.linalg.norm(a-b))
    go = (q["RGo"]+q["LGo"])/2; gn = (q["Pog"]+q["Me"])/2; co = (q["RSup"]+q["LSup"])/2 if "RSup" in q else np.full(3, np.nan)
    r = {"Pog_PNS_ant": q["Pog"][1], "PNS_Ba_depth": -q["Ba"][1],
         "SNA": sag_angle(q["S"]-q["N"], q["A"]-q["N"]), "SNB": sag_angle(q["S"]-q["N"], q["B"]-q["N"]),
         "S_N": L(q["S"], q["N"]), "S_Ba": L(q["S"], q["Ba"]), "ANS_PNS": L(q["ANS"], q["PNS"]), "Go_Me": L(go, q["Me"]),
         "N_Me": L(q["N"], q["Me"]), "S_Go_": L(q["S"], go), "NSBa": sag_angle(q["N"]-q["S"], q["Ba"]-q["S"])}
    if np.isfinite(co).all():
        r |= {"Co_Gn": L(co, gn), "Co_A": L(co, q["A"]), "Co_Go": L(co, go), "gonial": float(np.degrees(np.arccos(
            (co-go)@(q["Me"]-go)/np.linalg.norm(co-go)/np.linalg.norm(q["Me"]-go))))}
    return r


def build(d):
    rel = pd.read_csv(X.R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
    cond = pd.read_csv(bf.EXT/"condyle_all.csv", encoding="utf-8-sig").set_index("Patient")
    g = pd.DataFrame({p: geometry(bf.coords(int(p[2:]), rel.loc[p].to_dict() if p in rel.index else {}, cond.loc[p] if p in cond.index else None)[0])
                      for p in d.index}).T.astype(float)
    g["Jarabak"] = g.S_Go_/g.N_Me; g["McN_diff"] = g.Co_Gn-g.Co_A; g["ramus_body"] = g.Co_Go/g.Go_Me
    g["palate_cb"] = g.ANS_PNS/g.S_Ba; g["mand_cb"] = g.Co_Gn/g.S_N
    d = d.join(g.drop(columns="S_Go_"))
    gap = (d.Pog_PNS_ant+d.PNS_Ba_depth-d.Pog_Ba_ant).abs().max(); assert gap < 1e-3, gap      # 항등식 확인(features.csv 반올림 오차만 허용)
    return d


if __name__ == "__main__":
    X.SOURCE = "final_v2"; d = X.data(); X.size_axis(d); d = build(d[d.restored.astype(bool)].copy())
    d[["Pog_PNS_ant", "PNS_Ba_depth", "SNA", "SNB"]+LENGTHS+RATIOS].round(3).to_csv(OUT/"geometry.csv", encoding="utf-8-sig")
    print("ANB 대 SNA−SNB 상관", round(d.m_ANB.corr(d.SNA-d.SNB), 3))
    S = {"main59": d[d.analysis_list.astype(bool)], "sens85": d}; rows, coef = [], []
    for sn, s in S.items():
        for h, y in X.H.items():
            # B1·B2 분해: 성분 따로 넣은 모델 vs 확정 모델, 성분 계수
            for lab, add, ref in (("B1 Pog_Ba 분해 (ANB + 두 성분) vs 확정", ["Pog_PNS_ant", "PNS_Ba_depth"], ["m_ANB"]),
                                  ("B2 ANB 분해 (SNA + SNB) vs B0+ANB", ["SNA", "SNB"], [])):
                u = s.dropna(subset=[y]+X.B0+add+ref+["Pog_Ba_ant", "m_ANB"])
                cmp_ref = X.CONFIRMED if lab.startswith("B1") else ["m_ANB"]
                full = X.compare(u, y, X.B0, ref+add); base = X.compare(u, y, X.B0, cmp_ref)
                rows.append({"part": lab, "sample": sn, "hyp": h, "add": "+".join(ref+add), "n": len(u), "dR2": full["R2cv"]-base["R2cv"],
                             "R2cv_split": full["R2cv"], "R2cv_ref": base["R2cv"]})
                f = sm.OLS(u[y], sm.add_constant((u[X.B0+ref+add]-u[X.B0+ref+add].mean())/u[X.B0+ref+add].std())).fit()
                for c in ref+add: coef.append({"part": lab, "sample": sn, "hyp": h, "term": c, "beta_std": f.params[c], "p": f.pvalues[c]})
            for c in ("Pog_PNS_ant", "PNS_Ba_depth", "SNA", "SNB"):
                rows.append({"part": "B0 위 성분 단독", "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c])})
            # B3 비율 블록
            for c in LENGTHS+RATIOS:
                for lab, ref in (("B3 기존 인자 위", X.CONFIRMED), ("B3 기존 인자+크기 위", X.CONFIRMED+["SIZE_PC1"])):
                    rows.append({"part": lab, "kind": "비율·각도" if c in RATIOS else "길이", "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
    t = X.fdr(rows, ("sample", "hyp", "part")); t.round(4).to_csv(OUT/"B_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coef).round(4).to_csv(OUT/"B_decomposition_coef.csv", index=False, encoding="utf-8-sig")
    ind = [{"feature": c, "rho_SIZE": spearmanr(d[c], d.SIZE_PC1, nan_policy="omit")[0], "rho_ANB": spearmanr(d[c], d.m_ANB, nan_policy="omit")[0],
            "rho_PogBa": spearmanr(d[c], d.Pog_Ba_ant, nan_policy="omit")[0], "rho_Male": spearmanr(d[c], d.Male, nan_policy="omit")[0], "n": int(d[c].notna().sum())}
           for c in ["Pog_PNS_ant", "PNS_Ba_depth", "SNA", "SNB"]+LENGTHS+RATIOS]
    pd.DataFrame(ind).round(3).to_csv(OUT/"B_independence.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[t.part.str.startswith(("B1", "B2"))][["part", "sample", "hyp", "n", "dR2", "R2cv_split", "R2cv_ref"]].round(3).to_string(index=False))
    print(pd.DataFrame(coef).round(3).to_string(index=False))
    print(t[t.part == "B0 위 성분 단독"][["sample", "hyp", "add", "n", "dR2", "F_p", "coef_std"]].round(3).to_string(index=False))
    b3 = t[t.part.str.startswith("B3")]; print(b3[b3.F_p < .10][["part", "kind", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]].round(3).to_string(index=False))
    print(pd.DataFrame(ind).round(2).to_string(index=False))
