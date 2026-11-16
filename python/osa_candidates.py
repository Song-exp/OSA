"""C 단계: 후보 변수 상호 독립성 (사전 등록: 실험기록 "C 단계 사전 등록").
C1 상관 / C2 짝별 흡수 / C3 고유 기여(leave-one-out) / C4 전체 확장 모델.
출력: Result_py/analysis/osa_experiments_v2_final/C_candidates/
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import osa_experiments as X
import osa_addition as A
import osa_vertical as V
import osa_residual_shape as RS

OUT = X.R_/"analysis/osa_experiments_v2_final/C_candidates"; OUT.mkdir(parents=True, exist_ok=True)
CAND = {"SIZE_PC1": ["SIZE_PC1"], "oropharynx_min_area": ["oropharynx_min_area"], "soft_palate_thickness": ["soft_palate_thickness"],
        "ANBxMale": ["m_ANBxMale"], "ANBxTongue": ["tongue_c_ant", "m_ANBxtongue_c_ant"], "Co_Gn": ["Co_Gn"], "S_Ba": ["S_Ba"],
        "MAND_rPC4": ["MAND_rPC4"], "LAFH_ratio": ["LAFH_ratio"]}


def terms(names):
    return list(dict.fromkeys(c for n in names for c in CAND[n]))


if __name__ == "__main__":
    d = A.load(); d = V.vertical(d); sc, *_ = RS.residual_pcs(d, "MAND"); d = d.join(sc)
    cols = terms(CAND)+["m_ANB", "Pog_Ba_ant"]
    corr = d[[c for c in cols if not c.startswith("m_ANBx")]].corr(method="spearman"); corr.round(2).to_csv(OUT/"C1_corr.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250); print(corr.round(2).to_string())
    S = {"main59": d[d.analysis_list.astype(bool)], "sens85": d}; rows = []
    for sn, s0 in S.items():
        s = s0.dropna(subset=terms(CAND)+X.CONFIRMED+X.B0+["logAHI", "logAHI_SUP"])   # 모든 비교를 같은 환자에서
        for h, y in X.H.items():
            for a in CAND:
                add = [c for c in CAND[a] if c not in ("tongue_c_ant",)] if a == "ANBxTongue" else CAND[a]
                ref_a = ["tongue_c_ant"] if a == "ANBxTongue" else []
                rows.append({"part": "단독(기존 인자 위)", "sample": sn, "hyp": h, "cand": a, "given": "-", **X.compare(s, y, X.B0, add, ref_add=X.CONFIRMED+ref_a)})
                for b in CAND:
                    if b == a: continue
                    ref = list(dict.fromkeys(X.CONFIRMED+ref_a+terms([b]))); add2 = [c for c in add if c not in ref]
                    rows.append({"part": "C2 짝별", "sample": sn, "hyp": h, "cand": a, "given": b, **X.compare(s, y, X.B0, add2, ref_add=ref)})
                others = [b for b in CAND if b != a]; ref = list(dict.fromkeys(X.CONFIRMED+ref_a+terms(others))); add3 = [c for c in add if c not in ref]
                rows.append({"part": "C3 고유 기여", "sample": sn, "hyp": h, "cand": a, "given": "나머지 전부", **X.compare(s, y, X.B0, add3, ref_add=ref)})
            rows.append({"part": "C4 전체 확장", "sample": sn, "hyp": h, "cand": "후보 전부", "given": "-", **X.compare(s, y, X.B0, terms(CAND), ref_add=X.CONFIRMED)})
    t = pd.DataFrame(rows); t.round(4).to_csv(OUT/"C_results.csv", index=False, encoding="utf-8-sig")
    show = ["sample", "hyp", "cand", "n", "dR2", "share_better", "F_p"]
    print(t[t.part == "단독(기존 인자 위)"][show].round(3).to_string(index=False))
    pv = t[t.part == "C2 짝별"].pivot_table(index=["sample", "hyp", "cand"], columns="given", values="F_p"); pv.round(3).to_csv(OUT/"C2_pairwise_p.csv", encoding="utf-8-sig")
    print(pv.round(2).to_string())
    print(t[t.part.isin(["C3 고유 기여", "C4 전체 확장"])][["part"]+show].round(3).to_string(index=False))
