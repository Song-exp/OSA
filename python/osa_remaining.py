"""D5·D7·D8 (사전 등록: 실험기록 "D5·D7·D8 사전 등록").
D5 골성 비인두 → 비인두 단면적, 앙와위 초과 / D7 상대성장(크기 vs 크기에 따른 모양) / D8 기도 최소 위치·프로파일 모양.
출력: Result_py/analysis/shape_map/D5D7D8/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import osa_experiments as X
import osa_validate_holdout as VH
import osa_joint_axes as JA
from shape_ssm import gpa

OUT = X.R_/"analysis/shape_map/D5D7D8"; OUT.mkdir(parents=True, exist_ok=True)
REFS = {"기존 인자 위": X.CONFIRMED, "기존 인자+VERT_Z 위": X.CONFIRMED+["VERT_Z"]}


def ols(s, y, xs):
    u = s.dropna(subset=[y]+xs); Z = (u[xs]-u[xs].mean())/u[xs].std(); f = sm.OLS(u[y], sm.add_constant(Z)).fit()
    return len(u), f


if __name__ == "__main__":
    d, f = VH.load_frozen(); d["SUP_excess"] = d.logAHI_SUP-d.logAHI
    st = pd.read_csv(X.R_/"shape/fine/soft_tissue.csv", encoding="utf-8-sig").set_index("Patient")
    for c in ("nasopharynx_min_area", "nasopharynx_mean_area", "soft_palate_length"): d[c] = st[c].reindex(d.index)
    prof = pd.read_csv(X.R_/"shape/extra/airway_profile.csv", encoding="utf-8-sig").set_index("Patient")
    aw = [f"AW_p{i:02d}" for i in range(20)]; d["AW_min_pos"] = prof.AW_min_pos.reindex(d.index); d["Hy_sup"] = d.c_Hy_sup
    s85 = d[d.restored.astype(bool)]; S = {"main59": d[d.index.isin(f["discovery_ids"])], "sens85": s85}
    coef, ahi = [], []
    # D5 (i) 골성 비인두 → 비인두 단면
    for y in ("nasopharynx_min_area", "nasopharynx_mean_area"):
        n, fit = ols(s85, y, X.B0+["ANS_PNS", "PNS_Ba_depth", "S_Ba"])
        for c in ("ANS_PNS", "PNS_Ba_depth", "S_Ba"): coef.append({"part": "D5 골성 비인두 → 비인두", "y": y, "term": c, "n": n, "beta_std": fit.params[c], "p": fit.pvalues[c], "R2": fit.rsquared})
    # D8 최소 위치
    n, fit = ols(s85, "AW_min_pos", X.B0+["m_ANB", "Pog_Ba_ant", "VERT_Z", "soft_palate_length", "Hy_sup"])
    for c in ("m_ANB", "Pog_Ba_ant", "VERT_Z", "soft_palate_length", "Hy_sup"): coef.append({"part": "D8 최소 위치 ~ 뼈·연조직", "y": "AW_min_pos", "term": c, "n": n, "beta_std": fit.params[c], "p": fit.pvalues[c], "R2": fit.rsquared})
    # D8 기도 프로파일 모양 PC (표준화 PCA, 85명 비지도)
    P = s85[aw].dropna(); Z = (P-P.mean())/P.std(); u, sv, vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    for i in range(3):
        if vt[i, np.argmax(np.abs(vt[i]))] < 0: vt[i] *= -1
        d[f"AWshape_PC{i+1}"] = pd.Series(Z.to_numpy()@vt[i], index=P.index).reindex(d.index)
    awvar = (sv**2/(sv**2).sum())[:3]; pd.DataFrame(vt[:3].T, index=aw, columns=["PC1", "PC2", "PC3"]).round(3).to_csv(OUT/"D8_awshape_loadings.csv", encoding="utf-8-sig")
    # D7 상대성장
    ids, C, _, _ = JA.combined(("MAND", "MAX", "CB")); G, cs = gpa(C); Y = G.reshape(len(G), -1); Y = Y-Y.mean(0)
    sex = d.Male.reindex(ids).to_numpy(float); lcs = np.log(cs); Xa = np.c_[np.ones(len(ids)), lcs, sex]
    B = np.linalg.lstsq(Xa, Y, rcond=None)[0]; E = Y-Xa@B; X0 = np.c_[np.ones(len(ids)), sex]; E0 = Y-X0@np.linalg.lstsq(X0, Y, rcond=None)[0]
    allo_r2 = 1-(E**2).sum()/(E0**2).sum(); rng = np.random.default_rng(0); null = []
    for _ in range(499):
        xp = np.c_[np.ones(len(ids)), lcs[rng.permutation(len(ids))], sex]; Ep = Y-xp@np.linalg.lstsq(xp, Y, rcond=None)[0]; null.append(1-(Ep**2).sum()/(E0**2).sum())
    allo_p = (np.sum(np.array(null) >= allo_r2)+1)/500; b = B[1]/np.linalg.norm(B[1])
    d["AlloShape"] = pd.Series(Y@b, index=ids).reindex(d.index); d["logCS_all"] = pd.Series(lcs, index=ids).reindex(d.index)
    print("D7 상대성장: log CS가 설명하는 모양 분산(성별 보정) =", round(allo_r2, 4), "순열 p =", allo_p)
    print("상관(85명):", s85.join(d[["AlloShape", "logCS_all"]], rsuffix="_").pipe(lambda x: x)[["SIZE_PC1"]].join(d[["AlloShape", "logCS_all"]]).corr(method="spearman").round(2).to_string())
    # AHI 검정
    feats = {"D5": ["nasopharynx_min_area", "nasopharynx_mean_area", "PNS_Ba_depth", "ANS_PNS"], "D7": ["AlloShape", "logCS_all"],
             "D8": ["AW_min_pos", "AWshape_PC1", "AWshape_PC2", "AWshape_PC3"]}
    for sn, s in S.items():
        s = s.join(d[["AlloShape", "logCS_all", "AWshape_PC1", "AWshape_PC2", "AWshape_PC3"]], rsuffix="_dup") if "AlloShape" not in s else s
        s = d.loc[s.index]
        for y, ylab in ((X.H["H1"], "H1"), (X.H["H2"], "H2"), ("SUP_excess", "앙와위 초과")):
            for part, cs_ in feats.items():
                for c in cs_:
                    for lab, ref in REFS.items():
                        ahi.append({"part": lab, "stage": part, "sample": sn, "hyp": ylab, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
    t = X.fdr(ahi, ("sample", "hyp", "part")); t.round(4).to_csv(OUT/"D5D7D8_ahi.csv", index=False, encoding="utf-8-sig")
    cf = pd.DataFrame(coef); cf.round(4).to_csv(OUT/"D5D8_coef.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); print(cf.round(3).to_string(index=False)); print("기도 모양 PC 분산", np.round(awvar, 3).tolist())
    print(t[t.F_p < .10][["part", "stage", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]].round(3).to_string(index=False))
    pd.DataFrame([{"allometry_R2": allo_r2, "perm_p": allo_p}]).to_csv(OUT/"D7_allometry.csv", index=False, encoding="utf-8-sig")
