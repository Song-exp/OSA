"""D6: FH 없는 수직 패턴 (사전 등록: 실험기록 "D6 사전 등록").
D6-1 MAND_rPC4 ↔ 수직 계측 상관 / D6-2 PP_MP 하악 대응점 지도(앞뒤 위치·크기 고정) / D6-3 수직 계측 AHI 검정.
출력: Result_py/analysis/shape_map/D6_vertical/
"""
import base64, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import osa_experiments as X
import build_features as bf
import osa_proportions as P
import osa_residual_shape as RS
import osa_shape_map as SM

OUT = X.R_/"analysis/shape_map/D6_vertical"; OUT.mkdir(parents=True, exist_ok=True)
VERT = ["PP_MP", "SN_MP", "ANS_Me", "N_Me", "S_Go", "Co_Go", "LAFH_ratio", "Jarabak", "gonial"]


def ang(v):
    """정중 시상면(앞, 위) 방향각(도), 앞쪽 수평 = 0, 아래로 기울면 +."""
    return float(np.degrees(np.arctan2(-v[2], v[1])))


def vertical(d):
    rel = pd.read_csv(X.R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
    cond = pd.read_csv(bf.EXT/"condyle_all.csv", encoding="utf-8-sig").set_index("Patient"); rows = {}
    for p in d.index:
        q, _ = bf.coords(int(p[2:]), rel.loc[p].to_dict() if p in rel.index else {}, cond.loc[p] if p in cond.index else None)
        go = (q["RGo"]+q["LGo"])/2; mp = ang(q["Me"]-go)
        rows[p] = {"PP_MP": mp-ang(q["ANS"]-q["PNS"]), "SN_MP": mp-ang(q["N"]-q["S"]), "ANS_Me": float(np.linalg.norm(q["ANS"]-q["Me"])),
                   "S_Go": float(np.linalg.norm(q["S"]-go))}
    v = pd.DataFrame(rows).T.astype(float); d = P.build(d).join(v, rsuffix="_v")
    if "S_Go_v" in d: d["S_Go"] = d.pop("S_Go_v")
    d["LAFH_ratio"] = d.ANS_Me/d.N_Me
    return d


assert abs(ang(np.array([0, 1., -1.]))-45) < 1e-9                           # 앞으로 가며 아래로 45° = +45

if __name__ == "__main__":
    X.SOURCE = "final_v2"; d = X.data(); X.size_axis(d); d = vertical(d[d.restored.astype(bool)].copy())
    sc, *_ = RS.residual_pcs(d, "MAND"); d = d.join(sc)
    # D6-1
    c1 = pd.DataFrame([{"measure": m, "rho_MAND_rPC4": spearmanr(d.MAND_rPC4, d[m], nan_policy="omit")[0],
                        "p": spearmanr(d.MAND_rPC4, d[m], nan_policy="omit")[1], "rho_ANB": spearmanr(d.m_ANB, d[m], nan_policy="omit")[0],
                        "rho_SIZE": spearmanr(d.SIZE_PC1, d[m], nan_policy="omit")[0]} for m in VERT])
    c1.round(3).to_csv(OUT/"D6_1_rPC4_corr.csv", index=False, encoding="utf-8-sig"); print(c1.round(3).to_string(index=False))
    # D6-2 형상 지도: PP_MP, 공변량 나이·성별·BMI·크기·ANB·Pog_Ba
    ids, S, lab, use = SM.load_shapes(); dd = d.reindex(ids); cov = ["Age", "Male", "BMI", "SIZE_PC1", "m_ANB", "Pog_Ba_ant"]
    ok = use & dd[["PP_MP"]+cov].notna().all(1).to_numpy(); m = dd.loc[ok, "PP_MP"].to_numpy(float); m = (m-m.mean())/m.std()
    disp, pr2, tot, p = SM.vertex_regression(S[ok], m, dd.loc[ok, cov].to_numpy(float), np.random.default_rng(0))
    reg = pd.DataFrame([{"region": r, "mean_disp_mm": np.linalg.norm(disp[lab == r], axis=1).mean(), "ant_mm": disp[lab == r, 1].mean(),
                         "sup_mm": disp[lab == r, 2].mean()} for r in ("symphysis", "body", "gonial", "ramus_condyle")]).assign(total_partial_R2=tot, perm_p=p, n=int(ok.sum()))
    reg.round(3).to_csv(OUT/"D6_2_ppmp_map.csv", index=False, encoding="utf-8-sig"); print(reg.round(3).to_string(index=False))
    SM.figure(S[ok].mean(0), disp, pr2, lab, f"PP_MP +1 SD (나이·성별·BMI·크기·ANB·Pog_Ba 고정), 부분 R² {tot:.3f} (순열 p={p:.3f}), n={ok.sum()}", OUT/"PP_MP_map.png")
    # D6-3
    rows = []
    for sn, s in (("main59", d[d.analysis_list.astype(bool)]), ("sens85", d)):
        for h, y in X.H.items():
            for c in VERT:
                for lab2, ref in (("B0 위", []), ("기존 인자 위", X.CONFIRMED), ("기존 인자+크기 위", X.CONFIRMED+["SIZE_PC1"])):
                    rows.append({"part": lab2, "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
    t = X.fdr(rows, ("sample", "hyp", "part")); t.round(4).to_csv(OUT/"D6_3_ahi.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[t["add"] == "PP_MP"][["part", "sample", "hyp", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]].round(3).to_string(index=False))
    print(t[t.F_p < .10][["part", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]].round(3).to_string(index=False))
    d[VERT+["MAND_rPC4"]].round(3).to_csv(OUT/"D6_values.csv", encoding="utf-8-sig")
    img = base64.b64encode((OUT/"PP_MP_map.png").read_bytes()).decode()
    html = ['<!doctype html><meta charset="utf-8"><title>수직 패턴</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:0 16px}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 6px}</style>',
            "<h1>D6 FH 없는 수직 패턴</h1><h2>D6-1 MAND_rPC4 ↔ 수직 계측</h2>", c1.round(3).to_html(index=False),
            "<h2>D6-2 PP_MP 하악 지도</h2>", reg.round(3).to_html(index=False), f'<img src="data:image/png;base64,{img}">',
            "<h2>D6-3 AHI (p < 0.10)</h2>", t[t.F_p < .10].round(3).to_html(index=False)]
    (OUT/"index.html").write_text("".join(html), encoding="utf-8"); print("saved")
