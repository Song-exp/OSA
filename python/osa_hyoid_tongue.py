"""D3·D4: 설골–하악 연결, 구강 공간–혀 (사전 등록: 실험기록 "D3·D4 사전 등록").
D3-1 MP_H 하악 대응점 지도 / D3-2 설골–혀 상관 / D4 연조직 부하 비율 / AHI 검정(B0 위, 기존 인자 위, 기존 인자 + VERT_Z 위).
출력: Result_py/analysis/shape_map/D3D4/
"""
import base64, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import osa_experiments as X
import osa_validate_holdout as VH
import osa_shape_map as SM

OUT = X.R_/"analysis/shape_map/D3D4"; OUT.mkdir(parents=True, exist_ok=True)
D3 = ["MP_H", "Hy_Me", "Hy_Ba_dist", "Hy_sup", "Hy_ant"]; D4 = ["SoftLoad_MMV", "tongue_MMVhull", "InterGonial", "MMV_mL"]
MAIN = {"MP_H": 1, "SoftLoad_MMV": 1}


if __name__ == "__main__":
    d, f = VH.load_frozen()
    rel = pd.read_csv(X.R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
    st = pd.read_csv(X.R_/"shape/fine/soft_tissue.csv", encoding="utf-8-sig").set_index("Patient")
    for c in ("MP_H", "Hy_Me"):
        if c not in d: d[c] = rel[c].reindex(d.index)
    d["Hy_sup"], d["Hy_ant"] = d.c_Hy_sup, d.c_Hy_ant
    d["SoftLoad_MMV"] = (d.tongue_mL+st.soft_palate_volume.reindex(d.index))/d.MMV_mL
    d["tongue_MMVhull"] = d.tongue_mL/d.MMV_hull_mL
    s85 = d[d.restored.astype(bool)]
    # D3-2 상관
    tv = ["tongue_c_sup", "tongue_low_sup", "tongue_c_ant", "tongue_mL", "VERT_Z", "m_ANB", "Pog_Ba_ant", "SIZE_PC1", "BMI", "Age", "Male"]
    cor = pd.DataFrame({c: [spearmanr(s85[r], s85[c], nan_policy="omit")[0] for r in D3+D4] for c in tv}, index=D3+D4)
    cor.round(3).to_csv(OUT/"D3D4_corr.csv", encoding="utf-8-sig"); pd.set_option("display.width", 250); print(cor.round(2).to_string())
    # D3-1 형상 지도
    ids, S, lab, use = SM.load_shapes(); dd = d.reindex(ids); cov = ["Age", "Male", "BMI", "SIZE_PC1", "m_ANB", "Pog_Ba_ant"]
    ok = use & dd[["MP_H"]+cov].notna().all(1).to_numpy(); m = dd.loc[ok, "MP_H"].to_numpy(float); m = (m-m.mean())/m.std()
    disp, pr2, tot, p = SM.vertex_regression(S[ok], m, dd.loc[ok, cov].to_numpy(float), np.random.default_rng(0))
    reg = pd.DataFrame([{"region": r, "mean_disp_mm": np.linalg.norm(disp[lab == r], axis=1).mean(), "ant_mm": disp[lab == r, 1].mean(),
                         "sup_mm": disp[lab == r, 2].mean(), "lat_abs_mm": np.abs(disp[lab == r, 0]).mean()} for r in ("symphysis", "body", "gonial", "ramus_condyle")]).assign(total_partial_R2=tot, perm_p=p, n=int(ok.sum()))
    reg.round(3).to_csv(OUT/"D3_mph_map.csv", index=False, encoding="utf-8-sig"); print(reg.round(3).to_string(index=False))
    SM.figure(S[ok].mean(0), disp, pr2, lab, f"MP_H +1 SD (나이·성별·BMI·크기·ANB·Pog_Ba 고정), 부분 R² {tot:.3f} (순열 p={p:.3f}), n={ok.sum()}", OUT/"MP_H_map.png")
    # AHI
    rows = []
    for sn, s in (("main59", d[d.index.isin(f["discovery_ids"])]), ("sens85", s85)):
        for h, y in X.H.items():
            for c in D3+D4:
                for lab2, ref in (("B0 위", []), ("기존 인자 위", X.CONFIRMED), ("기존 인자+VERT_Z 위", X.CONFIRMED+["VERT_Z"])):
                    rows.append({"part": lab2, "kind": "사전 가설" if c in MAIN else "탐색", "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
    t = X.fdr(rows, ("sample", "hyp", "part", "kind")); t.round(4).to_csv(OUT/"D3D4_ahi.csv", index=False, encoding="utf-8-sig")
    show = ["part", "kind", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]
    print(t[t["add"].isin(MAIN)][show].round(3).to_string(index=False)); print(t[(t.F_p < .10) & ~t["add"].isin(MAIN)][show].round(3).to_string(index=False))
    img = base64.b64encode((OUT/"MP_H_map.png").read_bytes()).decode()
    html = ['<!doctype html><meta charset="utf-8"><title>설골·혀</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:0 16px}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:2px 5px}</style>',
            "<h1>D3 설골–하악, D4 구강 공간–혀</h1><h2>상관</h2>", cor.round(2).to_html(), "<h2>MP_H 하악 지도</h2>", reg.round(3).to_html(index=False),
            f'<img src="data:image/png;base64,{img}">', "<h2>AHI (사전 가설 + p < 0.10)</h2>", t[t["add"].isin(MAIN) | (t.F_p < .10)].round(3).to_html(index=False)]
    (OUT/"index.html").write_text("".join(html), encoding="utf-8"); print("saved")
