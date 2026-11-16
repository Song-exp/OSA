"""D-M5: 잔여 모양 — SSM 형상에서 ANB·Pog_Ba·SIZE_PC1·성별 성분을 뺀 뒤 PCA, 기존 인자 + 크기 위에서 AHI 검정.
사전 등록: 실험기록 "D-M5 사전 등록". 구조 = MAND, MAX, CB(참고). 출력: Result_py/analysis/shape_map/M5_residual/
"""
import base64, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import osa_experiments as X
from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import landmarks

OUT = X.R_/"analysis/shape_map/M5_residual"; OUT.mkdir(parents=True, exist_ok=True)
RESID_ON = ["m_ANB", "Pog_Ba_ant", "SIZE_PC1", "Male"]; K = 5
REF = X.CONFIRMED+["SIZE_PC1"]


def load(struct):
    """구조 SSM 대응점을 정중 좌표계(PNS 원점)로. 정합 실패 확정·의심 제외."""
    z = np.load(X.R_/f"shape/{struct}/points.npz"); ids = [f"P_{i}" for i in z["ids"]]; S = []
    for k, p in enumerate(ids):
        lps = z["frame"][k]@z["head_R"][k]+z["head_origin"][k]; o, R, _ = midline_frame(landmarks(int(p[2:]))[0]); S.append((lps-o)@R.T)
    q = pd.read_csv(X.R_/f"shape/{struct}/review/qc_review.csv", encoding="utf-8-sig").set_index("Patient").reindex(ids)
    bad = q.fit_flag.astype(str).str.lower().eq("true")
    for c in ("confirmed_fail_frame", "confirmed_fail_procrustes"): bad |= q[c].astype(str).str.lower().isin(["true", "1"])
    return ids, np.array(S), (~bad).to_numpy()


def residual_pcs(d, struct):
    ids, S, use = load(struct); dd = d.reindex(ids); ok = use & dd[RESID_ON].notna().all(1).to_numpy()
    Y = S[ok].reshape(ok.sum(), -1); C = np.c_[np.ones(ok.sum()), dd.loc[ok, RESID_ON].to_numpy(float)]
    E = Y-C@np.linalg.lstsq(C, Y, rcond=None)[0]; mu = E.mean(0)
    u, sv, vt = np.linalg.svd(E-mu, full_matrices=False); var = sv**2/(sv**2).sum()
    sc = pd.DataFrame((E-mu)@vt[:K].T/sv[:K]*np.sqrt(ok.sum()-1), index=np.array(ids)[ok], columns=[f"{struct}_rPC{i+1}" for i in range(K)])
    return sc, var[:K], vt[:K], sv[:K]/np.sqrt(ok.sum()-1), S[ok].mean(0)


def mode_figure(mean, v, sd, title, path):
    """잔여 PC ±2 SD를 평균 형상에 더한 시상·축 투영(잔여 성분만의 변화)."""
    M = v.reshape(-1, 3)*sd; fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, (i, j, nm) in zip(ax, ((1, 2, "시상(앞뒤–위아래)"), (0, 1, "축(좌우–앞뒤)"))):
        for sgn, col in ((-2, "tab:blue"), (2, "tab:red")): a.scatter(mean[:, i]+sgn*M[:, i], mean[:, j]+sgn*M[:, j], s=3, c=col, alpha=.5, label=f"{sgn:+d} SD")
        a.set_aspect("equal"); a.set_title(nm); a.legend()
    fig.suptitle(title); fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    X.SOURCE = "final_v2"; d = X.data(); X.size_axis(d); rows, info, keep = [], [], {}
    for st in ("MAND", "MAX", "CB"):
        sc, var, vt, sd, mean = residual_pcs(d, st); d = d.join(sc); keep[st] = (vt, sd, mean)
        info.append({"struct": st, "n": len(sc), **{f"var_rPC{i+1}": v for i, v in enumerate(var)}})
    S = {"main59": d[d.analysis_list.astype(bool)], "sens85": d[d.restored.astype(bool)]}
    for sn, s in S.items():
        for h, y in X.H.items():
            for st in ("MAND", "MAX", "CB"):
                cols = [f"{st}_rPC{i+1}" for i in range(K)]
                for c in cols: rows.append({"part": "PC 단독", "struct": st, "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=REF)})
                rows.append({"part": "구조 묶음", "struct": st, "sample": sn, "hyp": h, "add": f"{st}_rPC1-{K}", **X.compare(s, y, X.B0, cols, ref_add=REF)})
    t = X.fdr(rows, ("sample", "hyp", "part")); t.round(4).to_csv(OUT/"M5_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(info).round(3).to_csv(OUT/"M5_variance.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); print(pd.DataFrame(info).round(3).to_string(index=False))
    print(t[t.F_p < .10][["part", "struct", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]].round(3).to_string(index=False))
    print(t[t.part == "구조 묶음"][["struct", "sample", "hyp", "n", "dR2", "F_p", "q_BH"]].round(3).to_string(index=False))
    # 후보 선별 통과 PC 그림 (주 분석 p<0.10, ΔR²>0, 개선 ≥ 70%, 85명 같은 부호)
    figs = []
    for (c, h), g in t[t.part == "PC 단독"].groupby(["add", "hyp"]):
        m, e = g[g["sample"] == "main59"].iloc[0], g[g["sample"] == "sens85"].iloc[0]
        if m.F_p < .10 and m.dR2 > 0 and m.share_better >= .7 and np.sign(float(str(m.coef_std).split(";")[0])) == np.sign(float(str(e.coef_std).split(";")[0])):
            st, k = c.split("_rPC"); vt, sd, mean = keep[st]; f = OUT/f"{c}_{h}.png"
            mode_figure(mean, vt[int(k)-1], sd[int(k)-1], f"{c} ±2 SD (잔여 모양, {h} 후보)", f); figs.append(f)
    html = ['<!doctype html><meta charset="utf-8"><title>잔여 모양</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto;padding:0 16px}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 6px}</style>',
            "<h1>D-M5 잔여 모양: ANB·Pog_Ba·크기·성별로 설명되지 않는 형상</h1>", pd.DataFrame(info).round(3).to_html(index=False),
            "<h2>p < 0.10</h2>", t[t.F_p < .10].round(3).to_html(index=False)] + [f'<img src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}">' for f in figs]
    (OUT/"index.html").write_text("".join(html), encoding="utf-8"); print("후보 그림", [f.name for f in figs])
