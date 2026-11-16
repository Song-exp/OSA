"""E 단계(변경): AMASSS 벡터 공동 PC축 — MAND·MAX·CB 대응점을 한 벡터로 이어 PCA (사전 등록: 실험기록 "E 단계 설계 변경").
F1 형태(정중 좌표계 그대로) / F2 배치·모양(전체 구성 GPA 1회: 구조 간 상대 배치 보존) / F2-MM(MAND+MAX만).
축 해석(구조·부위별 변위, 하악–상악 상대 이동, 관련 변수 상관) + 확정 인자(+크기) 위 AHI 검정.
출력: Result_py/analysis/shape_map/E_joint_axes/
"""
import base64, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import osa_experiments as X
import osa_residual_shape as RS
import osa_validate_holdout as VH
import osa_vertical as V
from shape_local_roi import roi_labels
from shape_ssm import gpa

OUT = X.R_/"analysis/shape_map/E_joint_axes"; OUT.mkdir(parents=True, exist_ok=True)
K = 10; COL = {"MAND": "tab:red", "MAX": "tab:blue", "CB": "tab:green"}


def combined(structs):
    blocks, labs, structs_of, ids0 = [], [], [], None
    for st in structs:
        ids, S, use = RS.load(st); lab = roi_labels(st, np.load(X.R_/f"shape/{st}/points.npz"))[0]
        s = pd.Series(list(S), index=ids)[use]; blocks.append(s); labs.append(lab); structs_of += [st]*S.shape[1]
    common = sorted(set.intersection(*(set(b.index) for b in blocks)), key=lambda p: int(p[2:]))
    C = np.array([np.concatenate([b[p] for b in blocks]) for p in common])          # (n, 2048·k, 3)
    return common, C, np.concatenate(labs), np.array(structs_of)


def pca(Y, k):
    mu = Y.mean(0); u, s, vt = np.linalg.svd(Y-mu, full_matrices=False); sc = (Y-mu)@vt[:k].T
    return sc/sc.std(0, ddof=1), vt[:k], (s**2/(s**2).sum())[:k], s[:k]/np.sqrt(len(Y)-1)


def describe(name, mean, vt, sd, lab, st_of, var):
    rows = []
    for i in range(len(vt)):
        disp = vt[i].reshape(-1, 3)*sd[i]                                   # +1 SD당 변위(mm 또는 GPA 단위)
        r = {"axis": f"{name}_PC{i+1}", "var": var[i]}
        for st in np.unique(st_of):
            k = st_of == st; r[f"{st}_disp"] = np.linalg.norm(disp[k], axis=1).mean(); r[f"{st}_ant"] = disp[k, 1].mean(); r[f"{st}_sup"] = disp[k, 2].mean()
        if "MAND" in st_of and "MAX" in st_of:
            r["MAND_vs_MAX_ant"] = r["MAND_ant"]-r["MAX_ant"]; r["MAND_vs_MAX_sup"] = r["MAND_sup"]-r["MAX_sup"]
        regs = {f"{st}:{g}": np.linalg.norm(disp[(st_of == st) & (lab == g)], axis=1).mean() for st in np.unique(st_of) for g in np.unique(lab[st_of == st])}
        r["top_regions"] = ", ".join(sorted(regs, key=regs.get, reverse=True)[:3]); rows.append(r)
    return pd.DataFrame(rows)


def figure(mean, vt, sd, st_of, name, i, path):
    M = vt[i].reshape(-1, 3)*sd[i]; fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for a, sgn in zip(ax, (-2, 2)):
        for st in np.unique(st_of):
            k = st_of == st; P = mean[k]+sgn*M[k]; a.scatter(P[::3, 1], P[::3, 2], s=2, c=COL[st], label=st)
        a.scatter(mean[::6, 1], mean[::6, 2], s=1, c="lightgray", zorder=0); a.set_aspect("equal"); a.set_title(f"{sgn:+d} SD (회색 = 평균)"); a.legend()
    fig.suptitle(f"{name}_PC{i+1} 시상 투영"); fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    d, f = VH.load_frozen(); axes, descs, figs = [], [], []        # load_frozen이 수직·비율 계측까지 붙임
    for name, structs, mode in (("F1", ("MAND", "MAX", "CB"), "form"), ("F2", ("MAND", "MAX", "CB"), "gpa"), ("F2MM", ("MAND", "MAX"), "gpa")):
        ids, C, lab, st_of = combined(structs)
        if mode == "gpa":
            G, cs = gpa(C); Y = G.reshape(len(G), -1); mean = G.mean(0); axes.append(pd.DataFrame({f"{name}_logCS": np.log(cs)}, index=ids))
        else: Y = C.reshape(len(C), -1); mean = C.mean(0)
        sc, vt, var, sd = pca(Y, K); A = pd.DataFrame(sc, index=ids, columns=[f"{name}_PC{i+1}" for i in range(K)]); axes.append(A)
        de = describe(name, mean, vt, sd, lab, st_of, var); descs.append(de)
        for i in range(3):
            p = OUT/f"{name}_PC{i+1}.png"; figure(mean, vt, sd, st_of, name, i, p); figs.append(p)
        print(name, "n =", len(ids), "누적 분산 PC1~5", np.round(np.cumsum(var)[:5], 3).tolist())
    A = pd.concat(axes, axis=1); A.round(4).to_csv(OUT/"joint_axes.csv", encoding="utf-8-sig")
    De = pd.concat(descs); d = d.join(A); cols = list(A.columns)
    rel = ["m_ANB", "Pog_Ba_ant", "SIZE_PC1", "Male", "LAFH_ratio", "Co_Gn", "N_Me", "BMI"]
    s85 = d[d.restored.astype(bool)]
    cor = pd.DataFrame({r: [spearmanr(s85[c], s85[r], nan_policy="omit")[0] for c in cols] for r in rel}, index=cols)
    De = De.set_index("axis").join(cor); De.round(3).to_csv(OUT/"axis_description.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(De[["var", "MAND_vs_MAX_ant", "MAND_vs_MAX_sup", "top_regions"]+rel].round(2).to_string())
    S = {"main59": d[d.index.isin(f["discovery_ids"])], "sens85": s85}; rows = []
    for sn, s in S.items():
        for h, y in X.H.items():
            for lab2, ref in (("확정 위", X.CONFIRMED), ("확정+크기 위", X.CONFIRMED+["SIZE_PC1"])):
                for c in cols: rows.append({"part": lab2, "unit": "축", "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
                for name in ("F1", "F2", "F2MM"):
                    rows.append({"part": lab2, "unit": "PC1~5 묶음", "sample": sn, "hyp": h, "add": name, **X.compare(s, y, X.B0, [f"{name}_PC{i}" for i in range(1, 6)], ref_add=ref)})
    t = X.fdr(rows, ("sample", "hyp", "part", "unit")); t.round(4).to_csv(OUT/"E_joint_results.csv", index=False, encoding="utf-8-sig")
    show = ["part", "unit", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]
    print(t[t.F_p < .10][show].round(3).to_string(index=False)); print("q<0.05:", int((t.q_BH < .05).sum()))
    img = lambda p: base64.b64encode(p.read_bytes()).decode()
    html = ['<!doctype html><meta charset="utf-8"><title>AMASSS 공동 PC축</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:0 16px}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:2px 5px}</style>',
            "<h1>AMASSS 벡터 공동 PC축 (구조 간 배치)</h1><p>F1 = 형태(정중 좌표계), F2 = 전체 구성 GPA(구조 간 배치 보존, 크기 제거), F2MM = 하악+상악만. 빨강 MAND, 파랑 MAX, 초록 CB.</p>",
            "<h2>축 해석</h2>", De.round(2).to_html(), "<h2>AHI 검정 (p < 0.10)</h2>", t[t.F_p < .10].round(3).to_html(index=False)]
    html += [f"<h3>{p.stem}</h3><img src='data:image/png;base64,{img(p)}'>" for p in figs]
    (OUT/"index.html").write_text("".join(html), encoding="utf-8"); print("saved")
