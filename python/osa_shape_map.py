"""D1: 하악 후퇴의 형태적 근원 — ANB·Pog_Ba ↔ 하악 SSM 대응점 지도 (계측_형태_통합_탐구안 D1, M1·M2).

1) 대응점 2048개를 FH 머리 좌표계(points.npz)에서 Po 미사용 정중 좌표계(PNS 원점, cbct_eb_detect.midline_frame)로 옮긴다.
   두 판: 위치 포함(form, 정중 좌표계 그대로) / 모양만(크기 유지 Procrustes: 회전·이동만 제거, 평균 중심 크기로 환산).
2) 대응점마다 3D 좌표 ~ 계측(z) + 나이 + 성별 + BMI 회귀 → 계측 +1 SD당 변위 벡터(mm), 대응점 부분 R².
   전체 부분 R²(모든 좌표 합)를 Freedman–Lane 순열 999회로 검정.
3) 부위(턱끝·하악체·하악각·하악지과두, shape_local_roi.roi_labels)별 평균 변위(앞뒤·위아래·크기).
4) 분해: ANB·Pog_Ba ~ 하악체 길이(Go 중점–Pog) + 하악지 기울기(과두 최상점–Go, 정중 시상면) + 하악각(과두–Go–Me) + 공변량.
표본: SSM이 있는 85명(PSG 불필요). 계측은 최종 ANB(등급 A~D)와 Pog_Ba_ant.
출력: Result_py/analysis/shape_map/D1_mandible/ (그림 PNG, index.html, CSV)
"""
import base64, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import osa_experiments as X
import build_features as bf
from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import landmarks
from shape_local_roi import roi_labels

OUT = X.R_/"analysis/shape_map/D1_mandible"; OUT.mkdir(parents=True, exist_ok=True)
COV = ["Age", "Male", "BMI"]; NPERM = 999


def load_shapes():
    z = np.load(X.R_/"shape/MAND/points.npz"); ids = [f"P_{i}" for i in z["ids"]]; S = []
    for k, p in enumerate(ids):
        lps = z["frame"][k]@z["head_R"][k]+z["head_origin"][k]          # to_frame의 역변환: y = (x−o)R^T → x = yR + o
        o, R, _ = midline_frame(landmarks(int(p[2:]))[0]); S.append((lps-o)@R.T)
    S = np.array(S); lab = roi_labels("MAND", z)[0]
    rv = X.R_/"shape/MAND/review/qc_review.csv"
    q = pd.read_csv(rv, encoding="utf-8-sig").set_index("Patient").reindex(ids)                   # 정합 실패 확정·의심 환자 제외
    bad = q.fit_flag.astype(str).str.lower().eq("true")
    for c in ("confirmed_fail_frame", "confirmed_fail_procrustes"): bad |= q[c].astype(str).str.lower().isin(["true", "1"])
    use = (~bad).to_numpy()
    return ids, S, lab, use


def opa_noscale(S):
    """크기 유지 Procrustes: 각 형태를 평균에 회전·이동만 맞춤(반복). 단위 mm."""
    Y = S-S.mean(1, keepdims=True); ref = Y[0]
    for _ in range(10):
        for i in range(len(Y)):
            u, _, vt = np.linalg.svd(Y[i].T@ref); d = np.sign(np.linalg.det(u@vt)); Y[i] = Y[i]@(u@np.diag([1, 1, d])@vt)
        ref = Y.mean(0)
    return Y


def vertex_regression(Y, m, C, rng):
    """Y (n,M,3), m (n,) 표준화, C (n,k). 반환: 변위 (M,3) mm/SD, 대응점 부분 R² (M,), 전체 부분 R², 순열 p."""
    n, M, _ = Y.shape; Yf = Y.reshape(n, -1); Z0 = sm.add_constant(C); Z1 = np.c_[Z0, m]
    b0 = np.linalg.lstsq(Z0, Yf, rcond=None)[0]; e0 = Yf-Z0@b0; f0 = Z0@b0
    def fit(Yt):
        b = np.linalg.lstsq(Z1, Yt, rcond=None)[0]; e = Yt-Z1@b; return b, e
    b1, e1 = fit(Yf)
    sse0, sse1 = (e0**2).reshape(n, M, 3).sum((0, 2)), (e1**2).reshape(n, M, 3).sum((0, 2))
    pr2 = 1-sse1/sse0; tot = 1-sse1.sum()/sse0.sum()
    null = []
    for _ in range(NPERM):                                            # Freedman–Lane: 축소 모델 잔차를 섞음
        Yp = f0+e0[rng.permutation(n)]; _, ep = fit(Yp); null.append(1-(ep**2).sum()/((Yp-Z0@np.linalg.lstsq(Z0, Yp, rcond=None)[0])**2).sum())
    p = (np.sum(np.array(null) >= tot)+1)/(NPERM+1)
    return b1[-1].reshape(M, 3), pr2, tot, p


def angles(d):
    """하악체 길이·하악지 기울기·하악각 (정중 좌표계, build_features.coords)."""
    rel = pd.read_csv(X.R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
    cond = pd.read_csv(bf.EXT/"condyle_all.csv", encoding="utf-8-sig").set_index("Patient"); rows = {}
    for p in d.index:
        q, _ = bf.coords(int(p[2:]), rel.loc[p].to_dict() if p in rel.index else {}, cond.loc[p] if p in cond.index else None)
        if "RSup" not in q or not np.isfinite(q["RSup"]).all() or not np.isfinite(q["LSup"]).all(): continue
        go, sup = (q["RGo"]+q["LGo"])/2, (q["RSup"]+q["LSup"])/2; ram, bod = sup-go, q["Me"]-go
        rows[p] = {"body_len": float(np.linalg.norm(q["Pog"]-go)),
                   "ramus_incl": float(np.degrees(np.arctan2(-ram[1], ram[2]))),   # 위 방향 대비 뒤로 기운 각(+ = 하악지 상단이 뒤로)
                   "gonial_angle": float(np.degrees(np.arccos(ram@bod/np.linalg.norm(ram)/np.linalg.norm(bod)))),
                   "ramus_len": float(np.linalg.norm(ram))}
    return pd.DataFrame(rows).T


def figure(mean, disp, pr2, lab, title, path):
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5)); mag = np.linalg.norm(disp, axis=1); vmax = np.percentile(mag, 99)
    for a, (i, j, nm) in zip(ax, ((1, 2, "시상(앞뒤–위아래)"), (0, 1, "축(좌우–앞뒤)"), (0, 2, "관상(좌우–위아래)"))):
        sc = a.scatter(mean[:, i], mean[:, j], c=mag, s=6, cmap="viridis", vmin=0, vmax=vmax)
        k = np.argsort(-mag)[:120:3]
        a.quiver(mean[k, i], mean[k, j], disp[k, i], disp[k, j], angles="xy", scale_units="xy", scale=.15, color="red", width=.002)
        a.set_aspect("equal"); a.set_title(nm); a.set_xlabel(["좌(mm)", "앞(mm)", "위(mm)"][i]); a.set_ylabel(["좌(mm)", "앞(mm)", "위(mm)"][j])
    fig.colorbar(sc, ax=ax, label="계측 +1 SD당 변위 크기 (mm)", shrink=.8)
    fig.suptitle(title+"  (빨간 화살표 = 변위 방향, 길이 ×6.7)"); fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    X.SOURCE = "final_v2"; d = X.data(); X.size_axis(d); ids, S, lab, use = load_shapes()
    SIZE = "--size" in sys.argv                                       # 크기 보정 판: 공변량에 SIZE_PC1 추가
    if SIZE: COV = COV+["SIZE_PC1"]; OUT = OUT.parent/"D1_mandible_size"; OUT.mkdir(parents=True, exist_ok=True)
    d = d.reindex(ids); ok = use & d[["f_ANB" if "f_ANB" in d else "m_ANB", "Pog_Ba_ant"]+COV].notna().all(1).to_numpy()
    anb = "m_ANB"                                                     # final_v2에서 m_ANB = 최종 ANB(A~D)
    rng = np.random.default_rng(0); rows, figs = [], []
    shapes = {"위치 포함(정중 좌표계)": S[ok], "모양만(크기 유지 Procrustes)": opa_noscale(S[ok])}
    for meas in (anb, "Pog_Ba_ant"):
        m = d.loc[ok, meas].to_numpy(float); m = (m-m.mean())/m.std(); C = d.loc[ok, COV].to_numpy(float)
        for ver, Y in shapes.items():
            disp, pr2, tot, p = vertex_regression(Y, m, C, rng); mean = Y.mean(0)
            for r in ("symphysis", "body", "gonial", "ramus_condyle"):
                k = lab == r
                rows.append({"measure": meas, "version": ver, "region": r, "n": int(ok.sum()), "total_partial_R2": tot, "perm_p": p,
                             "mean_disp_mm": np.linalg.norm(disp[k], axis=1).mean(), "ant_mm": disp[k, 1].mean(), "sup_mm": disp[k, 2].mean(),
                             "lat_abs_mm": np.abs(disp[k, 0]).mean(), "mean_partial_R2": pr2[k].mean()})
            nm = f"{meas}_{'form' if ver.startswith('위치') else 'shape'}"
            figure(mean, disp, pr2, lab, f"{meas} +1 SD — {ver}, 전체 부분 R² {tot:.3f} (순열 p={p:.3f}), n={ok.sum()}", OUT/f"{nm}.png"); figs.append((nm, ver, meas))
            np.save(OUT/f"{nm}_disp.npy", disp)
    t = pd.DataFrame(rows); t.round(4).to_csv(OUT/"region_summary.csv", index=False, encoding="utf-8-sig"); print(t.round(3).to_string(index=False))

    # 4) 분해
    g = angles(d.loc[np.array(ids)[ok]]); dd = d.join(g, how="inner"); dec = []
    for meas in (anb, "Pog_Ba_ant"):
        s = dd.dropna(subset=[meas, "body_len", "ramus_incl", "gonial_angle"]+COV); Z = s[["body_len", "ramus_incl", "gonial_angle"]]; Z = (Z-Z.mean())/Z.std()
        f = sm.OLS(s[meas], sm.add_constant(pd.concat([Z, s[COV]], axis=1))).fit()
        for c in Z: dec.append({"measure": meas, "component": c, "n": len(s), "beta_per_SD": f.params[c], "p": f.pvalues[c], "R2_model": f.rsquared,
                                "spearman": s[[meas, c]].corr(method="spearman").iloc[0, 1]})
    dt = pd.DataFrame(dec); dt.round(4).to_csv(OUT/"decomposition.csv", index=False, encoding="utf-8-sig"); print(dt.round(3).to_string(index=False))
    g.describe().round(2).to_csv(OUT/"angles_describe.csv", encoding="utf-8-sig")
    ah = pd.DataFrame()
    if SIZE:   # 크기와 분리한 하악체 길이(크기 축·성별 잔차)가 AHI와 관련되는가
        f = sm.OLS(dd.body_len, sm.add_constant(dd[["SIZE_PC1", "Male"]])).fit(); dd["body_len_res"] = f.resid
        print("하악체 길이 ~ 크기 축 + 성별 R²", round(f.rsquared, 3))
        rows2 = []
        for sn, s2 in (("main59", dd[dd.analysis_list.astype(bool)]), ("sens85", dd)):
            for h, y in X.H.items():
                for lab2, add, ref in (("B0 위 하악체 길이", ["body_len"], []), ("B0 위 크기 보정 길이", ["body_len_res"], []),
                                       ("B0+크기 위 하악체 길이", ["body_len"], ["SIZE_PC1"]), ("기존 인자 위 크기 보정 길이", ["body_len_res"], X.CONFIRMED),
                                       ("기존 인자+크기 위 하악체 길이", ["body_len"], X.CONFIRMED+["SIZE_PC1"])):
                    rows2.append({"part": lab2, "sample": sn, "hyp": h, **X.compare(s2, y, X.B0, add, ref_add=ref)})
        ah = pd.DataFrame(rows2); ah.round(4).to_csv(OUT/"bodylen_ahi.csv", index=False, encoding="utf-8-sig")
        print(ah[["part", "sample", "hyp", "n", "dR2", "share_better", "F_p", "coef_std"]].round(3).to_string(index=False))

    img = lambda f: base64.b64encode((OUT/f"{f}.png").read_bytes()).decode()
    sub = " (크기 보정: 공변량에 SIZE_PC1 추가)" if SIZE else ""
    html = ['<!doctype html><meta charset="utf-8"><title>하악 후퇴 형태 지도</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:0 16px}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px}</style>',
            f"<h1>D1 하악 후퇴의 형태적 근원: ANB·Pog_Ba ↔ 하악 대응점 지도{sub}</h1>",
            "<p>대응점 2048개, 정중 좌표계(PNS 원점, Po 미사용). 공변량 나이·성별·BMI. 색 = 계측 +1 SD당 변위 크기, 빨간 화살표 = 방향.</p>",
            "<h2>부위별 요약</h2>", t.round(3).to_html(index=False), "<h2>분해: 하악체 길이·하악지 기울기·하악각</h2>", dt.round(3).to_html(index=False)]
    if len(ah): html += ["<h2>크기와 분리한 하악체 길이 ↔ AHI</h2>", ah.round(3).to_html(index=False)]
    for nm, ver, meas in figs: html += [f"<h2>{meas} — {ver}</h2>", f'<img src="data:image/png;base64,{img(nm)}">']
    (OUT/"index.html").write_text("".join(html), encoding="utf-8"); print("saved", OUT/"index.html")
