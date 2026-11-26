"""연조직 3부위(인두 기도·비강·혀)의 크기 제거 모양 PCA. 크기는 부피로 따로 두고, 모양만 PCA(n_components=0.95)로 압축한다.

모양 입력: 부위마다 20구간 단면적 프로파일 ÷ 그 환자의 평균 단면적(크기 제거, 구간별 표준화는 하지 않음).
  인두 기도: shape/extra/airway_profile.csv AW_p00~19 (Eb+3 mm → PNS 높이, 중심선 수직 단면, 밝기 규칙)
  비강:     shape/extra/nasal_profile.csv NC_p00~19 (PNS → ANS, TotalSegmentator 비강 라벨 관상 단면)
  혀:       이 스크립트가 만든다. TotalSegmentator head_muscles 9번(0.5 mL 미만 조각 제거)을 정중 좌표계로 옮겨
            혀뿌리(가장 뒤) → 혀끝(가장 앞)을 20구간으로 나눈 관상 단면적. 라벨이 영상 가장자리에 닿은 환자(잘림)는 PCA에서 뺀다.
그림: 예시 환자 P_2 정중 시상 단면 · 구간별 모양 설명률 · PC1~PC3 모양 변화 · 로딩(로딩² 기여율).
출력: Result_py/analysis/softtissue_pca/ (tongue_profile.csv 캐시, {region}_scores.csv, {region}_loadings.csv, variance.csv, softtissue_pca.png)
사용: python softtissue_profile_pca.py           python softtissue_profile_pca.py --retongue   (혀 프로파일 다시 계산)
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA

import cbct_pmca as pm
from cbct_eb_detect import midline_frame
from cbct_nav import ts_nasal
from cbct_patient_pipeline import CB, OUT as V2, landmarks
from check_landmarks import read_nrrd
from run_ali import ROOT
from shape_tongue_sinus import clean, coords, read, touches_edge

DEST = ROOT/"Result_py/analysis/softtissue_pca"; SH = ROOT/"Result_py/shape"
NB, EX = 20, 2
REGIONS = {"airway": ("인두 기도", "AW_p", "Eb+3 mm", "PNS"), "nasal": ("비강", "NC_p", "PNS", "ANS"), "tongue": ("혀", "TG_p", "혀뿌리", "혀끝")}


def tongue_profile(pid):
    img = read("head_muscles", pid)
    if img is None: return None
    sp = np.array(img.GetSpacing()); t = clean(sitk.GetArrayFromImage(img) == 9, float(np.prod(sp))/1000)
    if not t.any(): return None
    o, R, _ = midline_frame(landmarks(pid)[0]); ant = coords(img, t, o, R)[:, 1]
    edges = np.linspace(ant.min(), ant.max(), NB+1); cnt = np.histogram(ant, edges)[0]
    area = cnt*float(np.prod(sp))/(edges[1]-edges[0])                                    # mm³ ÷ 구간 폭(mm) = 관상 단면적(mm²)
    return {"Patient": f"P_{pid}", **{f"TG_p{i:02d}": a for i, a in enumerate(area)}, "TG_len": float(np.ptp(ant)),
            "TG_volume": float(t.sum()*np.prod(sp)), "TG_back_ant": float(ant.min()), "TG_cut": int(touches_edge(t))}


def tongue_table(redo=False):
    f = DEST/"tongue_profile.csv"
    if f.exists() and not redo: return pd.read_csv(f, encoding="utf-8-sig")
    ids = sorted(int(p.name[2:-7]) for p in (CB/"seg_totalseg/head_muscles").glob("P_*.nii.gz"))
    rows = []
    for pid in ids:
        print(pid, end=" ", flush=True)
        if (r := tongue_profile(pid)) is not None: rows.append(r)
    print(); d = pd.DataFrame(rows); d.round(3).to_csv(f, index=False, encoding="utf-8-sig"); return d


def profiles():
    aw = pd.read_csv(SH/"extra/airway_profile.csv", encoding="utf-8-sig").set_index("Patient")
    nc = pd.read_csv(SH/"extra/nasal_profile.csv", encoding="utf-8-sig").set_index("Patient")
    tg = tongue_table("--retongue" in sys.argv).set_index("Patient"); excluded = {"tongue": sorted(tg.index[tg.TG_cut == 1])}
    tg = tg[tg.TG_cut == 0]
    out = {}
    for key, d in (("airway", aw), ("nasal", nc), ("tongue", tg)):
        X = d[[f"{REGIONS[key][1]}{i:02d}" for i in range(NB)]].dropna()
        X = X[(X > 0).all(axis=1)]
        out[key] = X.div(X.mean(axis=1), axis=0)                                           # 크기 제거: 환자 평균 단면적 = 1
    return out, excluded


def fit(X):
    p = PCA(n_components=0.95, svd_solver="full").fit(X.to_numpy())
    S = p.transform(X.to_numpy()); var_bin = X.var(axis=0, ddof=1).to_numpy()
    expl_bin = (p.components_**2*S.var(axis=0, ddof=1)[:, None]).sum(0)/var_bin           # 구간별: 남긴 PC가 설명하는 분산 ÷ 그 구간 분산
    return p, S, expl_bin


def example_masks(pid=2):
    """P_2 정중 시상 단면(±1 mm)과 부위 마스크·구간 경계(정중 좌표: 앞, 위)."""
    P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); o, R, _ = midline_frame(P)
    a2, z2 = np.arange(-60, 70, .4), np.arange(-80, 45, .4); A, Z = np.meshgrid(a2, z2, indexing="ij")
    Q = o+np.stack([0*A, A, Z], -1).reshape(-1, 3)@R
    ct = map_coordinates(v, np.linalg.solve(dirs.T, (Q-org).T)[::-1], order=1, cval=-1000).reshape(A.shape)
    lat = R[0]    # 정중면은 비중격이라 비강 라벨이 거의 없다 → 좌우 ±15 mm를 겹쳐 그린다(그림 전용)
    nasal = np.any([ts_nasal(pid, Q+dl*lat).reshape(A.shape) for dl in np.arange(-15, 15.1, 1.5)], axis=0)
    hm = read("head_muscles", pid); lab = sitk.GetArrayFromImage(hm)
    k = np.rint(((Q-np.array(hm.GetOrigin()))@np.linalg.inv(np.array(hm.GetDirection()).reshape(3, 3)).T)/np.array(hm.GetSpacing())).astype(int)[:, ::-1]
    ok = ((k >= 0) & (k < lab.shape)).all(1); tongue = np.zeros(len(k), bool); tongue[ok] = lab[tuple(k[ok].T)] == 9; tongue = tongue.reshape(A.shape)
    from cbct_restore_extend import eb_row
    e = eb_row(pid, json.loads((V2/"eb_review/v2/summary.json").read_text(encoding="utf-8"))["offset_mm"])
    s = R@(P["S"]-o); Ls = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP); ant = np.arange(pm.ANT[0], pm.ANT[1]+1e-9, pm.STEP); sup = np.arange(e.Eb_sup, pm.TOP+1e-9, pm.STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup))
    air = pm.airway(vol, Ls, ant, sup, s[1], s[2]); air = air[np.abs(Ls) <= 1].any(0) & (sup >= e.Eb_sup+pm.BOTTOM_CENTER)[None] & (sup <= 0)[None]
    tg = pd.read_csv(DEST/"tongue_profile.csv", encoding="utf-8-sig").set_index("Patient").loc[f"P_{pid}"]
    ansA = (R@(P["ANS"]-o))[1]
    bounds = {"airway": ("h", np.linspace(e.Eb_sup+pm.BOTTOM_CENTER, 0, NB+1)), "nasal": ("v", np.linspace(0, ansA, NB+1)),
              "tongue": ("v", np.linspace(tg.TG_back_ant, tg.TG_back_ant+tg.TG_len, NB+1))}
    return ct, (a2, z2), {"airway": (air, ant, sup), "nasal": (nasal, a2, z2), "tongue": (tongue, a2, z2)}, bounds


COLORS = {"airway": "#1f77b4", "nasal": "#2ca02c", "tongue": "#d62728"}


def ct_panel(a, key, ex):
    """예시 환자 정중 시상 단면 위 부위 마스크와 20구간 경계 (보고서 roi_shape_pca_report.py도 사용)."""
    ct, (a2, z2), masks, bounds = ex; name, _, lo, hi = REGIONS[key]; c = COLORS[key]
    a.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=[a2[0], a2[-1], z2[0], z2[-1]])
    m, x, y = masks[key]; a.contourf(x, y, m.T.astype(float), [.5, 1.5], colors=[c], alpha=.45); a.contour(x, y, m.T.astype(float), [.5], colors=[c], linewidths=1)
    kind, bb = bounds[key]
    for b in bb: (a.axhline if kind == "h" else a.axvline)(b, color="w", lw=.4, alpha=.6)
    a.set_xlabel("앞뒤 (mm, PNS 원점, 오른쪽이 앞)"); a.set_ylabel("위아래 (mm)")
    a.set_xlim(-55, 65); a.set_ylim(-75, 40); a.text(.02, .02, f"흰 선 = 20구간 경계 ({lo} → {hi})", transform=a.transAxes, color="w", fontsize=9)


def draw(res, ex):
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    colors = COLORS
    fig = plt.figure(figsize=(26, 6.6*len(res)+1.6), facecolor="#f6f4f0")
    tot = ", ".join(f"{REGIONS[k][0]} {len(p.explained_variance_ratio_)}개 축 {p.explained_variance_ratio_.sum()*100:.1f}% (n={len(S)})" for k, (p, S, _, _) in res.items())
    fig.suptitle(f"연조직 3부위 크기 제거 모양 PCA  ·  PCA(n_components=0.95)\n{tot}", x=.02, ha="left", fontsize=20, fontweight="bold")
    gs = fig.add_gridspec(len(res), 4, width_ratios=[1.05, 1, 1.1, 1.5], left=.03, right=.99, top=1-1.5/fig.get_figheight(), bottom=.03, hspace=.45, wspace=.22)
    for r, (key, (p, S, eb, X)) in enumerate(res.items()):
        name, _, lo, hi = REGIONS[key]; c = colors[key]
        a = fig.add_subplot(gs[r, 0]); ct_panel(a, key, ex)
        a.set_title(f"{name}: 예시 환자 P_2 정중 시상 단면" + (" (비강은 좌우 ±15 mm 겹침)" if key == "nasal" else ""), fontsize=13, loc="left")
        b = fig.add_subplot(gs[r, 1]); xs = np.arange(NB)
        b.bar(xs, eb*100, color=plt.cm.viridis(eb), edgecolor="none"); b.set_ylim(0, 100); b.set_xticks([0, NB-1], [lo, hi])
        b.set_title("구간별 모양 설명률", fontsize=13, loc="left"); b.set_ylabel("남긴 PC가 설명하는 구간 분산 (%)")
        b.axhline(95, color="#888", ls=":", lw=.8)
        g = fig.add_subplot(gs[r, 2]); mu = X.mean().to_numpy(); g.plot(xs, mu, color="k", lw=2, label="평균 모양")
        for i, ls in zip(range(min(3, p.n_components_)), ("-", "--", ":")):
            sd = S[:, i].std(ddof=1)
            g.plot(xs, mu+EX*sd*p.components_[i], color=c, ls=ls, lw=1.4, label=f"PC{i+1} +{EX}SD")
            g.plot(xs, mu-EX*sd*p.components_[i], color="#888", ls=ls, lw=1.1, label=f"PC{i+1} -{EX}SD")
        g.axhline(1, color="#bbb", lw=.6); g.set_xticks([0, NB-1], [lo, hi]); g.set_ylabel("단면적 ÷ 환자 평균 단면적")
        g.set_title("PC1~PC3 모양 변화 (평균 ± 2 SD)", fontsize=13, loc="left"); g.legend(fontsize=8, ncol=2, loc="upper center", frameon=False)
        L = fig.add_subplot(gs[r, 3]); k = p.n_components_
        for i in range(k):
            contr = np.sign(p.components_[i])*p.components_[i]**2*100; sc = .45/max(np.abs(contr).max(), 1e-9)
            L.barh(np.arange(NB), contr*sc, left=i, color=c, height=.75)
            for j in np.flatnonzero(np.abs(contr) >= 5): L.text(i+contr[j]*sc+(.03 if contr[j] > 0 else -.03), j, f"{abs(contr[j]):.0f}", va="center", ha="left" if contr[j] > 0 else "right", fontsize=7)
            L.axvline(i, color="#999", lw=.5)
        L.set_xticks(range(k), [f"PC{i+1}\n{v*100:.1f}%" for i, v in enumerate(p.explained_variance_ratio_)], fontsize=8)
        L.set_yticks(range(NB), [f"{j+1:02d}" + (f" {lo}" if j == 0 else f" {hi}" if j == NB-1 else "") for j in range(NB)], fontsize=7); L.invert_yaxis()
        L.set_xlim(-.6, k-.4); L.set_title(f"PC1~PC{k} 로딩 (길이 = 로딩² 기여율, 5% 이상 수치)", fontsize=13, loc="left"); L.set_ylabel("구간")
    out = DEST/"softtissue_pca.png"; fig.savefig(out, dpi=80, facecolor=fig.get_facecolor()); plt.close(fig); return out


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    data, excluded = profiles(); res, var = {}, []
    for key, X in data.items():
        p, S, eb = fit(X); res[key] = (p, S, eb, X)
        pcs = [f"PC{i+1}" for i in range(p.n_components_)]
        pd.DataFrame(S, index=X.index, columns=pcs).round(5).to_csv(DEST/f"{key}_scores.csv", encoding="utf-8-sig")
        pd.DataFrame(p.components_.T, index=X.columns, columns=pcs).round(5).assign(bin_explained=eb.round(4)).to_csv(DEST/f"{key}_loadings.csv", encoding="utf-8-sig")
        var += [{"region": key, "PC": pc, "explained": v, "cumulative": cv, "n": len(S)} for pc, v, cv in zip(pcs, p.explained_variance_ratio_, np.cumsum(p.explained_variance_ratio_))]
        print(f"{REGIONS[key][0]}: n={len(S)}, PC {p.n_components_}개, 누적 {p.explained_variance_ratio_.sum()*100:.1f}%  ({', '.join(f'{v*100:.1f}' for v in p.explained_variance_ratio_)})")
    pd.DataFrame(var).round(5).to_csv(DEST/"variance.csv", index=False, encoding="utf-8-sig")
    (DEST/"excluded.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=1), encoding="utf-8")
    print("잘림으로 뺀 혀:", excluded["tongue"]); print("그림:", draw(res, example_masks()))


if __name__ == "__main__":
    main()
