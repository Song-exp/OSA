"""NAV 비강 분리 방식 비교 그림: 영상 규칙(NAV_500, NAV) vs TotalSegmentator 비강 라벨(NAV_ts). 결과 CSV는 바꾸지 않는다.

cbct_nav.measure와 같은 격자·경계·임계값으로 마스크를 다시 만들고, 부피가 nav.csv/new_alternatives.csv와 같은지 확인한 뒤 그린다.
색: 노랑 = 두 방식 모두, 빨강 = 규칙(NAV_500)만, 하늘 = TS 라벨(NAV_ts)만. 흰 선 = 경계 상자(N–Ba, S–PNS, Rh–ANS, ANS–PNS).
출력: Result_py/CBCT/patient_pipeline/v2/nav_review/v1/compare_fig/P_<n>.png, overview.png, summary.csv
사용: python nav_ts_vs_rule_fig.py 3 41 62 65 67 75
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

import cbct_nav as nv
from cbct_eb_detect import midline_frame
from cbct_evidence_figs import excel_col
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd

DEST = OUT/"nav_review/v1/compare_fig"
CMAP = ListedColormap([(0, 0, 0, 0), (1, .15, .15, .55), (0, .85, 1, .55), (1, .9, 0, .55)])    # 0 없음, 1 규칙만, 2 TS만, 3 둘 다


def masks(pid):
    """cbct_nav.measure와 같은 계산으로 (vol, Ls, ant, sup, pts, rh, 마스크 dict, 부피 dict)."""
    P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
    o, R, _ = midline_frame(P); pts = {k: R@(P[k]-o) for k in ("N", "Ba", "S", "ANS")}
    Ls = np.arange(-nv.HALF_L, nv.HALF_L+1e-9, nv.STEP); ant = np.arange(-40., pts["N"][1]+25, nv.STEP); sup = np.arange(0., pts["N"][2]+10, nv.STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup))
    ts = nv.ts_nasal(pid, o+G@R); assert ts is not None, f"P_{pid}: TS 라벨 없음"; ts = ts.reshape(vol.shape)
    mid = vol[np.abs(Ls) <= 1].max(0) > nv.BONE; rows = np.flatnonzero((sup <= pts["N"][2]) & (sup >= 5.))[::-1]
    a_lim = ant <= pts["N"][1]+20
    front = np.array([ant[(mid[:, k] & a_lim)].max() if (mid[:, k] & a_lim).any() else np.nan for k in rows])
    i = nv.rhinion(front, sup[rows]); assert i is not None, f"P_{pid}: Rh 못 찾음"
    rh = (front[i], sup[rows[i]])
    A, Z = np.meshgrid(ant, sup, indexing="ij"); N, Ba, S, ANS = ((pts[k][1], pts[k][2]) for k in ("N", "Ba", "S", "ANS"))
    box = nv.side_of_line(A, Z, N, Ba, (ANS[0]-10, 5.)) & nv.side_of_line(A, Z, S, (0, 0), (ANS[0]/2, 10.)) & nv.side_of_line(A, Z, rh, ANS, (ANS[0]/2, 10.)) & (nv.side_of_line(A, Z, N, rh, (ANS[0]/2, 10.)) | (Z < rh[1])) & (Z >= 0)
    M, V = {}, {}
    for name, thr in (("NAV", nv.AIR), ("NAV_500", -500.)):
        m, _ = nv.split_nasal((vol > -1200) & (vol < thr) & box[None], Ls)
        M[name] = (m if m is not None else np.zeros(vol.shape, bool)) & (np.abs(Ls) <= nv.LEAK_L)[:, None, None]
    M["NAV_ts"] = (vol > -1200) & (vol < nv.AIR) & box[None] & ts
    for k, m in M.items(): V[k] = float(m.sum())*nv.STEP**3
    return vol, Ls, ant, sup, pts, rh, M, V


def stored(pid):
    p = f"P_{pid}"
    for f in (OUT/"nav_review/v1/nav.csv", OUT/"restore_ext/v1/new_alternatives.csv"):
        d = pd.read_csv(f, encoding="utf-8-sig").set_index("Patient")
        if p in d.index: return d.loc[p]
    return None


def draw(pid, xl):
    vol, Ls, ant, sup, pts, rh, M, V = masks(pid)
    s = stored(pid)
    if s is not None:
        for k in V: assert np.isclose(V[k], s[k], rtol=1e-3), f"P_{pid} {k}: 재계산 {V[k]:.1f} ≠ 저장 {s[k]:.1f}"
    code = M["NAV_500"].astype(int)+2*M["NAV_ts"].astype(int)
    ansA = pts["ANS"][1]; ext_c = [Ls[0], Ls[-1], sup[0], sup[-1]]; ext_s = [ant[0], ant[-1], sup[0], sup[-1]]
    fig, ax = plt.subplots(1, 4, figsize=(26, 6.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    for a, f in zip(ax[:3], (.25, .5, .75)):
        j = int(np.argmin(np.abs(ant-f*ansA)))
        a.imshow(vol[:, j, :].T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=ext_c)
        a.imshow(code[:, j, :].T, origin="lower", cmap=CMAP, vmin=0, vmax=3, extent=ext_c, interpolation="nearest")
        a.set_title(f"관상면 {int(f*100)}% (PNS→ANS)", fontsize=13); a.set_xlabel("좌우 (mm)"); a.set_ylabel("ANS–PNS 위 높이 (mm)")
        a.set_xlim(-35, 35); a.set_ylim(sup[0], min(sup[-1], pts["N"][2]+5))
    k = int(np.argmin(np.abs(Ls))); a = ax[3]
    a.imshow(vol[k].T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=ext_s)
    a.imshow(code[np.abs(Ls) <= 2].max(0).T, origin="lower", cmap=CMAP, vmin=0, vmax=3, extent=ext_s, interpolation="nearest")
    for p, q in ((pts["N"][1:], pts["Ba"][1:]), ((0, 0), pts["S"][1:]), (rh, pts["ANS"][1:]), ((0, 0), pts["ANS"][1:])): a.plot([p[0], q[0]], [p[1], q[1]], "w", lw=1)
    for f in (.25, .5, .75): a.axvline(f*ansA, color="w", ls=":", lw=.8)
    a.plot(*rh, "w+", ms=12, mew=2); a.set_title("정중 시상면 (±2 mm, 점선 = 관상면 위치)", fontsize=13); a.set_xlabel("PNS→ANS 앞뒤 (mm)")
    x = f"{xl/1000:.1f}" if np.isfinite(xl) else "없음"
    fig.suptitle(f"P_{pid}   엑셀 {x} mL  |  NAV_500 {V['NAV_500']/1000:.1f}   NAV_ts {V['NAV_ts']/1000:.1f}   (NAV -400HU {V['NAV']/1000:.1f}) mL"
                 "     노랑 = 둘 다 · 빨강 = 규칙(NAV_500)만 · 하늘 = TS 라벨(NAV_ts)만", fontsize=14)
    fig.tight_layout(); out = DEST/f"P_{pid}.png"; fig.savefig(out, dpi=70); plt.close(fig)
    both = float((M["NAV_500"] & M["NAV_ts"]).sum())*nv.STEP**3
    return {"Patient": f"P_{pid}", "excel": xl, **V, "both": both, "rule_only": V["NAV_500"]-both, "ts_only": V["NAV_ts"]-both, "png": out.name}


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    DEST.mkdir(parents=True, exist_ok=True); xl = excel_col(nv.XL_NAV)
    rows = [draw(p, xl.get(p, np.nan)) for p in map(int, sys.argv[1:])]
    d = pd.DataFrame(rows); d.round(1).to_csv(DEST/"summary.csv", index=False, encoding="utf-8-sig")
    print((d.set_index("Patient").drop(columns="png")/1000).round(1).to_string())
    ims = [plt.imread(DEST/r["png"]) for r in rows]
    fig, ax = plt.subplots(len(ims), 1, figsize=(26, 6.8*len(ims)))
    for a, im in zip(np.atleast_1d(ax), ims): a.imshow(im); a.axis("off")
    fig.tight_layout(); fig.savefig(DEST/"overview.png", dpi=45); plt.close(fig)


if __name__ == "__main__":
    main()
