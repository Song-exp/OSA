"""NAV 탐색 계산. 경계 평면·임계값은 Roh 정의(A등급), 부비동 제거는 영상 규칙(B등급). 엑셀로 임계값을 맞추지 않는다.

좌표계: 정중 좌표계(Po 미사용, 원점 PNS). ANS–PNS는 항상 높이 0이다.
Rh: 정중 ±1 mm 뼈(>300) 앞쪽 끝을 N 8 mm 아래부터 ANS 쪽으로 내려가며, 앞쪽 끝이 3 mm 이상 뒤로 물러나기 직전의 행 (비골 하단 끝).
공기: -1200~AIR. 위 N–Ba 평면 아래, 아래 ANS–PNS 평면 위, 뒤 S–PNS 평면 앞(Se=Sella 가정), 앞 Rh–ANS 평면 뒤(Rh 위쪽은 N–Rh 평면 뒤).
부비동 제거: 공기를 SINUS_R(4 mm) 침식해 폭 8 mm 이하 개구를 끊고, 정중 ±8 mm와 겹치지 않는 덩어리를 복원해 뺀다. 남은 공기 중 정중과 이어진 성분이 비강. 부피는 좌우 ±30 mm 안에서만 재고, 그 밖이 10% 넘게 잘리면 누출 의심으로 표시. 사골동 칸은 분리되지 않는다.
후보 NAV_ts: 같은 경계 안 공기 중 TotalSegmentator head_glands_cavities 비강 라벨 안의 것 (라벨이 있는 환자만).
출력: Result_py/CBCT/patient_pipeline/v2/nav_review/v1/  사용: python cbct_nav.py [환자번호...] (없으면 전원)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from cbct_eb_detect import midline_frame
from cbct_evidence_figs import excel_col
from cbct_molar_section import PILOT
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd
from run_ali import ascii_path

DEST = OUT/"nav_review/v1"
STEP = .4; HALF_L = 45.; BONE = 300.; JUMP = 3.; SKIP_BELOW_N = 8.; AIR = -400.; SINUS_R = 4.; RIM = 4; LEAK_L = 30.; LEAK_FRAC = .1
XL_NAV = 29
TS_NASAL = (14, 15)    # head_glands_cavities: nasal_cavity_right, nasal_cavity_left


def rhinion(front, sup):
    """front·sup: 위→아래 순서. 뼈 앞쪽 끝이 JUMP 넘게 뒤로 물러나기 직전 행의 인덱스. 못 찾으면 None."""
    f = np.where(np.isfinite(front), front, -np.inf)
    for i in range(len(f)-1):
        if sup[0]-sup[i] >= SKIP_BELOW_N and f[i] > -np.inf and f[i+1] < f[i]-JUMP:
            return i
    return None


def side_of_line(A, Z, p, q, ref):
    """p→q 직선(앞, 위)에서 ref와 같은 쪽 불리언."""
    f = lambda a, z: (q[0]-p[0])*(z-p[1])-(q[1]-p[1])*(a-p[0])
    return f(A, Z)*f(*ref) > 0


def split_nasal(air, Ls, step=STEP):
    """(비강 마스크, 부비동·바깥 공기 덩어리 수). 정중에서 떨어진 큰 공기 덩어리를 복원해 뺀 나머지 중 정중과 이어진 성분."""
    lab, n = ndi.label(ndi.distance_transform_edt(air, sampling=step) > SINUS_R)
    near = set(np.unique(lab[np.abs(Ls) < 8]))-{0}; big = [k for k in range(1, n+1) if k not in near]
    gone = ndi.binary_dilation((ndi.distance_transform_edt(~np.isin(lab, big), sampling=step) <= SINUS_R+step) & air, mask=air, iterations=RIM) if big else np.zeros_like(air)
    rest = air & ~gone    # RIM: 덩어리 가장자리 공기까지 공기 안에서만 넓혀 없앤다(뼈는 넘지 않음)
    lab, _ = ndi.label(rest); seeds = set(np.unique(lab[np.abs(Ls) < 8]))-{0}
    return (np.isin(lab, sorted(seeds)), len(big)) if seeds else (None, 0)


def ts_nasal(pid, pts_lps):
    """TotalSegmentator 비강 라벨(좌14·우15)을 격자 점(LPS, N×3)에 최근접 표본화. 파일이 없으면 None."""
    f = CB/f"seg_totalseg/head_glands_cavities/P_{pid}.nii.gz"
    if not f.exists(): return None
    im = sitk.ReadImage(str(ascii_path(f))); a = sitk.GetArrayFromImage(im)
    org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
    k = np.rint(((pts_lps-org)@np.linalg.inv(D).T)/sp).astype(int)[:, ::-1]; ok = ((k >= 0) & (k < a.shape)).all(1)
    out = np.zeros(len(k), bool); out[ok] = np.isin(a[tuple(k[ok].T)], TS_NASAL)
    return out


def measure(pid, keep_imgs=True):
    r = {"Patient": f"P_{pid}"}; P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
    o, R, _ = midline_frame(P); pts = {k: R@(P[k]-o) for k in ("N", "Ba", "S", "ANS")}
    Ls = np.arange(-HALF_L, HALF_L+1e-9, STEP); ant = np.arange(-40., pts["N"][1]+25, STEP); sup = np.arange(0., pts["N"][2]+10, STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup))
    ts = ts_nasal(pid, o+G@R); ts = ts.reshape(vol.shape) if ts is not None else None
    mid = vol[np.abs(Ls) <= 1].max(0) > BONE; rows = np.flatnonzero((sup <= pts["N"][2]) & (sup >= 5.))[::-1]
    a_lim = ant <= pts["N"][1]+20
    front = np.array([ant[(mid[:, k] & a_lim)].max() if (mid[:, k] & a_lim).any() else np.nan for k in rows])
    i = rhinion(front, sup[rows])
    if i is None: r["reason"] = "no_rhinion"; return r, None
    rh = (front[i], sup[rows[i]]); r.update(Rh_ant=rh[0], Rh_sup=rh[1], N_to_Rh=float(np.hypot(pts["N"][1]-rh[0], pts["N"][2]-rh[1])))
    A, Z = np.meshgrid(ant, sup, indexing="ij"); N, Ba, S, ANS = ((pts[k][1], pts[k][2]) for k in ("N", "Ba", "S", "ANS"))
    box = side_of_line(A, Z, N, Ba, (ANS[0]-10, 5.)) & side_of_line(A, Z, S, (0, 0), (ANS[0]/2, 10.)) & side_of_line(A, Z, rh, ANS, (ANS[0]/2, 10.)) & (side_of_line(A, Z, N, rh, (ANS[0]/2, 10.)) | (Z < rh[1])) & (Z >= 0)
    for name, thr in (("NAV", AIR), ("NAV_500", -500.)):
        air = (vol > -1200) & (vol < thr) & box[None]
        m, n_s = split_nasal(air, Ls)
        r[name] = float(m[np.abs(Ls) <= LEAK_L].sum())*STEP**3 if m is not None else np.nan
        if name == "NAV" and m is not None: r["NAV_uncapped"] = float(m.sum())*STEP**3
        if name == "NAV": r["n_sinus_seeds"] = n_s; main = m; r["max_L"] = float(np.abs(Ls[m.any((1, 2))]).max()) if m is not None and m.any() else np.nan
    if ts is not None:
        for name, thr in (("NAV_ts", AIR), ("NAV_ts_500", -500.)):
            r[name] = float(((vol > -1200) & (vol < thr) & box[None] & ts).sum())*STEP**3
        r["TS_label_mL"] = float(ts.sum())*STEP**3/1000
    if main is None: r["reason"] = "no_nasal_seed"; return r, None
    r["reason"] = "leak_suspected" if r["NAV_uncapped"] > (1+LEAK_FRAC)*r["NAV"] else ""
    img = None
    if keep_imgs:
        a2, z2 = np.arange(-50, pts["N"][1]+25, STEP), np.arange(-5, pts["N"][2]+10, STEP); AA, ZZ = np.meshgrid(a2, z2, indexing="ij")
        Q = np.stack([0*AA, AA, ZZ], -1).reshape(-1, 3)
        img = (map_coordinates(v, np.linalg.solve(dirs.T, (o+Q@R-org).T)[::-1], order=1, cval=-1000).reshape(AA.shape), a2, z2, main.any(0), ant, sup, pts, rh, main.sum(1), Ls, r["NAV"])
    return r, img


def ts_figure(ids):
    """PNS–ANS 사이 25·50·75% 위치의 관상면: TotalSegmentator 비강 라벨(초록)과 그 안 공기(<AIR, 노랑)."""
    fig, ax = plt.subplots(len(ids), 3, figsize=(21, 6*len(ids)), squeeze=False)
    for i, pid in enumerate(ids):
        P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); o, R, _ = midline_frame(P); ans = (R@(P["ANS"]-o))[1]
        Ls, Zs = np.arange(-40, 40, .3), np.arange(-15, 65, .3); LL, ZZ = np.meshgrid(Ls, Zs, indexing="ij")
        for j, f in enumerate((.25, .5, .75)):
            Q = o+np.stack([LL, 0*LL+f*ans, ZZ], -1).reshape(-1, 3)@R
            ct = map_coordinates(v, np.linalg.solve(dirs.T, (Q-org).T)[::-1], order=1, cval=-1000).reshape(LL.shape); lab = ts_nasal(pid, Q).reshape(LL.shape)
            a = ax[i, j]; a.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000, vmax=1200, extent=[Ls[0], Ls[-1], Zs[0], Zs[-1]])
            a.contour(Ls, Zs, lab.T.astype(float), [.5], colors="lime", linewidths=1); a.contour(Ls, Zs, (lab & (ct < AIR)).T.astype(float), [.5], colors="yellow", linewidths=.6)
            a.set_title(f"P_{pid}: coronal at {int(f*100)}% PNS→ANS", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT/"evidence_review/v1/figs/nav_ts_coronal.png", dpi=55); plt.close(fig)


def main():
    if sys.argv[1:2] == ["--tsfig"]: ts_figure([int(a) for a in sys.argv[2:]]); return
    ids = [int(a) for a in sys.argv[1:]] or [int(p[2:]) for p, i in pd.read_csv(OUT/"cohort.csv", dtype=str, encoding="utf-8-sig")[["Patient", "Include"]].values if i.lower() == "true"]
    DEST.mkdir(parents=True, exist_ok=True); xl = excel_col(XL_NAV); rows = []; imgs = {}; full = not sys.argv[1:]
    for pid in ids:
        print(pid, end=" ", flush=True); r, im = measure(pid); r["NAV_xl"] = xl.get(pid, np.nan); r["Set"] = "pilot" if pid in PILOT else "eval"; rows.append(r)
        if im is not None and (not full or pid in PILOT or (r["NAV_xl"] == r["NAV_xl"] and abs(r["NAV"]/r["NAV_xl"]-1) > .25)): imgs[pid] = im
    print(); d = pd.DataFrame(rows); d.round(2).to_csv(DEST/("nav.csv" if full else "nav_pilot.csv"), index=False, encoding="utf-8-sig")
    n = int(np.ceil(len(imgs)/4)); fig, ax = plt.subplots(n, 8, figsize=(48, 6*n), squeeze=False)
    for a in ax.flat: a.axis("off")
    for k, (pid, (ct, a2, z2, sag, ant, sup, pts, rh, cor, Ls, nav)) in enumerate(imgs.items()):
        a, b = ax.flat[2*k], ax.flat[2*k+1]; a.axis("on"); b.axis("on"); x = d.set_index("Patient").NAV_xl[f"P_{pid}"]
        a.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000, vmax=1200, extent=[a2[0], a2[-1], z2[0], z2[-1]]); a.contour(ant, sup, sag.T.astype(float), [.5], colors="lime", linewidths=.8)
        for p, q, c in ((pts["N"][1:], pts["Ba"][1:], "y"), ((0, 0), pts["S"][1:], "y"), (rh, pts["ANS"][1:], "c"), ((0, 0), pts["ANS"][1:], "r")): a.plot([p[0], q[0]], [p[1], q[1]], c, lw=1)
        a.plot(*rh, "r+", ms=13, mew=2); a.set_title(f"P_{pid}: NAV {nav/1000:.1f} mL vs Excel {x/1000:.1f}", fontsize=13)
        b.imshow(cor.T, origin="lower", cmap="magma", extent=[Ls[0], Ls[-1], sup[0], sup[-1]]); b.set_title("nasal mask, coronal projection", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"evidence_review/v1/figs/"/("nav_sections.png" if full else "nav_pilot.png"), dpi=45); plt.close(fig)
    for c in [c for c in ("NAV", "NAV_500", "NAV_ts", "NAV_ts_500") if c in d]:
        g = d.dropna(subset=[c, "NAV_xl"]); e = g[c]-g.NAV_xl
        for s in ("pilot", "eval"):
            h = g[g.Set == s]; ee = h[c]-h.NAV_xl
            if len(h): print(c, s, "n", len(h), "median auto/xl", round(h[c].median()), round(h.NAV_xl.median()), "bias", round(ee.mean()), "MAE", round(ee.abs().mean()), "MedAPE", round((ee.abs()/h.NAV_xl).median()*100, 1), "% r", round(h[c].corr(h.NAV_xl), 2))
    print(d[[c for c in ("Patient", "Rh_ant", "Rh_sup", "N_to_Rh", "NAV", "NAV_uncapped", "NAV_500", "NAV_ts", "NAV_ts_500", "TS_label_mL", "NAV_xl", "n_sinus_seeds", "max_L", "reason") if c in d]].round(1).to_string())


if __name__ == "__main__":
    main()
