"""PMCA 탐색 계산. 전원 동일 규칙, 환자별 예외 없음. 엑셀은 규칙에 쓰지 않는다. result.json은 바꾸지 않는다.

입력 Eb: eb_review/v2/eb_candidates.csv (미계측 환자는 계산하지 않음). 좌표계: cbct_eb_detect.midline_frame(Po 미사용, 원점 PNS).
기도: 공기 -1200~-400 (Roh PAV 임계값). 범위: Eb 높이 이상, S–PNS 선 뒤쪽(Se=Sella 가정), PNS 코로나면 뒤쪽.
      정중 ±8 mm와 가장 많이 겹치는 3차원 연결 성분 하나.
PMCA: (주 후보) 기도 중심선에 수직인 단면의 최소 넓이 / (대안) 좌표계 축상 단면의 최소 넓이.
측정 높이: PNS 높이 이하, 경계 잘림을 피하려고 아래쪽 여유 중심선 5 mm·축상 1 mm.
출력: Result_py/CBCT/patient_pipeline/v2/pmca_review/v1/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label, map_coordinates

from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import CB, OUT, digest, landmarks, save
from check_landmarks import read_nrrd

DEST = OUT/"pmca_review/v1"
STEP, HALF_L, AIR = .4, 30.0, (-1200, -400)
ANT = (-45.0, 0.0); TOP = 25.0; SMOOTH_MM = 5.0; BOTTOM_CENTER, BOTTOM_AXIAL = 5.0, 1.0


def airway(vol, Ls, ant, sup, s_ant, s_sup):
    """vol (L, ant, sup)에서 기도 성분. S–PNS 선 뒤쪽만 남긴다(원점 PNS)."""
    A, Z = np.meshgrid(ant, sup, indexing="ij")
    behind = (s_ant*Z-s_sup*A)*np.sign(s_ant*(-30.0)-s_sup*(-30.0)) > 0    # (-30,-30)은 항상 인두 쪽
    air = (vol > AIR[0]) & (vol < AIR[1]) & behind[None]
    lab, n = label(air)
    if not n: return None
    counts = np.bincount(lab[np.abs(Ls) < 8].ravel(), minlength=n+1); counts[0] = 0
    return lab == counts.argmax() if counts.max() else None


def sections(main, Ls, ant, sup):
    """(축상 넓이 배열, 중심선 수직 넓이 배열, 중심선 (L, ant) 배열). NaN = 측정 불가."""
    axial = main.sum((0, 1))*STEP**2
    cen = np.full((len(sup), 2), np.nan)
    for k in np.flatnonzero(axial):
        i, j = np.nonzero(main[..., k]); cen[k] = Ls[i].mean(), ant[j].mean()
    ok = np.flatnonzero(np.isfinite(cen[:, 0]))
    if ok.size < 5: return axial, np.full(len(sup), np.nan), cen
    cen = np.c_[np.interp(np.arange(len(sup)), ok, cen[ok, 0]), np.interp(np.arange(len(sup)), ok, cen[ok, 1])]
    w = max(1, int(SMOOTH_MM/STEP)) | 1
    cen = np.c_[[np.convolve(np.pad(c, w//2, mode="edge"), np.ones(w)/w, mode="valid") for c in cen.T]].T
    pts = np.c_[cen, sup]; tan = np.gradient(pts, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    g = np.arange(-HALF_L, HALF_L+1e-9, STEP); U, V = np.meshgrid(g, g, indexing="ij"); perp = np.full(len(sup), np.nan)
    for k, (p, t) in enumerate(zip(pts, tan)):
        a = np.cross(t, [1., 0., 0.]); a /= np.linalg.norm(a); b = np.cross(t, a)
        q = p+U[..., None]*a+V[..., None]*b
        idx = np.stack([(q[..., 0]-Ls[0])/STEP, (q[..., 1]-ant[0])/STEP, (q[..., 2]-sup[0])/STEP])
        perp[k] = (map_coordinates(main.astype(np.float32), idx, order=1, cval=0) > .5).sum()*STEP**2
    return axial, perp, cen


def pick(area, sup, lo, hi):
    """lo≤sup≤hi에서 0보다 큰 최소 넓이와 그 높이."""
    m = (sup >= lo) & (sup <= hi) & np.isfinite(area) & (area > 0)
    if not m.any(): return np.nan, np.nan
    i = np.flatnonzero(m)[area[m].argmin()]
    return float(area[i]), float(sup[i])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    eb = pd.read_csv(OUT/"eb_review/v2/eb_candidates.csv", encoding="utf-8-sig"); rows, imgs = [], {}
    for _, e in eb.iterrows():
        pid = int(e.Patient[2:]); print(pid, flush=True)
        r = {"Patient": e.Patient, "Eb_method": e.method, "Eb_sup": e.Eb_sup}
        if e.method == "none": rows.append(r); continue
        P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
        o, R, _ = midline_frame(P); s = R@(P["S"]-o)
        Ls = np.arange(-HALF_L, HALF_L+1e-9, STEP); ant = np.arange(ANT[0], ANT[1]+1e-9, STEP); sup = np.arange(e.Eb_sup, TOP+1e-9, STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
        vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup))
        m = airway(vol, Ls, ant, sup, s[1], s[2])
        if m is None: r["note"] = "no_airway"; rows.append(r); continue
        ax, pe, _ = sections(m, Ls, ant, sup)
        r["PMCA_center"], r["z_center"] = pick(pe, sup, e.Eb_sup+BOTTOM_CENTER, 0.0)
        r["PMCA_axial"], r["z_axial"] = pick(ax, sup, e.Eb_sup+BOTTOM_AXIAL, 0.0)
        r["min_at_bottom_center"] = bool(r["z_center"]-(e.Eb_sup+BOTTOM_CENTER) < 3)
        r["min_at_bottom_axial"] = bool(r["z_axial"]-(e.Eb_sup+BOTTOM_AXIAL) < 3)
        rows.append(r); imgs[pid] = (m.sum(0), ant, sup, r["z_center"], r["z_axial"])
    d = pd.DataFrame(rows); d.round(2).to_csv(DEST/"pmca_candidates.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(6, 9, figsize=(36, 30))
    for a in ax.flat: a.axis("off")
    for a, (pid, (p, ant, sup, zc, za)) in zip(ax.flat, imgs.items()):
        a.axis("on"); a.imshow(p.T, origin="lower", cmap="magma", extent=[ant[0], ant[-1], sup[0], sup[-1]])
        a.axhline(zc, color="lime", lw=2); a.axhline(za, color="cyan", lw=1.5, ls="--"); a.set_title(f"P_{pid}", fontsize=18); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(DEST/"pmca_airway_masks.png", dpi=45); plt.close(fig)
    save(DEST/"summary.json", {"n": int(d.PMCA_center.notna().sum()), "status": "exploratory; Eb 위치 미검증; 전문가 확인 전; result.json 미적용",
         "eb_sha256": digest(OUT/"eb_review/v2/eb_candidates.csv"), "script_sha256": digest(Path(__file__))})
    print(d[["PMCA_center", "PMCA_axial"]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
