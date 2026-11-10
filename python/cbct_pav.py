"""PAV 탐색 계산. Roh 정의 A등급을 그대로 쓰고 엑셀로 임계값을 맞추지 않는다. result.json은 바꾸지 않는다.

공기 -1200~-400 안에서 S–PNS 평면 뒤쪽(Se=Sella 가정), PNS 관상면 뒤쪽(구강 공기 제외, PMCA와 동일), 아래 경계 평면 위쪽을 먼저 자르고
정중 ±8 mm와 가장 많이 겹치는 3차원 연결 성분의 부피를 잰다.
아래 경계: (주) Eb를 지나 FH에 평행 / (민감도) Eb를 지나 Po 미사용 정중 좌표계 수평면.
입력 Eb: eb_review/v2/eb_candidates.csv. 출력: Result_py/CBCT/patient_pipeline/v2/pav_review/v1/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

import cbct_pmca as pm
from cbct_eb_detect import midline_frame
from cbct_evidence_figs import excel_col
from cbct_molar_section import PILOT
from cbct_fh_repair import repair
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd
from osa.ceph import head_frame

DEST = OUT/"pav_review/v1"
XL_PAV = 30
BELOW = 20.    # FH가 정중 좌표계에 대해 기울어 Eb 평면이 아래로 내려갈 수 있는 여유 (mm)


def above_plane(Ls, ant, sup, point, normal):
    """격자 점이 point를 지나고 법선 normal인 평면의 위쪽인지."""
    return (Ls[:, None, None]-point[0])*normal[0]+(ant[None, :, None]-point[1])*normal[1]+(sup[None, None, :]-point[2])*normal[2] >= 0


def volume(vol, Ls, ant, sup, s, keep):
    v = vol.copy(); v[~keep] = 0
    m = pm.airway(v, Ls, ant, sup, s[1], s[2])
    return m, (float(m.sum())*pm.STEP**3 if m is not None else np.nan)


def main():
    DEST.mkdir(parents=True, exist_ok=True); xl = excel_col(XL_PAV)
    eb = pd.read_csv(OUT/"eb_review/v2/eb_candidates.csv", encoding="utf-8-sig"); rows, imgs = [], {}
    for _, e in eb.iterrows():
        pid = int(e.Patient[2:]); r = {"Patient": e.Patient, "Eb_method": e.method, "PAV_xl": xl.get(pid, np.nan), "Set": "pilot" if pid in PILOT else "eval"}
        if e.method == "none": rows.append(r); continue
        print(pid, end=" ", flush=True)
        P = repair(landmarks(pid)[0])[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
        o, R, _ = midline_frame(P); s = R@(P["S"]-o); fh = R@head_frame(P)[1][2]; eb_m = np.array([0., e.Eb_ant, e.Eb_sup])
        r["FH_vs_midline_deg"] = float(np.degrees(np.arccos(np.clip(abs(fh[2])/np.linalg.norm(fh), 0, 1))))
        Ls = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP); ant = np.arange(pm.ANT[0], pm.ANT[1]+1e-9, pm.STEP); sup = np.arange(e.Eb_sup-BELOW, pm.TOP+1e-9, pm.STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
        vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup))
        fh_up = fh if fh[2] > 0 else -fh
        m, r["PAV_fh"] = volume(vol, Ls, ant, sup, s, above_plane(Ls, ant, sup, eb_m, fh_up))
        _, r["PAV_midline"] = volume(vol, Ls, ant, sup, s, above_plane(Ls, ant, sup, eb_m, np.array([0., 0., 1.])))
        r["touches_image_bottom"] = bool(m is not None and m[..., 0].any())
        rows.append(r)
        if m is not None and (r["PAV_xl"] == r["PAV_xl"] or pid in PILOT):
            A, Z = np.meshgrid(np.arange(-70, 40, .4), np.arange(-95, 35, .4), indexing="ij"); Q = np.stack([0*A, A, Z], -1).reshape(-1, 3)
            imgs[pid] = (map_coordinates(v, np.linalg.solve(dirs.T, (o+Q@R-org).T)[::-1], order=1, cval=-1000).reshape(A.shape), m.any(0), ant, sup, s, eb_m, fh_up, r["PAV_fh"], r["PAV_xl"])
    print(); d = pd.DataFrame(rows); d.round(2).to_csv(DEST/"pav.csv", index=False, encoding="utf-8-sig")
    n = int(np.ceil(len(imgs)/5)); fig, ax = plt.subplots(n, 5, figsize=(30, 6.5*n), squeeze=False)
    for a in ax.flat: a.axis("off")
    for a, (pid, (ct, mid, ant, sup, s, eb_m, fh, pav, x)) in zip(ax.flat, imgs.items()):
        a.axis("on"); a.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000, vmax=1200, extent=[-70, 40, -95, 35]); a.contour(ant, sup, mid.T.astype(float), [.5], colors="lime", linewidths=1)
        a.plot([0, s[1]], [0, s[2]], "y", lw=1); xs = np.array([-65., 25.]); a.plot(xs, eb_m[2]-(xs-eb_m[1])*fh[1]/fh[2], "c", lw=1); a.plot(eb_m[1], eb_m[2], "r+", ms=13, mew=2)
        a.set_xlim(-70, 40); a.set_ylim(-95, 35); a.set_title(f"P_{pid}: PAV {pav/1000:.1f} mL vs Excel {x/1000:.1f} mL" if x == x else f"P_{pid}: PAV {pav/1000:.1f} mL (no Excel)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT/"evidence_review/v1/figs/pav_sections.png", dpi=60); plt.close(fig)
    for c in ("PAV_fh", "PAV_midline"):
        g = d.dropna(subset=[c, "PAV_xl"]); e = g[c]-g.PAV_xl
        print(c, "n", len(g), "median auto/xl", round(g[c].median()), round(g.PAV_xl.median()), "bias", round(e.mean()), "MAE", round(e.abs().mean()), "MedAPE", round((e.abs()/g.PAV_xl).median()*100, 1), "% r", round(g[c].corr(g.PAV_xl), 2), "rho", round(g[c].corr(g.PAV_xl, method="spearman"), 2))
    print(d[["Patient", "Eb_method", "PAV_fh", "PAV_midline", "PAV_xl", "FH_vs_midline_deg", "touches_image_bottom"]].dropna(subset=["PAV_xl"]).round(1).to_string())


if __name__ == "__main__":
    main()
