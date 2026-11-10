"""검토용 원본 영상 그림 생성. 좌표·계측값은 바꾸지 않는다.

출력: Result_py/CBCT/patient_pipeline/v2/evidence_review/v1/figs/
  fov_P_n.png   DICOM 원본 정중부 최소·최대 투영과 영상 최하단 6 mm 축상 투영 (Eb 미계측 판단 근거)
  pmca_P_n.png  최소 단면 위치의 정중 시상면·축상면과 기도 영역 윤곽, 높이별 넓이 곡선
사용: python cbct_evidence_figs.py fov 1 83 68 ... / python cbct_evidence_figs.py pmca 29 82 44 38
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

import cbct_pmca as pm
from cbct_eb_detect import midline_frame
from cbct_landmark_review import panel
from cbct_fh_repair import repair
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd
from osa.cbct import read_cbct

ROOT = Path(__file__).resolve().parents[1]
DEST = OUT/"evidence_review/v1/figs"
XL_PMCA, XL_PAL = 31, 32


def excel_col(col):
    x = pd.read_excel(ROOT/"Patient_Info_Data_87명_261019.xlsx", header=None).iloc[3:]
    x = x[x[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    return dict(zip(x[0].astype(int), pd.to_numeric(x[col], errors="coerce")))


def fov(pid):
    folder = next((ROOT/"Data/CBCT").glob(f"P_{pid}_*")); d = read_cbct(folder, top_only=True)
    v = d.rescaled.astype(np.float32); t = d.slice_table; sp = d.voxel_size_mm; z = t.PosZ.to_numpy()
    y0, x0 = t.PosY.iloc[0], t.PosX.iloc[0]; P = landmarks(pid)[0]
    i = int(round((P["PNS"][0]-x0)/sp[1])); sag = v[:, i-8:i+9, :]; ext = [y0, y0+v.shape[0]*sp[0], z.min(), z.max()]
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for a, im, lo, hi, n in [(ax[0], sag.min(1), -1000, 500, "midline +-3 mm minIP (air)"), (ax[1], sag.max(1), -500, 1500, "midline +-3 mm maxIP (bone)")]:
        a.imshow(im.T, origin="lower", cmap="gray", vmin=lo, vmax=hi, extent=ext); a.plot(P["PNS"][1], P["PNS"][2], "r+", ms=14, mew=2); a.plot(P["Me"][1], P["Me"][2], "c+", ms=14, mew=2)
        a.set_xlim(P["PNS"][1]-45, P["PNS"][1]+55); a.set_ylim(z.min()-3, P["PNS"][2]+15); a.set_title(n); a.set_xlabel("LPS y (mm)"); a.set_ylabel("LPS z (mm)")
    n = int(round(6/sp[2])); ax[2].imshow(v[:, :, :n].min(2), origin="lower", cmap="gray", vmin=-1000, vmax=500, extent=[x0, x0+v.shape[1]*sp[1], y0, y0+v.shape[0]*sp[0]])
    ax[2].set_title("lowest 6 mm of volume, axial minIP"); ax[2].set_xlabel("LPS x (mm)"); ax[2].set_ylabel("LPS y (mm)")
    fig.suptitle(f"P_{pid}: slices {v.shape[2]}, z {z.min():.0f}-{z.max():.0f} mm, voxel {sp[0]:.2f} mm; red PNS, cyan Me"); fig.tight_layout(); fig.savefig(DEST/f"fov_P_{pid}.png", dpi=90); plt.close(fig)


def pmca(pid, xl):
    c = pd.read_csv(OUT/"pmca_review/v1/pmca_candidates.csv", encoding="utf-8-sig").set_index("Patient").loc[f"P_{pid}"]
    P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); o, R, _ = midline_frame(P); s = R@(P["S"]-o)
    def ct(L, A, Z):
        return map_coordinates(v, np.linalg.solve(dirs.T, (o+np.stack([L, A, Z], -1).reshape(-1, 3)@R-org).T)[::-1], order=1, cval=-1000).reshape(L.shape)
    Ls, ant, sup = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP), np.arange(pm.ANT[0], pm.ANT[1]+1e-9, pm.STEP), np.arange(c.Eb_sup, pm.TOP+1e-9, pm.STEP)
    main = pm.airway(ct(*np.meshgrid(Ls, ant, sup, indexing="ij")), Ls, ant, sup, s[1], s[2]); ax_a, pe, _ = pm.sections(main, Ls, ant, sup)
    a2, z2 = np.arange(-70, 40, pm.STEP), np.arange(-95, 35, pm.STEP); A, Z = np.meshgrid(a2, z2, indexing="ij")
    fig, ax = plt.subplots(1, 4, figsize=(26, 7))
    ax[0].imshow(ct(0*A, A, Z).T, origin="lower", cmap="gray", vmin=-1000, vmax=1200, extent=[a2[0], a2[-1], z2[0], z2[-1]])
    ax[0].contour(ant, sup, main[np.abs(Ls) < 1].any(0).T.astype(float), [.5], colors="lime", linewidths=1)
    ax[0].axhline(c.z_center, color="yellow", lw=1.2); ax[0].axhline(c.Eb_sup, color="cyan", lw=1, ls="--"); ax[0].axhline(0, color="red", lw=.8, ls=":"); ax[0].set_title("midsagittal; yellow = min section, cyan = Eb plane, red = PNS level")
    ax[0].set_xlabel("anterior + (mm)"); ax[0].set_ylabel("superior + (mm)")
    for a, z, area, n in [(ax[1], c.z_center, c.PMCA_center, "centerline-min level"), (ax[2], c.z_axial, c.PMCA_axial, "axial-min level")]:
        l3, a3 = np.arange(-40, 40, pm.STEP), np.arange(-60, 30, pm.STEP); LL, AA = np.meshgrid(l3, a3, indexing="ij")
        a.imshow(ct(LL, AA, 0*LL+z).T, origin="lower", cmap="gray", vmin=-1000, vmax=1200, extent=[l3[0], l3[-1], a3[0], a3[-1]])
        a.contour(Ls, ant, main[..., int(round((z-sup[0])/pm.STEP))].T.astype(float), [.5], colors="lime", linewidths=1); a.set_title(f"{n}: z {z:.1f}, area {area:.1f} mm2"); a.set_xlabel("left + (mm)"); a.set_ylabel("anterior + (mm)")
    m = (sup >= c.Eb_sup) & (sup <= 0)
    ax[3].plot(pe[m], sup[m], "g", label="centerline-perpendicular"); ax[3].plot(ax_a[m], sup[m], "c--", label="axial"); ax[3].axvline(xl, color="orange", label=f"Excel PMCA {xl:.1f}"); ax[3].set_xlabel("area (mm2)"); ax[3].set_ylabel("superior + (mm)"); ax[3].legend(); ax[3].set_title("area by level (Eb to PNS)")
    fig.suptitle(f"P_{pid}: Eb method {c.Eb_method}; exploratory, not expert reviewed"); fig.tight_layout(); fig.savefig(DEST/f"pmca_P_{pid}.png", dpi=90); plt.close(fig)


def teeth_points(pid):
    """landmarks_teeth의 ALI 치아 점 (LPS mm)."""
    out = {}
    for f in sorted((CB/"landmarks_teeth").glob(f"P_{pid}_*.mrk.json")):
        for m in json.loads(f.read_text(encoding="utf-8"))["markups"]:
            for c in m["controlPoints"]:
                out[c["label"]] = np.asarray(c["position"], float)*(np.array([-1, -1, 1]) if m.get("coordinateSystem") == "RAS" else 1)
    return out


def molar(ids):
    """ALI 제1대구치 점을 치아별 관상·시상 단면에 표시."""
    teeth, colors = ["UR6", "UL6", "LR6", "LL6"], {"O": "red", "R": "cyan", "MB": "lime", "DB": "yellow", "MP": "orange"}
    for plane in ("coronal", "sagittal"):
        fig, ax = plt.subplots(len(teeth), len(ids), figsize=(6*len(ids), 6*len(teeth)), squeeze=False)
        for j, pid in enumerate(ids):
            q = teeth_points(pid); v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd")
            for i, t in enumerate(teeth):
                marks = {t+k: (q[t+k], c, "+") for k, c in colors.items() if t+k in q}
                panel(ax[i, j], v, dirs, org, np.mean([m[0] for m in marks.values()], axis=0), plane, marks, half=20)
                ax[i, j].set_title(f"P_{pid} {t} {plane}")
        fig.tight_layout(); fig.savefig(DEST/f"ali_molar_{plane}.png", dpi=70); plt.close(fig)


def molar_seg(ids):
    """UNIVERSALLAB 제1대구치 분할 윤곽과 ALI O(빨강)·R(하늘)을 R 점을 지나는 관상면에 표시."""
    import SimpleITK as sitk
    from run_ali import ascii_path
    labels = {"UR6": 3, "UL6": 14, "LR6": 30, "LL6": 19}; name = "UNIVERSALLAB_MODEL"
    fig, ax = plt.subplots(len(labels), len(ids), figsize=(6*len(ids), 6*len(labels)), squeeze=False)
    for j, pid in enumerate(ids):
        im = sitk.ReadImage(str(ascii_path(CB/f"teeth_work/{name}/in/P_{pid}_0000.nii.gz"))); ct = sitk.GetArrayFromImage(im)
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ascii_path(CB/f"seg_teeth/{name}/P_{pid}.nii.gz")))); q = teeth_points(pid); org = np.array(im.GetOrigin()); sp = np.array(im.GetSpacing())
        for i, (t, l) in enumerate(labels.items()):
            a = ax[i, j]; c = np.argwhere(seg == l)
            y = int(round((q[t+"R"][1]-org[1])/sp[1])) if t+"R" in q and 0 <= (q[t+"R"][1]-org[1])/sp[1] < seg.shape[1] else int(c[:, 1].mean())
            ext = [org[0], org[0]+ct.shape[2]*sp[0], org[2], org[2]+ct.shape[0]*sp[2]]
            a.imshow(ct[:, y, :], origin="lower", cmap="gray", vmin=-700, vmax=2200, extent=ext)
            xs, zs = org[0]+np.arange(ct.shape[2])*sp[0], org[2]+np.arange(ct.shape[0])*sp[2]
            if (seg[:, y, :] == l).any(): a.contour(xs, zs, (seg[:, y, :] == l).astype(float), [.5], colors="lime", linewidths=1.2)
            for k, col in (("O", "red"), ("R", "cyan")):
                if t+k in q: a.plot(q[t+k][0], q[t+k][2], "+", color=col, ms=12, mew=2)
            cx, cz = org[0]+c[:, 2].mean()*sp[0], org[2]+c[:, 0].mean()*sp[2]; a.set_xlim(cx-18, cx+18); a.set_ylim(cz-18, cz+18); a.set_title(f"P_{pid} {t} (label {l}, {(seg == l).sum()*sp.prod()/1000:.2f} mL)")
    fig.tight_layout(); fig.savefig(DEST/"molar_seg_coronal.png", dpi=70); plt.close(fig)


def tdi(ids):
    """제1대구치 높이 머리 좌표계 관상면. 윗줄 상악, 아랫줄 하악. 하늘색 저항중심(분지부), 노랑 협측 바깥면, 초록 제1대구치 분할."""
    import SimpleITK as sitk
    from osa.ceph import head_frame
    from run_ali import ascii_path
    name = "UNIVERSALLAB_MODEL"; D = OUT/"tdi_review/v1"; m = pd.read_csv(D/"molars.csv", encoding="utf-8-sig"); t = pd.read_csv(D/"tdi.csv", encoding="utf-8-sig").set_index("Patient")
    labels = {"UR6": 3, "UL6": 14, "LR6": 30, "LL6": 19}; rows = int(np.ceil(len(ids)/6))*2
    fig, ax = plt.subplots(rows, 6, figsize=(30, 5*rows), squeeze=False)
    for a in ax.ravel(): a.axis("off")
    for n, pid in enumerate(ids):
        im = sitk.ReadImage(str(ascii_path(CB/f"teeth_work/{name}/in/P_{pid}_0000.nii.gz"))); ct = sitk.GetArrayFromImage(im)
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ascii_path(CB/f"seg_teeth/{name}/P_{pid}.nii.gz")))); org, sp = np.array(im.GetOrigin()), np.array(im.GetSpacing()); Dm = np.array(im.GetDirection()).reshape(3, 3)
        o, R = head_frame(repair(landmarks(pid)[0])[0]); g = m[m.Patient == f"P_{pid}"].set_index("tooth")
        for r, (arch, tt) in enumerate((("Mx", ("UR6", "UL6")), ("Mn", ("LR6", "LL6")))):
            a = ax[2*(n//6)+r, n % 6]; a.axis("on"); h = g.loc[list(tt)]
            if h.A.isna().all(): a.set_title(f"P_{pid} {arch}: no molar found"); continue
            A0, Z0 = h.A.mean(), h.Z.mean(); Ls, Zs = np.arange(-50, 50, .25), np.arange(Z0-22, Z0+22, .25); LL, ZZ = np.meshgrid(Ls, Zs, indexing="ij")
            xyz = o+np.stack([LL, 0*LL+A0, ZZ], -1).reshape(-1, 3)@R; ijk = (((xyz-org)@np.linalg.inv(Dm).T)/sp)[:, ::-1].T
            a.imshow(map_coordinates(ct, ijk, order=1, cval=-1000).reshape(LL.shape).T, origin="lower", cmap="gray", vmin=-500, vmax=2200, extent=[Ls[0], Ls[-1], Zs[0], Zs[-1]])
            sg = map_coordinates(seg, ijk, order=0).reshape(LL.shape)
            for tn in tt:
                if (sg == labels[tn]).any(): a.contour(Ls, Zs, (sg == labels[tn]).T.astype(float), [.5], colors="lime", linewidths=.8)
            a.plot(h.L, h.Z, "c+", ms=13, mew=2); a.plot(h.surf_L, h.Z, "y+", ms=13, mew=2)
            a.set_title(f"P_{pid} {arch}: {' / '.join(h.method.fillna('none').str[:5])}  rc {t.loc[f'P_{pid}', arch+'AW_rc']:.1f}" + (f"; TDI {t.loc[f'P_{pid}', 'TDI_rc']:.1f} vs xl {t.loc[f'P_{pid}', 'TDI_xl']:.1f}" if r else ""), fontsize=11)
    fig.tight_layout(); fig.savefig(DEST/"tdi_sections.png", dpi=65); plt.close(fig)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    kind, ids = sys.argv[1], [int(a) for a in sys.argv[2:]]
    xl = excel_col(XL_PMCA) if kind == "pmca" else None
    if kind == "molar": molar(ids); molar_seg(ids); sys.exit()
    if kind == "tdi": tdi(ids); sys.exit()
    for pid in ids:
        print(kind, pid, flush=True); fov(pid) if kind == "fov" else pmca(pid, xl[pid])
