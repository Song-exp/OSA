"""제1대구치 분지부(저항중심)와 치조폭 후보, TDI. 전 환자 동일 규칙이며 엑셀로 임계값을 맞추지 않는다.

분지부: 치아 마스크를 머리 좌표계 수평 단면으로 치관 최대 단면부터 치근 쪽으로 내려가며,
        넓이 3 mm² 이상 덩어리가 둘 이상인 상태가 1.5 mm 지속되는 첫 높이의 바로 윗 단면 중심 (Koo 2017: 다근치 저항중심 = 분지부)
후보 ① surface: 분지부에서 좌우축으로 협측에 나가 뼈·치아 마스크가 끝나는 곳 사이 거리 (틈 1 mm 허용)
대체: 분지부를 못 찾으면 최대 단면에서 중앙값 거리만큼 내려간 단면 중심. ALI R이 같은 치아 2 mm 안일 때만 인정
후보 ② rc: 좌우 분지부 사이 거리 (Koo 2017, Ma 2023)
출력: Result_py/CBCT/patient_pipeline/v2/tdi_review/v1/{molars.csv, tdi.csv}
"""
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

from cbct_evidence_figs import ROOT, excel_col, teeth_points
from cbct_fh_repair import repair
from cbct_patient_pipeline import CB, OUT, landmarks
from osa.ceph import head_frame
from run_ali import ascii_path

TEETH = {"UR6": 3, "UL6": 14, "LR6": 30, "LL6": 19}
MODEL = "UNIVERSALLAB_MODEL"
DEST = OUT/"tdi_review/v1"
XL_TDI = 28
MIN_VOL_ML, MAX_VOL_ML = .3, 2.    # 범위 밖은 결손·부분 분할 또는 다른 치아와 합쳐진 분할로 본다
FIXED_MM = {"U": 7.5, "L": 6.0}    # 대체 단계: 분지부를 찾은 치아의 최대 단면~분지부 거리 중앙값 (엑셀 미사용)
MIN_CROWN_MM = 3.   # 최대 단면에서 분지부까지 이보다 짧으면 치관이 분할되지 않은 것으로 본다
BONE_MAX_LABEL = 55    # 치아 1~52, 하악 53, 상악 54, 하악관 55 (그 위 라벨은 쓰지 않음)
WIDTH, MAX_DZ = (35., 70.), 6.    # 좌우 저항중심 폭·높이 차의 해부학적 범위(mm)
PILOT = (5, 7, 9, 24, 30, 32, 41, 44, 50, 54, 60, 69)    # docs/CBCT_측정_계획.md 7-6


def furcation(q, crown_low, res, dz=.5, min_area=3., persist=1.5):
    """q: N×3 (좌, 앞, 위). 분지부 좌표, 치관 쪽 길이, 치근 길이. 못 찾으면 None."""
    zs = np.arange(q[:, 2].min(), q[:, 2].max()+dz, dz); zs = zs if crown_low else zs[::-1]
    ij = np.floor((q[:, :2]-q[:, :2].min(0))/res).astype(int); shape = tuple(ij.max(0)+1)
    n, area, cent = [], [], []
    for z in zs:
        m = np.abs(q[:, 2]-z) <= dz/2; g = np.zeros(shape, bool); g[tuple(ij[m].T)] = True
        lab, k = ndi.label(ndi.binary_closing(g), np.ones((3, 3)))
        a = np.bincount(lab.ravel(), minlength=k+1)[1:]*res**2
        n.append(int((a >= min_area).sum())); area.append(g.sum()); cent.append(q[m].mean(0) if m.any() else np.full(3, np.nan))
    n = np.array(n); start = int(np.argmax(area)); w = max(int(round(persist/dz)), 1)    # 교두 분리를 피하려고 최대 단면부터 시작
    for i in range(start+1, len(n)-w+1):
        if (n[i:i+w] >= 2).all():
            f = cent[i-1].copy(); f[2] = zs[i-1]
            return f, abs(zs[i-1]-zs[start]), abs(zs[-1]-zs[i-1])
    return None


def level_center(q, crown_low, mm, dz=.5):
    """치관 최대 단면에서 치근 쪽으로 mm 내려간 단면의 중심. 분지부를 못 찾은 치아의 대체값."""
    z = np.arange(q[:, 2].min(), q[:, 2].max()+dz, dz); z0 = z[np.argmax(np.histogram(q[:, 2], np.r_[z-dz/2, z[-1]+dz/2])[0])]
    target = z0+(mm if crown_low else -mm); m = np.abs(q[:, 2]-target) <= dz/2
    if not m.any(): return None
    c = q[m].mean(0); c[2] = target
    return c


def buccal_run(mask_at_s, s, gap=1.):
    """s(mm) 순서의 불리언 배열에서 시작점과 이어진 구간의 끝 거리 (틈 gap 이하 허용)."""
    idx = np.flatnonzero(mask_at_s); ds = s[1]-s[0]
    if not len(idx): return np.nan
    jumps = np.flatnonzero(np.diff(idx)*ds > gap)
    return float(s[idx[jumps[0]]] if len(jumps) else s[idx[-1]])


def measure(pid):
    im = sitk.ReadImage(str(ascii_path(CB/f"seg_teeth/{MODEL}/P_{pid}.nii.gz"))); seg = sitk.GetArrayFromImage(im)
    org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
    P = repair(landmarks(pid)[0])[0]; o, R = head_frame(P); pts = teeth_points(pid); bone = (seg > 0) & (seg <= BONE_MAX_LABEL); Dinv = np.linalg.inv(D)
    rows = []
    for t, l in TEETH.items():
        vox = np.argwhere(seg == l); r = {"Patient": f"P_{pid}", "tooth": t, "vol_mL": len(vox)*sp.prod()/1000}
        lps = org+(vox[:, ::-1]*sp)@D.T
        for k in "OR":
            r[f"d_{k}"] = float(np.linalg.norm(lps-pts[t+k], axis=1).min()) if t+k in pts and len(vox) else np.nan
        q = (lps-o)@R.T; valid = MIN_VOL_ML <= r["vol_mL"] <= MAX_VOL_ML
        f = furcation(q, t[0] == "U", sp.max()) if valid else None
        fixed = level_center(q, t[0] == "U", FIXED_MM[t[0]]) if valid else None
        if f and f[1] < MIN_CROWN_MM: f = None
        if not f and not r.get("d_R", np.inf) <= 2.: fixed = None    # 대체값은 ALI R이 같은 치아 분할 2 mm 안일 때만(치아 식별 교차 확인)
        r["method"] = "furcation" if f else ("fixed_level" if fixed is not None else "missing")
        r["status"] = "ok" if r["method"] != "missing" else "missing_or_partial"
        if fixed is not None: r["fixed_L"] = fixed[0]
        if f or fixed is not None:
            c = f[0] if f else fixed
            if f: r["crown_mm"], r["root_mm"] = f[1:]
            r.update(zip(("L", "A", "Z"), c))
            if t+"R" in pts: r["d_R_furc"] = float(np.linalg.norm((pts[t+"R"]-o)@R.T-c))
            side = -1. if t[1] == "R" else 1.; s = np.arange(0, 40, .1)
            ray = o+(c+np.outer(s*side, [1, 0, 0]))@R
            k = np.round(((ray-org)@Dinv.T)/sp).astype(int)[:, ::-1]; ok = ((k >= 0) & (k < seg.shape)).all(1)
            hit = np.zeros(len(s), bool); hit[ok] = bone[tuple(k[ok].T)]
            e = buccal_run(hit, s); r["buccal_mm"] = e; r["surf_L"] = c[0]+side*e
        rows.append(r)
    return rows


def pair_ok(a, b):
    """좌우 치아 쌍의 해부학적 범위. 벗어나면 치아 라벨 오류로 본다."""
    return WIDTH[0] <= a.L-b.L <= WIDTH[1] and abs(a.Z-b.Z) <= MAX_DZ


def widths(g):
    x = g.set_index("tooth"); out = {}
    for arch, a, b in (("Mx", "UL6", "UR6"), ("Mn", "LL6", "LR6")):
        ok = all(t in x.index and x.status[t] == "ok" for t in (a, b)) and pair_ok(x.loc[a], x.loc[b])
        out[f"{arch}AW_surface"] = x.surf_L[a]-x.surf_L[b] if ok else np.nan
        out[f"{arch}AW_rc"] = x.L[a]-x.L[b] if ok else np.nan
    return out


def score(d, c):
    e = (d[c]-d.TDI_xl).dropna()
    return {"n": len(e), "bias": e.mean(), "MAE": e.abs().mean(), "median_abs": e.abs().median(), "r": d[c].corr(d.TDI_xl)}


def main():
    ids = [int(a) for a in sys.argv[1:]] or [int(p[2:]) for p, i in pd.read_csv(OUT/"cohort.csv", dtype=str, encoding="utf-8-sig")[["Patient", "Include"]].values if i.lower() == "true"]
    DEST.mkdir(parents=True, exist_ok=True); rows = []
    for pid in ids:
        rows += measure(pid); print(pid, end=" ", flush=True)
    m = pd.DataFrame(rows); m.to_csv(DEST/"molars.csv", index=False, encoding="utf-8-sig")
    xl = excel_col(XL_TDI); t = []
    for p, g in m.groupby("Patient", sort=False):
        w = widths(g); w.update(Patient=p, TDI_surface=w["MxAW_surface"]-w["MnAW_surface"], TDI_rc=w["MxAW_rc"]-w["MnAW_rc"], TDI_xl=xl.get(int(p[2:]), np.nan), Set="pilot" if int(p[2:]) in PILOT else "eval"); t.append(w)
    t = pd.DataFrame(t); t.to_csv(DEST/"tdi.csv", index=False, encoding="utf-8-sig")
    print("\n", m.status.value_counts().to_dict())
    for s in ("pilot", "eval"):
        for c in ("TDI_surface", "TDI_rc"): print(s, c, {k: round(v, 2) for k, v in score(t[t.Set == s], c).items()})


if __name__ == "__main__":
    main()
