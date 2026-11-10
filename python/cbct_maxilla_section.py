"""상악 제1대구치 관상 단면의 PHI·NCWR 후보. 정의가 C등급이라 후보를 미리 고정하고 파일럿으로만 고른다.

단면: tdi_review/v1/molars.csv 의 UR6·UL6 저항중심 앞뒤 평균, 머리 좌표계 관상면.
구개 천장: 좌우 저항중심 가운데 ±1 mm 열에서 교합면 위로 처음 만나는 상악 라벨 하단(중앙값).
PH = 구개 천장 높이 - 교합 높이(ALI UR6O·UL6O 평균)
MxOW: oc(UR6O–UL6O), mb(UR6MB–UL6MB), rc(저항중심 사이)
비강저: 정중 양옆 3~12 mm 열(비중격 제외)에서 구개 천장 위로 처음 만나는 공기(<-400)의 높이 중앙값
NCW: 비강저~+15 mm 행에서 정중 양옆 공기 2D 연결 성분의 좌우 폭 최댓값 (상악동 개구 위쪽 제외)
NCW_wall: 구개 위 3~25 mm 행에서 양쪽 상악동 공기 안쪽 끝 사이 폭의 최댓값 (공기 폭이 비갑개·개구에 흔들려 파일럿 그림을 보고 추가)
NCW_ts: 같은 단면에서 TotalSegmentator 비강 라벨(좌우 합)의 최대 좌우 폭 (비갑개 포함 벽 사이 폭)
MxBW: jugal(저항중심~비강저에서 외측 윤곽이 가장 안쪽인 점의 폭). 비강저 높이 끝 폭은 협골이 포함되어 버렸다
출력: Result_py/CBCT/patient_pipeline/v2/phi_ncwr_review/v1/{phi_ncwr.csv, score.csv}
"""
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from cbct_evidence_figs import excel_col, teeth_points
from cbct_nav import ts_nasal
from cbct_molar_section import MODEL, PILOT, pair_ok
from cbct_fh_repair import repair
from cbct_patient_pipeline import CB, OUT, landmarks
from osa.ceph import head_frame
from run_ali import ascii_path

DEST = OUT/"phi_ncwr_review/v1"
STEP = .25; AIR = -400.; MAXILLA = 54; NCW_RANGE = 15.; SIDE = (3., 12.); SINUS_MIN = 15.
XL = {"NCWR": 26, "PHI": 27}


def first_true_up(col, start):
    """start 이후 처음 참인 인덱스. 없으면 None."""
    i = np.flatnonzero(col[start:])
    return int(i[0])+start if len(i) else None


def row_extent(mask, row, Ls):
    """row 행 참 영역의 좌우 끝 (왼쪽 끝 L, 오른쪽 끝 L). 없으면 nan."""
    i = np.flatnonzero(mask[:, row])
    return (Ls[i[0]], Ls[i[-1]]) if len(i) else (np.nan, np.nan)


def sinus_medial_width(air, Ls, Lc, rows):
    """행마다 양쪽 상악동 공기(정중에서 15 mm 밖의 가장 긴 공기 구간)의 안쪽 끝 사이 폭. 최댓값과 그 행을 돌려준다."""
    best = (np.nan, -1, np.nan, np.nan)
    for k in rows:
        e = []
        for sign in (-1, 1):
            i = np.flatnonzero(air[:, k] & (sign*(Ls-Lc) > SINUS_MIN))
            runs = np.split(i, np.flatnonzero(np.diff(i) > 1)+1) if len(i) else []
            if not runs: break
            run = max(runs, key=len); e.append(run[0] if sign == 1 else run[-1])    # 안쪽 끝
        if len(e) == 2 and not Ls[e[1]]-Ls[e[0]] <= best[0]: best = (Ls[e[1]]-Ls[e[0]], k, Ls[e[0]], Ls[e[1]])
    return best


def jugal_width(mx, Ls, Zs, Lc, u, fl):
    """저항중심 높이~비강저에서 좌우 외측 윤곽이 가장 안쪽인 점 사이의 폭. 대구치보다 안쪽인 행은 분할 끊김으로 버린다."""
    outer = {}
    for name, sign, furc in (("R", -1, u.L.UR6), ("L", 1, u.L.UL6)):
        vals = []
        for k in range(int(np.searchsorted(Zs, u.Z.mean())), fl+1):
            i = np.flatnonzero(mx[:, k] & (sign*(Ls-Lc) > 0))
            v = (sign*(Ls[i]-Lc)).max() if len(i) else np.nan
            vals.append(v if v >= sign*(furc-Lc) else np.nan)
        if not np.isfinite(vals).any(): return {"reason": "no_lateral_contour"}
        outer[name] = np.nanmin(ndi.median_filter(np.nan_to_num(vals, nan=99.), 13))    # 3 mm 중앙값, 끊긴 행은 99로 두어 최솟값에서 빠짐
    return {"MxBW_jugal": outer["R"]+outer["L"]+STEP}


def section(pid, m):
    """(ct, maxilla, Ls, Zs, 머리 좌표계 점들). 상악 대구치 저항중심이 없으면 None."""
    u = m[(m.Patient == f"P_{pid}") & m.tooth.isin(["UR6", "UL6"]) & m.status.eq("ok")].set_index("tooth")
    if len(u) < 2 or not pair_ok(u.loc["UL6"], u.loc["UR6"]): return None
    im = sitk.ReadImage(str(ascii_path(CB/f"teeth_work/{MODEL}/in/P_{pid}_0000.nii.gz"))); ct = sitk.GetArrayFromImage(im)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(ascii_path(CB/f"seg_teeth/{MODEL}/P_{pid}.nii.gz"))))
    org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
    o, R = head_frame(repair(landmarks(pid)[0])[0]); A0 = u.A.mean(); Lc = (u.L.UR6+u.L.UL6)/2
    Ls, Zs = np.arange(Lc-55, Lc+55, STEP), np.arange(u.Z.mean()-25, u.Z.mean()+55, STEP); LL, ZZ = np.meshgrid(Ls, Zs, indexing="ij")
    Q = o+np.stack([LL, 0*LL+A0, ZZ], -1).reshape(-1, 3)@R; ijk = (((Q-org)@np.linalg.inv(D).T)/sp)[:, ::-1].T
    ts = ts_nasal(pid, Q); ts = ts.reshape(LL.shape) if ts is not None else None
    q = {k: R@(v-o) for k, v in teeth_points(pid).items()}
    return map_coordinates(ct, ijk, order=1, cval=0).reshape(LL.shape), map_coordinates(seg, ijk, order=0).reshape(LL.shape) == MAXILLA, Ls, Zs, q, u, Lc, ts


def measure(pid, m):
    s = section(pid, m)
    if s is None: return {"Patient": f"P_{pid}", "reason": "no_upper_molars"}
    ct, mx, Ls, Zs, q, u, Lc, ts = s; r = {"Patient": f"P_{pid}", "reason": ""}
    if ts is not None and ts.any():
        ext = {k: np.ptp(Ls[ts[:, k]])+STEP for k in range(ts.shape[1]) if ts[:, k].any()}; k = max(ext, key=ext.get)
        r.update(NCW_ts=float(ext[k]), Z_ts=Zs[k], ts_l=Ls[ts[:, k]].min(), ts_r=Ls[ts[:, k]].max())
    if not all(k in q for k in ("UR6O", "UL6O")) or not (u.d_O <= 2.).all(): r["reason"] = "ali_occlusal_off_tooth"; return r
    occ = (q["UR6O"][2]+q["UL6O"][2])/2; mid = np.abs(Ls-Lc) <= 1.
    start = int(np.searchsorted(Zs, occ))
    tops = [first_true_up(mx[i], start) for i in np.flatnonzero(mid)]; tops = [t for t in tops if t is not None]
    if not tops: r["reason"] = "no_palate"; return r
    pal = int(np.median(tops)); r["PH"] = Zs[pal]-occ
    r["MxOW_oc"] = abs(q["UL6O"][0]-q["UR6O"][0]); r["MxOW_rc"] = u.L.UL6-u.L.UR6
    if "UR6MB" in q and "UL6MB" in q: r["MxOW_mb"] = abs(q["UL6MB"][0]-q["UR6MB"][0])
    air = ct < AIR; side = (np.abs(Ls-Lc) >= SIDE[0]) & (np.abs(Ls-Lc) <= SIDE[1])
    floors = [first_true_up(air[i], pal) for i in np.flatnonzero(side)]; floors = [f for f in floors if f is not None]
    if not floors: r["reason"] = "no_nasal_air"; return r
    fl = int(np.median(floors)); r["floor_above_palate"] = Zs[fl]-Zs[pal]
    n = int(round(NCW_RANGE/STEP)); box = np.zeros_like(air); box[:, fl:fl+n] = air[:, fl:fl+n]
    lab, _ = ndi.label(box); ids = np.unique(lab[np.ix_(np.flatnonzero(side), range(fl, fl+n))]); ids = ids[ids > 0]
    if not len(ids): r["reason"] = "no_nasal_component"; return r
    nas = np.isin(lab, ids)
    ext = {k: np.ptp(Ls[nas[:, k]])+STEP for k in range(fl, min(fl+n, nas.shape[1])) if nas[:, k].any()}
    best = max(ext, key=ext.get); r["NCW"] = float(ext[best]); i = np.flatnonzero(nas[:, best])
    r.update(Z_occ=occ, Z_palate=Zs[pal], Z_floor=Zs[fl], Z_ncw=Zs[best], ncw_l=Ls[i[0]], ncw_r=Ls[i[-1]], Lc=Lc)
    r.update(jugal_width(mx, Ls, Zs, Lc, u, fl))
    a, b = int(round(3/STEP)), int(round(25/STEP)); w, k, wl, wr = sinus_medial_width(air, Ls, Lc, range(pal+a, min(pal+b, air.shape[1])))
    r.update(NCW_wall=w, Z_wall=Zs[k] if k >= 0 else np.nan, wall_l=wl, wall_r=wr)
    return r


def figure(ids):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = pd.read_csv(OUT/"tdi_review/v1/molars.csv", encoding="utf-8-sig"); d = pd.read_csv(DEST/"phi_ncwr.csv", encoding="utf-8-sig").set_index("Patient")
    fig, ax = plt.subplots(int(np.ceil(len(ids)/4)), 4, figsize=(24, 6.5*int(np.ceil(len(ids)/4))), squeeze=False)
    for a in ax.ravel(): a.axis("off")
    for a, pid in zip(ax.ravel(), ids):
        ct, mx, Ls, Zs, q, u, Lc, ts = section(pid, m); r = d.loc[f"P_{pid}"]; a.axis("on")
        a.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=[Ls[0], Ls[-1], Zs[0], Zs[-1]]); a.contour(Ls, Zs, mx.T.astype(float), [.5], colors="orange", linewidths=.6)
        for z, c in ((r.Z_occ, "r"), (r.Z_palate, "lime"), (r.Z_floor, "cyan")): a.axhline(z, color=c, lw=.8)
        a.plot([r.ncw_l, r.ncw_r], [r.Z_ncw, r.Z_ncw], "m", lw=1); a.plot([r.wall_l, r.wall_r], [r.Z_wall, r.Z_wall], "y", lw=1); a.plot([r.ts_l, r.ts_r], [r.Z_ts, r.Z_ts], "lime", lw=2)
        if ts is not None: a.contour(Ls, Zs, ts.T.astype(float), [.5], colors="lime", linewidths=.8)
        a.plot(u.L, u.Z, "c+", ms=11, mew=2)
        a.set_title(f"P_{pid}: PH {r.PH:.1f}, NCW {r.NCW:.1f}, MxBW {r.MxBW_jugal:.1f}; PHI {r.PHI_rc:.1f} vs {r.PHI_xl:.1f}, NCWR ts {r.NCWR_ts:.1f} (wall {r.NCWR_wall:.1f}) vs {r.NCWR_xl:.1f}", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"evidence_review/v1/figs/phi_ncwr_sections.png", dpi=65); plt.close(fig)


def main():
    if sys.argv[1:2] == ["--fig"]: figure([int(a) for a in sys.argv[2:]]); return

    ids = [int(a) for a in sys.argv[1:]] or [int(p[2:]) for p, i in pd.read_csv(OUT/"cohort.csv", dtype=str, encoding="utf-8-sig")[["Patient", "Include"]].values if i.lower() == "true"]
    m = pd.read_csv(OUT/"tdi_review/v1/molars.csv", encoding="utf-8-sig"); DEST.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame([measure(p, m) for p in ids]); d["Set"] = ["pilot" if int(p[2:]) in PILOT else "eval" for p in d.Patient]
    xl = {k: excel_col(c) for k, c in XL.items()}
    for k in XL: d[k+"_xl"] = [xl[k].get(int(p[2:]), np.nan) for p in d.Patient]
    for c in ("oc", "mb", "rc"):
        if f"MxOW_{c}" in d: d[f"PHI_{c}"] = d.PH/d[f"MxOW_{c}"]*100
    d["NCWR_jugal"] = d.NCW/d.MxBW_jugal*100; d["NCWR_wall"] = d.NCW_wall/d.MxBW_jugal*100; d["NCWR_ts"] = d.NCW_ts/d.MxBW_jugal*100
    d.to_csv(DEST/"phi_ncwr.csv", index=False, encoding="utf-8-sig")
    rows = []
    for v in XL:
        for c in [x for x in d if x.startswith(v+"_") and not x.endswith("_xl")]:
            for s in ("pilot", "eval"):
                g = d[d.Set == s]; e = (g[c]-g[v+"_xl"]).dropna()
                rows.append({"var": c, "set": s, "n": len(e), "bias": e.mean(), "MAE": e.abs().mean(), "median_abs": e.abs().median(), "r": g[c].corr(g[v+"_xl"])})
    sc = pd.DataFrame(rows); sc.to_csv(DEST/"score.csv", index=False, encoding="utf-8-sig"); print(sc.round(2).to_string()); print(d.reason.value_counts().to_dict())


if __name__ == "__main__":
    main()
