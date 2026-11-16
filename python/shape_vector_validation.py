"""세밀 벡터의 수치 검증: 서로 독립적인 방법끼리 비교해 신뢰할 변수와 확인할 환자를 고른다. 육안 점검은 여기서 걸린 사례만.

1) 인두 라벨(TotalSegmentator 11·12·13) vs AMASSS 상기도(UAW, CBCT 학습): 같은 높이 범위·PNS 뒤쪽에서 Dice, 대칭 표면 거리(평균·95%)
2) 같은 범위(Eb+5 mm ~ PNS) 부피·단면 프로파일: TotalSegmentator 인두 vs 우리 공기 임계값 기도(airway_profile) → ICC·r·Bland-Altman, 환자별 곡선 상관
3) 치아 분할 제1대구치 중심 vs ALI 치아 랜드마크(별도 모델) 저항중심(R): 위치 오차(mm)
4) 연구개 내부 일관성: PNS 부착 간격, 길이·두께 robust 이상치
5) SSM: 구조별 Procrustes 거리(평균 형태와) robust 이상치 + 정합 잔차
출력: Result_py/shape/validation/ (csv, summary.json, report.md)
사용: python shape_vector_validation.py [환자번호...]
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

from cbct_accuracy_experiments import stats, wait_ram
from cbct_eb_detect import midline_frame
from cbct_evidence_figs import teeth_points
from cbct_patient_pipeline import CB, OUT, landmarks, save
from run_ali import ascii_path
from shape_ssm import gpa

ROOT = Path(__file__).resolve().parents[1]; SH = ROOT/"Result_py/shape"; DEST = SH/"validation"
PHARYNX = (11, 12, 13); NB = 20


def surf_dist(a, b, sp):
    """두 마스크 경계 사이 대칭 거리 (평균, 95%), mm."""
    ba, bb = a & ~binary_erosion(a), b & ~binary_erosion(b)
    if not ba.any() or not bb.any(): return np.nan, np.nan
    da = distance_transform_edt(~bb, sampling=sp)[ba]; db = distance_transform_edt(~ba, sampling=sp)[bb]; d = np.r_[da, db]
    return float(d.mean()), float(np.percentile(d, 95))


def pharynx(pid, P, ap):
    """AMASSS UAW 격자에서 두 분할 비교 (PNS 뒤, 두 마스크 높이 범위가 겹치는 곳)."""
    uaw = sitk.ReadImage(str(ascii_path(CB/f"seg/P_{pid}_seg_UAW.nrrd")))
    ts = sitk.ReadImage(str(ascii_path(CB/f"seg_totalseg/head_glands_cavities/P_{pid}.nii.gz")))
    ts = sitk.Resample(ts, uaw, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    A = sitk.GetArrayFromImage(uaw) > 0; T = np.isin(sitk.GetArrayFromImage(ts), PHARYNX); del ts
    box = np.argwhere(A | T)
    if not len(box): return {}
    lo, hi = np.maximum(box.min(0)-3, 0), box.max(0)+4; A, T = A[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], T[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sp = np.array(uaw.GetSpacing())[::-1]; org, D = np.array(uaw.GetOrigin()), np.array(uaw.GetDirection()).reshape(3, 3)
    o, R, _ = midline_frame(P)
    k = np.stack(np.meshgrid(*[np.arange(l, h, dtype=np.float32) for l, h in zip(lo, hi)], indexing="ij"), -1)    # (k,j,i)
    q = ((org+(k[..., ::-1]*sp[::-1])@D.T)-o)@R.T; del k
    ant, sup = q[..., 1], q[..., 2]; del q
    za, zt = sup[A], sup[T]
    if not len(za) or not len(zt): return {"dice": np.nan}
    zlo, zhi = max(za.min(), zt.min()), min(za.max(), zt.max())
    roi = (ant < 0) & (sup >= zlo) & (sup <= zhi); a, t = A & roi, T & roi
    r = {"dice": float(2*(a & t).sum()/max(a.sum()+t.sum(), 1)), "uaw_ml": float(a.sum()*sp.prod()/1000), "ts_ml": float(t.sum()*sp.prod()/1000),
         "overlap_height_mm": float(zhi-zlo)}
    r["surf_mean_mm"], r["surf_p95_mm"] = surf_dist(a, t, sp)
    # 우리 임계값 기도 범위(Eb+5 mm ~ PNS)에서 TotalSegmentator 부피·단면 프로파일
    if ap is not None and np.isfinite(ap.get("AW_length", np.nan)):
        lo_s = -ap["AW_length"]+5; m = T & (sup >= lo_s) & (sup <= 0)
        r["ts_ml_same_range"] = float(m.sum()*sp.prod()/1000)
        z = sup[m]; e = np.linspace(lo_s, 0, NB+1); area = np.histogram(z, e)[0]*sp.prod()/((e[1]-e[0]))
        ours = np.array([ap.get(f"AW_p{i:02d}", np.nan) for i in range(NB)]); ok = np.isfinite(ours) & (area > 0)
        r["profile_r"] = float(np.corrcoef(area[ok], ours[ok])[0, 1]) if ok.sum() >= 5 else np.nan
    return r


def molars(pid, P, teeth):
    o, R, _ = midline_frame(P); tp = teeth_points(pid); out = []
    for t, k in ((3, "UR6R"), (14, "UL6R"), (19, "LL6R"), (30, "LR6R")):
        row = teeth[(teeth.Patient == f"P_{pid}") & (teeth.tooth == t)]
        if row.empty or not bool(row.label_ok.iloc[0]) or k not in tp: continue
        c = row[["c_left", "c_ant", "c_sup"]].to_numpy(float)[0]; a = R@(tp[k]-o); d = c-a
        out.append({"Patient": f"P_{pid}", "tooth": k[:3], "err_mm": float(np.linalg.norm(d)), "d_left": d[0], "d_ant": d[1], "d_sup": d[2]})
    return out


def robust_flags(d, cols, z=3.5):
    out = []
    for c in cols:
        x = d[c]; med = x.median(); mad = 1.4826*(x-med).abs().median()
        if mad > 0: out += [{"Patient": p, "variable": c, "value": v, "robust_z": (v-med)/mad} for p, v in zip(d.Patient, x) if abs((v-med)/mad) > z]
    return out


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ids = [int(a) for a in sys.argv[1:]] or sorted(int(p.parent.name[2:]) for p in OUT.glob("P_*/result.json"))
    ap = pd.read_csv(SH/"extra/airway_profile.csv", encoding="utf-8-sig").set_index("Patient")
    teeth = pd.read_csv(SH/"fine/teeth_long.csv", encoding="utf-8-sig"); ph, mo = [], []
    for pid in ids:
        wait_ram(6, f"P_{pid}"); p = f"P_{pid}"; print(p, end=" ", flush=True); P = landmarks(pid)[0]
        ph.append({"Patient": p, **pharynx(pid, P, ap.loc[p].to_dict() if p in ap.index else None)}); mo += molars(pid, P, teeth); gc.collect()
    ph, mo = pd.DataFrame(ph), pd.DataFrame(mo); ph.round(4).to_csv(DEST/"pharynx_agreement.csv", index=False, encoding="utf-8-sig")
    mo.round(3).to_csv(DEST/"teeth_vs_ali.csv", index=False, encoding="utf-8-sig")
    # 2) 같은 범위 부피 일치도
    j = ph.set_index("Patient").join(ap.AW_volume/1000)
    vol = stats(j.ts_ml_same_range.to_numpy(float), j.AW_volume.to_numpy(float)) if "ts_ml_same_range" in j else {}
    # 4) 연구개
    st = pd.read_csv(SH/"fine/soft_tissue.csv", encoding="utf-8-sig")
    sp_flags = robust_flags(st, ["soft_palate_length", "soft_palate_thickness", "soft_palate_angle", "soft_palate_PNS_gap"])
    # 5) SSM
    ssm = []
    for s in ("MAND", "CB", "MAX", "UAW"):
        f = SH/s/"points.npz"
        if not f.exists(): continue
        z = np.load(f); X, _ = gpa(z["frame"]); dist = np.sqrt(((X-X.mean(0))**2).sum((1, 2)))
        qc = pd.read_csv(SH/s/"qc.csv", encoding="utf-8-sig").set_index("Patient")
        d = pd.DataFrame({"Patient": [f"P_{i}" for i in z["ids"]], "procrustes_dist": dist}); d["fit_mean_mm"] = d.Patient.map(qc.fit_mean_mm)
        for r in robust_flags(d, ["procrustes_dist", "fit_mean_mm"]): ssm.append({"struct": s, **r})
    pd.DataFrame(sp_flags).round(3).to_csv(DEST/"softpalate_flags.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ssm).round(4).to_csv(DEST/"ssm_outliers.csv", index=False, encoding="utf-8-sig")
    q = lambda s: {k: round(float(v), 3) for k, v in s.describe(percentiles=[.1, .5, .9]).items()} if len(s.dropna()) else {}
    summ = {"n": len(ids),
            "pharynx_TS_vs_AMASSS": {"dice": q(ph.dice), "surf_mean_mm": q(ph.get("surf_mean_mm", pd.Series(dtype=float))), "surf_p95_mm": q(ph.get("surf_p95_mm", pd.Series(dtype=float)))},
            "pharynx_volume_same_range_vs_threshold": {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in vol.items()},
            "airway_profile_curve_r": q(ph.get("profile_r", pd.Series(dtype=float))),
            "molar_seg_vs_ALI_mm": q(mo.err_mm) if len(mo) else {},
            "softpalate_flags": len(sp_flags), "softpalate_PNS_gap_mm": q(st.soft_palate_PNS_gap),
            "ssm_outliers": pd.DataFrame(ssm).groupby("struct").size().to_dict() if ssm else {}}
    save(DEST/"summary.json", summ); print("\n", json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
