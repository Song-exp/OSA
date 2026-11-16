"""혀·저작근(TotalSegmentator head_muscles)과 상악동·전두동(craniofacial_structures) 피처. 전 환자 같은 규칙.

좌표: 정중 좌표계(Po 미사용, 원점 PNS, 축 = 좌·앞·위, mm). 좌우는 정중 좌표계 L 부호(+ = 왼쪽).
피처
- 상악동: 좌우 라벨 부피, 라벨 안 공기(< -400) 부피와 비율(점막 비후·액체가 있으면 공기 비율이 낮다), 전두동 부피
- 혀: 부피, 무게중심(앞뒤·위아래), 가장 뒤 끝의 앞뒤 위치(혀뿌리, PNS 기준), 가장 아래 끝 높이, 앞뒤 길이·높이,
      혀 ÷ 하악 부피 비(AMASSS 하악, 선행연구의 tongue/mandible volume ratio)
- 저작근: 교근·측두근·바깥/안쪽 익돌근 좌우 부피 (이복근은 너무 작아 제외)
- 작은 조각 제거: 0.5 mL 미만 연결 덩어리는 오검출로 보고 버린다(P_5 상악동 가장자리 5·18 복셀 조각)
- 잘림 표시: 라벨이 영상 가장자리(2 복셀 안)에 닿으면 *_cut = 1 (촬영 범위에 잘려 부피가 작게 나왔을 수 있음)
출력: Result_py/shape/fine/tongue_sinus_muscles.csv
사용: python shape_tongue_sinus.py            (상악동 분할이 있는 환자 전부. 혀·저작근 분할이 있으면 그 피처도 포함)
"""
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import CB, landmarks
from run_ali import ROOT, ascii_path

OUT = ROOT/"Result_py/shape/fine/tongue_sinus_muscles.csv"
AIR = -400.
MUSCLES = {"masseter": (1, 2), "temporalis": (3, 4), "lat_pterygoid": (5, 6), "med_pterygoid": (7, 8)}    # (오른쪽, 왼쪽)


def read(task, pid):
    f = CB/f"seg_totalseg/{task}/P_{pid}.nii.gz"
    return sitk.ReadImage(str(ascii_path(f))) if f.exists() else None


def coords(img, mask, o, R):
    """마스크 복셀의 정중 좌표계 좌표 (N×3)."""
    idx = np.argwhere(mask)[:, ::-1].astype(float)    # (x, y, z) 인덱스
    org, sp, D = np.array(img.GetOrigin()), np.array(img.GetSpacing()), np.array(img.GetDirection()).reshape(3, 3)
    return ((org+(idx*sp)@D.T)-o)@R.T


def clean(mask, vox_ml, min_ml=.5):
    lab, n = ndi.label(mask)
    if n == 0: return mask
    keep = np.flatnonzero(ndi.sum(mask, lab, range(1, n+1))*vox_ml >= min_ml)+1
    return np.isin(lab, keep)


def touches_edge(mask, k=2):
    return bool(mask[:k].any() or mask[-k:].any() or mask[:, :k].any() or mask[:, -k:].any() or mask[:, :, :k].any() or mask[:, :, -k:].any())


def measure(pid):
    cf, hm = read("craniofacial_structures", pid), read("head_muscles", pid)
    if cf is None: return None
    P = landmarks(pid)[0]; o, R, _ = midline_frame(P)
    ct = sitk.ReadImage(str(ascii_path(CB/f"nrrd/P_{pid}.nrrd")))
    r = {"Patient": f"P_{pid}"}
    # 상악동·전두동
    a = sitk.GetArrayFromImage(cf); v = float(np.prod(cf.GetSpacing()))/1000
    hu = sitk.GetArrayFromImage(sitk.Resample(ct, cf, sitk.Transform(), sitk.sitkLinear, 0.))
    s = clean(a == 5, v); fr = clean(a == 6, v)
    if s.any():
        L = np.zeros(a.shape); L[s] = coords(cf, s, o, R)[:, 0]
        for side, m in (("R", s & (L < 0)), ("L", s & (L > 0))):
            r[f"sinus_max_{side}_mL"] = m.sum()*v; r[f"sinus_max_{side}_air_mL"] = (m & (hu < AIR)).sum()*v
            r[f"sinus_max_{side}_air_frac"] = r[f"sinus_max_{side}_air_mL"]/r[f"sinus_max_{side}_mL"] if m.any() else np.nan
        r["sinus_max_cut"] = int(touches_edge(s))
    r["sinus_frontal_mL"] = fr.sum()*v; r["sinus_frontal_cut"] = int(touches_edge(fr)) if fr.any() else np.nan
    if hm is None: return r    # 혀·저작근 분할이 아직 없으면 상악동·전두동만
    # 혀·저작근
    b = sitk.GetArrayFromImage(hm); v2 = float(np.prod(hm.GetSpacing()))/1000; t = clean(b == 9, v2)
    r["tongue_mL"] = t.sum()*v2
    if t.any():
        q = coords(hm, t, o, R)
        r.update(tongue_c_ant=q[:, 1].mean(), tongue_c_sup=q[:, 2].mean(), tongue_back_ant=q[:, 1].min(), tongue_low_sup=q[:, 2].min(),
                 tongue_len=np.ptp(q[:, 1]), tongue_height=np.ptp(q[:, 2]), tongue_cut=int(touches_edge(t)))
    for name, (rr, ll) in MUSCLES.items():
        mr, ml = clean(b == rr, v2), clean(b == ll, v2)
        r[f"{name}_R_mL"], r[f"{name}_L_mL"] = mr.sum()*v2, ml.sum()*v2
        r[f"{name}_cut"] = int(touches_edge(mr | ml))
    mf = CB/f"seg/P_{pid}_seg_MAND.nrrd"
    if mf.exists():
        m = sitk.ReadImage(str(ascii_path(mf))); r["mandible_mL"] = float((sitk.GetArrayFromImage(m) > 0).sum())*float(np.prod(m.GetSpacing()))/1000
        r["tongue_mandible_ratio"] = r["tongue_mL"]/r["mandible_mL"] if r["mandible_mL"] else np.nan
    return r


def main():
    ids = [int(a) for a in sys.argv[1:]] or sorted(int(f.name[2:-7]) for f in (CB/"seg_totalseg/craniofacial_structures").glob("P_*.nii.gz"))
    rows = [r for pid in ids if (r := measure(pid)) is not None]
    d = pd.DataFrame(rows); OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.csv"); d.round(3).to_csv(tmp, index=False, encoding="utf-8-sig"); tmp.replace(OUT)
    print(len(d), "patients ->", OUT); print(d.describe().T[["count", "mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
