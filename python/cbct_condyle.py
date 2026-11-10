"""과두 측면각·축상각. 전 환자 동일 규칙, 엑셀로 임계값을 맞추지 않는다.

입력: AMASSS 하악 분할 seg/P_n_seg_MAND.nrrd, ALI Go, 머리 좌표계(FH 보정 후, 원점 N, 행 = 좌·앞·위).
Sup(과두 최상점): 같은 쪽 Go 앞뒤 -10~+15 mm, 좌우 ±15 mm 범위에서 가장 높은 하악 마스크 점(근돌기 제외).
측면각 (PPT 15): Sup을 지나는 MSP 평행 시상면(±1 mm)에서 Sup과 연결된 하악 단면을 위에서 아래로 따라가며,
               Sup 아래 10 mm 안의 과두 머리 최대폭 행보다 아래, 22 mm까지 중 앞뒤 폭이 가장 좁은 행의 중점을 Mid(경부 중점)로 둔다. Sup–Mid 선과 FH 평행선의 예각.
축상각 (C등급, 정의 그림 잘림): Sup 3 mm 아래 축상 단면(±0.5 mm)의 과두 단면 장축(주성분)과 좌우축의 예각.
출력: Result_py/CBCT/patient_pipeline/v2/condyle_review/v1/condyle.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from cbct_evidence_figs import excel_col
from cbct_fh_repair import repair
from cbct_molar_section import PILOT
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd
from osa.ceph import head_frame

DEST = OUT/"condyle_review/v1"
STEP = .3; NECK = (4., 22.); HEAD = 10.; AXIAL_BELOW = 3.
XL = {"Axial_R": 41, "Axial_L": 42, "Lat_R": 43, "Lat_L": 44}


def neck(sag, ant, sup, top):
    """sag: (ant, sup) 불리언. top: (ant 인덱스, sup 인덱스). 과두 머리 최대폭 행(최상점 아래 HEAD mm 안) 아래에서
    NECK[1] mm까지 앞뒤 폭이 가장 좁은 행의 (중점 ant, sup, 폭). 못 찾으면 None."""
    lab, _ = ndi.label(sag); comp = lab == lab[top]
    prev = (top[0], top[0]); rows = []
    for k in range(top[1], -1, -1):
        i = np.flatnonzero(comp[:, k])
        if not len(i): break
        runs = np.split(i, np.flatnonzero(np.diff(i) > 1)+1); hit = [r for r in runs if r[0] <= prev[1]+1 and r[-1] >= prev[0]-1]
        if not hit: break
        r = max(hit, key=len); prev = (r[0], r[-1]); depth = sup[top[1]]-sup[k]
        if depth > NECK[1]: break
        rows.append((depth, (r[-1]-r[0]+1)*STEP, (ant[r[0]]+ant[r[-1]])/2, sup[k]))
    head = [x for x in rows if x[0] <= HEAD]
    if not head: return None
    h = max(head, key=lambda x: x[1]); below = [x for x in rows if x[0] > h[0]]
    if not below: return None
    n = min(below, key=lambda x: x[1])
    return n[2], n[3], n[1]


def measure(pid):
    r = {"Patient": f"P_{pid}"}; P = repair(landmarks(pid)[0])[0]; o, R = head_frame(P); m, dirs, org = read_nrrd(CB/f"seg/P_{pid}_seg_MAND.nrrd")
    m = (m > 0).astype(np.float32)
    for side, go in (("R", "RGo"), ("L", "LGo")):
        g = R@(np.asarray(P[go], float)-o)
        Ls, ant, sup = np.arange(g[0]-15, g[0]+15, STEP), np.arange(g[1]-10, g[1]+15, STEP), np.arange(g[2]+20, g[2]+80, STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
        v = map_coordinates(m, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=0, cval=0).reshape(len(Ls), len(ant), len(sup)) > 0
        if not v.any(): r[f"reason_{side}"] = "no_mask"; continue
        k = np.flatnonzero(v.any((0, 1))).max(); top = np.argwhere(v[..., k]); i, j = top[np.argmin(np.abs(top-top.mean(0)).sum(1))]    # 최상단 복셀 중 중심에 가장 가까운 것
        r.update({f"Sup_{side}_L": Ls[i], f"Sup_{side}_ant": ant[j], f"Sup_{side}_sup": sup[k]})
        sag = v[np.abs(Ls-Ls[i]) <= 1.].any(0)
        nk = neck(sag, ant, sup, (j, k))
        if nk is None: r[f"reason_{side}"] = "no_neck"; continue
        da, dz = ant[j]-nk[0], sup[k]-nk[1]
        r.update({f"Mid_{side}_ant": nk[0], f"Mid_{side}_sup": nk[1], f"neck_w_{side}": nk[2], f"Lat_{side}": float(np.degrees(np.arctan2(abs(dz), abs(da)))),
                  f"Lat_{side}_tilt": "forward" if da > 0 else "backward"})
        z = int(np.argmin(np.abs(sup-(sup[k]-AXIAL_BELOW)))); ax = v[..., max(z-1, 0):z+2].any(2)
        lab, _ = ndi.label(ax); ii = np.argwhere(ax)
        if len(ii) > 20:
            c = lab[tuple(ii[np.argmin(np.hypot(ii[:, 0]-i, ii[:, 1]-j))])]; pts = np.argwhere(lab == c)*STEP
            e = np.linalg.eigh(np.cov((pts-pts.mean(0)).T))[1][:, -1]; r[f"Axial_{side}"] = float(np.degrees(np.arctan2(abs(e[1]), abs(e[0]))))
    return r


def main():
    ids = [int(a) for a in sys.argv[1:]] or [int(p[2:]) for p, i in pd.read_csv(OUT/"cohort.csv", dtype=str, encoding="utf-8-sig")[["Patient", "Include"]].values if i.lower() == "true"]
    DEST.mkdir(parents=True, exist_ok=True); rows = []
    for pid in ids:
        print(pid, end=" ", flush=True); rows.append(measure(pid))
    d = pd.DataFrame(rows); d["Set"] = ["pilot" if int(p[2:]) in PILOT else "eval" for p in d.Patient]
    for k, c in XL.items(): d[k+"_xl"] = [excel_col(c).get(int(p[2:]), np.nan) for p in d.Patient]
    d.to_csv(DEST/"condyle.csv", index=False, encoding="utf-8-sig"); print()
    for k in XL:
        for s in ("pilot", "eval"):
            g = d[d.Set == s].dropna(subset=[k, k+"_xl"]); e = g[k]-g[k+"_xl"]
            print(k, s, "n", len(g), "auto med", round(g[k].median(), 1), "xl med", round(g[k+"_xl"].median(), 1), "bias", round(e.mean(), 2), "MAE", round(e.abs().mean(), 2), "r", round(g[k].corr(g[k+"_xl"]), 2))
    print(d.filter(like="reason").apply(pd.Series.value_counts).to_dict() if d.filter(like="reason").shape[1] else "no failures")


if __name__ == "__main__":
    main()
