"""Me(그리고 A·B·Pog)가 뼈 마스크 밖에 찍히는 원인 조사: 마스크 경계 vs CT 뼈 경계 vs ALI 점.

각 점에서 바깥쪽 방향(Me = 아래, A·B·Pog = 앞)으로 CT 값과 AMASSS 마스크를 0.1 mm 간격으로 읽어
  CT 경계  : 뼈 안쪽에서 바깥으로 가며 HU가 처음 400 아래로 떨어지는 위치 (+ 같은 구간에서 밝기 경사가 가장 큰 위치)
  마스크 경계: 마스크가 처음 0이 되는 위치
를 찾고, ALI 점이 각각에서 몇 mm 바깥인지 비교합니다. (+ = 바깥쪽)
판단: 마스크 경계가 CT 경계보다 안쪽이면 "마스크가 작게 잡힘", CT 경계와 마스크가 같고 ALI만 바깥이면 "ALI 위치/정의 차이".
결과: Result_py/pilot/diag_boundary.csv
"""
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from osa.ceph import head_frame
from pilot_candidates import load
from pilot_diagnose import points
from pilot_measure import OUT, PILOT

CHECK = {"Me": ("MAND", np.array([0, 0, -1.0])), "Pog": ("MAND", np.array([0, 1.0, 0])),
         "B": ("MAND", np.array([0, 1.0, 0])), "A": ("MAX", np.array([0, 1.0, 0]))}
HU_EDGE = 400


def line(vol, dirs, org, o, R, start_h, direction, ts, order):
    head = start_h + ts[:, None] * direction
    lps = o + head @ R
    ijk = np.linalg.solve(dirs.T, (lps - org).T)
    return map_coordinates(vol, ijk[::-1], order=order, mode="nearest")


def main():
    rows = []
    for p in PILOT:
        P = points(p)
        ct, mand, maxm, dirs, org = load(p)
        masks = {"MAND": mand, "MAX": maxm}
        o, R = head_frame(P)
        ctf = ct.astype(np.float32)
        for lm, (s, dvec) in CHECK.items():
            h = R @ (P[lm] - o)
            ts = np.arange(-8, 10.001, 0.1)                       # 점 기준 안쪽 8 mm ~ 바깥 10 mm
            hu = line(ctf, dirs, org, o, R, h, dvec, ts, 1)
            mk = line(masks[s], dirs, org, o, R, h, dvec, ts, 0) > 0
            inside = np.flatnonzero(mk & (ts <= 0))               # 점 안쪽에서 마스크가 있는 곳부터 시작
            start = inside[-1] if inside.size else 0
            m_out = np.flatnonzero(~mk[start:])
            mask_edge = ts[start + m_out[0]] if m_out.size else np.nan
            bone = np.flatnonzero(hu[: start + 1] >= HU_EDGE)     # 안쪽 뼈에서 출발
            b0 = bone[-1] if bone.size else start
            h_out = np.flatnonzero(hu[b0:] < HU_EDGE)
            hu_edge = ts[b0 + h_out[0]] if h_out.size else np.nan
            seg = slice(max(b0 - 20, 0), min(b0 + 120, ts.size))
            grad_edge = ts[seg][np.argmin(np.gradient(hu[seg]))]   # 바깥으로 가며 가장 급히 어두워지는 곳
            rows.append({"Patient": f"P_{p}", "Landmark": lm, "ALI_minus_maskedge": -mask_edge,
                         "ALI_minus_HU400edge": -hu_edge, "ALI_minus_gradedge": -grad_edge,
                         "mask_minus_HU400": mask_edge - hu_edge, "HU_at_ALI": hu[np.argmin(np.abs(ts))]})
    d = pd.DataFrame(rows)
    d.round(2).to_csv(OUT / "diag_boundary.csv", index=False, encoding="utf-8-sig")
    print("점과 경계의 거리 (mm, + = ALI 점이 경계보다 바깥) · 마스크−HU400 (+ = 마스크가 더 바깥)\n")
    print(d.groupby("Landmark")[["ALI_minus_maskedge", "ALI_minus_HU400edge", "ALI_minus_gradedge",
                                 "mask_minus_HU400", "HU_at_ALI"]].median().round(2).to_string())
    print("\nMe 환자별\n" + d[d.Landmark == "Me"].drop(columns="Landmark").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
