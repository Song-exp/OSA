"""CBCT 랜드마크로 머리 기준 좌표계를 만들고 Roh (2025) 그림 1A 변수를 계산합니다. 계획 문서 2장, 4-1.

좌표계 (Jung 2025, Invivo AFOR):
  FH  = RPo, LPo, ROr 세 점의 평면 (fh="jung", 기본)
        대안 fh="yao4": 좌우 Po·좌우 Or 4점의 최소제곱 평면 (Yao 2023) / fh="lor": RPo, LPo, LOr (Dot 2022)
  MSP = FH에 수직이고 N과 Ba를 지나는 평면
  원점 N, 축 = (좌+, 앞+, 위+) [mm]
입력 좌표: DICOM 환자 좌표 LPS (mm), {이름: (x, y, z)}
각도는 MSP에 투영해 계산(2D)이 기본, ANB와 MP-FH는 3D 값도 함께 냅니다.
"""
from __future__ import annotations

import numpy as np

unit = lambda v: v / np.linalg.norm(v)


def fh_normal(p: dict, fh: str = "jung") -> np.ndarray:
    g = lambda k: np.asarray(p[k], float)
    if fh == "yao4":
        pts = np.vstack([g(k) for k in ("RPo", "LPo", "ROr", "LOr")])
        return np.linalg.svd(pts - pts.mean(0))[2][-1]
    o = g("ROr") if fh == "jung" else g("LOr")
    return unit(np.cross(g("LPo") - g("RPo"), o - g("RPo")))


def head_frame(p: dict, fh: str = "jung") -> tuple[np.ndarray, np.ndarray]:
    """(원점, 3x3 행렬: 행 = 좌, 앞, 위 단위벡터)."""
    rpo, lpo, n, ba = (np.asarray(p[k], float) for k in ("RPo", "LPo", "N", "Ba"))
    sup = fh_normal(p, fh)
    if sup @ (n - rpo) < 0:                      # N은 FH보다 위 → 위쪽을 향하게 (영상 방향과 무관)
        sup = -sup
    left = unit(np.cross(n - ba, sup))           # MSP 법선: N–Ba와 FH 법선을 모두 포함하는 평면의 법선
    if left @ (lpo - rpo) < 0:
        left = -left
    ant = np.cross(sup, left)
    if ant @ (n - ba) < 0:
        ant = -ant
    return n, np.vstack([left, ant, sup])


def to_head(p: dict, fh: str = "jung") -> dict:
    o, R = head_frame(p, fh)
    return {k: R @ (np.asarray(v, float) - o) for k, v in p.items()}


def _msp_angle(v):
    """MSP 투영 벡터의 방향각: 아래(−위)에서 앞쪽으로 돌아간 각 [deg]."""
    return np.degrees(np.arctan2(v[1], -v[2]))


def measures(p: dict, fh: str = "jung") -> dict:
    h = to_head(p, fh)
    out = {}
    if {"A", "B"} <= h.keys():
        out["ANB"] = _msp_angle(h["A"]) - _msp_angle(h["B"])
        a, b = h["A"], h["B"]
        out["ANB_3D"] = np.sign(out["ANB"]) * np.degrees(np.arccos(unit(a) @ unit(b)))
    # Nperp: N을 지나 FH에 수직인 선 → 앞뒤 좌표 차이 (앞쪽 +)
    for k, name in (("A", "Nperp_A"), ("Pog", "Nperp_Pog")):
        if k in h:
            out[name] = h[k][1]
    if {"Me", "RGo", "LGo"} <= h.keys():
        go = (h["RGo"] + h["LGo"]) / 2
        d = go - h["Me"]                         # Me → Go (뒤·위로 향함)
        out["MPFH_GoMe"] = np.degrees(np.arctan2(d[2], -d[1]))
        nrm = unit(np.cross(h["RGo"] - h["Me"], h["LGo"] - h["Me"]))
        out["MPFH_3Dplane"] = np.degrees(np.arccos(abs(nrm[2])))
    if {"PNS", "Eb"} <= h.keys():
        out["PAL_3D"] = np.linalg.norm(h["Eb"] - h["PNS"])
        out["PAL_MSP"] = np.linalg.norm((h["Eb"] - h["PNS"])[1:])
    return out


if __name__ == "__main__":
    # 자가 점검: 알려진 기하에서 값이 맞는지, 머리를 임의로 돌리고 옮겨도 값이 같은지
    t = np.radians
    ref = {"N": (0, 0, 0), "Ba": (0, -90, -30), "RPo": (-55, -80, -25), "LPo": (55, -80, -25), "ROr": (-35, -10, -25), "LOr": (35, -10, -25),
           "A": (0, 60 * np.sin(t(-2)), -60 * np.cos(t(2))),
           "B": (0, 100 * np.sin(t(-7)), -100 * np.cos(t(7))),
           "Pog": (0, -8, -115), "Me": (0, -12, -122), "RGo": (-45, -80, -122 + 68 * np.tan(t(28))),
           "LGo": (45, -80, -122 + 68 * np.tan(t(28)))}
    # 위 좌표는 이미 (좌, 앞, 위) 기준이므로 LPS로 바꿔 넣음: x=좌, y=−앞, z=위
    lps = {k: np.array([v[0], -v[1], v[2]]) for k, v in ref.items()}
    m = measures(lps)
    assert abs(m["ANB"] - 5) < 1e-9, m["ANB"]
    assert abs(m["Nperp_A"] - 60 * np.sin(t(-2))) < 1e-9 and abs(m["Nperp_Pog"] + 8) < 1e-9
    assert abs(m["MPFH_GoMe"] - 28) < 1e-9, m["MPFH_GoMe"]
    assert abs(m["MPFH_3Dplane"] - 28) < 1e-9, m["MPFH_3Dplane"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        Q *= np.sign(np.linalg.det(Q))           # 회전만 (반사 제외)
        shift = rng.normal(scale=50, size=3)
        moved = {k: Q @ v + shift for k, v in lps.items()}
        m2 = measures(moved)
        assert all(abs(m[k] - m2[k]) < 1e-9 for k in m), (m, m2)
        for fh in ("yao4", "lor"):             # 좌우 대칭 기하에서는 세 FH 정의가 같은 값
            m3 = measures(moved, fh)
            assert all(abs(m[k] - m3[k]) < 1e-9 for k in m), (fh, m, m3)
    print("ceph 자가 점검 통과:", {k: round(v, 3) for k, v in m.items()})
