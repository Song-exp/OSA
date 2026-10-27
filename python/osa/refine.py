"""정중 랜드마크(A, B, Pog) 보정: AMASSS 하악·상악 마스크의 앞쪽 윤곽선 위에서 정의대로 다시 잡습니다.

정의 (Jung 2025 표 1, Roh 그림 1A)와 구현:
  Pog: 턱끝에서 가장 앞쪽 점        → Me 위 3~20 mm 높이에서 하악 앞 윤곽이 가장 앞으로 나온 점
  B  : 턱끝–하악 치조부 사이 가장 오목한 점 → Pog 위 2~18 mm에서 하악 앞 윤곽이 가장 뒤로 들어간 점
  A  : ANS–상악 치조부 사이 가장 오목한 점  → ANS 아래 2~15 mm에서 상악 앞 윤곽이 가장 뒤로 들어간 점
  "앞/뒤"는 머리 좌표계(FH·MSP, osa.ceph)의 앞뒤 축. 높이 범위는 해부학적 거리로 미리 고정했고 엑셀을 보고 정하지 않음.
  윤곽선 = 정중면 ±1 mm 두께에서 각 높이마다 마스크가 가장 앞에 있는 위치 (0.25 mm 격자).
안전장치: ALI 점에서 10 mm 넘게 옮겨지거나, 범위 안 윤곽이 절반 넘게 비면 보정하지 않고 표시만 함.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .ceph import head_frame

STEP, SLAB = 0.25, 1.0
WINDOWS = {"Pog": ("Me", 3, 20, "max"), "B": ("Pog", 2, 18, "min"), "A": ("ANS", -15, -2, "min")}
MAX_SHIFT = 10.0


def _sample(mask, dirs, origin, o, R, ant, sup):
    """머리 좌표 격자(좌우 = 정중면 ±SLAB, 앞뒤 ant, 높이 sup)에서 마스크를 읽어 좌우 방향 최댓값."""
    xs = np.arange(-SLAB, SLAB + 1e-9, STEP)
    L, A, S = np.meshgrid(xs, ant, sup, indexing="ij")
    head = np.stack([L, A, S], -1).reshape(-1, 3)
    lps = o + head @ R                                  # R 행 = 좌, 앞, 위 → 환자 좌표
    ijk = np.linalg.solve(dirs.T, (lps - origin).T)     # (i, j, k)
    vals = map_coordinates(mask, ijk[::-1], order=0, mode="constant", cval=0)   # mask는 (k, j, i)
    return vals.reshape(L.shape).max(0) > 0             # (ant, sup)


def front_contour(mask, dirs, origin, o, R, sup, ant_range):
    ant = np.arange(*ant_range, STEP)
    m = _sample(mask, dirs, origin, o, R, ant, sup)
    has = m.any(0)
    front = np.where(has, ant[::-1][np.argmax(m[::-1], axis=0)], np.nan)   # 가장 앞쪽(ant 최대) 마스크 위치
    return front


def refine(P: dict, mand, maxm, dirs, origin, fh: str = "jung") -> tuple[dict, dict]:
    """(보정된 좌표 dict, 점별 정보). 좌표는 LPS mm."""
    o, R = head_frame(P, fh)
    h = {k: R @ (np.asarray(v, float) - o) for k, v in P.items()}
    out, info = dict(P), {}
    for name in ("Pog", "B", "A"):                      # Pog를 먼저 (B 범위의 기준)
        ref, lo, hi, mode = WINDOWS[name]
        base = h[ref] if ref != "Pog" else R @ (np.asarray(out["Pog"], float) - o)
        sup = np.arange(base[2] + lo, base[2] + hi + 1e-9, STEP)
        a0 = h[name][1]
        front = front_contour(mand if name != "A" else maxm, dirs, origin, o, R, sup, (a0 - 25, a0 + 25))
        ok = np.isfinite(front)
        rec = {"window_coverage": ok.mean()}
        if ok.mean() < 0.5:
            info[name] = {**rec, "status": "윤곽 부족 → ALI 유지"}
            continue
        best = np.nanmax(front) if mode == "max" else np.nanmin(front)
        ties = np.flatnonzero(np.abs(front - best) <= STEP / 2)   # 윤곽이 격자 단위로 계단져 동률이 생김
        i = ties[len(ties) // 2]                                   # → 동률 구간의 가운데
        new_h = np.array([0.0, front[i], sup[i]])
        shift = np.linalg.norm(new_h - h[name])
        rec["shift_mm"] = shift
        if shift > MAX_SHIFT:
            info[name] = {**rec, "status": f"이동 {shift:.1f} mm > {MAX_SHIFT} → ALI 유지"}
            continue
        out[name] = o + new_h @ R
        info[name] = {**rec, "status": "보정"}
    return out, info


if __name__ == "__main__":
    # 자가 점검: 앞쪽으로 오목한 가짜 턱 윤곽에서 B·Pog가 정의 위치에 잡히는지
    from .ceph import to_head
    n = 120
    dirs, origin = np.eye(3), np.array([-60.0, -60.0, -60.0])   # 1 mm 격자, LPS
    mand = np.zeros((n, n, n), np.uint8)
    maxm = np.zeros_like(mand)
    z = np.arange(n) + origin[2]                       # 높이(LPS z = 위)
    y = np.arange(n) + origin[1]                       # LPS y(+ = 뒤)
    for k, zz in enumerate(z):
        # 앞 윤곽 ant(z): Pog(z=-40)에서 가장 앞(ant=10), B(z=-28)에서 가장 뒤(ant=4), 그 위로 다시 앞
        if -50 <= zz <= -15:
            ant = 10 - 6 * np.exp(-((zz + 28) / 5) ** 2) if zz > -40 else 10 - 0.3 * (zz + 40) ** 2 / 10
            mand[k, (-y <= ant) & (-y >= ant - 15), 50:70] = 1
        if -12 <= zz <= 5:                              # 상악: A(z=-6)에서 가장 뒤(ant=6)
            ant = 12 - 6 * np.exp(-((zz + 6) / 3) ** 2)
            maxm[k, (-y <= ant) & (-y >= ant - 15), 50:70] = 1
    P = {"N": (0, -10, 40), "Ba": (0, 80, 0), "RPo": (-50, 70, 10), "LPo": (50, 70, 10), "ROr": (-30, 0, 10),
         "Me": (0, -5, -48), "Pog": (0, -14, -40), "B": (0, -2, -25), "ANS": (0, -14, 1), "A": (0, -3, -4)}
    P = {k: np.array(v, float) for k, v in P.items()}
    newP, info = refine(P, mand, maxm, dirs, origin)
    h = to_head(newP)
    # 이 기하에서 머리 좌표 = (x, -(y+10), z-40). 기대값: B z=-28, A z=-6 → 높이 -68, -46
    assert abs(h["B"][2] + 68) < 1.5 and abs(h["A"][2] + 46) < 1.5, (h["B"], h["A"], info)
    assert all(v["status"] == "보정" for v in info.values()), info
    print("refine 자가 점검 통과:", {k: v["status"] for k, v in info.items()})
