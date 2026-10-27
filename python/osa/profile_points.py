"""A·B·Pog 후보 ③·④a·④b: 앞쪽 뼈 윤곽 f(z)에서 현(chord) 기준 최심점으로 다시 잡습니다.

후보 정의는 docs/파일럿_자동계측_1차.md 7장에 결과를 보기 전에 고정했습니다.
  윤곽 표현  msp: 마스크 정중면 ±1 mm / sil: 마스크 정중면 ±25 mm 겹침 / drr: CT 좌우 합(모의 측면 X-ray)의 뼈 경계
  점 선택    Pog = Me 위 3~20 mm의 가장 앞 점
             B   = 현(Pog, Pog 위 8~30 mm의 가장 앞 점)에서 뒤로 가장 먼 점
             A   = 현(ANS −3~0 mm의 가장 앞 점, ANS 아래 8~25 mm의 가장 앞 점)에서 뒤로 가장 먼 점
좌표: 머리 좌표계(좌, 앞, 위; osa.ceph.head_frame). 찾은 점의 좌우 좌표는 정중면(0)에 둡니다.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

from .ceph import head_frame

STEP = 0.25
SLABS = {"msp": 1.0, "sil": 25.0}
DRR_HALF, DRR_STEP = 40.0, 0.5
DRR_WIN = (-8.0, 2.0)                  # sil 외곽 기준 (뒤, 앞) mm


def sample(vol, dirs, origin, o, R, xs, ant, sup, order=0):
    """머리 좌표 격자 (xs, ant, sup)에서 볼륨(k, j, i 배열)을 읽음."""
    L, A, S = np.meshgrid(xs, ant, sup, indexing="ij")
    lps = o + np.stack([L, A, S], -1).reshape(-1, 3) @ R
    ijk = np.linalg.solve(dirs.T, (lps - origin).T)
    return map_coordinates(vol, ijk[::-1], order=order, mode="constant", cval=0).reshape(L.shape)


def _front(m2d, ant):
    """(ant, sup) 불리언 → 높이별 가장 앞쪽 True 위치 (없으면 NaN)."""
    has = m2d.any(0)
    return np.where(has, ant[::-1][np.argmax(m2d[::-1], axis=0)], np.nan)


def mask_profile(mask, geo, ant, sup, slab):
    xs = np.arange(-slab, slab + 1e-9, STEP if slab <= 2 else 1.0)
    return _front(sample(mask, *geo, xs, ant, sup).max(0) > 0, ant)


def drr_image(ct, geo, ant, sup):
    xs = np.arange(-DRR_HALF, DRR_HALF + 1e-9, DRR_STEP)
    v = sample(ct.astype(np.float32), *geo, xs, ant, sup, order=1)
    return (np.clip(v, -1000, 3000) + 1000).sum(0) * DRR_STEP           # (ant, sup)


def drr_profile(drr, sil_front, ant):
    g = -np.gradient(gaussian_filter1d(drr, 2, axis=0), axis=0)           # 뒤로 갈수록 증가 = 양수
    out = np.full(sil_front.shape, np.nan)
    for j, f in enumerate(sil_front):
        if not np.isfinite(f):
            continue
        w = (ant >= f + DRR_WIN[0]) & (ant <= f + DRR_WIN[1])
        if w.any():
            out[j] = ant[w][np.argmax(g[w, j])]
    return out


def _argext(f, sup, lo, hi, mode="max"):
    w = (sup >= lo) & (sup <= hi) & np.isfinite(f)
    if not w.any():
        return None
    idx = np.flatnonzero(w)
    best = f[idx].max() if mode == "max" else f[idx].min()
    ties = idx[np.abs(f[idx] - best) <= STEP / 2]
    return ties[len(ties) // 2]


def chord_deepest(f, sup, i0, i1):
    """sup[i0]~sup[i1] 사이에서 두 끝점을 이은 직선보다 뒤로 가장 먼 윤곽 점의 인덱스."""
    a, b = sorted((i0, i1))
    seg = np.arange(a + 1, b)
    seg = seg[np.isfinite(f[seg])]
    if seg.size == 0:
        return None
    line = f[a] + (f[b] - f[a]) * (sup[seg] - sup[a]) / (sup[b] - sup[a])
    depth = line - f[seg]
    ties = seg[np.abs(depth - depth.max()) <= STEP / 4]
    return ties[len(ties) // 2]


def pick(f_max, f_mand, sup, h) -> dict:
    """윤곽(상악 f_max, 하악 f_mand)에서 Pog·B·A (머리 좌표 [0, ant, sup])."""
    out = {}
    ip = _argext(f_mand, sup, h["Me"][2] + 3, h["Me"][2] + 20)
    if ip is not None:
        out["Pog"] = np.array([0.0, f_mand[ip], sup[ip]])
        iu = _argext(f_mand, sup, sup[ip] + 8, sup[ip] + 30)
        ib = chord_deepest(f_mand, sup, ip, iu) if iu is not None else None
        if ib is not None:
            out["B"] = np.array([0.0, f_mand[ib], sup[ib]])
    z = h["ANS"][2]
    i_top, i_bot = _argext(f_max, sup, z - 3, z), _argext(f_max, sup, z - 25, z - 8)
    if i_top is not None and i_bot is not None:
        ia = chord_deepest(f_max, sup, i_bot, i_top)
        if ia is not None:
            out["A"] = np.array([0.0, f_max[ia], sup[ia]])
    return out


def candidates(P: dict, ct, mand, maxm, dirs, origin, fh="jung") -> tuple[dict, dict]:
    """{후보: 보정 좌표(LPS) dict}, {후보: 그림용 자료}. 후보 = msp, sil, drr."""
    o, R = head_frame(P, fh)
    h = {k: R @ (np.asarray(v, float) - o) for k, v in P.items()}
    ants = [h[k][1] for k in ("A", "B", "Pog", "ANS")]
    ant = np.arange(min(ants) - 20, max(ants) + 15, STEP)
    sup = np.arange(h["Me"][2] - 8, h["ANS"][2] + 6, STEP)
    geo = (dirs, origin, o, R)
    prof = {k: {"MAX": mask_profile(maxm, geo, ant, sup, s), "MAND": mask_profile(mand, geo, ant, sup, s)}
            for k, s in SLABS.items()}
    drr = drr_image(ct, geo, ant, sup)
    prof["drr"] = {j: drr_profile(drr, prof["sil"][j], ant) for j in ("MAX", "MAND")}
    out = {}
    for k, pr in prof.items():
        pts = pick(pr["MAX"], pr["MAND"], sup, h)
        out[k] = {**P, **{n: o + v @ R for n, v in pts.items()}}
    return out, {"ant": ant, "sup": sup, "drr": drr, "prof": prof, "o": o, "R": R, "h": h}


if __name__ == "__main__":
    # 자가 점검: 현 기준 최심점이 오목부 바닥을 찾고, 기울어진 윤곽에서도 FH 최후방점과 다르게 동작하는지
    sup = np.arange(0, 20.001, STEP)
    f = 10 - 4 * np.exp(-((sup - 8) / 3) ** 2) + 0.6 * sup      # 위로 갈수록 앞으로 기운 윤곽 + 오목부(z=8 부근)
    i = chord_deepest(f, sup, 0, len(sup) - 1)
    assert abs(sup[i] - 8) < 1.5, sup[i]
    assert abs(sup[np.argmin(f)] - sup[i]) > 0.5          # 기울어진 윤곽에서는 최후방점과 다른 높이
    print(f"profile_points 자가 점검 통과: 현 기준 최심점 z={sup[i]:.2f}, FH 최후방점 z={sup[np.argmin(f)]:.2f}")
