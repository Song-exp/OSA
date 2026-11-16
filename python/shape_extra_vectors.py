"""형태 연구 남은 벡터: 인두 기도 프로파일, 비강 프로파일, 상대관계(설골·하악). 85명, 환자 한 명씩.

좌표계: Po를 쓰지 않는 정중 좌표계(cbct_eb_detect.midline_frame, 원점 PNS, 축 좌·앞·위 mm). Po 오류의 영향을 받지 않는다.
기도 프로파일: 범위 = Eb(후두개곡) 높이+5 mm ~ PNS 높이 (PAV·PMCA와 같은 Roh 범위; 하인두 아래쪽 잘림 문제를 피함).
  PMCA와 같은 기도 성분(cbct_pmca.airway)·중심선 수직 단면(cbct_pmca.sections)을 쓰고, Eb→PNS를 0→1로 정규화한 20구간 단면적.
  요약: 길이(높이), 부피, 최소 단면적과 그 위치, 평균 단면적. Eb 미계측 환자는 빈칸.
비강 프로파일: TotalSegmentator 비강 라벨(cbct_nav.ts_nasal), PNS→ANS를 0→1로 정규화한 20구간의 관상 단면적 + 부피·최소 단면.
상대관계: 설골(cbct_eb_detect.hyoid)·Eb·Me·Pog·Go(좌우 평균)의 정중 좌표(앞·위), 설골–하악평면(Go–Me) 거리(MP-H), 설골–Me 거리.
Eb는 cbct_restore_extend.eb_row(동결된 설골 보정량)로 85명 모두 같은 규칙으로 다시 구한다.
출력: Result_py/shape/extra/{airway_profile,nasal_profile,relation}.csv
사용: python shape_extra_vectors.py [환자번호...]
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

import cbct_pmca as pm
from cbct_accuracy_experiments import wait_ram
from cbct_eb_detect import midline_frame
from cbct_nav import ts_nasal
from cbct_patient_pipeline import CB, OUT, landmarks
from cbct_restore_extend import eb_row
from check_landmarks import read_nrrd

DEST = Path(__file__).resolve().parents[1]/"Result_py/shape/extra"
NB = 20; GRID = np.linspace(0, 1, NB)


def profile(pos, area, lo, hi):
    """pos(높이·앞뒤 좌표)에서 lo→hi를 0→1로 정규화해 NB 구간 단면적 (0 넓이는 결측)."""
    a = np.where(area > 0, area, np.nan); t = (pos-lo)/(hi-lo); ok = np.isfinite(a) & (t >= 0) & (t <= 1)
    if ok.sum() < 5: return np.full(NB, np.nan)
    o = np.argsort(t[ok]); return np.interp(GRID, t[ok][o], a[ok][o])


def airway(pid, e, v, dirs, org, o, R, s):
    r = {}
    if not np.isfinite(e.Eb_sup): return r
    Ls = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP); ant = np.arange(pm.ANT[0], pm.ANT[1]+1e-9, pm.STEP); sup = np.arange(e.Eb_sup, pm.TOP+1e-9, pm.STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup))
    m = pm.airway(vol, Ls, ant, sup, s[1], s[2]); del vol
    if m is None: return r
    ax, perp, _ = pm.sections(m, Ls, ant, sup)
    lo = e.Eb_sup+pm.BOTTOM_CENTER; rng = (sup >= lo) & (sup <= 0)    # PMCA와 같이 아래 경계 5 mm는 잘림을 피해 뺀다
    p = profile(sup, perp, lo, 0.0)
    r |= {f"AW_p{i:02d}": x for i, x in enumerate(p)}
    pr = np.where(perp[rng] > 0, perp[rng], np.nan)
    r |= {"AW_length": -e.Eb_sup, "AW_volume": float(ax[rng].sum()*pm.STEP), "AW_min": float(np.nanmin(pr)) if np.isfinite(pr).any() else np.nan,
          "AW_min_pos": float(GRID[np.nanargmin(p)]) if np.isfinite(p).any() else np.nan, "AW_mean": float(np.nanmean(pr)) if np.isfinite(pr).any() else np.nan}
    return r


def nasal(pid, P, o, R):
    ans = R@(P["ANS"]-o); L = np.arange(-30, 30.01, .5); A = np.arange(0, ans[1]+1e-9, .5); S = np.arange(0, 45.01, .5)
    G = np.stack(np.meshgrid(L, A, S, indexing="ij"), -1).reshape(-1, 3)
    lab = ts_nasal(pid, o+G@R)
    if lab is None or not lab.any(): return {}
    lab = lab.reshape(len(L), len(A), len(S)); area = lab.sum((0, 2))*.25; p = profile(A, area, 0.0, ans[1])
    return {**{f"NC_p{i:02d}": x for i, x in enumerate(p)}, "NC_volume": float(lab.sum()*.125), "NC_length": float(ans[1]),
            "NC_min": float(np.nanmin(p)), "NC_min_pos": float(GRID[np.nanargmin(p)]), "NC_mean": float(np.nanmean(p))}


def relation(e, P, o, R):
    q = {k: R@(P[k]-o) for k in ("Me", "Pog", "RGo", "LGo", "N", "S", "Ba", "ANS")}; go = (q["RGo"]+q["LGo"])/2
    r = {f"{k}_{ax}": q[k][i] for k in ("Me", "Pog", "N", "S", "Ba") for ax, i in (("ant", 1), ("sup", 2))}
    r |= {"Go_ant": go[1], "Go_sup": go[2], "Eb_ant": e.Eb_ant, "Eb_sup": e.Eb_sup}
    if "hy_ant" in e and np.isfinite(e.get("hy_ant", np.nan)):
        h = np.array([e.hy_ant, e.hy_sup]); a, b = go[1:], q["Me"][1:]; d = b-a
        r |= {"Hy_ant": h[0], "Hy_sup": h[1], "MP_H": float(abs(d[0]*(h[1]-a[1])-d[1]*(h[0]-a[0]))/np.linalg.norm(d)),   # 설골 → Go–Me 직선 거리
              "Hy_Me": float(np.linalg.norm(h-b))}
    return r


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    offset = json.loads((OUT/"eb_review/v2/summary.json").read_text(encoding="utf-8"))["offset_mm"]
    ids = [int(a) for a in sys.argv[1:]] or sorted(int(p.parent.name[2:]) for p in OUT.glob("P_*/result.json"))
    out = {k: [] for k in ("airway_profile", "nasal_profile", "relation")}
    for pid in ids:
        wait_ram(4, f"P_{pid}"); p = f"P_{pid}"; print(p, end=" ", flush=True)
        P = landmarks(pid)[0]; o, R, _ = midline_frame(P); s = R@(P["S"]-o)
        e = eb_row(pid, offset); v, dirs, org = read_nrrd(CB/f"nrrd/{p}.nrrd"); v = v.astype(np.float32)
        out["airway_profile"].append({"Patient": p, "Eb_method": e.method, **airway(pid, e, v, dirs, org, o, R, s)}); del v
        out["nasal_profile"].append({"Patient": p, **nasal(pid, P, o, R)})
        out["relation"].append({"Patient": p, **relation(e, P, o, R)}); gc.collect()
    for k, rows in out.items():
        f = DEST/f"{k}.csv"; d = pd.DataFrame(rows)
        if sys.argv[1:] and f.exists():    # 일부 환자만 다시 계산하면 기존 표에 합친다
            old = pd.read_csv(f, encoding="utf-8-sig"); d = pd.concat([old[~old.Patient.isin(d.Patient)], d], ignore_index=True)
        d.round(4).to_csv(f, index=False, encoding="utf-8-sig")
        print(f"\n{k}: {len(d)}명, 값 있는 환자 {int(d.drop(columns=['Patient', 'Eb_method'], errors='ignore').notna().any(axis=1).sum())}")


if __name__ == "__main__":
    main()
