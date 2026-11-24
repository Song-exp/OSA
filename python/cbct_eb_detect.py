"""Eb(Yao 2023: most inferior point of epiglottic vallecula) 영상 분석 후보. 전원 동일 규칙, 환자별 예외 없음.

엑셀은 규칙에 쓰지 않는다. 좌표·result.json은 바꾸지 않는다. 전문가 확인 전 탐색용.
1) 정중면: N·S·Ba·ANS·PNS 최소제곱 평면(Po 미사용). 앞뒤 축 = PNS→ANS, 원점 = PNS.
2) 설골: 정중 ±3 mm 최대투영 > 400 (Dong 2019, 복셀 0.40 mm 최적값), 탐색창 가장자리에 닿지 않는 가장 큰 덩어리의 뒤쪽 1/3 최상단.
3) 후두개곡: 정중 ±6 mm 최소투영 공기(< -400, Roh PAV 상한)에서 후두개로 갈린 앞쪽 줄기의 최하점(method=vallecula). 탐색 높이 = 설골 몸통 아래 끝 ~ 기준점 +22 mm.
4) 후두개곡 공기가 없으면: vallecula 환자의 (Eb−설골) 중앙값을 더한 추정(method=hyoid_offset). 높이·거리 근사일 뿐이다.
5) 설골을 못 찾거나 Eb가 영상 밖이면 미계측(method=none). 기도 공기가 영상 아래 끝까지 이어지면 reason=FOV_bottom.
출력: Result_py/CBCT/patient_pipeline/v2/eb_review/v2/
"""
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label, map_coordinates

from cbct_patient_pipeline import CB, OUT, digest, landmarks, save
from check_landmarks import read_nrrd

DEST = OUT/"eb_review/v2"
STEP, AIR, BONE, MIN_ROWS = .4, -400, 400, 2    # MIN_ROWS 2: 민감도 확인(엑셀 PAL)으로 3에서 낮춤. BONE: Dong 2019 BMC Oral Health, 복셀 0.40 mm 최적 최소 기준(우리 0.39 mm). 이전 450은 근거 기록 없음
GAP = (0.8, 10.0)              # 후두개 두께로 인정할 앞뒤 간격 mm. 하한 0.8(영상 2칸): 민감도 확인(엑셀 PAL)으로 1.0에서 낮춤. 상한은 실제로 거르는 것이 없음
ANT, SUP = (-45, 25), (-95, -35)   # PNS 기준 영상 범위 mm. 뒤쪽 끝 −45: 민감도 확인(엑셀 PAL)·육안 판정으로 −35에서 넓힘(기도 뒤쪽이 잘리던 환자)
HYOID_ANT = (-33, 12)
MIDLINE = ["N", "S", "Ba", "ANS", "PNS"]


def midline_frame(P):
    """(원점 PNS, 3x3 행 = 좌·앞·위, 평면 잔차 최대 mm). Po를 쓰지 않는다."""
    pts = np.array([P[k] for k in MIDLINE]); c = pts.mean(0)
    left = np.linalg.svd(pts-c)[2][-1]; left = -left if left[0] < 0 else left
    d = P["ANS"]-P["PNS"]; ant = d-(d@left)*left; ant /= np.linalg.norm(ant)
    sup = np.cross(ant, left); sup = -sup if sup@(P["N"]-P["PNS"]) < 0 else sup
    return P["PNS"], np.vstack([left, ant, sup]), float(np.abs((pts-c)@left).max())


def runs(row):
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)-1))


def hyoid(mx, ant, sup):
    """(면적, 뒤쪽 1/3 앞뒤 평균, 그 최상단, H 앞뒤, H 위아래, 몸통 아래 끝 높이) 또는 None.
    뒤쪽 1/3 최상단은 Eb 탐색의 내부 기준(설골 덩어리 고정), H(가장 앞쪽·위쪽 점, 두부계측 문헌 정의)는 설골 피처용."""
    win = (ant > HYOID_ANT[0]) & (ant < HYOID_ANT[1]); lo, hi = np.flatnonzero(win)[[0, -1]]
    lab, n = label((mx > BONE) & win[:, None]); best = None
    for i in range(1, n+1):
        xs, zs = np.nonzero(lab == i); area = xs.size*STEP**2
        if xs.min() <= lo or xs.max() >= hi or zs.min() == 0 or zs.max() == len(sup)-1: continue
        if 8 <= area <= 150 and (xs.max()-xs.min())*STEP < 18 and (best is None or area > best[0]):
            post = ant[xs] <= np.percentile(ant[xs], 33)
            h = np.argmax(ant[xs]+sup[zs])    # 문헌 설골점 H: 몸통의 가장 앞쪽·위쪽 점(앞·위 45° 방향 최대)
            best = (area, float(ant[xs][post].mean()), float(sup[zs][post].max()), float(ant[xs][h]), float(sup[zs][h]), float(sup[zs].min()))
    return best


def vallecula(air, ant, sup, hy, c_lo=-18.0, c_hi=2.0, run_back=25.0):
    """(앞뒤, 높이, 검출 행 수) 또는 None. sup 오름차순이므로 첫 검출이 최하점.
    c_lo·c_hi: 앞 갈래 중심의 설골 기준 앞뒤 범위, run_back: 설골 기준 이보다 뒤에서 끝나는 공기 줄은 무시(민감도 확인용 인자)."""
    lab, n = label(air)
    if not n: return None
    main = np.bincount(lab.ravel())[1:].argmax()+1; hits = []
    for k, z in enumerate(sup):
        if not hy[5] <= z <= hy[2]+22: continue    # 아래 한계 = 설골 몸통 아래 끝(후두개곡 바닥인 설골–후두개 인대가 몸통에 붙음). 이전: 기준점 −3 mm
        rs = [r for r in runs(lab[:, k] == main) if ant[r[1]] >= hy[1]-run_back]
        if len(rs) < 2: continue
        back, front = rs[-2], rs[-1]; c = ant[(front[0]+front[1])//2]
        if GAP[0] <= (front[0]-back[1]-1)*STEP <= GAP[1] and hy[1]+c_lo <= c <= hy[1]+c_hi: hits.append((float(c), float(z)))
    return (*hits[0], len(hits)) if hits else None


def air_reaches_bottom(air, mip, ant):
    """정중 기도 범위(PNS −35~+5 mm)의 영상 최하단 행에 공기가 있으면 True."""
    for i in np.flatnonzero((ant > -35) & (ant < 5)):
        ok = np.flatnonzero(np.isfinite(mip[i]))
        if ok.size and air[i, ok[0]]: return True
    return False


def analyse(pid):
    P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
    o, R, resid = midline_frame(P)
    ant, sup, Ls = np.arange(*ANT, STEP), np.arange(*SUP, STEP), np.arange(-6, 6+1e-9, STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-3000).reshape(len(Ls), len(ant), len(sup))
    vol[vol < -2500] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mip, mx = np.nanmin(vol, 0), np.nanmax(vol[np.abs(Ls) <= 3], 0)
    air = mip < AIR
    r = {"Patient": f"P_{pid}", "plane_residual_mm": resid, "method": "none", "reason": ""}
    hy = hyoid(np.nan_to_num(mx, nan=-3000), ant, sup)
    if hy:
        r |= {"hy_ant": hy[1], "hy_sup": hy[2], "H_ant": hy[3], "H_sup": hy[4]}; val = vallecula(air, ant, sup, hy)
        if val and val[2] >= MIN_ROWS: r |= {"Eb_ant": val[0], "Eb_sup": val[1], "n_rows": val[2], "method": "vallecula"}
    else:
        r["reason"] = "FOV_bottom" if air_reaches_bottom(air, mip, ant) else "hyoid_not_found"
    return r, (mip, ant, sup), (o, R), (v.shape, dirs, org)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(OUT/"cohort.csv", dtype=str, keep_default_na=False)
    ids = [int(p[2:]) for p in cohort.loc[cohort.Include.str.lower().isin(["true", "1"]), "Patient"]]
    frozen = set(pd.read_csv(OUT/"fh_repair/v2/comparison.csv", encoding="utf-8-sig").Patient)    # 동결 54명만(나머지는 cbct_restore_extend가 고정 보정량으로 처리)
    ids = [i for i in ids if f"P_{i}" in frozen]
    rows, imgs, frames = [], {}, {}
    for pid in ids:
        print(pid, flush=True); r, img, fr, geom = analyse(pid); rows.append(r); imgs[pid] = img; frames[pid] = (fr, geom)
    d = pd.DataFrame(rows); o_ = d[d.method == "vallecula"]
    off = float((o_.Eb_ant-o_.hy_ant).median()), float((o_.Eb_sup-o_.hy_sup).median())
    m = (d.method == "none") & d.hy_ant.notna()
    d.loc[m, ["Eb_ant", "Eb_sup"]] = np.c_[d.hy_ant[m]+off[0], d.hy_sup[m]+off[1]]; d.loc[m, "method"] = "hyoid_offset"
    for i, r in d[d.Eb_ant.notna()].iterrows():    # 좌표 저장, 영상 밖 후보는 미계측
        pid = int(r.Patient[2:]); (o, R), (shape, dirs, org) = frames[pid]
        q = o+np.array([0, r.Eb_ant, r.Eb_sup])@R; ijk = np.linalg.solve(dirs.T, q-org)
        if np.any(ijk < 0) or np.any(ijk > np.array(shape[::-1])-1):
            d.loc[i, ["method", "reason"]] = ["none", "Eb_outside_image"]; d.loc[i, ["Eb_ant", "Eb_sup"]] = np.nan; continue
        d.loc[i, ["Eb_x", "Eb_y", "Eb_z"]] = q
    d["PAL_mm"] = np.hypot(d.Eb_ant, d.Eb_sup)
    d.round(2).to_csv(DEST/"eb_candidates.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(int(np.ceil(len(ids)/9)), 9, figsize=(36, 5*np.ceil(len(ids)/9)), squeeze=False)
    for a in ax.flat[len(ids):]: a.axis("off")
    for a, (_, r), pid in zip(ax.flat, d.iterrows(), ids):
        mip, ant, sup = imgs[pid]
        a.imshow(mip.T, origin="lower", cmap="gray", vmin=-1000, vmax=500, extent=[ant[0], ant[-1], sup[0], sup[-1]])
        if pd.notna(r.get("hy_ant")): a.plot(r.hy_ant, r.hy_sup, "y+", ms=16, mew=2)
        if pd.notna(r.get("Eb_ant")): a.plot(r.Eb_ant, r.Eb_sup, "g+" if r.method == "vallecula" else "c+", ms=22, mew=3)
        a.set_title(f"{r.Patient} {r.method} {r.reason}", fontsize=17); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(DEST/"eb_all_patients.png", dpi=45); plt.close(fig)
    summary = {"counts": d.method.value_counts().to_dict(), "reasons": d[d.method == "none"].set_index("Patient").reason.to_dict(),
               "offset_mm": {"ant": round(off[0], 2), "sup": round(off[1], 2)}, "definition": "Yao 2023 Diagnostics 13, Table: Most inferior point of epiglottic vallecula",
               "status": "exploratory; 전문가 확인 전; result.json 미적용", "cohort_sha256": digest(OUT/"cohort.csv"), "script_sha256": digest(Path(__file__))}
    save(DEST/"summary.json", summary); print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
