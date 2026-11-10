"""Pog–Me 거리 QC 표시(P_7·P_12·P_27) 원본 영상 검토 자료. 좌표·계측값은 바꾸지 않는다.

출력: Result_py/CBCT/patient_pipeline/v2/chin_review/v1/
  flagged_vs_reference.png  표시 3명 + 참고 1명의 턱끝 시상·관상 단면 (B·Pog·Gn·Me)
  all_patients_sagittal.png 현재 Include 전원의 턱끝 시상 단면 (Pog·Me)
  bone_distance.csv         각 점에서 가장 가까운 고밀도 골 복셀(>=1000)까지 거리(mm)
골 거리는 밝기 문턱 하나에 의존하는 탐색 지표이며 정답 판정 기준이 아니다.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from cbct_patient_pipeline import CB, OUT, digest, landmarks, save
from cbct_landmark_review import panel
from check_landmarks import read_nrrd

FLAGGED, REFERENCE, BONE = [7, 12, 27], 33, 1000  # 참고: Include 환자 중 QC 미표시, 단면이 깨끗한 예
DEST = OUT/"chin_review/v1"


def bone_distance(v, dirs, org, point, radius=15):
    sp = np.diag(dirs); ijk = np.rint((point-org)/sp).astype(int); h = np.rint(radius/sp).astype(int)
    lo = np.maximum(ijk-h, 0); hi = np.minimum(ijk+h+1, np.array(v.shape[::-1]))
    bone = v[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] >= BONE   # 배열 순서 z,y,x
    if not bone.any(): return np.nan
    return float(ndimage.distance_transform_edt(~bone, sampling=sp[::-1])[tuple((ijk-lo)[::-1])])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(OUT/"cohort.csv", dtype=str, keep_default_na=False)
    ids = [int(p[2:]) for p in cohort.loc[cohort.Include.str.lower().isin(["true", "1"]), "Patient"]]
    data = {pid: (landmarks(pid)[0], *read_nrrd(CB/f"nrrd/P_{pid}.nrrd")) for pid in ids}

    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    for j, pid in enumerate(FLAGGED+[REFERENCE]):
        p, v, dirs, org = data[pid]
        marks = {"B": (p["B"], "yellow", "+"), "Pog": (p["Pog"], "red", "+"),
                 "Gn": (p["Gn"], "orange", "x"), "Me": (p["Me"], "cyan", "+")}
        center = (p["Pog"]+p["Me"])/2
        for i, view in enumerate(["sagittal", "coronal"]):
            panel(ax[i, j], v, dirs, org, center, view, marks, half=22)
        ax[0, j].set_title(f"P_{pid} {'QC flagged' if pid in FLAGGED else 'reference (not flagged)'}")
    fig.tight_layout(); fig.savefig(DEST/"flagged_vs_reference.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(int(np.ceil(len(ids)/9)), 9, figsize=(36, 4*np.ceil(len(ids)/9)), squeeze=False)
    for a in ax.flat[len(ids):]: a.axis("off")
    rows = []
    for a, pid in zip(ax.flat, ids):
        p, v, dirs, org = data[pid]
        panel(a, v, dirs, org, (p["Pog"]+p["Me"])/2, "sagittal",
              {"Pog": (p["Pog"], "red", "+"), "Me": (p["Me"], "cyan", "+")}, half=20)
        a.set_title(f"P_{pid}"+(" (QC flagged)" if pid in FLAGGED else ""), fontsize=18)
        a.get_legend().remove(); a.set_xlabel(""); a.set_ylabel("")
        rows.append({"Patient": f"P_{pid}", "PogMe_mm": float(np.linalg.norm(p["Pog"]-p["Me"])),
                     **{f"{k}_to_bone_mm": bone_distance(v, dirs, org, p[k]) for k in ["B", "Pog", "Gn", "Me"]}})
    fig.tight_layout(); fig.savefig(DEST/"all_patients_sagittal.png", dpi=60); plt.close(fig)
    pd.DataFrame(rows).round(2).to_csv(DEST/"bone_distance.csv", index=False, encoding="utf-8-sig")
    save(DEST/"provenance.json", {"patients": [f"P_{p}" for p in ids], "flagged": FLAGGED, "reference": REFERENCE,
                                   "bone_threshold": BONE, "cohort_sha256": digest(OUT/"cohort.csv"),
                                   "script_sha256": digest(__import__("pathlib").Path(__file__)),
                                   "note": "자동 좌표 검토 자료. 후보 좌표·계측값 변경 없음"})
    print(json.dumps({"output": str(DEST), "patients": len(ids)}))


def chin_candidates(P, mand, dirs, org):
    """하악 마스크 정중 슬랩(±1 mm)에서 Pog·Me 후보. 골 위에 있는 B·Gn을 탐색 범위 기준으로 쓴다(엑셀 미사용).
    Pog: Gn~B 높이의 아래 절반에서 앞 윤곽이 가장 앞인 점 / Me: Gn 앞 3 mm~뒤 10 mm에서 가장 낮은 점."""
    from osa.ceph import head_frame
    from osa.refine import _sample, STEP
    o, R = head_frame(P); b, g = (R@(P[k]-o) for k in ("B", "Gn"))
    sup = np.arange(g[2], g[2]+0.5*(b[2]-g[2])+1e-9, STEP)
    ant = np.arange(min(b[1], g[1])-10, max(b[1], g[1])+20, STEP)
    m = _sample(mand, dirs, org, o, R, ant, sup)
    front = np.where(m.any(0), ant[::-1][np.argmax(m[::-1], axis=0)], -np.inf)
    t = np.flatnonzero(front >= front.max()-STEP/2); i = t[len(t)//2]
    ant2, sup2 = np.arange(g[1]-10, g[1]+3, STEP), np.arange(g[2]-15, g[2]+5, STEP)
    low = np.where((m2 := _sample(mand, dirs, org, o, R, ant2, sup2)).any(1), sup2[np.argmax(m2, axis=1)], np.inf)
    t = np.flatnonzero(low <= low.min()+STEP/2); j = t[len(t)//2]
    if not (np.isfinite(front[i]) and np.isfinite(low[j])): raise ValueError("턱끝 마스크 윤곽 없음")
    return {"Pog": o+np.array([0, front[i], sup[i]])@R, "Me": o+np.array([0, ant2[j], low[j]])@R}, (o, R)


def recompute(pids=(7, 12, 27, 55), threshold_mm=2.0):
    """후보와 ALI의 앞뒤·상하 차이가 비표시 환자 분포를 넘는 점만 바꿔 4지표 재계산. result.json은 건드리지 않는다."""
    from osa.ceph import measures
    from pilot_candidates import load
    from pilot_measure import excel
    from cbct_patient_pipeline import PAIRS, qc
    cohort = pd.read_csv(OUT/"cohort.csv", dtype=str, keep_default_na=False)
    ids = [int(p[2:]) for p in cohort.loc[cohort.Include.str.lower().isin(["true", "1"]), "Patient"]]
    offs, cand = [], {}
    for pid in ids:   # 같은 규칙을 전원에 적용해 ALI와의 평소 차이(체계적 편향)를 먼저 잰다
        P = landmarks(pid)[0]; ct, mand, _, dirs, org = load(pid)
        try: c, (o, R) = chin_candidates(P, mand, dirs, org)
        except ValueError: continue
        cand[pid] = (P, c)
        offs += [{"Patient": f"P_{pid}", "Point": k, "d_ant": (R@(c[k]-P[k]))[1], "d_sup": (R@(c[k]-P[k]))[2]} for k in c]
    offs = pd.DataFrame(offs); base = offs[~offs.Patient.isin([f"P_{p}" for p in pids])]
    med = base.groupby("Point")[["d_ant", "d_sup"]].median()
    profile = json.loads((OUT/"qc_reference.json").read_text(encoding="utf-8")); xl = excel()
    rows, flags = [], {}
    fig, ax = plt.subplots(1, len(pids), figsize=(6*len(pids), 6), squeeze=False)
    for a, pid in zip(ax[0], pids):   # 빨강·하늘 = ALI, 초록·노랑 x = 후보(교체 여부와 무관하게 표시)
        P, c = cand[pid]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd")
        panel(a, v, dirs, org, (c["Pog"]+c["Me"])/2, "sagittal",
              {"ALI Pog": (P["Pog"], "red", "+"), "ALI Me": (P["Me"], "cyan", "+"),
               "cand Pog": (c["Pog"], "lime", "x"), "cand Me": (c["Me"], "yellow", "x")}, half=20)
        a.set_title(f"P_{pid} candidates (not approved)")
    fig.tight_layout(); fig.savefig(DEST/"candidates_sagittal.png", dpi=110); plt.close(fig)
    for pid in pids:
        P, c = cand[pid]; o = offs[offs.Patient == f"P_{pid}"].set_index("Point")
        # 비표시 환자 중앙값에서 threshold 넘게 벗어난 점만 교체(편향만큼은 ALI 정의 차이로 보고 남긴다)
        moved = [k for k in c if np.linalg.norm((o.loc[k, ["d_ant", "d_sup"]]-med.loc[k]).to_numpy(float)) > threshold_mm]
        Q = {**P, **{k: c[k] for k in moved}}
        before, after = measures(P), measures(Q)
        flags[f"P_{pid}"] = {"moved": moved, "automatic_qc": qc(P, profile), "candidate_qc": qc(Q, profile),
                             "candidate_lps_mm": {k: c[k].round(3).tolist() for k in moved}}
        for k, col in PAIRS.items():
            e = float(xl.loc[pid][col])
            rows.append({"Patient": f"P_{pid}", "Moved": "+".join(moved) or "-", "Variable": k,
                         "Automatic": before[k], "Candidate": after[k], "Excel": e,
                         "Abs_auto_minus_excel": abs(before[k]-e), "Abs_cand_minus_excel": abs(after[k]-e)})
    out = pd.DataFrame(rows).round(2)
    out.to_csv(DEST/"candidate_comparison.csv", index=False, encoding="utf-8-sig")
    offs.round(2).to_csv(DEST/"candidate_offsets_all.csv", index=False, encoding="utf-8-sig")
    save(DEST/"candidate_evidence.json", {"rule": chin_candidates.__doc__, "threshold_mm": threshold_mm,
         "unflagged_median_offset_mm": med.round(2).to_dict(), "patients": flags,
         "status": "candidate_requires_expert_review; result.json 미적용",
         "selected_by": "mask-contour rule + assistant visual check; not a specialist annotation",
         "cohort_sha256": digest(OUT/"cohort.csv"), "script_sha256": digest(__import__("pathlib").Path(__file__))})
    print(med.round(2).to_string()); print(out.to_string(index=False)); print(json.dumps(flags, ensure_ascii=False, default=float)[:1500])


if __name__ == "__main__":
    import sys
    recompute() if "--candidates" in sys.argv else main()
