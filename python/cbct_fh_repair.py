"""FH 점(Po·Or) 이상값 대체. 전 환자 동일 규칙, 엑셀 미사용.

정중 좌표계(Po 미사용, 원점 PNS)에서 좌우 점의 (|좌우|, 앞, 위)가 코호트 중앙값에서 Z_CUT MAD를 넘으면 이상값으로 보고,
반대쪽 점을 정중면에 대칭시킨 좌표로 바꾼다. 양쪽 모두 이상이면 바꾸지 않고 표시만 한다.
기준 분포: Result_py/CBCT/patient_pipeline/v2/fh_repair/v1/reference.json (Include 환자 ALI 좌표로 한 번 고정)
"""
import json
import sys

import numpy as np
import pandas as pd

from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import OUT, landmarks

DEST = OUT/"fh_repair/v1"
PAIRS = (("RPo", "LPo"), ("ROr", "LOr"))
Z_CUT = 5.


def local(P, k):
    o, R, _ = midline_frame(P); q = R@(np.asarray(P[k], float)-o)
    return np.array([abs(q[0]), q[1], q[2]]), o, R


def build_reference(ids):
    rows = {k: np.array([local(landmarks(p)[0], k)[0] for p in ids]) for pair in PAIRS for k in pair}
    ref = {k: {"median": np.median(v, 0).tolist(), "mad": (1.4826*np.median(np.abs(v-np.median(v, 0)), 0)).tolist()} for k, v in rows.items()}
    DEST.mkdir(parents=True, exist_ok=True); (DEST/"reference.json").write_text(json.dumps({"n": len(ids), "z_cut": Z_CUT, "points": ref}, indent=1), encoding="utf-8")


def repair(P, ref=None):
    """(고친 좌표 dict, [(점, z, 대체 여부)])."""
    ref = ref or json.loads((DEST/"reference.json").read_text(encoding="utf-8"))["points"]; out = dict(P); log = []
    for a, b in PAIRS:
        z = {k: float(np.max(np.abs((local(P, k)[0]-ref[k]["median"])/ref[k]["mad"]))) for k in (a, b)}
        bad = [k for k in (a, b) if z[k] > Z_CUT]
        if len(bad) == 1:
            k = bad[0]; good = b if k == a else a; q, o, R = local(P, good)
            mirrored = R@(np.asarray(P[good], float)-o); mirrored[0] *= -1
            out[k] = o+mirrored@R; log.append((k, round(z[k], 1), "mirrored"))
        elif bad: log += [(k, round(z[k], 1), "both_outlying_not_changed") for k in bad]
    return out, log


def figure(ids):
    """원래 점(빨강)과 대칭 대체점(하늘)을 각 위치의 시상·관상면에 표시."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cbct_landmark_review import panel
    from cbct_patient_pipeline import CB
    from check_landmarks import read_nrrd
    fig, ax = plt.subplots(len(ids), 4, figsize=(24, 6*len(ids)), squeeze=False)
    for i, pid in enumerate(ids):
        P = landmarks(pid)[0]; Q, log = repair(P); v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd")
        for k, _, _ in log:
            other = {"RPo": "LPo", "LPo": "RPo", "ROr": "LOr", "LOr": "ROr"}[k]
            marks = {k+" ALI": (P[k], "red", "+"), k+" mirrored": (Q[k], "cyan", "x"), other: (P[other], "yellow", "+")}
            for j, (c, plane) in enumerate(((P[k], "sagittal"), (P[k], "coronal"), (Q[k], "sagittal"), (Q[k], "coronal"))):
                panel(ax[i, j], v, dirs, org, np.asarray(c, float), plane, marks, half=35); ax[i, j].set_title(f"P_{pid} {k}: {plane} at {'ALI' if j < 2 else 'mirrored'} point")
    fig.tight_layout(); fig.savefig(OUT/"evidence_review/v1/figs/fh_po_mirror.png", dpi=60); plt.close(fig)


if __name__ == "__main__":
    ids = [int(p[2:]) for p, i in pd.read_csv(OUT/"cohort.csv", dtype=str, encoding="utf-8-sig")[["Patient", "Include"]].values if i.lower() == "true"]
    if sys.argv[1:2] == ["--build"]: build_reference(ids)
    if sys.argv[1:2] == ["--fig"]: figure([int(x) for x in sys.argv[2:]]); sys.exit()
    for pid in ids:
        _, log = repair(landmarks(pid)[0])
        if log: print(f"P_{pid}", log)
