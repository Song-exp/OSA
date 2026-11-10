"""Po 상하 위치 점검: Po 바로 아래 외이도(어두운 구간) 윗경계에서 Po까지의 높이를 54명 전원에서 잰다.
좌표·계측값·result.json은 바꾸지 않는다. 엑셀 값은 후보 선택에 쓰지 않는다.

높이 = Po 중심 3.5 mm 정사각 기둥의 복셀 중앙값이 문턱(150) 아래인 가장 위 복셀부터 Po까지의 LPS z 거리.
후보 = 같은 쪽 전원 중앙값에서 2 mm 넘게 높은 Po를 (외이도 윗경계 + 그 중앙값) 높이로 내림. x·y는 유지.
출력: Result_py/CBCT/patient_pipeline/v2/po_height_review/v1/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cbct_landmark_review import panel
from cbct_patient_pipeline import CB, OUT, PAIRS, digest, landmarks, qc, save
from check_landmarks import read_nrrd
from osa.ceph import measures
from pilot_measure import excel

DEST = OUT/"po_height_review/v1"
DARK, HALF, BELOW, MARGIN = 150, 4, 15.0, 2.0


def canal_top(v, org, sp, P):
    """(Po에서 외이도 윗경계까지 높이 mm, 윗경계 z). 어두운 구간이 없으면 (nan, nan)."""
    i, j, k = np.rint((P-org)/sp).astype(int); n = int(BELOW/sp[2])
    prof = np.array([np.median(v[kk, j-HALF:j+HALF+1, i-HALF:i+HALF+1]) for kk in range(k-n, k+1)])
    dark = np.flatnonzero(prof < DARK)
    if not dark.size: return np.nan, np.nan
    return (n-dark.max())*sp[2], P[2]-(n-dark.max())*sp[2]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(OUT/"cohort.csv", dtype=str, keep_default_na=False)
    ids = [int(p[2:]) for p in cohort.loc[cohort.Include.str.lower().isin(["true", "1"]), "Patient"]]
    data, rows = {}, []
    for pid in ids:
        p = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); sp = np.diag(dirs)
        data[pid] = p
        for side in ("RPo", "LPo"):
            gap, top = canal_top(v, org, sp, p[side])
            rows.append({"Patient": f"P_{pid}", "Side": side, "gap_mm": gap, "canal_top_z": top, "Po_z": p[side][2]})
    d = pd.DataFrame(rows); med = d.groupby("Side").gap_mm.median()
    d["excess_mm"] = d.gap_mm-d.Side.map(med)
    d.round(2).to_csv(DEST/"po_canal_gap.csv", index=False, encoding="utf-8-sig")
    flagged = d[d.excess_mm > MARGIN]
    profile = json.loads((OUT/"qc_reference.json").read_text(encoding="utf-8")); xl = excel()
    comp, ev = [], {}
    for pid in sorted({int(x[2:]) for x in flagged.Patient}):
        p = data[pid]; q = dict(p); moved = {}
        for _, r in flagged[flagged.Patient == f"P_{pid}"].iterrows():
            q[r.Side] = p[r.Side].copy(); q[r.Side][2] = r.canal_top_z+med[r.Side]
            moved[r.Side] = {"from": p[r.Side].round(3).tolist(), "to": q[r.Side].round(3).tolist(), "down_mm": float(r.excess_mm)}
        before, after = measures(p), measures(q)
        ev[f"P_{pid}"] = {"moved": moved, "automatic_qc": qc(p, profile), "candidate_qc": qc(q, profile)}
        for k, col in PAIRS.items():
            e = float(xl.loc[pid][col])
            comp.append({"Patient": f"P_{pid}", "Moved": "+".join(moved), "Variable": k, "Automatic": before[k],
                         "Candidate": after[k], "Excel": e, "Abs_auto_minus_excel": abs(before[k]-e),
                         "Abs_cand_minus_excel": abs(after[k]-e)})
    comp = pd.DataFrame(comp).round(2); comp.to_csv(DEST/"candidate_comparison.csv", index=False, encoding="utf-8-sig")
    cases = [(int(r.Patient[2:]), r.Side) for _, r in flagged.iterrows()]+[(33, "LPo")]   # 마지막 = 중앙값 근처 참고
    fig, ax = plt.subplots(2, len(cases), figsize=(5.5*len(cases), 11), squeeze=False)
    for j, (pid, side) in enumerate(cases):
        p = data[pid]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); r = d[(d.Patient == f"P_{pid}") & (d.Side == side)].iloc[0]
        cand = p[side].copy(); cand[2] = r.canal_top_z+med[side]
        marks = {"ALI": (p[side], "red", "+"), "candidate": (cand, "lime", "x")}
        for i, plane in enumerate(["sagittal", "coronal"]):
            panel(ax[i, j], v, dirs, org, p[side], plane, marks, half=16)
            ax[i, j].axhline(r.canal_top_z, color="yellow", lw=0.8, ls="--")
        ax[0, j].set_title(f"P_{pid} {side}: gap {r.gap_mm:.1f} mm" + (" (reference)" if pid == 33 else ""))
    fig.tight_layout(); fig.savefig(DEST/"po_height_views.png", dpi=90); plt.close(fig)
    save(DEST/"evidence.json", {"rule": __doc__, "median_gap_mm": med.round(2).to_dict(), "margin_mm": MARGIN,
         "patients": ev, "status": "candidate_requires_expert_review; result.json 미적용",
         "cohort_sha256": digest(OUT/"cohort.csv"), "script_sha256": digest(Path(__file__))})
    print(med.round(2).to_dict()); print(flagged.round(2).to_string(index=False)); print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
