"""거리 QC 통과 환자 중 엑셀 차이가 큰 사례 점검. 좌표·계측값은 바꾸지 않는다.

출력: Result_py/CBCT/patient_pipeline/v2/qcpass_review/v1/
  landmark_checks.csv  환자별 4지표 자동−엑셀 차이, FH 정의별(jung/lor/yao4) 차이,
                       Nperp 차이를 각도로 환산한 값(tilt_A·tilt_Pog), 좌우 Or 높이차, 점→고밀도 골 거리
  or_views.png / n_po_views.png / p32_anb.png  상위 사례 단면
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cbct_chin_review import bone_distance
from cbct_landmark_review import panel
from cbct_patient_pipeline import CB, OUT, PAIRS, digest, landmarks, save
from check_landmarks import read_nrrd
from osa.ceph import measures, to_head
from pilot_measure import excel

DEST = OUT/"qcpass_review/v1"
POINTS = ["N", "A", "B", "RPo", "LPo", "ROr", "LOr", "RGo", "LGo"]


def views(name, cases, half=20):
    fig, ax = plt.subplots(2, len(cases), figsize=(5*len(cases), 10), squeeze=False)
    for j, (pid, keys, center, planes) in enumerate(cases):
        p = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd")
        c = np.mean([p[k] for k in center], axis=0)
        marks = {k: (p[k], col, "+") for k, col in zip(keys, ["red", "cyan", "lime", "orange"])}
        for i, plane in enumerate(planes):
            panel(ax[i, j], v, dirs, org, c, plane, marks, half=half)
            ax[i, j].set_title(f"P_{pid} {'/'.join(keys)} {plane}")
    fig.tight_layout(); fig.savefig(DEST/name, dpi=90); plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(OUT/"cohort.csv", dtype=str, keep_default_na=False)
    ids = [int(p[2:]) for p in cohort.loc[cohort.Include.str.lower().isin(["true", "1"]), "Patient"]]
    status = {f"P_{pid}": json.loads((OUT/f"P_{pid}/result.json").read_text(encoding="utf-8"))
              ["variants"]["automatic"]["release_status"] for pid in ids}
    xl, rows = excel(), []
    for pid in ids:
        p = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); h = to_head(p)
        r = {"Patient": f"P_{pid}", "QC": status[f"P_{pid}"]}
        for fh in ("jung", "lor", "yao4"):
            m = measures(p, fh)
            r |= {f"{k}_diff_{fh}": m[k]-float(xl.loc[pid][c]) for k, c in PAIRS.items() if fh == "jung" or k != "ANB"}
        # Nperp 차이를 N 아래 높이로 나눠 각도로 환산: FH 방향 차이라면 MP-FH 차이와 반대 부호로 함께 움직인다
        r["tilt_A_deg"] = np.degrees(np.arctan2(r["Nperp_A_diff_jung"], -h["A"][2]))
        r["tilt_Pog_deg"] = np.degrees(np.arctan2(r["Nperp_Pog_diff_jung"], -h["Pog"][2]))
        r["ROr_minus_LOr_sup_mm"] = h["ROr"][2]-h["LOr"][2]
        r |= {f"{k}_to_bone_mm": bone_distance(v, dirs, org, p[k]) for k in POINTS}
        rows.append(r)
    d = pd.DataFrame(rows).round(2); d.to_csv(DEST/"landmark_checks.csv", index=False, encoding="utf-8-sig")
    q = d[d.QC == "qc_pass_pending_visual_review"]
    summary = {"n_qc_pass": len(q),
               "corr_MPFH_vs_tilt_Pog": round(q.MPFH_GoMe_diff_jung.corr(q.tilt_Pog_deg), 2),
               "corr_MPFH_vs_tilt_A": round(q.MPFH_GoMe_diff_jung.corr(q.tilt_A_deg), 2),
               "median_abs_diff_by_fh": {fh: {k: round(q[f"{k}_diff_{fh}"].abs().median(), 2) for k in PAIRS if k != "ANB"}
                                         for fh in ("jung", "lor", "yao4")},
               "max_bone_distance_mm": {k: float(q[f"{k}_to_bone_mm"].max()) for k in POINTS}}
    views("or_views.png", [(pid, ["ROr", "LOr"], ["ROr", "LOr"], ["coronal", "sagittal"]) for pid in (86, 4, 5, 81, 52, 83)], half=40)
    views("n_po_views.png", [(4, ["RPo"], ["RPo"], ["sagittal", "coronal"]), (86, ["RPo"], ["RPo"], ["sagittal", "coronal"]),
                             (62, ["N"], ["N"], ["sagittal", "axial"]), (74, ["N"], ["N"], ["sagittal", "axial"]),
                             (83, ["N"], ["N"], ["sagittal", "axial"])])
    views("p32_anb.png", [(32, ["A", "B", "N"], ["A", "B"], ["sagittal", "coronal"])], half=30)
    save(DEST/"summary.json", {**summary, "cohort_sha256": digest(OUT/"cohort.csv"), "script_sha256": digest(Path(__file__)),
                               "note": "탐색적 점검. 엑셀 원 수작업 좌표 없음. 좌표·계측 변경 없음"})
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
