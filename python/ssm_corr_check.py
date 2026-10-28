"""전달 오차가 큰 사례의 대응점 점검 그림: 환자 대응점(회색), 템플릿 랜드마크에 해당하는 대응점(빨강), ALI 랜드마크(파랑 X).
빨강이 해부학적으로 맞는 자리에 있고 파랑이 벗어났으면 ALI 오류, 반대면 대응점 미끄러짐. 판정은 사람이 함.
각 랜드마크마다 전달 오차가 중앙값에 가까운 환자 1명을 비교용으로 같이 그림.
결과: Result_py/shape/<구조>/review/corr_check.png
사용: python ssm_corr_check.py --struct MAND
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from check_landmarks import read_points
from shape_ssm import TE_LMS, subsample, surface

ROOT = Path(__file__).resolve().parents[1]
CASES = {"MAND": [("P_7", "Me"), ("P_7", "Pog"), ("P_20", "B"), ("P_20", "RGo"), ("P_53", "B"), ("P_77", "B")],
         "CB": [("P_7", "S"), ("P_12", "S"), ("P_18", "S"), ("P_42", "S"), ("P_69", "S"), ("P_56", "N"), ("P_56", "S")]}
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="MAND")
    ap.add_argument("--half", type=float, default=25.0, help="그림 범위: 랜드마크 중심 ±mm")
    ap.add_argument("--slab", type=float, default=4.0, help="단면 두께: 화면에 수직인 방향 ±mm")
    a = ap.parse_args()
    src = ROOT / "Result_py/shape" / a.struct
    z = np.load(src / "points.npz")
    ids = [f"P_{p}" for p in z["ids"]]
    q = pd.read_csv(src / "review/qc_review.csv", index_col=0, encoding="utf-8-sig")
    tpl = int(z["template_id"])
    s, P = surface(tpl, a.struct)
    tpl_lps = subsample(s, z["frame"].shape[1])
    near = {l: int(np.argmin(np.linalg.norm(tpl_lps - P[l], axis=1))) for l in TE_LMS[a.struct]}

    rows = []
    for pid, lm in CASES[a.struct]:
        ref = (q[f"te_{lm}"] - q[f"te_{lm}"].median()).abs().idxmin()
        rows += [(pid, lm, "검수"), (ref, lm, "비교(중앙값)")]
    fig, ax = plt.subplots(len(rows), 2, figsize=(8, 3.6 * len(rows)))
    for r, (pid, lm, tag) in enumerate(rows):
        i = ids.index(pid)
        o, R = z["head_origin"][i], z["head_R"][i]
        pts = z["frame"][i]
        L = (read_points(int(pid[2:])).loc[lm, ["x", "y", "z"]].to_numpy(float) - o) @ R.T
        c = pts[near[lm]]
        te = np.linalg.norm(c - L)
        assert abs(te - q.loc[pid, f"te_{lm}"]) < 0.01, (pid, lm, te)   # qc.csv 값과 같은지
        box = np.all(np.abs(pts - L) < a.half, axis=1)
        for j, (h, v, d, name) in enumerate(((1, 2, 0, "옆 (가로 = 앞)"), (0, 2, 1, "앞 (가로 = 좌)"))):
            sl = box & (np.abs(pts[:, d] - L[d]) < a.slab)          # 랜드마크를 지나는 얇은 판만: 안쪽·바깥 면이 겹치지 않게
            ax[r, j].scatter(pts[sl, h], pts[sl, v], s=4, c="0.5")
            ax[r, j].scatter(c[h], c[v], s=60, c="red", label="대응점")
            ax[r, j].scatter(L[h], L[v], s=80, c="blue", marker="x", label="ALI")
            ax[r, j].set_aspect("equal")
            ax[r, j].set_title(f"{pid} {lm} [{tag}] 전달 오차 {te:.1f} mm · {name}", fontsize=9)
        ax[r, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{a.struct} 대응점 점검 (머리 좌표계, 랜드마크 ±{a.half:.0f} mm, 단면 ±{a.slab:.0f} mm). Po 실패 환자는 좌표계가 기울어 있을 수 있음", fontsize=10)
    fig.tight_layout()
    fig.savefig(src / "review/corr_check.png", dpi=90)
    print(f"저장: {src / 'review/corr_check.png'} ({len(rows)}행)")


if __name__ == "__main__":
    main()
