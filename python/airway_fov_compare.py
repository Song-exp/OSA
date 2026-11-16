"""기도 아래쪽이 영상 밖으로 잘린 4명(Eb 미계측)과 기도가 끝까지 보이는 대조 환자 비교 그림.

정중 시상 단면(Po 미사용 정중 좌표계, 원점 PNS)에 AMASSS 상기도 분할을 색으로 칠하고, PNS·Eb·설골·Me와 영상 범위 밖을 표시한다.
대조 환자: 후두개곡을 직접 찾았고(vallecula) 기도 길이(Eb–PNS)가 85명 중앙값에 가까운 기존 54명 중 3명.
출력: Result_py/shape/validation/airway_fov/ (airway_fov_compare.png, index.html)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import CB, OUT, landmarks
from cbct_restore_extend import eb_row
from check_landmarks import read_nrrd
from shape_visual_review import lab_sample
from shape_vector_validation import DEST as VAL, SH

DEST = VAL/"airway_fov"; CUT = ["P_1", "P_20", "P_50", "P_68"]; REF = ["P_38", "P_7", "P_16"]
AIR, OUTSIDE = "#eb6834", "#3a3a37"


def panel(ax, p, offset, soft):
    pid = int(p[2:]); P = landmarks(pid)[0]; o, R, _ = midline_frame(P); e = eb_row(pid, offset)
    v, dirs, org = read_nrrd(CB/f"nrrd/{p}.nrrd"); v = v.astype(np.float32)
    A, S = np.arange(-95, 45, .4), np.arange(-135, 30, .4); G = np.stack(np.meshgrid(A, S, indexing="ij"), -1).reshape(-1, 2)
    Q = o+np.c_[np.zeros(len(G)), G]@R
    ct = map_coordinates(v, np.linalg.solve(dirs.T, (Q-org).T)[::-1], order=1, cval=-5000).reshape(len(A), len(S)); del v
    air = lab_sample(CB/f"seg/{p}_seg_UAW.nrrd", Q).reshape(len(A), len(S)) > 0
    ext = [A[0], A[-1], S[0], S[-1]]
    ax.imshow(np.where(ct < -4000, np.nan, ct).T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=ext)
    ax.imshow(np.where(ct < -4000, 1, np.nan).T, origin="lower", cmap=ListedColormap([OUTSIDE]), extent=ext, alpha=.9)    # 영상 범위 밖
    ax.imshow(np.where(air, 1, np.nan).T, origin="lower", cmap=ListedColormap([AIR]), extent=ext, alpha=.6)
    me = R@(P["Me"]-o)
    marks = [("PNS", 0, 0, "w"), ("Me", me[1], me[2], "w")]
    if np.isfinite(e.get("Eb_sup", np.nan)): marks.append(("Eb", e.Eb_ant, e.Eb_sup, "#1baf7a"))
    if "hy_ant" in e and np.isfinite(e.get("hy_ant", np.nan)): marks.append(("설골", e.hy_ant, e.hy_sup, "#eda100"))
    for t, x, y, c in marks:
        ax.plot(x, y, "+", color=c, ms=13, mew=2.2); ax.annotate(t, (x, y), color=c, fontsize=10, fontweight="bold", xytext=(5, 4), textcoords="offset points")
    bottom = S[np.flatnonzero((ct > -4000).any(0))[0]] if (ct > -4000).any() else np.nan
    hp = soft.loc[p]
    title = f"{p} · 하인두 높이 {hp.hypopharynx_height:.0f} mm · " + ("Eb·설골 영상 밖" if p in CUT else f"Eb–PNS {-e.Eb_sup:.0f} mm")
    ax.set_title(title, fontsize=11, color="#0b0b0b"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#c8c7c2")
    return {"Patient": p, "group": "아래쪽 잘림" if p in CUT else "대조", "image_bottom_mm_from_PNS": float(bottom), "Me_sup": float(me[2]),
            "Eb_sup": float(e.get("Eb_sup", np.nan)), "hyoid_found": bool("hy_ant" in e and np.isfinite(e.get("hy_ant", np.nan))),
            "airway_ml_in_slice_view": float(air.sum()*.16/1000), "hypopharynx_height_mm": float(hp.hypopharynx_height)}


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; DEST.mkdir(parents=True, exist_ok=True)
    offset = json.loads((OUT/"eb_review/v2/summary.json").read_text(encoding="utf-8"))["offset_mm"]
    soft = pd.read_csv(SH/"fine/soft_tissue.csv", encoding="utf-8-sig").set_index("Patient")
    fig, ax = plt.subplots(2, 4, figsize=(19, 12.5)); rows = []
    for a, p in zip(ax[0], CUT): rows.append(panel(a, p, offset, soft)); print(p, end=" ", flush=True)
    for a, p in zip(ax[1], REF): rows.append(panel(a, p, offset, soft)); print(p, end=" ", flush=True)
    leg = ax[1, 3]; leg.axis("off")
    for i, (c, t) in enumerate(((AIR, "기도 (AMASSS 상기도 분할)"), (OUTSIDE, "영상 범위 밖 (촬영되지 않음)"), ("#1baf7a", "Eb 후두개곡 (기도 프로파일 아래 경계)"), ("#eda100", "설골"))):
        leg.add_patch(plt.Rectangle((.05, .78-i*.14), .1, .08, color=c, transform=leg.transAxes)); leg.text(.2, .82-i*.14, t, fontsize=12, va="center", transform=leg.transAxes)
    leg.text(.05, .14, "윗줄: Eb를 정할 수 없는 4명\n아랫줄: 기도가 Eb 아래까지 보이는 대조 3명\n(기도 길이가 85명 중앙값에 가까운 환자)", fontsize=11, transform=leg.transAxes)
    fig.suptitle("정중 시상 단면의 기도: 아래쪽이 영상 밖으로 잘린 환자 vs 끝까지 보이는 환자", fontsize=15)
    fig.tight_layout(); fig.savefig(DEST/"airway_fov_compare.png", dpi=75); plt.close(fig)
    t = pd.DataFrame(rows); t.round(1).to_csv(DEST/"airway_fov_compare.csv", index=False, encoding="utf-8-sig")
    tb = t.round(1).to_html(index=False, border=0)
    (DEST/"index.html").write_text(f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>기도 촬영 범위 비교</title>
<style>body{{font:15px/1.6 "Malgun Gothic",sans-serif;max-width:1500px;margin:20px auto;padding:0 16px;background:#fcfcfb;color:#0b0b0b}}img{{max-width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse;font-size:.88rem}}td,th{{border:1px solid #ddd;padding:4px 8px;text-align:right}}th{{background:#eef1f4}}</style></head><body>
<h1>기도 아래쪽이 잘린 4명 vs 대조 3명</h1>
<p>주황색이 CBCT에서 기도로 분할된 부분(AMASSS 상기도), 짙은 회색이 촬영되지 않은 영상 범위 밖입니다. 위 4명은 기도가 영상 아래 끝까지 이어진 채로 끝나 후두개곡(Eb)과 설골이 보이지 않습니다.
아래 대조 환자는 기도가 Eb(녹색 +) 아래까지 보입니다. Eb는 기도 프로파일·PAL·PAV·PMCA의 아래 경계라, 위 4명은 이 값들을 정의할 수 없습니다.</p>
<a href="airway_fov_compare.png" target="_blank"><img src="airway_fov_compare.png" alt="비교 그림"></a>
<h2>수치</h2>{tb}<p>image_bottom_mm_from_PNS: 정중 단면에서 영상이 끝나는 높이(PNS 기준, 음수 = 아래). 대조 환자의 Eb 높이보다 위에서 끝나면 Eb를 볼 수 없습니다.</p>
</body></html>""", encoding="utf-8")
    print("\nsaved", DEST); print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
