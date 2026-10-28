"""구조별 색 분할 QC 몽타주 (반투명 채우기 + 윤곽). cbct_eda.py 캐시의 정중 시상 단면만 읽으므로 몇 초면 끝남.
결과: Result_py/cbct_eda/montage_overlay_all.png (전원), montage_overlay_review.png (자동 QC 검수 대상만, 사유 표시)
얼굴 윤곽이 보이므로 내부용. 환자를 자동으로 제외하지 않음.
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/cbct_eda"
QC_COLORS = {"MAND": "red", "MAX": "dodgerblue", "CB": "limegreen", "UAW": "yellow"}
NAMES = {"MAND": "하악", "MAX": "상악", "CB": "두개저", "UAW": "상기도"}
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def overlay(ax, raw, masks, alpha=0.3, lw=0.6, window=(-1000, 2000)):
    """원본 위에 구조별 반투명 채우기와 진한 윤곽."""
    ax.imshow(raw, cmap="gray", vmin=window[0], vmax=window[1])
    rgba = np.zeros(raw.shape + (4,))
    for s, c in QC_COLORS.items():
        m = masks.get(s)
        if m is not None and m.any():
            rgba[m] = to_rgba(c, alpha)
    ax.imshow(rgba)
    for s, c in QC_COLORS.items():
        m = masks.get(s)
        if m is not None and m.any():
            ax.contour(m, levels=[0.5], colors=c, linewidths=lw)


def legend():
    return [Patch(color=c, label=NAMES[s]) for s, c in QC_COLORS.items()]


def montage(ids, d, path, title, reasons=False, ncol=10):
    nrow = int(np.ceil(len(ids) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * (2.6 if reasons else 2.1)), squeeze=False)
    for k, axx in enumerate(ax.ravel()):
        axx.axis("off")
        if k >= len(ids):
            continue
        p = ids[k]
        z = np.load(OUT / f"cache/{p}_thumb.npz")
        overlay(axx, z["raw"], {s: z[f"m_{s}"] for s in QC_COLORS if f"m_{s}" in z})
        rev = d.loc[p, "review_items"]
        t = p + ("\n" + "\n".join(textwrap.wrap(rev, 26)[:3]) if reasons and rev else "")
        axx.set_title(t, fontsize=6 if reasons else 8, color="red" if rev else "black")
    fig.legend(handles=legend(), loc="upper right", ncol=4, fontsize=9)
    fig.suptitle(title, fontsize=11, x=0.35)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    d = pd.read_csv(OUT / "cbct_eda_patient.csv", index_col=0, encoding="utf-8-sig")
    d["review_items"] = d.review_items.fillna("")
    ids = sorted(d.index, key=lambda p: int(p[2:]))
    rev = [p for p in ids if d.loc[p, "review_items"]]
    montage(ids, d, OUT / "montage_overlay_all.png", f"정중 시상 단면 · 구조별 분할 (n={len(ids)}) · 빨간 제목 = 자동 QC 검수 대상 (제외 아님)")
    montage(rev, d, OUT / "montage_overlay_review.png", f"자동 QC 검수 대상 {len(rev)}명 · 정중 시상 단면 한 장만 보이므로 전체는 qc_viewer.py로 확인",
            reasons=True, ncol=6)
    print(f"저장: {OUT / 'montage_overlay_all.png'}, {OUT / 'montage_overlay_review.png'} (검수 대상 {len(rev)}명)")


if __name__ == "__main__":
    main()
