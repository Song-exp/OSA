"""ALI 랜드마크 결과 확인: 좌표표, 해부학 타당성 점검(계획 7-3), 검수용 그림.

입력: Result_py/CBCT/landmarks/P_n*.mrk.json, Result_py/CBCT/nrrd/P_n.nrrd
출력: Result_py/CBCT/landmarks/P_n_points.csv, P_n_check.png, 화면에 점검 결과
점검:
  - 뼈 위 여부: 점 주변 3 mm 안 최대 HU (낮으면 공기·연조직에 떠 있음)
  - 좌우 대칭: 정중점(N, Ba, S, ANS, PNS)으로 맞춘 평면에서 좌우 점까지 거리 차이
  - 거리 범위: ANS–PNS, N–Me, 좌우 Po 간 거리 (성인 문헌 대략값, 크게 벗어나면 표시)
사용: python check_landmarks.py 5
"""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LM = ROOT / "Result_py/CBCT/landmarks"
MID = ["N", "S", "Ba", "ANS", "PNS", "A", "B", "Pog", "Me", "Gn"]
PAIRS = [("RPo", "LPo"), ("ROr", "LOr"), ("RGo", "LGo"), ("RCo", "LCo")]
KEY = ["N", "S", "Ba", "RPo", "LPo", "ROr", "LOr", "ANS", "PNS", "A", "B", "Pog", "Me", "RGo", "LGo"]   # Roh 변수에 쓰는 점만 그림
RANGES = {("ANS", "PNS"): (40, 65), ("N", "Me"): (100, 145), ("RPo", "LPo"): (85, 125), ("S", "N"): (60, 80)}


def read_nrrd(path):
    b = Path(path).read_bytes()
    i = b.index(b"\n\n")
    h = dict(l.split(": ", 1) for l in b[:i].decode().splitlines()[1:] if ": " in l)
    size = [int(x) for x in h["sizes"].split()]
    dirs = np.array([[float(v) for v in d.strip("()").split(",")] for d in re.findall(r"\([^)]*\)", h["space directions"])])
    origin = np.array([float(v) for v in h["space origin"].strip("()").split(",")])
    a = np.frombuffer(gzip.decompress(b[i + 2:]), "<i2").reshape(size[::-1])   # (k, j, i)
    return a, dirs, origin


def read_points(pid) -> pd.DataFrame:
    rows = []
    for f in sorted(LM.glob(f"P_{pid}_*.mrk.json")):
        for m in json.loads(f.read_text())["markups"]:
            ras = m.get("coordinateSystem", "LPS") == "RAS"
            for p in m["controlPoints"]:
                x = np.array(p["position"], float) * ([-1, -1, 1] if ras else 1)   # → LPS
                rows.append({"Label": p["label"], "x": x[0], "y": x[1], "z": x[2]})
    return pd.DataFrame(rows).drop_duplicates("Label").set_index("Label")


def main(pid):
    a, dirs, origin = read_nrrd(ROOT / f"Result_py/CBCT/nrrd/P_{pid}.nrrd")
    pts = read_points(pid)
    to_ijk = lambda x: np.linalg.solve(dirs.T, x - origin)
    sp = np.linalg.norm(dirs, axis=1)
    r = int(np.ceil(3 / sp.min()))
    out = []
    for lab, p in pts.iterrows():
        i, j, k = np.round(to_ijk(p.values)).astype(int)
        inside = 0 <= i < a.shape[2] and 0 <= j < a.shape[1] and 0 <= k < a.shape[0]
        hu = a[max(k - r, 0):k + r + 1, max(j - r, 0):j + r + 1, max(i - r, 0):i + r + 1].max() if inside else np.nan
        out.append({"Label": lab, "x": p.x, "y": p.y, "z": p.z, "inside": inside, "maxHU_3mm": hu,
                    "flag": "" if inside and hu > 300 else ("FOV밖" if not inside else "뼈 아님?")})
    t = pd.DataFrame(out).set_index("Label")
    # 정중면: 정중점들에 맞춘 평면 (SVD, 법선 = 최소 분산 방향)
    mid = pts.loc[pts.index.intersection(MID)].values
    if len(mid) >= 3:
        c = mid.mean(0)
        nrm = np.linalg.svd(mid - c)[2][-1]
        t["dist_MSP"] = (pts.values - c) @ nrm
        print(f"정중점 {len(mid)}개의 평면 이탈 최대 {np.abs((mid - c) @ nrm).max():.1f} mm")
        for a_, b_ in PAIRS:
            if a_ in t.index and b_ in t.index:
                d = abs(abs(t.dist_MSP[a_]) - abs(t.dist_MSP[b_]))
                print(f"  대칭 {a_}/{b_}: 정중면 거리 차이 {d:.1f} mm {'← 확인' if d > 5 else ''}")
    for (p, q), (lo, hi) in RANGES.items():
        if p in pts.index and q in pts.index:
            d = np.linalg.norm(pts.loc[p] - pts.loc[q])
            print(f"  거리 {p}–{q}: {d:.1f} mm (대략 {lo}~{hi}) {'← 확인' if not lo <= d <= hi else ''}")
    t.round(2).to_csv(LM / f"P_{pid}_points.csv", encoding="utf-8-sig")
    print(t.round(1).to_string())
    draw(a, dirs, origin, pts.loc[pts.index.intersection(KEY, sort=False)], to_ijk, LM / f"P_{pid}_check.png")


def draw(a, dirs, origin, pts, to_ijk, path):
    """랜드마크마다 축상·관상·시상 단면 3장 (점 위치에서 자름)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(pts)
    fig, ax = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    ax = np.atleast_2d(ax)
    for row, (lab, p) in enumerate(pts.iterrows()):
        i, j, k = np.clip(np.round(to_ijk(p.values)).astype(int), 0, np.array(a.shape[::-1]) - 1)
        views = [(a[k], (i, j), "axial"), (a[:, j, :], (i, k), "coronal"), (a[:, :, i], (j, k), "sagittal")]
        for c, (img, (u, v), name) in enumerate(views):
            w = 60   # 주변 ±60 voxel (약 ±23 mm)
            v0, u0 = max(v - w, 0), max(u - w, 0)
            ax[row, c].imshow(img[v0:v + w, u0:u + w], cmap="gray", vmin=-1000, vmax=2000, origin="lower")
            ax[row, c].plot(u - u0, v - v0, "r+", ms=14, mew=2)
            ax[row, c].set_title(f"{lab} {name}", fontsize=8)
            ax[row, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    print(f"그림: {path}")


if __name__ == "__main__":
    main(sys.argv[1])
