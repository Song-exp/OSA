"""형태 연구 Track A (Global): 구조 하나(기본 하악)의 통계적 형태모델(SSM).

1. 표면: AMASSS 마스크 → marching cubes → 머리 좌표계(Jung FH·MSP, 원점 N, mm)로 변환
2. 대응점: 템플릿(중앙 크기 환자)의 표면 점 M개를 각 환자에 CPD(강체 → 비강체)로 정합한 뒤,
           정합된 각 점에서 가장 가까운 환자 표면 점을 그 환자의 대응점으로 씀 (환자 표면 위의 점)
3. 정렬 두 버전 (EDA 계획 4장)
   B 'frame'    : 머리 좌표계 그대로 → 크기·위치 보존
   A 'procrustes': 일반화 Procrustes(이동·회전·크기 제거) → 순수 형태. 중심 크기(centroid size)는 따로 저장
4. PCA: 설명분산, 환자별 PC 점수, PC1~3의 평균 ±2SD 형태 그림
결과물이 결과변수(PSG)를 쓰지 않으므로 대응점·템플릿은 전체 환자로 한 번 만듭니다. PCA는 예측 평가 때 폴드 안에서 다시 합니다.
결과: Result_py/shape/<구조>/ (points.npz, pc_scores_*.csv, variance_*.csv, modes_*.png, qc.csv)
사용: python shape_ssm.py [--struct MAND] [--n 2048]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pycpd import DeformableRegistration, RigidRegistration
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from check_landmarks import read_points
from osa.ceph import head_frame
from run_ali import ascii_path

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "Result_py/CBCT/seg"


def surface(p, struct):
    """머리 좌표계의 표면 점 (정점, mm)."""
    img = sitk.ReadImage(str(ascii_path(SEG / f"P_{p}_seg_{struct}.nrrd")))
    m = sitk.GetArrayFromImage(img) > 0                     # (k, j, i)
    lab, n = label(m)                                        # 떨어진 조각(턱받침·설골 등) 제거: 가장 큰 덩어리만
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    k, j, i = np.nonzero(m)
    lo = np.maximum(np.array([k.min(), j.min(), i.min()]) - 1, 0)
    crop = np.pad(m[lo[0]:k.max() + 2, lo[1]:j.max() + 2, lo[2]:i.max() + 2], 1)   # 영상 가장자리에 닿아도 표면이 닫히게
    lo = lo - 1
    v, _, _, _ = marching_cubes(crop.astype(np.uint8), 0.5, step_size=2)
    kji = v + lo
    lps = np.array([img.TransformContinuousIndexToPhysicalPoint(tuple(map(float, x[::-1]))) for x in kji])
    P = {r.Index: np.array([r.x, r.y, r.z]) for r in read_points(p).itertuples()}
    o, R = head_frame(P)
    return (lps - o) @ R.T                                   # 행 = 점, 열 = (좌, 앞, 위)


def subsample(x, n, seed=0):
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), min(n, len(x)), replace=False)]


def correspond(template, target, n_target=5000):
    """템플릿 점 → 환자 표면 대응점, 정합 잔차(mm)."""
    X = subsample(target, n_target)
    rig = RigidRegistration(X=X, Y=template, scale=True)
    Y1, _ = rig.register()
    dfm = DeformableRegistration(X=X, Y=Y1, alpha=2.0, beta=8.0, max_iterations=80)
    Y2, _ = dfm.register()
    dist, idx = cKDTree(target).query(Y2)
    return target[idx], dist


def gpa(shapes, iters=10):
    """일반화 Procrustes: (n, M, 3) → 정렬된 형태, 중심 크기."""
    X = shapes - shapes.mean(1, keepdims=True)
    size = np.sqrt((X ** 2).sum((1, 2)))
    X = X / size[:, None, None]
    ref = X[0]
    for _ in range(iters):
        for i in range(len(X)):
            u, _, vt = np.linalg.svd(X[i].T @ ref)
            d = np.sign(np.linalg.det(u @ vt))
            X[i] = X[i] @ (u @ np.diag([1, 1, d]) @ vt)
        ref = X.mean(0)
        ref /= np.sqrt((ref ** 2).sum())
    return X, size


def pca(X):
    flat = X.reshape(len(X), -1)
    mean = flat.mean(0)
    u, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    var = s ** 2 / (len(X) - 1)
    return mean, vt, var, (flat - mean) @ vt.T


def modes_figure(mean, vt, var, M, path, title, k=3):
    fig, ax = plt.subplots(k, 2, figsize=(9, 3.2 * k))
    for c in range(min(k, len(var))):
        for j, (a, b, lab) in enumerate(((1, 2, "옆 (앞→오른쪽)"), (0, 1, "아래에서 (좌→오른쪽)"))):
            for sd, col in ((-2, "tab:blue"), (0, "0.6"), (2, "tab:red")):
                sh = (mean + sd * np.sqrt(var[c]) * vt[c]).reshape(M, 3)
                ax[c, j].scatter(sh[:, a], sh[:, b], s=1, c=col, alpha=0.5)
            ax[c, j].set_aspect("equal")
            ax[c, j].set_title(f"PC{c + 1} ({100 * var[c] / var.sum():.1f}%) {lab}", fontsize=9)
    fig.suptitle(title + "  파랑 -2SD · 회색 평균 · 빨강 +2SD")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="MAND")
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    out = ROOT / "Result_py/shape" / a.struct
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(f.name.split("_")[1]) for f in SEG.glob(f"P_*_seg_{a.struct}.nrrd")
                 if list((ROOT / "Result_py/CBCT/landmarks").glob(f"P_{f.name.split('_')[1]}_lm_Pred_*.mrk.json")))
    print(f"{a.struct}: 환자 {len(ids)}명", flush=True)
    surf = {p: surface(p, a.struct) for p in ids}
    sizes = {p: np.sqrt(((s - s.mean(0)) ** 2).sum(1).mean()) for p, s in surf.items()}
    tpl_id = sorted(sizes, key=sizes.get)[len(sizes) // 2]   # ponytail: 중앙 크기 환자 1명을 템플릿으로. 평균 형태로 반복 갱신은 표본이 늘면 추가
    template = subsample(surf[tpl_id], a.n)
    shapes, qc = [], []
    for p in ids:
        pts, dist = correspond(template, surf[p])
        shapes.append(pts)
        qc.append({"Patient": f"P_{p}", "template": p == tpl_id, "fit_mean_mm": dist.mean(), "fit_p95_mm": np.percentile(dist, 95),
                   "unique_ratio": len(np.unique(pts, axis=0)) / len(pts)})
        print(f"  P_{p}: 정합 잔차 평균 {dist.mean():.2f} mm", flush=True)
    shapes = np.array(shapes)
    np.savez_compressed(out / "points.npz", ids=np.array(ids), frame=shapes, template_id=tpl_id)
    pd.DataFrame(qc).round(3).to_csv(out / "qc.csv", index=False, encoding="utf-8-sig")

    gp, csize = gpa(shapes)
    for ver, X in (("frame", shapes), ("procrustes", gp)):
        mean, vt, var, scores = pca(X)
        k = min(10, len(var))
        pd.DataFrame(scores[:, :k], index=[f"P_{p}" for p in ids], columns=[f"PC{i + 1}" for i in range(k)]) \
            .assign(**({"CentroidSize": csize} if ver == "procrustes" else {})).round(4) \
            .to_csv(out / f"pc_scores_{ver}.csv", encoding="utf-8-sig")
        ev = pd.DataFrame({"PC": np.arange(1, len(var) + 1), "var_ratio": var / var.sum(),
                           "cumulative": np.cumsum(var) / var.sum()})
        ev.round(4).to_csv(out / f"variance_{ver}.csv", index=False, encoding="utf-8-sig")
        modes_figure(mean, vt, var, a.n, out / f"modes_{ver}.png", f"{a.struct} SSM ({ver}, n={len(ids)})")
        print(f"[{ver}] 누적 설명분산 PC1~5: {np.round(ev.cumulative[:5].to_numpy(), 3).tolist()}")
    q = pd.DataFrame(qc)
    print(f"정합 잔차 평균 {q.fit_mean_mm.mean():.2f} mm (최대 환자 {q.fit_mean_mm.max():.2f}), 대응점 중복 없는 비율 중앙값 {q.unique_ratio.median():.2f}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
