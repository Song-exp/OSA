"""형태 연구 Track A (Global): 구조 하나(기본 하악)의 통계적 형태모델(SSM).

1. 표면: AMASSS 마스크 → marching cubes (환자 LPS mm)
2. 대응점: 템플릿(중앙 크기 환자)의 표면 점 M개를 각 환자에 CPD(강체 → 비강체)로 정합한 뒤,
           정합된 각 점에서 가장 가까운 환자 표면 점을 그 환자의 대응점으로 씀 (환자 표면 위의 점)
   CPD 출발 자세는 Po를 쓰지 않는 정중선 좌표계(N·S·Ba·ANS·PNS)로 맞춤.
   ALI Po가 틀린 환자(P_37 등)에서 FH 좌표계로 시작하면 CPD가 엉뚱한 곳에 멈췄기 때문 (정합 잔차 8 mm).
3. 정렬 두 버전 (EDA 계획 4장)
   B 'frame'    : 같은 대응점을 머리 좌표계(Jung FH·MSP, 원점 N)로 → 크기·위치 보존. FH는 ALI Po에 의존
   A 'procrustes': 일반화 Procrustes(이동·회전·크기 제거) → 순수 형태. 중심 크기(centroid size)는 따로 저장
4. PCA: 설명분산, 환자별 PC 점수, PC1~3의 평균 ±2SD 형태 그림
결과물이 결과변수(PSG)를 쓰지 않으므로 대응점·템플릿은 전체 환자로 한 번 만듭니다. PCA는 예측 평가 때 폴드 안에서 다시 합니다.
결과: Result_py/shape/<구조>/ (points.npz: 대응점·좌표계 변환, pc_scores_*.csv, variance_*.csv, modes_*.png, qc.csv)
qc.csv의 fit_flag: 정합 잔차가 중앙값의 3배를 넘으면 표시 / po_flag: 랜드마크 QC에서 Po 이탈 (frame 버전 오염)
  te_*: 랜드마크 전달 오차 (대응점이 같은 해부학적 위치를 가리키는지. 잔차가 작아도 점이 표면을 따라 미끄러질 수 있음)
  fit_flag·te 표시는 검수용이며 PCA에서 빼지 않음. frame PCA만 po_flag 환자를 축 계산에서 빼고 투영 (pc_scores의 in_pca).
  주 분석용 PCA(확정 실패만 제외)는 ssm_review.py가 review/에 따로 만듦.
재개: 환자별 대응점(idx, 잔차)을 cache/P_n.npz에 바로 저장 (임시 파일 → 이름 바꾸기). 방법·템플릿·점 수·표면 점 수가
  모두 같고 정상적으로 읽힐 때만 재사용. 메모리: 표면은 환자마다 다시 만들고 버림. 환자별 메모리는 qc.csv의 M_ 열
사용: python shape_ssm.py [--struct MAND] [--n 2048]  |  python shape_ssm.py --selftest
"""
import argparse
import json
import tracemalloc
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

from cbct_eda import atomic_save, working_set_mb
from check_landmarks import read_points
from osa.ceph import head_frame, unit
from run_ali import ascii_path

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "Result_py/CBCT/seg"
TE_LMS = {"MAND": ["Me", "Pog", "B", "Gn", "RGo", "LGo"], "CB": ["S", "Ba", "N"], "MAX": ["ANS", "PNS", "A"]}
CACHE_METHOD = "midline_frame+cpd_rigid_scale+deformable_a2_b8_it80+nearest_v1"   # 대응점 방법을 바꾸면 이 이름도 바꿀 것


def load_corr(f, meta):
    """캐시된 대응점: 파일이 완전하고 meta가 모두 같을 때만 (idx, dist), 아니면 None."""
    if not f.exists():
        return None
    try:
        with np.load(f) as z:
            if json.loads(str(z["meta"])) != meta:
                return None
            return z["idx"], z["dist"]
    except Exception as ex:
        print(f"  {f.name} 캐시 손상 ({ex}) → 다시 계산", flush=True)
        return None


def midline_frame(P):
    """Po를 쓰지 않는 좌표계: 원점 N, 축 (좌, 앞, 위). 좌우 = N·S·Ba·ANS·PNS 최소제곱 평면의 법선."""
    mid = np.array([P[k] for k in ("N", "S", "Ba", "ANS", "PNS")])
    lat = np.linalg.svd(mid - mid.mean(0))[2][-1]
    lat = lat * np.sign(lat[0])                               # LPS x+ = 환자 왼쪽
    ant = P["ANS"] - P["PNS"]
    ant = unit(ant - (ant @ lat) * lat)
    return P["N"], np.array([lat, ant, np.cross(ant, lat)])   # 위 = 앞 × 좌 (LPS에서 z+)


def surface(p, struct):
    """환자 LPS 좌표의 표면 점 (정점, mm)과 랜드마크."""
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
    return lps, {r.Index: np.array([r.x, r.y, r.z]) for r in read_points(p).itertuples()}


def to_frame(x, frame):
    o, R = frame
    return (x - o) @ R.T                                     # 행 = 점, 열 = (좌, 앞, 위)


def subsample(x, n, seed=0):
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), min(n, len(x)), replace=False)]


def correspond(template, target, n_target=5000):
    """템플릿 점 → 환자 표면 대응점의 번호, 정합 잔차(mm). 둘 다 같은 좌표계여야 함."""
    X = subsample(target, n_target)
    rig = RigidRegistration(X=X, Y=template, scale=True)
    Y1, _ = rig.register()
    dfm = DeformableRegistration(X=X, Y=Y1, alpha=2.0, beta=8.0, max_iterations=80)
    Y2, _ = dfm.register()
    dist, idx = cKDTree(target).query(Y2)
    return idx, dist


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


def pca(X, use):
    """use 환자로만 축을 만들고, 전원을 그 축에 투영."""
    flat = X.reshape(len(X), -1)
    mean = flat[use].mean(0)
    u, s, vt = np.linalg.svd(flat[use] - mean, full_matrices=False)
    var = s ** 2 / (use.sum() - 1)
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


def selftest():
    """정중선 좌표계: 머리를 아무렇게나 돌리고 옮겨도 좌표가 같아야 하고, 축 방향이 (좌, 앞, 위)여야 함."""
    P = {"N": [0, -90, 30], "S": [0, -20, 20], "Ba": [1, 10, -15], "ANS": [0, -85, -20], "PNS": [0, -40, -18]}
    P = {k: np.array(v, float) for k, v in P.items()}
    o, R = midline_frame(P)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and R[0, 0] > 0.99 and R[1, 1] < -0.99 and R[2, 2] > 0.99
    x = np.random.default_rng(0).normal(size=(50, 3)) * 30
    a, b, c = np.radians([25, -40, 70])
    Q = (np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
         @ np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
         @ np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]))
    t = np.array([5, -7, 12.0])
    P2 = {k: Q @ v + t for k, v in P.items()}
    assert np.allclose(to_frame(x, (o, R)), to_frame(x @ Q.T + t, midline_frame(P2)), atol=1e-8)
    print("selftest 통과")


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="MAND")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = ROOT / "Result_py/shape" / a.struct
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(f.name.split("_")[1]) for f in SEG.glob(f"P_*_seg_{a.struct}.nrrd")
                 if list((ROOT / "Result_py/CBCT/landmarks").glob(f"P_{f.name.split('_')[1]}_lm_Pred_*.mrk.json")))
    print(f"{a.struct}: 환자 {len(ids)}명", flush=True)
    po = pd.read_csv(ROOT / "Result_py/eval/eval_qc.csv", index_col=0, encoding="utf-8-sig").qc_flag.fillna("")
    # 1차: 좌표계와 크기만 (표면은 버림 → 메모리 절약). 표면은 2차에서 환자마다 다시 만듦 (marching cubes는 결정적)
    lms, mframe, hframe, sizes, npts = {}, {}, {}, {}, {}
    for p in ids:
        s, P = surface(p, a.struct)
        mframe[p], hframe[p], lms[p] = midline_frame(P), head_frame(P), P
        sizes[p], npts[p] = np.sqrt(((s - s.mean(0)) ** 2).sum(1).mean()), len(s)
    tpl_id = sorted(sizes, key=sizes.get)[len(sizes) // 2]   # ponytail: 중앙 크기 환자 1명을 템플릿으로. 평균 형태로 반복 갱신은 표본이 늘면 추가
    tpl_lps = subsample(surface(tpl_id, a.struct)[0], a.n)
    template = to_frame(tpl_lps, mframe[tpl_id])
    # 대응점 검증: 템플릿에서 랜드마크에 가장 가까운 대응점이 각 환자에서 그 환자의 ALI 랜드마크와 몇 mm 떨어지는지
    near = {l: int(np.argmin(np.linalg.norm(tpl_lps - lms[tpl_id][l], axis=1))) for l in TE_LMS.get(a.struct, [])}
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    meta = {"method": CACHE_METHOD, "template_id": tpl_id, "n": a.n}
    shapes, qc = [], []
    tracemalloc.start()
    for p in ids:
        tracemalloc.reset_peak()
        surf = surface(p, a.struct)[0]
        f = cache / f"P_{p}.npz"
        hit = load_corr(f, {**meta, "n_surface": len(surf)})
        if hit:
            idx, dist = hit
        else:
            idx, dist = correspond(template, to_frame(surf, mframe[p]))

            def write(t):
                with open(t, "wb") as fh:
                    np.savez_compressed(fh, idx=idx, dist=dist, meta=json.dumps({**meta, "n_surface": len(surf)}))
            atomic_save(f, write)
        pts = surf[idx]                                      # 환자 LPS의 대응점
        te = {f"te_{l}": np.linalg.norm(pts[i] - lms[p][l]) for l, i in near.items()}
        shapes.append(to_frame(pts, hframe[p]))
        qc.append({"Patient": f"P_{p}", "template": p == tpl_id, "fit_mean_mm": dist.mean(), "fit_p95_mm": np.percentile(dist, 95),
                   "unique_ratio": len(np.unique(idx)) / len(idx),
                   "po_flag": any(s in po.get(f"P_{p}", "") for s in ("RPo", "LPo")),
                   **te, "te_median_mm": np.median(list(te.values())) if te else np.nan,
                   "cached": bool(hit), "M_numpy_peak_MB": tracemalloc.get_traced_memory()[1] / 2 ** 20,
                   "M_ws_MB": working_set_mb()})
        print(f"  P_{p}: 정합 잔차 평균 {dist.mean():.2f} mm{' (캐시)' if hit else ''} | 메모리 {qc[-1]['M_ws_MB']:.0f} MB", flush=True)
        del surf, pts
    tracemalloc.stop()
    shapes = np.array(shapes)
    q = pd.DataFrame(qc)
    q["fit_flag"] = q.fit_mean_mm > 3 * q.fit_mean_mm.median()
    np.savez_compressed(out / "points.npz", ids=np.array(ids), frame=shapes, template_id=tpl_id,
                        midline_origin=np.array([mframe[p][0] for p in ids]), midline_R=np.array([mframe[p][1] for p in ids]),
                        head_origin=np.array([hframe[p][0] for p in ids]), head_R=np.array([hframe[p][1] for p in ids]))
    q.round(3).to_csv(out / "qc.csv", index=False, encoding="utf-8-sig")
    if q.fit_flag.any():
        print("정합 실패 의심:", q.loc[q.fit_flag, "Patient"].tolist())

    gp, csize = gpa(shapes)   # ponytail: 정합 실패 환자도 GPA 평균에 들어감. 실패가 많아지면 GPA에서도 제외
    for ver, X in (("frame", shapes), ("procrustes", gp)):
        use = ~q.po_flag.to_numpy() if ver == "frame" else np.ones(len(q), bool)   # 잔차 표시는 검수용일 뿐 제외하지 않음. 주 분석 PCA는 ssm_review.py
        mean, vt, var, scores = pca(X, use)
        k = min(10, len(var))
        pd.DataFrame(scores[:, :k], index=[f"P_{p}" for p in ids], columns=[f"PC{i + 1}" for i in range(k)]) \
            .assign(**({"CentroidSize": csize} if ver == "procrustes" else {}), in_pca=use).round(4) \
            .to_csv(out / f"pc_scores_{ver}.csv", encoding="utf-8-sig")
        ev = pd.DataFrame({"PC": np.arange(1, len(var) + 1), "var_ratio": var / var.sum(),
                           "cumulative": np.cumsum(var) / var.sum()})
        ev.round(4).to_csv(out / f"variance_{ver}.csv", index=False, encoding="utf-8-sig")
        modes_figure(mean, vt, var, a.n, out / f"modes_{ver}.png", f"{a.struct} SSM ({ver}, n={use.sum()})")
        print(f"[{ver}] 누적 설명분산 PC1~5: {np.round(ev.cumulative[:5].to_numpy(), 3).tolist()}")
    print(f"정합 잔차 평균 {q.fit_mean_mm.mean():.2f} mm (최대 환자 {q.fit_mean_mm.max():.2f}), 대응점 중복 없는 비율 중앙값 {q.unique_ratio.median():.2f}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
