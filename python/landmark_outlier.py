"""이상 환자의 오차가 '시작점(랜드마크)'에서 왔는지 진단합니다. 진단용이며, 평가 결과를 바꾸지 않습니다.

머리 좌표계는 Po로 만들기 때문에, Po가 틀린 환자를 그 좌표계에서 보면 모든 점이 같이 밀립니다.
그래서 좌표계를 쓰지 않습니다.
1. 정상 환자(QC 통과 + ANB 오차 ≤3° + MP-FH 오차 ≤5°)의 랜드마크 16개로 평균 배치를 만듭니다 (일반화 Procrustes, mm 유지).
2. 환자마다 점 하나씩 빼고 나머지로 평균 배치에 맞춘 뒤(유사 변환), 뺀 점이 평균 위치에서 몇 mm 떨어졌는지 잽니다.
   틀린 점이 정렬을 끌고 가지 않게, 범위를 넘은 점은 가장 큰 것부터 하나씩 정렬에서 빼고 다시 맞춥니다.
3. 같은 거리를 정상 환자에게도 재서(그 환자를 뺀 평균으로), 점마다 정상 범위(99백분위)를 정합니다.
4. 범위를 넘는 점을 평균 배치에서 예측한 위치로 바꿔 ANB·MP-FH를 다시 계산합니다.
   엑셀에 가까워지면 그 점이 원인입니다. 넘는 점이 없는데 오차가 크면 큰 실패는 아니고,
   해부학적 개인차(수 mm) 안의 위치 차이, 정의 차이, 또는 엑셀 쪽 문제입니다.
결과: Result_py/eval/landmark_outlier.csv, landmark_outlier_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from batch_cbct_pipeline import LMS
from check_landmarks import read_points
from osa.ceph import measures
from pilot_measure import excel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/eval"
EXTRA = ["P_32", "P_41", "P_69"]   # 파일럿 검수 대상


def similarity(src, dst):
    """src → dst 유사 변환 (Umeyama). 반환: s, R, t (y = s R x + t)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, d, vt = np.linalg.svd(b.T @ a)
    S = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    R = u @ S @ vt
    s = (d * np.diag(S)).sum() / (a ** 2).sum()
    return s, R, md - s * R @ ms


def mean_shape(shapes, iters=10):
    ref = shapes[0]
    for _ in range(iters):
        al = []
        for x in shapes:
            s, R, t = similarity(x, ref)
            al.append((s * (R @ x.T)).T + t)
        new = np.mean(al, 0)
        size = np.mean([np.sqrt(((x - x.mean(0)) ** 2).sum()) for x in shapes])
        new = (new - new.mean(0)) / np.sqrt(((new - new.mean(0)) ** 2).sum()) * size   # 평균 크기(mm)로
        ref = new
    return ref


def loo_residuals(x, mu, skip=None):
    """점마다: 그 점(과 skip 점들)을 뺀 정렬 후 평균 위치와의 거리(mm), 환자 공간에서의 예측 위치."""
    skip = np.zeros(len(x), bool) if skip is None else skip
    res, pred = np.zeros(len(x)), np.zeros_like(x)
    for i in range(len(x)):
        k = (np.arange(len(x)) != i) & ~skip
        s, R, t = similarity(x[k], mu[k])
        res[i] = np.linalg.norm(s * R @ x[i] + t - mu[i])
        pred[i] = R.T @ (mu[i] - t) / s
    return res, pred


def diagnose(x, mu, lim):
    """가장 많이 벗어난 점부터 하나씩 정렬에서 빼며 다시 맞춤 (큰 실패 하나가 다른 점까지 끌고 가지 않게)."""
    bad = np.zeros(len(x), bool)
    while True:
        res, pred = loo_residuals(x, mu, bad)
        r = np.where(bad, 0, res / lim)
        if r.max() <= 1 or bad.sum() >= len(x) // 2:
            return res, pred, bad
        bad[np.argmax(r)] = True


def main():
    qc = pd.read_csv(OUT / "eval_qc.csv", index_col=0, encoding="utf-8-sig").qc_flag.fillna("")
    xl = excel()
    pts, meas = {}, {}
    for pid in qc.index:
        P = read_points(int(pid[2:]))
        pts[pid] = np.array([P.loc[l, ["x", "y", "z"]].to_numpy(float) for l in LMS])
        m = measures(dict(zip(LMS, pts[pid])))
        meas[pid] = {"ANB": m["ANB"], "MPFH": m["MPFH_GoMe"],
                     "Excel_ANB": xl.loc[int(pid[2:]), "ANB"], "Excel_MPFH": xl.loc[int(pid[2:]), "MPFH"]}
    me = pd.DataFrame(meas).T.astype(float)
    big = ((me.ANB - me.Excel_ANB).abs() > 3) | ((me.MPFH - me.Excel_MPFH).abs() > 5)
    target = (qc != "") | big | qc.index.isin(EXTRA)
    ref_ids = list(qc.index[~target])
    mu = mean_shape(np.array([pts[p] for p in ref_ids]))

    ref_res = []
    for p in ref_ids:   # 정상 범위: 그 환자를 뺀 평균 배치로
        mu_p = mean_shape(np.array([pts[q] for q in ref_ids if q != p]), iters=3)
        ref_res.append(loo_residuals(pts[p], mu_p)[0])
    p99 = np.percentile(ref_res, 99, axis=0)

    rows, summ = [], []
    for p in qc.index[target]:
        res, pred, bad = diagnose(pts[p], mu, p99)
        for l, r, lim, b in zip(LMS, res, p99, bad):
            rows.append({"Patient": p, "Landmark": l, "dev_mm": r, "normal_p99_mm": lim, "outside": b})
        fixed = pts[p].copy()
        fixed[bad] = pred[bad]
        mf = measures(dict(zip(LMS, fixed)))
        e = me.loc[p]
        summ.append({"Patient": p, "qc_flag": qc[p], "outside_landmarks": ", ".join(np.array(LMS)[bad]),
                     "worst": f"{LMS[int(np.argmax(res / p99))]} {res.max():.0f}mm",
                     "ANB_auto": e.ANB, "ANB_fixed": mf["ANB"], "ANB_excel": e.Excel_ANB,
                     "MPFH_auto": e.MPFH, "MPFH_fixed": mf["MPFH_GoMe"], "MPFH_excel": e.Excel_MPFH})
    pd.DataFrame(rows).round(2).to_csv(OUT / "landmark_outlier.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summ).round(1)
    s.to_csv(OUT / "landmark_outlier_summary.csv", index=False, encoding="utf-8-sig")
    print(f"정상 기준 {len(ref_ids)}명, 진단 대상 {len(s)}명")
    print("정상 범위(99백분위, mm):", dict(zip(LMS, p99.round(1))))
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
