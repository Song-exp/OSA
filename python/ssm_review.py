"""SSM QC 검수표와 주 분석용 PCA. shape_ssm.py 결과를 읽기만 하고 review/ 폴더에 따로 저장합니다.

QC 지표 (따로 관리)
  fit_flag : CPD 정합 잔차 평균 > 중앙값 × 3                → 검수 표시 (자동 제외 아님)
  te_flag  : 랜드마크 전달 오차(점별 또는 중앙값) 로버스트 z > 3.5 → 검수 표시 (자동 제외 아님)
             대응점이 미끄러졌는지, ALI 랜드마크가 틀렸는지는 원본 영상으로 확인해야 함
  po_flag  : 랜드마크 QC의 Po 이탈 (frame 버전만 해당)
확정 실패 (CONFIRMED): 근거가 확인된 환자만 사람이 여기에 적음. 확정 실패는 PCA 축과 주 분석에서 모두 빠지고,
  점수는 투영값을 진단용으로만 남김 (main_analysis = False).
  검수 표시만 있고 확정되지 않은 환자는 주 분석에 남음.
결과: Result_py/shape/<구조>/review/ (qc_review.csv, pc_scores_{frame,procrustes}.csv, variance_*.csv)
사용: python ssm_review.py --struct MAND
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shape_ssm import gpa

ROOT = Path(__file__).resolve().parents[1]
# 확정 실패: {버전: {환자: 근거}}. 버전 "all"은 frame·procrustes 둘 다
CONFIRMED = {
    "MAND": {"frame": {p: "ALI Po 이탈 확인 (landmark_outlier.py, 평가_자동계측_결과.md 7장) → 머리 좌표계 오염"
                       for p in ("P_16", "P_20", "P_23", "P_35", "P_37")}},
    "CB": {"frame": {p: "ALI Po 이탈 확인 → 머리 좌표계 오염" for p in ("P_16", "P_20", "P_23", "P_35", "P_37")}},
}
# 원본 영상 확인이 필요한 개별 메모
NOTES = {"P_7": "Me·Pog 전달 오차 약 10 mm (정합 잔차는 정상) → 대응점 미끄러짐인지 ALI Me·Pog 오류인지 원본 영상 확인"}


def robust_z(x):
    mad = 1.4826 * (x - x.median()).abs().median()
    return (x - x.median()) / mad if mad > 0 else x * 0


def pca(flat, use):
    mean = flat[use].mean(0)
    _, s, vt = np.linalg.svd(flat[use] - mean, full_matrices=False)
    var = s ** 2 / (use.sum() - 1)
    return var, (flat - mean) @ vt.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="MAND")
    a = ap.parse_args()
    src = ROOT / "Result_py/shape" / a.struct
    out = src / "review"
    out.mkdir(exist_ok=True)
    q = pd.read_csv(src / "qc.csv", encoding="utf-8-sig").set_index("Patient")
    z = np.load(src / "points.npz")
    ids = [f"P_{p}" for p in z["ids"]]
    assert ids == list(q.index)

    te_cols = [c for c in q.columns if c.startswith("te_")]
    te_z = pd.DataFrame({c: robust_z(q[c]) for c in te_cols})
    q["te_flag"] = (te_z > 3.5).any(axis=1)
    q["te_flag_items"] = te_z.apply(lambda r: ", ".join(c[3:] for c in te_cols if r[c] > 3.5), axis=1)
    q["note"] = [NOTES.get(p, "") for p in q.index]
    q["review"] = q.fit_flag | q.te_flag | (q.note != "")
    conf = CONFIRMED.get(a.struct, {})

    shapes = z["frame"]
    gp, csize = gpa(shapes)
    summary = []
    for ver, X in (("frame", shapes), ("procrustes", gp)):
        failed = {**conf.get("all", {}), **conf.get(ver, {})}
        q[f"confirmed_fail_{ver}"] = [failed.get(p, "") for p in q.index]
        main_ = np.array([p not in failed for p in ids])
        var, scores = pca(X.reshape(len(X), -1), main_)
        k = min(10, len(var))
        sc = pd.DataFrame(scores[:, :k], index=ids, columns=[f"PC{i + 1}" for i in range(k)])
        if ver == "procrustes":
            sc["CentroidSize"] = csize
        sc["main_analysis"] = main_                      # False = 투영 점수, 진단용
        sc.round(4).to_csv(out / f"pc_scores_{ver}.csv", encoding="utf-8-sig")
        ev = pd.DataFrame({"PC": np.arange(1, len(var) + 1), "var_ratio": var / var.sum(), "cumulative": np.cumsum(var) / var.sum()})
        ev.round(4).to_csv(out / f"variance_{ver}.csv", index=False, encoding="utf-8-sig")
        summary.append(f"[{ver}] PCA 사용 {main_.sum()}명 / 전체 {len(ids)}명 (확정 실패 {len(failed)}명 제외), "
                       f"누적 설명분산 PC1~5 {np.round(ev.cumulative[:5].to_numpy(), 3).tolist()}")
    q.round(3).to_csv(out / "qc_review.csv", encoding="utf-8-sig")

    print(f"{a.struct}: 환자 {len(q)}명")
    print("\n정합 잔차 평균(mm) 분포:", q.fit_mean_mm.describe(percentiles=[.05, .5, .95]).round(2).to_dict())
    print("랜드마크 전달 오차 중앙값(mm) 분포:", q.te_median_mm.describe(percentiles=[.05, .5, .95]).round(2).to_dict())
    print("점별 전달 오차 중앙값(mm):", q[te_cols].median().round(1).to_dict(), "| 95백분위:", q[te_cols].quantile(.95).round(1).to_dict())
    cols = ["fit_mean_mm", "te_median_mm", "fit_flag", "te_flag_items", "po_flag", "note"]
    print("\n검수 대상:\n" + q.loc[q.review | q.po_flag, cols].round(2).to_string())
    print("\n" + "\n".join(summary))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
