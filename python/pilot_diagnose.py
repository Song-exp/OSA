"""파일럿 자동 계측 오차 진단 0~2단계 (계획: docs/파일럿_자동계측_1차.md 5장).

0단계: 촬영 자세(FH·정중면이 스캐너 축에서 기운 정도), 2D 투영 vs 3D 계측 차이
1단계: 랜드마크별 뼈 표면 거리 + 해당 마스크 존재 + 가장 가까운 구조가 기대 구조인지 (AMASSS 마스크)
       표면 거리는 "정확도"가 아니라 이상 징후입니다. 구성점(접선 교차점 등)에는 적용하지 않습니다.
2단계: 민감도(랜드마크를 머리 좌표 축으로 1 mm 옮겼을 때 계측값 변화, 단독·동시 이동),
       ALI 그대로의 MAE 부트스트랩 신뢰구간 (파일럿은 방법 선택에 쓰므로 최종 성능 아님)
입력: ALI 랜드마크(read_points), Result_py/CBCT/seg/P_n_seg_{MAND,MAX,CB}.nrrd, 엑셀
결과: Result_py/pilot/diag_orientation.csv, diag_surface.csv, diag_sensitivity.csv
사용: python pilot_diagnose.py [0 1 2]   (기본: 전부)
"""
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

from check_landmarks import read_points
from osa.ceph import head_frame, measures
from pilot_measure import OUT, PILOT, SEG, VARS, excel
from run_ali import ascii_path

MEAS = {"ANB": "ANB", "Nperp_A": "Nperp_A", "Nperp_Pog": "Nperp_Pog", "MPFH_GoMe": "MPFH"}
EXPECTED = {"N": "CB", "Ba": "CB", "RPo": "CB", "LPo": "CB", "ROr": "MAX", "LOr": "MAX", "ANS": "MAX",
            "PNS": "MAX", "A": "MAX", "B": "MAND", "Pog": "MAND", "Me": "MAND", "Gn": "MAND", "RGo": "MAND", "LGo": "MAND"}
MOVE = {"N": "as", "A": "as", "B": "as", "Pog": "as", "Me": "as", "RGo": "as", "LGo": "as",
        "RPo": "asl", "LPo": "asl", "ROr": "asl"}          # a=앞, s=위, l=좌
AXIS = {"l": 0, "a": 1, "s": 2}


def points(p) -> dict:
    return {k: r[["x", "y", "z"]].to_numpy(float) for k, r in read_points(p).iterrows()}


# ------------------------------------------------------------ 0단계
def stage0(xl):
    rows = []
    for p in PILOT:
        P = points(p)
        _, R = head_frame(P)
        left, ant, sup = R
        m = measures(P)
        rows.append({"Patient": f"P_{p}",
                     "pitch_deg": np.degrees(np.arcsin(ant[2])),            # + = 얼굴이 위로
                     "roll_deg": np.degrees(np.arcsin(left[2])),            # + = 왼쪽이 위로
                     "yaw_deg": np.degrees(np.arctan2(ant[0], -ant[1])),    # + = 얼굴이 왼쪽으로
                     "ANB_2D": m["ANB"], "ANB_3D": m["ANB_3D"], "ANB_2D_minus_3D": m["ANB"] - m["ANB_3D"],
                     "MPFH_line_minus_plane": m["MPFH_GoMe"] - m["MPFH_3Dplane"],
                     **{f"err_{k}": m[k] - xl.loc[p, v] for k, v in MEAS.items()}})
    d = pd.DataFrame(rows)
    d.round(2).to_csv(OUT / "diag_orientation.csv", index=False, encoding="utf-8-sig")
    print("[0단계] 촬영 자세 (°)\n" + d[["pitch_deg", "roll_deg", "yaw_deg"]].describe().loc[["mean", "std", "min", "max"]].round(1).to_string())
    print("\n2D − 3D 차이: ANB 최대 |%.2f|°, MP-FH(직선−평면) 최대 |%.2f|°" %
          (d.ANB_2D_minus_3D.abs().max(), d.MPFH_line_minus_plane.abs().max()))
    c = d[["pitch_deg", "roll_deg", "yaw_deg"]].abs().corrwith(d.filter(like="err_").abs().mean(axis=1), method="spearman")
    print("자세 크기 vs 평균 |오차| 순위상관 (n=12, 참고용):", c.round(2).to_dict())
    print(d[["Patient", "pitch_deg", "roll_deg", "yaw_deg", "err_ANB", "err_MPFH_GoMe"]].round(1).to_string(index=False))


# ------------------------------------------------------------ 1단계
def signed_distance(mask, img, pt, half_mm=15.0):
    """점에서 마스크 표면까지 거리(mm). 마스크 안이면 음수. 주변 ±half_mm만 계산."""
    sp = np.array(img.GetSpacing())                               # (i, j, k)
    ijk = np.round(img.TransformPhysicalPointToContinuousIndex(tuple(map(float, pt)))).astype(int)
    r = np.ceil(half_mm / sp).astype(int)
    lo, hi = np.maximum(ijk - r, 0), np.minimum(ijk + r + 1, mask.shape[::-1])
    if np.any(ijk < 0) or np.any(ijk >= mask.shape[::-1]):
        return np.nan
    crop = mask[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] > 0          # (k, j, i)
    c = tuple((ijk - lo)[::-1])
    if not crop.any():
        return np.inf
    if crop[c]:
        return -distance_transform_edt(crop, sampling=sp[::-1])[c]
    return distance_transform_edt(~crop, sampling=sp[::-1])[c]


def stage1():
    rows = []
    for p in PILOT:
        P = points(p)
        imgs = {s: sitk.ReadImage(str(ascii_path(SEG / f"P_{p}_seg_{s}.nrrd"))) for s in ("MAND", "MAX", "CB")}
        arrs = {s: sitk.GetArrayFromImage(i) for s, i in imgs.items()}
        for lm, exp in EXPECTED.items():
            if lm not in P:
                continue
            d = {s: signed_distance(arrs[s], imgs[s], P[lm]) for s in arrs}
            near = min(d, key=lambda s: abs(d[s]) if np.isfinite(d[s]) else 1e9)
            rows.append({"Patient": f"P_{p}", "Landmark": lm, "Expected": exp, "dist_mm": d[exp],
                         "mask_within_15mm": np.isfinite(d[exp]), "nearest": near, "roi_match": near == exp,
                         **{f"d_{s}": v for s, v in d.items()}})
    d = pd.DataFrame(rows)
    d.round(2).to_csv(OUT / "diag_surface.csv", index=False, encoding="utf-8-sig")
    g = d.groupby("Landmark", sort=False)
    s = pd.DataFrame({"기대 마스크": g.Expected.first(),
                      "15mm 안 마스크 있음": g.mask_within_15mm.mean().map("{:.0%}".format),
                      "가장 가까운 구조 일치": g.roi_match.mean().map("{:.0%}".format),
                      "|거리| 중앙값": g.dist_mm.apply(lambda x: x[np.isfinite(x)].abs().median()).round(1),
                      "|거리| > 2mm 환자": g.apply(lambda x: ", ".join(x.Patient[np.isfinite(x.dist_mm) & (x.dist_mm.abs() > 2)])),
                      })
    print("[1단계] 랜드마크–기대 뼈 표면 거리 (음수=뼈 안)\n" + s.to_string())
    bad = d[~d.roi_match]
    if len(bad):
        print("\n가장 가까운 구조가 기대와 다른 경우:\n" + bad[["Patient", "Landmark", "Expected", "nearest", "d_MAND", "d_MAX", "d_CB"]].round(1).to_string(index=False))


# ------------------------------------------------------------ 2단계
def stage2(xl):
    rows, base_rows = [], []
    for p in PILOT:
        P = points(p)
        base = measures(P)
        base_rows.append({"Patient": f"P_{p}", **{k: base[k] - xl.loc[p, v] for k, v in MEAS.items()}})
        _, R = head_frame(P)
        moves = [((lm,), axes) for lm, axes in MOVE.items()]
        moves += [(("A", "B"), "as"), (("A", "B", "N"), "as"), (("A", "B", "N", "Pog"), "as")]   # 같은 방향 동시 이동
        for lms, axes in moves:
            for ax in axes:
                Q = dict(P)
                for lm in lms:
                    Q[lm] = P[lm] + R[AXIS[ax]]                         # 머리 좌표 축 방향으로 1 mm
                m = measures(Q)
                rows.append({"Patient": f"P_{p}", "Move": "+".join(lms), "Axis": {"a": "앞", "s": "위", "l": "좌"}[ax],
                             **{k: m[k] - base[k] for k in MEAS}})
    d = pd.DataFrame(rows)
    d.round(3).to_csv(OUT / "diag_sensitivity.csv", index=False, encoding="utf-8-sig")
    t = d.groupby(["Move", "Axis"], sort=False)[list(MEAS)].mean()
    print("[2단계] 1 mm 이동 시 계측값 변화 (파일럿 평균; ANB·MP-FH는 °, Nperp는 mm)\n" + t.round(2).to_string())

    # 엑셀과의 차이를 "그 점 하나만 앞뒤로 움직여 설명한다면 몇 mm인가" (가상 환산, 원인 확정 아님)
    e = pd.DataFrame(base_rows).set_index("Patient")
    sens = d[(d.Axis == "앞")].groupby(["Patient", "Move"]).ANB.first().unstack()
    print("\nANB 차이를 한 점의 앞뒤 이동으로 환산한 크기 (mm, 중앙값|IQR) — 가상 환산이며 원인을 확정하지 않음")
    for lm in ("A", "B", "N"):
        need = (-e.ANB / sens[lm]).abs()
        print(f"  {lm}: {need.median():.1f} mm (IQR {need.quantile(.25):.1f}–{need.quantile(.75):.1f})")

    rng = np.random.default_rng(0)
    print("\nALI 그대로의 MAE와 환자 단위 부트스트랩 95% CI (n=12, 불안정할 수 있음 · 방법 선택용, 최종 성능 아님)")
    for k in MEAS:
        a = e[k].abs().to_numpy()
        bs = [rng.choice(a, a.size).mean() for _ in range(5000)]
        print(f"  {k}: MAE {a.mean():.2f} (95% CI {np.percentile(bs, 2.5):.2f}–{np.percentile(bs, 97.5):.2f}),"
              f" 평균 차이 {e[k].mean():+.2f}, 차이 SD {e[k].std():.2f}")


if __name__ == "__main__":
    stages = {int(a) for a in sys.argv[1:]} or {0, 1, 2}
    xl = excel()
    if 0 in stages:
        stage0(xl)
    if 1 in stages:
        stage1()
    if 2 in stages:
        stage2(xl)
