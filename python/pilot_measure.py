"""파일럿 12명 자동 계측과 엑셀 대조 (계획 7장, 파일럿 단계만).

입력: Result_py/CBCT/landmarks/P_n_lm_Pred_*.mrk.json (run_ali.py), 엑셀
변수: ANB, Nperp_A, Nperp_Pog, MP-FH (Go–Me 직선 = B등급 주 정의, 3D 평면 = 대안)
FH 구성 3가지: jung (RPo·LPo·ROr), yao4 (좌우 Po·Or 4점 최소제곱), lor (RPo·LPo·LOr)
FH 기울기 진단: 머리를 좌우축으로 θ만큼 돌렸을 때 각 FH 의존 변수가 엑셀과 맞는 θ를 따로 구함.
  세 변수의 θ가 비슷하면 "랜드마크는 맞고 기준면만 기울어짐", 제각각이면 개별 랜드마크 문제.
보정 버전 "jung+refine": AMASSS 하악·상악 마스크 윤곽으로 A·B·Pog를 다시 잡음 (osa/refine.py)
결과: Result_py/pilot/pilot_measures.csv, pilot_errors.csv, pilot_tilt.csv, pilot_refine_info.csv
평가 세트(71명)에는 쓰지 않습니다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from check_landmarks import read_points
from osa.ceph import measures, to_head
from osa.refine import refine
from run_ali import ascii_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/pilot"
PILOT = [5, 7, 9, 24, 30, 32, 41, 44, 50, 54, 60, 69]
VARS = {"ANB": 22, "Nperp_A": 23, "Nperp_Pog": 24, "MPFH": 25}
FHS = ["jung", "yao4", "lor"]
SEG = ROOT / "Result_py/CBCT/seg"


def excel() -> pd.DataFrame:
    x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None).iloc[3:]
    x = x[x[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    return pd.DataFrame({k: pd.to_numeric(x[c], errors="coerce").values for k, c in VARS.items()},
                        index=x[0].astype(int).values)


def rotated(h: dict, deg: float) -> dict:
    """머리 좌표(좌, 앞, 위)를 좌우축으로 회전 (+ = 얼굴이 위로 들림)."""
    t = np.radians(deg)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return {k: R @ v for k, v in h.items()}


def fh_vars(h: dict) -> dict:
    """이미 머리 좌표인 점에서 FH 의존 변수 계산 (ceph.measures와 같은 공식)."""
    go = (h["RGo"] + h["LGo"]) / 2
    d = go - h["Me"]
    return {"Nperp_A": h["A"][1], "Nperp_Pog": h["Pog"][1], "MPFH": np.degrees(np.arctan2(d[2], -d[1]))}


def tilt(h: dict, ref: pd.Series) -> dict:
    grid = np.arange(-15, 15.001, 0.05)
    vals = [fh_vars(rotated(h, g)) for g in grid]
    return {f"theta_{k}": grid[int(np.argmin([abs(v[k] - ref[k]) for v in vals]))] for k in ("Nperp_A", "Nperp_Pog", "MPFH")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xl = excel()
    rows, tilts, infos = [], [], []
    for p in PILOT:
        pts = read_points(p)
        need = {"N", "Ba", "RPo", "LPo", "ROr", "LOr", "A", "B", "Pog", "Me", "RGo", "LGo"}
        missing = need - set(pts.index)
        if missing:
            print(f"P_{p}: 랜드마크 없음 {sorted(missing)} → 건너뜀")
            continue
        P = {k: r[["x", "y", "z"]].to_numpy(float) for k, r in pts.iterrows()}
        # 마스크 기반 A·B·Pog 보정
        rd = lambda s: sitk.ReadImage(str(ascii_path(SEG / f"P_{p}_seg_{s}.nrrd")))
        ref = rd("MAND")
        dirs = np.array(ref.GetDirection()).reshape(3, 3).T * np.array(ref.GetSpacing())[:, None]
        Pr, info = refine(P, sitk.GetArrayFromImage(ref), sitk.GetArrayFromImage(rd("MAX")), dirs,
                          np.array(ref.GetOrigin()))
        infos += [{"Patient": f"P_{p}", "Point": k, **v} for k, v in info.items()]
        for fh, pts_ in [(f, P) for f in FHS] + [("jung+refine", Pr)]:
            m = measures(pts_, fh.split("+")[0])
            rows.append({"Patient": f"P_{p}", "FH": fh, "ANB": m["ANB"], "Nperp_A": m["Nperp_A"],
                         "Nperp_Pog": m["Nperp_Pog"], "MPFH_GoMe": m["MPFH_GoMe"], "MPFH_3Dplane": m["MPFH_3Dplane"],
                         **{f"Excel_{k}": xl.loc[p, k] for k in VARS}})
        tilts.append({"Patient": f"P_{p}", **tilt(to_head(P, "jung"), xl.loc[p])})
    d = pd.DataFrame(rows)
    d.round(2).to_csv(OUT / "pilot_measures.csv", index=False, encoding="utf-8-sig")

    err = []
    for fh, g in d.groupby("FH", sort=False):
        for auto, ref in [("ANB", "ANB"), ("Nperp_A", "Nperp_A"), ("Nperp_Pog", "Nperp_Pog"),
                          ("MPFH_GoMe", "MPFH"), ("MPFH_3Dplane", "MPFH")]:
            e = g[auto] - g[f"Excel_{ref}"]
            r = np.corrcoef(g[auto], g[f"Excel_{ref}"])[0, 1]
            err.append({"FH": fh, "Variable": auto, "N": len(e), "MeanDiff": e.mean(), "SD": e.std(),
                        "MAE": e.abs().mean(), "MaxAbs": e.abs().max(), "r": r})
    e = pd.DataFrame(err).round(2)
    e.to_csv(OUT / "pilot_errors.csv", index=False, encoding="utf-8-sig")
    t = pd.DataFrame(tilts)
    t.to_csv(OUT / "pilot_tilt.csv", index=False, encoding="utf-8-sig")

    print("자동 − 엑셀 오차 (파일럿)\n" + e.to_string(index=False))
    print("\nFH 기울기 진단: 엑셀과 맞추려면 필요한 회전 θ (°, jung FH 기준)\n" + t.to_string(index=False))
    th = t.filter(like="theta")
    print(f"\n환자별 θ 범위(세 변수 간 최대−최소) 중앙값: {(th.max(axis=1) - th.min(axis=1)).median():.1f}°"
          f" | θ 평균 {th.mean().round(1).to_dict()} | 환자 간 SD {th.std().round(1).to_dict()}")
    agree = lambda a, b, cut: ((a >= cut) == (b >= cut)).mean()
    for fh in ("jung", "jung+refine"):
        c = d[d.FH == fh]
        print(f"분류 일치 ({fh}): ANB≥4 {agree(c.ANB, c.Excel_ANB, 4):.0%}"
              f" | MP-FH≥30 {agree(c.MPFH_GoMe, c.Excel_MPFH, 30):.0%}")
    ri = pd.DataFrame(infos)
    ri.round(2).to_csv(OUT / "pilot_refine_info.csv", index=False, encoding="utf-8-sig")
    print("\n보정 상태:", ri.status.value_counts().to_dict(), f"| 평균 이동 {ri.shift_mm.mean():.1f} mm")
    if (ri.status != "보정").any():
        print(ri[ri.status != "보정"].round(2).to_string(index=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
