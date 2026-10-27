"""A·B·Pog가 같은 높이의 뼈 앞쪽 윤곽(정중면 ±1 mm, AMASSS 마스크)보다 앞/뒤로 몇 mm 떨어졌는지.

양수 = 윤곽보다 앞(연조직 쪽), 음수 = 윤곽보다 뒤(뼈 안쪽). 머리 좌표계(FH·MSP)의 앞뒤 축 기준.
2단계 민감도 분석의 가설("A는 측정자보다 뒤, Pog는 앞")을 점 자체로 확인하는 보조 점검입니다.
결과: Result_py/pilot/diag_front_offset.csv
"""
import numpy as np
import pandas as pd
import SimpleITK as sitk

from osa.ceph import head_frame
from osa.refine import front_contour
from pilot_diagnose import points
from pilot_measure import OUT, PILOT, SEG
from run_ali import ascii_path


def main():
    rows = []
    for p in PILOT:
        P = points(p)
        o, R = head_frame(P)
        imgs = {s: sitk.ReadImage(str(ascii_path(SEG / f"P_{p}_seg_{s}.nrrd"))) for s in ("MAND", "MAX")}
        ref = imgs["MAND"]
        dirs = np.array(ref.GetDirection()).reshape(3, 3).T * np.array(ref.GetSpacing())[:, None]
        org = np.array(ref.GetOrigin())
        for lm, s in (("A", "MAX"), ("B", "MAND"), ("Pog", "MAND")):
            h = R @ (P[lm] - o)
            f = front_contour(sitk.GetArrayFromImage(imgs[s]), dirs, org, o, R, np.array([h[2]]), (h[1] - 25, h[1] + 25))[0]
            rows.append({"Patient": f"P_{p}", "Landmark": lm, "offset_ant_mm": h[1] - f, "lateral_mm": h[0]})
    d = pd.DataFrame(rows)
    d.round(2).to_csv(OUT / "diag_front_offset.csv", index=False, encoding="utf-8-sig")
    t = d.pivot(index="Patient", columns="Landmark", values="offset_ant_mm").loc[[f"P_{p}" for p in PILOT]]
    print("ALI 점 − 뼈 앞 윤곽 (mm, + = 윤곽보다 앞)\n" + t.round(1).to_string())
    print("\n평균 / 중앙값:", t.mean().round(2).to_dict(), "/", t.median().round(2).to_dict())
    print("정중면에서 좌우로 벗어난 거리 |x| 중앙값:", d.groupby("Landmark").lateral_mm.apply(lambda x: x.abs().median()).round(1).to_dict())


if __name__ == "__main__":
    main()
