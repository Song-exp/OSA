"""미팅용 대표 환자 그림: qc_viewer의 그리기를 그대로 써서, 문제 부위를 지나는 세 단면을 PNG로 저장.
위치: 랜드마크 이름이면 그 점, "frag:<구조>"면 그 구조 마스크의 두 번째로 큰 덩어리 중심, None이면 시상 N·나머지 가운데.
정상 예시는 EDA·SSM 검수 표시가 없고 하악 정합 잔차가 중앙값에 가장 가까운 환자를 자동으로 고름.
결과: Result_py/cbct_eda/qc_snapshots/rep_<n>_<P>.png (얼굴이 보이므로 내부용)
"""
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

import qc_viewer as qv
from check_landmarks import read_points
from qc_montage import QC_COLORS
from run_ali import ascii_path

CASES = [  # (환자, 위치, 랜드마크 표시, 설명)
    (None, None, False, "정상 예시 (검수 표시 없음, 하악 정합 잔차 중앙값)"),
    ("P_7", "Me", True, "턱 금속 + ALI Me·Pog가 뼈 밖 (전달 오차 약 10 mm)"),
    ("P_51", "frag:CB", False, "두개저 마스크 조각화 + 두개저 정합 잔차 상위 10%"),
    ("P_20", "RPo", True, "ALI Po 실패 + 하악 정합 잔차 최대 (1.31 mm)"),
    ("P_22", "frag:UAW", False, "상기도 마스크 여러 덩어리"),
    ("P_50", None, False, "촬영 조건 이탈 (0.3 mm 복셀, 15 cm 범위, 잡음)"),
]


def normal_case():
    eda, _ = qv.qc_tables()
    q = pd.read_csv(qv.ROOT / "Result_py/shape/MAND/review/qc_review.csv", index_col=0, encoding="utf-8-sig")
    c = pd.read_csv(qv.ROOT / "Result_py/shape/CB/review/qc_review.csv", index_col=0, encoding="utf-8-sig")
    ok = [p for p in q.index if not eda.get(p, "") and not (q.review | q.po_flag)[p] and not (c.review | c.po_flag)[p]]
    return (q.loc[ok, "fit_mean_mm"] - q.fit_mean_mm.median()).abs().idxmin()


def read_case(pid):
    """qc_viewer.Viewer.load와 같은 읽기 (동기)."""
    img = sitk.ReadImage(str(ascii_path(qv.CB / f"nrrd/{pid}.nrrd")))
    a = sitk.GetArrayFromImage(img)
    lab = np.zeros(a.shape, np.uint8)
    for b, s in enumerate(QC_COLORS):
        f = qv.CB / f"seg/{pid}_seg_{s}.nrrd"
        if f.exists():
            lab |= (sitk.GetArrayFromImage(sitk.ReadImage(str(ascii_path(f)))) > 0).astype(np.uint8) << b
    lm = {k: np.array(img.TransformPhysicalPointToContinuousIndex(tuple(r[["x", "y", "z"]].astype(float))))
          for k, r in read_points(int(pid[2:])).iterrows()}
    return a, lab, lm, img.GetSpacing()


def target(where, lab, lm):
    if where is None:
        return None
    if where.startswith("frag:"):
        b = list(QC_COLORS).index(where[5:])
        cc, n = ndimage.label((lab >> b) & 1)
        assert n >= 2, where
        sizes = ndimage.sum(np.ones_like(cc), cc, range(1, n + 1))
        k, j, i = ndimage.center_of_mass(cc == 1 + np.argsort(sizes)[-2])
        return [int(k), int(j), int(i)]
    i, j, k = lm[where]
    return [int(round(k)), int(round(j)), int(round(i))]


def main():
    v = qv.Viewer()
    v.withdraw()
    out = qv.EDA / "qc_snapshots"
    out.mkdir(exist_ok=True)
    rows = []
    for n, (pid, where, show_lm, desc) in enumerate(CASES, 1):
        pid = pid or normal_case()
        a, lab, lm, sp = read_case(pid)
        v.pid = pid
        v._loaded(pid, a, lab, lm, sp)
        v.pos = target(where, lab, lm) or v.pos
        v.show_lm.set(show_lm)
        v.redraw()
        v.fig.suptitle(f"{pid} · {desc}", fontsize=11)
        f = out / f"rep_{n}_{pid}.png"
        v.fig.savefig(f, dpi=110)
        rows.append({"n": n, "patient": pid, "position_kji": v.pos, "desc": desc, "file": f.name})
        print(n, pid, v.pos, desc)
    pd.DataFrame(rows).to_csv(out / "rep_cases.csv", index=False, encoding="utf-8-sig")
    v.destroy()


if __name__ == "__main__":
    main()
