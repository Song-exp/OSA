"""CBCT를 3D Slicer용 NRRD로 내보냅니다. 환자 식별정보(DICOM 헤더)는 들어가지 않습니다.

DICOM에 프리앰블이 없어 Slicer DICOM 가져오기가 실패할 수 있어서, 파이썬 리더로 읽은 값을 그대로 씁니다.
값: HU (rescale 적용, int16), 좌표: DICOM 환자 좌표(LPS, mm) 그대로
사용: python export_nrrd.py 5 7 9  → Result_py/CBCT/nrrd/P_5.nrrd ...
"""
import gzip
import sys
from pathlib import Path

import numpy as np

from osa.cbct import read_cbct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/CBCT/nrrd"


def write_nrrd(d, path: Path):
    t = d.slice_table
    o = t[[f"Orient{i}" for i in range(1, 7)]].iloc[0].to_numpy()
    pos = t[["PosX", "PosY", "PosZ"]].to_numpy()
    normal = np.cross(o[:3], o[3:])
    rs, cs, z = d.voxel_size_mm
    assert d.regular_spacing, "간격이 불규칙한 볼륨은 NRRD 한 장으로 표현할 수 없습니다."
    # 배열 (row, col, slice) → NRRD 축 순서 (col, row, slice): col이 가장 빠르게 변함
    data = np.round(d.rescaled).astype("<i2").transpose(2, 0, 1)
    vec = lambda v: "(" + ",".join(f"{x:.6f}" for x in v) + ")"
    header = "\n".join([
        "NRRD0004", "type: short", "dimension: 3", "space: left-posterior-superior",
        f"sizes: {data.shape[2]} {data.shape[1]} {data.shape[0]}",
        f"space directions: {vec(o[:3] * cs)} {vec(o[3:] * rs)} {vec(normal * z)}",
        "kinds: domain domain domain", "endian: little", "encoding: gzip",
        f"space origin: {vec(pos[0])}", "", ""])
    path.write_bytes(header.encode("ascii") + gzip.compress(data.tobytes(), 6))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for pid in sys.argv[1:]:
        folder = next((ROOT / "Data/CBCT").glob(f"P_{pid}_*"))
        p = OUT / f"P_{pid}.nrrd"
        write_nrrd(read_cbct(folder), p)
        print(f"{p}  {p.stat().st_size / 1e6:.0f} MB")
