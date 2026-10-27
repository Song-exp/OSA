"""CBCT DICOM 시리즈 읽기. Matlab/Function/CBCT.m 변환.

원본 voxel과 환자 좌표를 유지하고 보간하지 않습니다.
배열 순서는 MATLAB과 같게 (row, column, slice) 입니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.uid import ImplicitVRLittleEndian

# 장비가 쓴 UID에 선행 0이 있어(예: .00902.) 표준 위반 경고가 슬라이스마다 납니다. 값은 그대로 읽습니다.
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE


def _read(f, stop_before_pixels=False):
    """이 데이터의 DICOM은 128바이트 프리앰블/파일 메타가 없는 implicit VR little-endian입니다."""
    ds = pydicom.dcmread(f, stop_before_pixels=stop_before_pixels, force=True)
    if "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    return ds


@dataclass
class CBCTData:
    source_folder: str
    volume: np.ndarray            # 원본 저장값 (rows, cols, slices)
    rescaled: np.ndarray          # float32, slope/intercept 적용
    unit: str
    slice_table: pd.DataFrame     # 슬라이스별 파일, 좌표, 방향, rescale
    pixel_spacings: np.ndarray    # (slices, 2) [row, column] mm
    voxel_size_mm: np.ndarray     # [row, column, slice], 불규칙하면 NaN
    slice_spacing_mm: np.ndarray
    regular_spacing: bool
    volume_dimension: int
    first_header: pydicom.Dataset
    series_uid: str
    created_at: datetime = field(default_factory=datetime.now)
    schema_version: str = "CBCT-1.0"
    voxel_size_order: str = "row, column, slice"


def read_cbct(folder, top_only: bool = False) -> CBCTData:
    """top_only=True면 하위 폴더를 무시합니다 (P_34, P_40: 하위 폴더에 다른 사람 영상이 섞여 있음)."""
    folder = Path(folder)
    assert folder.is_dir(), f"CBCT 폴더가 없습니다: {folder}"
    files = sorted(f for f in (folder.glob("*") if top_only else folder.rglob("*")) if f.is_file())
    headers = []
    for f in files:
        try:
            ds = _read(f, stop_before_pixels=True)
        except Exception:
            continue
        if "Rows" in ds and "SeriesInstanceUID" in ds:
            headers.append((f, ds))
    series = {str(ds.SeriesInstanceUID) for _, ds in headers}
    assert len(series) == 1, f"DICOM 시리즈가 {len(series)}개입니다. 한 시리즈만 포함된 환자 폴더가 필요합니다: {folder}"
    n = len(headers)
    assert n >= 2, "단일 프레임 DICOM 슬라이스가 2개 이상 필요합니다."
    first_header = headers[0][1]

    pos = np.zeros((n, 3)); ori = np.zeros((n, 6)); spacing = np.zeros((n, 2))
    slopes = np.ones(n); inter = np.zeros(n); has = np.zeros(n, bool); types = [""] * n
    for k, (f, ds) in enumerate(headers):
        assert int(getattr(ds, "NumberOfFrames", 1)) == 1, f"다중 프레임 DICOM은 별도 처리가 필요합니다: {f}"
        assert "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds, \
            f"환자 좌표/방향 정보가 없습니다: {f}"
        pos[k] = [float(x) for x in ds.ImagePositionPatient]
        ori[k] = [float(x) for x in ds.ImageOrientationPatient]
        spacing[k] = [float(x) for x in ds.PixelSpacing]
        hs, hi = "RescaleSlope" in ds, "RescaleIntercept" in ds
        assert hs == hi, f"Slope/Intercept 중 하나만 존재합니다: {f}"
        if hs:
            slopes[k], inter[k], has[k] = float(ds.RescaleSlope), float(ds.RescaleIntercept), True
        types[k] = str(getattr(ds, "RescaleType", ""))
    assert np.isfinite(slopes).all() and np.isfinite(inter).all(), "잘못된 픽셀 변환 계수입니다."
    assert np.abs(ori - ori[0]).max() < 1e-6, "슬라이스 방향이 서로 다릅니다."

    # 슬라이스 법선 방향 좌표 오름차순 (MATLAB dicomreadVolume 정렬과 동일한지 verify_matlab.py로 확인)
    normal = np.cross(ori[0, :3], ori[0, 3:])
    normal /= np.linalg.norm(normal)
    order = np.argsort(pos @ normal, kind="stable")
    headers = [headers[i] for i in order]
    pos, ori, spacing = pos[order], ori[order], spacing[order]
    slopes, inter, has = slopes[order], inter[order], has[order]
    types = [types[i] for i in order]

    first = _read(headers[0][0]).pixel_array
    volume = np.empty(first.shape + (n,), dtype=first.dtype)
    for k, (f, _) in enumerate(headers):
        volume[:, :, k] = first if k == 0 else _read(f).pixel_array
    rescaled = (volume.astype(np.float64) * slopes + inter).astype(np.float32)

    slice_spacing = np.abs(np.diff(pos @ normal))
    assert (slice_spacing > 0).all(), "중복된 슬라이스 평면입니다."
    z = float(np.median(slice_spacing))
    regular = bool(np.all(np.abs(slice_spacing - z) <= max(1e-4, z * 1e-3)))
    regular_px = bool(np.abs(spacing - spacing[0]).max() < 1e-6)
    voxel = np.array([spacing[0, 0], spacing[0, 1], z if regular else np.nan])
    if not regular_px:
        voxel[:2] = np.nan
    unit = "Rescaled DICOM values"
    if not has.any():
        unit = "Raw DICOM values"
    elif all(t.strip().upper() == "HU" for t in types):
        unit = "HU"
    table = pd.DataFrame({"Slice": np.arange(1, n + 1), "SourceFile": [str(f) for f, _ in headers],
                          "PosX": pos[:, 0], "PosY": pos[:, 1], "PosZ": pos[:, 2],
                          **{f"Orient{i + 1}": ori[:, i] for i in range(6)},
                          "RescaleSlope": slopes, "RescaleIntercept": inter,
                          "HasRescale": has, "RescaleType": types})
    print(f"CBCT: {volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]} voxels; {unit}")
    return CBCTData(str(folder), volume, rescaled, unit, table, spacing, voxel, slice_spacing,
                    regular and regular_px, int(np.argmax(np.ptp(pos, axis=0))) + 1,
                    first_header, series.pop())


def draw_cbct_figure(data: CBCTData, pid, max_grid=160, surface_fraction=0.35, max_faces=100_000):
    """main_YJ_PSG_All.m의 drawCBCTFigure/drawCBCTSurface. 중앙 단면 3개 + 표시용 3D 표면.

    MATLAB FIG의 마우스 회전 대신 고정 시점(view 135, 20) PNG용 그림입니다.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    v = data.rescaled
    dims = v.shape
    mid = [(d + 1) // 2 - 1 for d in dims]     # MATLAB ceil(dims/2), 0-based
    planes = [v[:, :, mid[2]], v[mid[0], :, :].T, v[:, mid[1], :].T]
    labels = [("Column (voxel)", "Row (voxel)"), ("Column (voxel)", "Slice (voxel)"),
              ("Row (voxel)", "Slice (voxel)")]
    titles = [f"Row-column | slice {mid[2] + 1}", f"Column-slice | row {mid[0] + 1}",
              f"Row-slice | column {mid[1] + 1}"]
    finite = np.concatenate([p[np.isfinite(p)] for p in planes])
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    dz = np.diff(data.slice_table.PosZ.to_numpy())
    # DICOM 환자 Z는 머리 방향으로 증가. 높은 Z가 위로 오게 표시합니다.
    origin = "lower" if (dz > 1e-6).all() else "upper"

    fig = plt.figure(figsize=(14, 9.5), constrained_layout=True)
    for k in range(3):
        ax = fig.add_subplot(2, 2, k + 1)
        im = ax.imshow(planes[k], cmap="gray", vmin=lo, vmax=hi, origin="upper" if k == 0 else origin,
                       aspect="equal", interpolation="nearest")
        ax.set_xlabel(labels[k][0]); ax.set_ylabel(labels[k][1]); ax.set_title(titles[k])
        fig.colorbar(im, ax=ax, label=data.unit)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    step = max(1, -(-max(dims) // max_grid))
    small = v[::step, ::step, ::step]
    threshold = lo + surface_fraction * (hi - lo)
    try:
        s = 1
        verts, faces, _, _ = marching_cubes(small, threshold, step_size=s)
        while len(faces) > max_faces:        # MATLAB reducepatch 대신 격자를 더 성기게
            s += 1
            verts, faces, _, _ = marching_cubes(small, threshold, step_size=s)
    except (ValueError, RuntimeError):
        ax.text2D(0.5, 0.5, "No surface at selected display threshold", transform=ax.transAxes, ha="center")
        verts = None
    if verts is not None:
        r, c, k = (verts[:, i] * step for i in range(3))
        n = dims[2]
        pos = data.slice_table[["PosX", "PosY", "PosZ"]].to_numpy()
        origin3 = np.column_stack([np.interp(k, np.arange(n), pos[:, j]) for j in range(3)])
        sp = np.column_stack([np.interp(k, np.arange(n), data.pixel_spacings[:, j]) for j in range(2)])
        o = data.slice_table[[f"Orient{i}" for i in range(1, 7)]].to_numpy()[0]
        xyz = origin3 + (c * sp[:, 1])[:, None] * o[:3] + (r * sp[:, 0])[:, None] * o[3:]
        ax.add_collection3d(Poly3DCollection(xyz[faces], facecolors=[(0.88, 0.83, 0.72)],
                                             linewidths=0, shade=True))
        lo3, hi3 = xyz.min(0), xyz.max(0)
        ax.set_xlim(lo3[0], hi3[0]); ax.set_ylim(lo3[1], hi3[1]); ax.set_zlim(lo3[2], hi3[2])
        ax.set_box_aspect(hi3 - lo3)
        ax.view_init(elev=20, azim=135)
    ax.set_xlabel("Patient X (mm)"); ax.set_ylabel("Patient Y (mm)"); ax.set_zlabel("Patient Z (mm)")
    ax.set_title(f"3D surface\nDisplay threshold: {threshold:.1f} ({data.unit})")
    fig.suptitle(f"P_{pid} | CBCT | {dims[0]} x {dims[1]} x {dims[2]}")
    return fig
