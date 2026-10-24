"""전체 환자 일괄 처리. Matlab/main_YJ_PSG_All.m 변환.

사용 (python 폴더에서):
    python -m osa.batch psg                      # 전체 PSG
    python -m osa.batch psg --ids 4 7 12 --overwrite
    python -m osa.batch cbct
결과: Result_py/<PSG|CBCT>/h5/P_<id>.h5, Result_py/<PSG|CBCT>/Figure/P_<id>[_tabNN].png
기존 MATLAB 결과(Result/)와 원본(Data/)은 건드리지 않습니다.

저장 형식 (HDF5):
  - 신호 시간축은 저장하지 않고 segments(시작 시각, 샘플 수, fs)로 복원합니다 (load_psg).
  - CBCT RescaledVolume은 저장하지 않고 slope/intercept로 복원합니다 (load_cbct).
  - PlotData(표시용 축약 신호)는 저장하지 않습니다. 그림은 PNG로 저장됩니다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NS = np.timedelta64(1, "ns")


# ------------------------------------------------------------ HDF5 표 저장/읽기
def write_table(group: h5py.Group, df: pd.DataFrame):
    group.attrs["columns"] = json.dumps(list(df.columns))
    group.attrs.update({f"attr_{k}": str(v) for k, v in df.attrs.items()})
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            ds = group.create_dataset(c, data=col.to_numpy("datetime64[ns]").view("int64"))
            ds.attrs["kind"] = "datetime64[ns]"
        elif pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            group.create_dataset(c, data=col.to_numpy())
        else:
            group.create_dataset(c, data=col.astype(str).to_numpy().astype(object),
                                 dtype=h5py.string_dtype())


def read_table(group: h5py.Group) -> pd.DataFrame:
    out = {}
    for c in json.loads(group.attrs["columns"]):
        ds = group[c]
        v = ds[()]
        if ds.attrs.get("kind") == "datetime64[ns]":
            v = v.view("datetime64[ns]")
        elif h5py.check_string_dtype(ds.dtype):
            v = ds.asstr()[()]
        out[c] = v
    df = pd.DataFrame(out)
    df.attrs = {k[5:]: v for k, v in group.attrs.items() if k.startswith("attr_")}
    return df


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _atomic(path: Path, write):
    """임시 파일에 다 쓴 뒤 이름을 바꿉니다. 중간에 실패해도 반쪽 파일이 남지 않습니다."""
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ------------------------------------------------------------ PSG
def save_psg(path: Path, rec, events, stages, counts, save_info: dict):
    from .naming import standard_names
    std, used, common = standard_names(rec.format, [c.name for c in rec.channels])

    def write(tmp):
        with h5py.File(tmp, "w") as f:
            sig = f.create_group("Signals")
            for key in common:          # 없는 공통 신호도 available=False로 남깁니다 (MATLAB과 동일)
                if key not in std:
                    sig.create_group(key).attrs.update(available=False, system=rec.format)
            for i, (key, ch) in enumerate(zip(std, rec.channels)):
                g = sig.create_group(key)
                g.attrs.update(available=True, unit=ch.unit, fs=ch.fs, sourceName=ch.name,
                               sourceIndex=i + 1, system=rec.format, sourceFile=ch.source_file,
                               metadata=_json(ch.metadata))
                g.create_dataset("data", data=ch.data, compression="lzf", shuffle=True, chunks=True)
                write_table(g.create_group("segments"), ch.segments)
            write_table(f.create_group("SignalMap"), pd.DataFrame({
                "SourceIndex": np.arange(1, len(std) + 1), "SourceName": [c.name for c in rec.channels],
                "StandardName": std, "CommonSignal": used}))
            write_table(f.create_group("ChannelTable"), rec.channel_table)
            write_table(f.create_group("Events"), events)
            write_table(f.create_group("SleepStages"), stages)
            write_table(f.create_group("EventCounts"), counts)
            ri = f.create_group("RecordingInfo")
            ri.attrs.update(schemaVersion=rec.schema_version, format=rec.format,
                            sourceFolder=rec.source_folder, sourceFiles=_json(rec.source_files),
                            createdAt=str(rec.created_at))
            if rec.format == "EBM":
                ri.attrs["ebmScalingVersion"] = "SpO2-percent-1"
            write_table(ri.create_group("excludedChannels"), rec.excluded_channels)
            if rec.annotations is not None:
                write_table(ri.create_group("annotations"), rec.annotations)
            f.create_group("SaveInfo").attrs.update({k: str(v) for k, v in save_info.items()})
    _atomic(path, write)


def load_psg(path) -> dict:
    """저장한 PSG h5를 읽습니다. Signals[key] = dict(data, time, fs, unit, ...)."""
    from .psg_read import segment_times
    out = {}
    with h5py.File(path, "r") as f:
        out["Signals"] = {}
        for key, g in f["Signals"].items():
            s = dict(g.attrs)
            if s.get("available"):
                s["data"] = g["data"][()]
                s["segments"] = seg = read_table(g["segments"])
                s["time"] = np.concatenate([segment_times(t, n, fs) for t, n, fs in
                                            zip(seg.StartTime, seg.Samples, seg.Fs)])
                s["metadata"] = json.loads(s["metadata"])
            out["Signals"][key] = s
        for name in ("SignalMap", "ChannelTable", "Events", "SleepStages", "EventCounts"):
            out[name] = read_table(f[name])
        out["RecordingInfo"] = dict(f["RecordingInfo"].attrs)
        out["SaveInfo"] = dict(f["SaveInfo"].attrs)
    return out


def process_psg(folder: Path, h5: Path, fig_dir: Path, pid: int, opt):
    import matplotlib.pyplot as plt
    from .psg_plot import plot_psg
    from .psg_read import read_psg

    expected = "EBM" if folder.name.upper().endswith("_E") else "NDF"
    rec = read_psg(folder)
    assert rec.format == expected, f"폴더 접미사와 실제 파일 형식이 다릅니다: {folder}"
    figs, events, stages, _, counts = plot_psg(
        rec, channels_per_tab=opt.channels_per_tab, max_plot_points=opt.max_plot_points,
        skip_initial_seconds=opt.skip_initial_seconds, y_axis_range_fraction=opt.y_axis_range_fraction)
    save_info = {"SchemaVersion": "PSG-batch-py-1.0", "SourceFolder": folder, "SavedAt": datetime.now(),
                 "H5File": h5, "FigureFolder": fig_dir, "ChannelCount": len(rec.channels),
                 "SkipInitialSeconds": opt.skip_initial_seconds,
                 "EventCountScope": "Full exported scoring; no first-minute exclusion",
                 "UnitPolicy": "Original reader values and units retained; variable names only are harmonized"}
    save_psg(h5, rec, events, stages, counts, save_info)
    for k, fig in enumerate(figs, 1):
        fig.savefig(fig_dir / f"P_{pid}_tab{k:02d}.png", dpi=opt.dpi)
        plt.close(fig)
    return len(rec.channels)


# ------------------------------------------------------------ CBCT
def save_cbct(path: Path, d, save_info: dict):
    def write(tmp):
        with h5py.File(tmp, "w") as f:
            f.create_dataset("Volume", data=d.volume, compression="lzf", chunks=True)
            write_table(f.create_group("SliceTable"), d.slice_table)
            f.create_dataset("PixelSpacings", data=d.pixel_spacings)
            f.create_dataset("VoxelSizeMm", data=d.voxel_size_mm)
            f.create_dataset("SliceSpacingMm", data=d.slice_spacing_mm)
            f.attrs.update(SourceFolder=d.source_folder, Unit=d.unit, RegularSpacing=d.regular_spacing,
                           VolumeDimension=d.volume_dimension, SeriesInstanceUID=d.series_uid,
                           SchemaVersion=d.schema_version, CreatedAt=str(d.created_at),
                           VoxelSizeOrder=d.voxel_size_order,
                           FirstSourceHeader=d.first_header.to_json(suppress_invalid_tags=True))
            f.create_group("SaveInfo").attrs.update({k: str(v) for k, v in save_info.items()})
    _atomic(path, write)


def load_cbct(path) -> dict:
    """Volume(원본) + RescaledVolume(float32, 슬라이스별 slope/intercept 적용)을 복원합니다."""
    with h5py.File(path, "r") as f:
        out = {k: f[k][()] for k in ("Volume", "PixelSpacings", "VoxelSizeMm", "SliceSpacingMm")}
        out["SliceTable"] = st = read_table(f["SliceTable"])
        out.update(dict(f.attrs))
    out["RescaledVolume"] = (out["Volume"].astype(np.float64) * st.RescaleSlope.to_numpy()
                             + st.RescaleIntercept.to_numpy()).astype(np.float32)
    return out


def process_cbct(folder: Path, h5: Path, fig_dir: Path, pid: int, opt):
    import matplotlib.pyplot as plt
    from .cbct import draw_cbct_figure, read_cbct

    d = read_cbct(folder)
    fig = draw_cbct_figure(d, pid)
    save_cbct(h5, d, {"SchemaVersion": "CBCT-batch-py-1.0", "SavedAt": datetime.now(),
                      "SourceFolder": folder, "H5File": h5,
                      "FigureDescription": "Three central sections and display isosurface; complete volume in H5"})
    fig.savefig(fig_dir / f"P_{pid}.png", dpi=opt.dpi)
    plt.close(fig)
    return d.volume.shape[2]


# ------------------------------------------------------------ 공통 루프
def run(kind: str, opt) -> pd.DataFrame:
    kind = kind.upper()
    data_root = Path(opt.data_root) if opt.data_root else ROOT / "Data" / kind
    pattern = r"^P_(\d+)(?:_.*)?_([EN])$" if kind == "PSG" else r"^P_(\d+)_CBCT$"
    found = {}
    for d in data_root.iterdir():
        m = re.match(pattern, d.name)
        if d.is_dir() and m:
            pid = int(m.group(1))
            assert pid not in found, f"같은 P_번호를 가진 폴더가 둘 이상입니다: P_{pid}"
            found[pid] = d
    assert found, f"{data_root}에 환자 폴더가 없습니다."
    if opt.ids:
        missing = set(opt.ids) - set(found)
        assert not missing, f"찾지 못한 환자 번호: {sorted(missing)}"
    ids = sorted(i for i in found if not opt.ids or i in opt.ids)

    out = ROOT / "Result_py" / kind
    h5_dir, fig_dir = out / "h5", out / "Figure"
    h5_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    process = process_psg if kind == "PSG" else process_cbct
    rows = []
    for k, pid in enumerate(ids, 1):
        t = time.time()
        h5 = h5_dir / f"P_{pid}.h5"
        print(f"\n[{kind} {k}/{len(ids)}] {found[pid]}", flush=True)
        row = {"Patient": f"P_{pid}", "Folder": found[pid].name, "Status": "", "Count": 0, "Message": ""}
        if h5.exists() and not opt.overwrite:
            row.update(Status="Skipped", Message="H5 already exists (use --overwrite)")
        else:
            try:
                row.update(Status="Saved", Count=process(found[pid], h5, fig_dir, pid, opt), Message=str(h5))
            except Exception as e:
                traceback.print_exc()
                row.update(Status="Failed", Message=f"{type(e).__name__}: {e}")
                if isinstance(e, OSError) and e.errno == 28:   # 디스크 공간 부족이면 중단
                    break
        row["ElapsedSeconds"] = round(time.time() - t, 1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string())
    print(f"완료 {(summary.Status == 'Saved').sum()} / 기존 결과 {(summary.Status == 'Skipped').sum()} "
          f"/ 실패 {(summary.Status == 'Failed').sum()}")
    return summary


def main():
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["psg", "cbct"])
    p.add_argument("--ids", type=int, nargs="*", help="환자 번호 (생략 시 전체)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--data-root", help="기본값: Data/PSG 또는 Data/CBCT")
    p.add_argument("--channels-per-tab", type=int, default=7)
    p.add_argument("--max-plot-points", type=int, default=100_000)
    p.add_argument("--skip-initial-seconds", type=float, default=180)
    p.add_argument("--y-axis-range-fraction", type=float, default=0.50)
    p.add_argument("--dpi", type=int, default=100)
    args = p.parse_args()
    run(args.kind, args)


if __name__ == "__main__":
    main()
