"""Python 변환 결과를 기존 MATLAB 결과(Result/*/mat)와 비교합니다. 원본/기존 결과는 읽기만 합니다.

사용: python verify_matlab.py psg [ID ...]     # ID 생략 시 전체
      python verify_matlab.py cbct [ID ...]
결과 요약: Result_py/verify_<psg|cbct>.csv
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from osa.matv73 import MatFile
from osa.naming import standard_names
from osa.psg_events import read_events
from osa.psg_read import read_psg

ROOT = Path(__file__).resolve().parents[1]
NS = np.timedelta64(1, "ns")


def _ids(folder: Path, pattern: str, wanted):
    found = {}
    for d in folder.iterdir():
        m = re.match(pattern, d.name)
        if d.is_dir() and m:
            found[int(m.group(1))] = d
    return {i: found[i] for i in sorted(found) if not wanted or i in wanted}


def _max_dt_us(a, b) -> float:
    a, b = np.asarray(a, "datetime64[ns]"), np.asarray(b, "datetime64[ns]")
    if a.shape != b.shape:
        return np.inf
    return float(np.abs((a - b) / NS).max()) / 1000 if a.size else 0.0


def verify_psg(pid: int, folder: Path) -> dict:
    row = {"Patient": f"P_{pid}"}
    rec = read_psg(folder)
    events, stages, counts = read_events(rec)
    row["ExcludedChannels"] = len(rec.excluded_channels)
    mat = ROOT / "Result" / "PSG" / "mat" / f"P_{pid}.mat"
    if not mat.exists():    # MATLAB이 처리하지 못한 환자 (P_23)
        row.update(Channels=len(rec.channels), Events=len(events), Pass=None, Error="No MATLAB result")
        return row
    m = MatFile(mat)
    try:
        std, _, _ = standard_names(rec.format, [c.name for c in rec.channels])
        mat_keys = {k for k in m.f["Signals"] if m[f"Signals/{k}/available"]}
        # 기존 EBM 결과 일부는 SpO2 % 보정/단위 표기 이전 MATLAB 버전으로 저장됨
        # (RecordingInfo.ebmScalingVersion 없음). 그 경우 구버전 규칙으로 비교합니다.
        legacy = rec.format == "EBM" and "ebmScalingVersion" not in m.f["RecordingInfo"]
        row["LegacyEBM"] = legacy
        row["Channels"] = len(std)
        row["ChannelSetEqual"] = set(std) == mat_keys
        data_bad, max_data, max_time, fs_bad, unit_bad = [], 0.0, 0.0, [], []
        for key, ch in zip(std, rec.channels):
            if key not in mat_keys:
                continue
            g = f"Signals/{key}"
            md = np.asarray(m[f"{g}/data"], dtype=np.float64).ravel()
            if md.shape != ch.data.shape:
                data_bad.append(key)
                continue
            expect, unit = ch.data, ch.unit
            if legacy:
                unit = "EbmRead-scaled (verify unit)"
                if ch.unit == "%":
                    expect = ch.data * 1e-9
            diff = np.nanmax(np.abs(md - expect)) if md.size else 0.0
            ok = diff == 0 or (legacy and ch.unit == "%" and np.allclose(md, expect, rtol=1e-12, atol=0))
            max_data = max(max_data, 0.0 if ok else diff)
            if not ok:
                data_bad.append(key)
            max_time = max(max_time, _max_dt_us(m[f"{g}/time"], ch.time))
            if not np.isclose(float(np.ravel(m[f"{g}/fs"])[0]), ch.fs, equal_nan=True, rtol=0, atol=1e-9):
                fs_bad.append(key)
            if m[f"{g}/unit"] != unit:
                unit_bad.append(key)
        row.update(DataMismatch=";".join(data_bad), MaxDataAbsDiff=max_data,
                   MaxTimeDiffUs=max_time, FsMismatch=";".join(fs_bad), UnitMismatch=";".join(unit_bad))

        me, ms, mc = m["Events"], m["SleepStages"], m["EventCounts"]
        row["Events"] = len(events)
        row["EventsEqual"] = (len(me) == len(events) and
                              list(np.atleast_1d(me.EventType)) == list(events.EventType) and
                              _max_dt_us(me.StartTime, events.StartTime) < 1 and
                              _max_dt_us(me.EndTime, events.EndTime) < 1)
        row["StagesEqual"] = (len(ms) == len(stages) and
                              np.array_equal(np.asarray(ms.Level, float), stages.Level.to_numpy(), equal_nan=True) and
                              _max_dt_us(ms.StartTime, stages.StartTime) < 1)
        row["CountsEqual"] = np.array_equal(np.asarray(mc.Count, float), counts.Count.to_numpy(), equal_nan=True)
    finally:
        m.close()
    row["Pass"] = bool(row["ChannelSetEqual"] and not row["DataMismatch"] and row["MaxTimeDiffUs"] < 1
                       and not row["FsMismatch"] and not row["UnitMismatch"]
                       and row["EventsEqual"] and row["StagesEqual"] and row["CountsEqual"])
    return row


def verify_cbct(pid: int, folder: Path) -> dict:
    from osa.cbct import read_cbct
    row = {"Patient": f"P_{pid}"}
    r = read_cbct(folder)
    m = MatFile(ROOT / "Result" / "CBCT" / "mat" / f"P_{pid}.mat")
    try:
        g = "CBCTData"
        mv = m.f[f"{g}/Volume"][()].T                      # HDF5는 MATLAB 차원 역순
        row["Shape"] = "x".join(map(str, r.volume.shape))
        row["VolumeEqual"] = mv.shape == r.volume.shape and np.array_equal(mv, r.volume)
        mr = m.f[f"{g}/RescaledVolume"][()].T
        row["RescaledEqual"] = mr.shape == r.rescaled.shape and np.array_equal(mr, r.rescaled)
        mvox = np.asarray(m[f"{g}/VoxelSizeMm"], float).ravel()
        row["VoxelEqual"] = np.allclose(mvox, r.voxel_size_mm, equal_nan=True, atol=1e-9)
        st = m[f"{g}/SliceTable"]
        pos = np.asarray(st.PatientPositionMm.tolist() if isinstance(st.PatientPositionMm, pd.Series)
                         else st.PatientPositionMm, float).reshape(-1, 3)
        row["PositionsEqual"] = np.allclose(pos, r.slice_table[["PosX", "PosY", "PosZ"]].to_numpy(), atol=1e-6)
        row["Unit"] = r.unit
        row["UnitEqual"] = m[f"{g}/Unit"] == r.unit
    finally:
        m.close()
    row["Pass"] = bool(row["VolumeEqual"] and row["RescaledEqual"] and row["VoxelEqual"]
                       and row["PositionsEqual"] and row["UnitEqual"])
    return row


def main():
    kind = sys.argv[1].lower()
    wanted = {int(x) for x in sys.argv[2:]}
    if kind == "psg":
        targets, fn = _ids(ROOT / "Data" / "PSG", r"^P_(\d+)(?:_.*)?_([EN])$", wanted), verify_psg
    else:
        targets, fn = _ids(ROOT / "Data" / "CBCT", r"^P_(\d+)_CBCT$", wanted), verify_cbct
    out = ROOT / "Result_py" / f"verify_{kind}.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for k, (pid, folder) in enumerate(targets.items(), 1):
        t = time.time()
        print(f"\n[{k}/{len(targets)}] P_{pid}", flush=True)
        try:
            row = fn(pid, folder)
        except Exception as e:
            traceback.print_exc()
            row = {"Patient": f"P_{pid}", "Pass": False, "Error": f"{type(e).__name__}: {e}"}
        row["Seconds"] = round(time.time() - t, 1)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    print(f"\n통과 {int(df.Pass.sum())} / {len(df)}  → {out}")


if __name__ == "__main__":
    main()
