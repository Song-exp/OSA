"""판독 이벤트/수면단계 읽기. Matlab/Function/PSG_Events.m 변환. 원본 파일은 읽기만 합니다.

NOX: Data.ndb(SQLite)의 활성 scoring revision, .NET ticks (100 ns, 0001-01-01 기준).
EBM: RemLogic export TXT. EDF: 같은 이름의 TXT 이벤트 / CSV 수면단계.
"""
from __future__ import annotations

import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .psg_read import Recording

NS = np.timedelta64(1, "ns")
COUNT_NAMES = ["APNEA-CENTRAL", "APNEA-MIXED", "APNEA-OBSTRUCTIVE",
               "AROUSAL-RERA", "AROUSAL-RESP", "HYPOPNEA"]
STAGE_GROUPS = [("W", "SLEEP-S0", "SLEEP-WAKE"), ("REM", "R", "SLEEP-REM"),
                ("N1", "SLEEP-S1", "SLEEP-N1"), ("N2", "SLEEP-S2", "SLEEP-N2"),
                ("N3", "SLEEP-S3", "SLEEP-N3"), ("N4", "SLEEP-S4", "SLEEP-N4")]
DOTNET_EPOCH_TICKS = 621_355_968_000_000_000   # 1970-01-01의 .NET ticks


def read_events(rec: Recording):
    """Returns (events, stages, event_counts) DataFrames."""
    ev, st = [], []
    start = rec.channel_table.StartTime.min().to_datetime64()
    stop = rec.channel_table.EndTime.max().to_datetime64()
    folder = Path(rec.source_folder)
    count_available, source = False, ""

    if rec.format == "NDF":
        db_file = folder / "Data.ndb"
        if not db_file.is_file():
            warnings.warn("No Data.ndb: events unavailable.")
            return _finish(ev, st, False, "", start, stop)
        db = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
        try:
            rows, source = _read_nox_markers(db)
        finally:
            db.close()
        count_available = True
        for a, b, kind in rows:
            a, b = _ticks_time(a), _ticks_time(b)
            assert b >= a, "Negative NOX event duration."
            _append(ev, st, a, b, str(kind).upper())

    elif rec.format == "EBM":
        for i, name in enumerate(["EN-Apnea Events.txt", "EN-Stage Events.txt"]):
            f = folder / name
            if not f.is_file():
                warnings.warn(f"Missing scoring: {f}")
                continue
            if i == 0:
                count_available, source = True, str(f)
            txt = _read_text(f, "cp949")
            m = re.search(r"Recording Date:\s*(\d{4}-\d{2}-\d{2})", txt)
            assert m, f"Missing recording date: {f}"
            base = np.datetime64(m.group(1), "ns")
            lines = txt.splitlines()
            # RemLogic은 Position 열이 있는 5열 / 없는 4열을 모두 출력합니다. 헤더에서 열을 찾습니다.
            header = None
            for k, line in enumerate(lines):
                cols = [c.strip() for c in line.split("\t")]
                low = [c.lower() for c in cols]
                if "event" in low and any(c.startswith("time") for c in low):
                    tcol = [j for j, c in enumerate(low) if c.startswith("time")]
                    ecol = [j for j, c in enumerate(low) if c == "event"]
                    dcol = [j for j, c in enumerate(low) if c.startswith("duration")]
                    assert len(tcol) == len(ecol) == len(dcol) == 1, \
                        f"Missing or ambiguous Time/Event/Duration columns: {f}"
                    header, tcol, ecol, dcol = k, tcol[0], ecol[0], dcol[0]
                    break
            assert header is not None, f"Scoring column header not found: {f}"
            need = max(tcol, ecol, dcol) + 1
            prev = None
            for k in range(header + 1, len(lines)):
                if not lines[k].strip():
                    continue
                cols = lines[k].split("\t")
                assert len(cols) >= need, f"Incomplete scoring row {k + 1}: {f}"
                a = _clock_time(cols[tcol], base, start, stop, prev)
                prev = a
                d = float(cols[dcol])
                assert np.isfinite(d) and d >= 0, f"Invalid event duration at row {k + 1}: {f}"
                _append(ev, st, a, a + np.int64(round(d * 1e9)) * NS, cols[ecol].strip().upper())

    elif rec.format == "EDF":
        stem = Path(rec.source_files[0]).stem
        base = start.astype("datetime64[D]").astype("datetime64[ns]")
        f = folder / f"{stem}.txt"
        if f.is_file():
            count_available, source = True, str(f)
            for line in _read_text(f, "utf-8").splitlines():
                cols = line.split("\t")
                if len(cols) < 6 or cols[0].strip().lower() in \
                        ("time", "time [hh:mm:ss]", "start", "start time", "onset"):
                    continue
                name = cols[5].strip().upper()
                if not name:
                    continue
                a = _clock_time(cols[0], base, start, stop)
                parts = [float(x) for x in cols[4].split(":")]
                assert len(parts) <= 3, "Invalid EDF event duration."
                d = sum(p * 60 ** (len(parts) - 1 - i) for i, p in enumerate(parts))
                _append(ev, st, a, a + np.int64(round(d * 1e9)) * NS, name)
        else:
            warnings.warn("Matching EDF event TXT not found; no event shading.")
        f = folder / f"{stem}.csv"
        if f.is_file():
            times, levels = [], []
            for line in _read_text(f, "utf-8").splitlines()[1:]:
                cols = line.split(",")
                if len(cols) < 3:
                    continue
                levels.append(_stage_level(cols[2].strip().replace('"', "")))
                times.append(_clock_time(cols[1], base, start, stop))
            if len(times) >= 2:
                interval = np.median(np.diff(np.array(times)) / NS)
                assert interval > 0, "Invalid stage order."
                ends = times[1:] + [min(times[-1] + np.int64(interval) * NS, stop)]
                st = [(a, b, lv) for a, b, lv in zip(times, ends, levels)]
    return _finish(ev, st, count_available, source, start, stop)


def _finish(ev, st, count_available, source, start, stop):
    events = pd.DataFrame(ev, columns=["StartTime", "EndTime", "DurationSeconds", "EventType", "Group"])
    stages = pd.DataFrame(st, columns=["StartTime", "EndTime", "Level"])
    for df in (events, stages):
        for c in ("StartTime", "EndTime"):
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    # 전체 판독 기준 개수 (표시용 앞부분 제외나 기록 범위 자르기 전)
    counts = pd.DataFrame({"EventType": COUNT_NAMES,
                           "Count": [float((events.EventType == n).sum()) if count_available else np.nan
                                     for n in COUNT_NAMES]})
    counts.attrs = {"Scope": "Full exported scoring; no first-minute exclusion", "Source": source}
    events = events.sort_values("StartTime", kind="stable").reset_index(drop=True)
    stages = stages.sort_values("StartTime", kind="stable").reset_index(drop=True)
    # 판독이 원신호보다 조금 길 수 있습니다. 겹치는 구간만 남깁니다.
    outside = (events.EndTime < start) | (events.StartTime > stop)
    if outside.any():
        warnings.warn(f"Excluded {int(outside.sum())} events outside signal coverage.")
    events = events[~outside].reset_index(drop=True)
    stages = stages[~((stages.EndTime < start) | (stages.StartTime > stop))].reset_index(drop=True)
    print(f"Scoring: {len(events)} events, {len(stages)} sleep-stage intervals.")
    return events, stages, counts


def _append(ev, st, a, b, name):
    if name.startswith("SLEEP-"):
        st.append((a, b, _stage_level(name)))
        return
    # 같은 의미의 라벨만 통일하고, 나머지 종류는 그대로 둡니다.
    name = {"RERA": "AROUSAL-RERA", "AROUSAL-RESPIRATORY": "AROUSAL-RESP"}.get(name, name)
    if "HYPOPNEA" in name:
        group = "HYPOPNEA"
    elif "APNEA" in name:
        group = "APNEA"
    elif name.startswith("AROUSAL"):
        group = "AROUSAL"
    else:
        group = "OTHER"
    ev.append((a, b, (b - a) / np.timedelta64(1, "s"), name, group))


def _read_nox_markers(db):
    """NOX export는 revision 방식 또는 단일 temporary scoring group 방식입니다."""
    q = lambda sql: db.execute(sql).fetchall()
    names = {r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table'")}
    if {"scoring_revision", "scoring_marker"} <= names:
        revisions = [r[0] for r in q("SELECT id FROM scoring_revision WHERE is_deleted=0")]
        assert revisions, "No active NOX scoring revision."
        revision = []
        if len(revisions) == 1:
            revision = revisions
        elif {"temporary_scoring_group", "temporary_scoring_group_to_key"} <= names:
            groups = [r[0] for r in q("SELECT id FROM temporary_scoring_group WHERE is_deleted=0")]
            if len(groups) == 1:
                current = {r[0] for r in q(
                    f"SELECT key_id FROM temporary_scoring_group_to_key WHERE group_id={groups[0]}")}
                for rid in revisions:
                    keys = {r[0] for r in q(
                        f"SELECT key_id FROM scoring_revision_to_key WHERE revision_id={rid}")}
                    if current and current == keys:
                        revision.append(rid)
        assert len(revision) == 1, \
            "Cannot uniquely match NOX revisions to the current scoring group. Explicit review required."
        table = "scoring_marker"
        keys_sql = f"SELECT key_id FROM scoring_revision_to_key WHERE revision_id={revision[0]}"
        source = f"Data.ndb: scoring_revision {revision[0]}"
    elif {"temporary_scoring_marker", "temporary_scoring_group", "temporary_scoring_group_to_key"} <= names:
        groups = [r[0] for r in q("SELECT id FROM temporary_scoring_group WHERE is_deleted=0")]
        assert len(groups) == 1, "Expected one active NOX temporary scoring group."
        table = "temporary_scoring_marker"
        keys_sql = f"SELECT key_id FROM temporary_scoring_group_to_key WHERE group_id={groups[0]}"
        source = f"Data.ndb: temporary_scoring_group {groups[0]}"
    else:
        raise ValueError("No supported NOX scoring tables found.")
    rows = q(f"SELECT CAST(m.starts_at AS TEXT), CAST(m.ends_at AS TEXT), m.type FROM {table} m "
             f"WHERE m.is_deleted=0 AND m.key_id IN ({keys_sql}) ORDER BY m.starts_at")
    print(f"Scoring source: {source}")
    return rows, source


def _stage_level(name) -> float:
    name = str(name).upper()
    for level, group in enumerate(STAGE_GROUPS):
        if name in group:
            return float(level)
    return np.nan


def _ticks_time(value) -> np.datetime64:
    text = str(value)
    assert text.isdigit(), "Invalid ticks."
    return np.datetime64((int(text) - DOTNET_EPOCH_TICKS) * 100, "ns")


def _clock_time(text, base, start, stop, previous=None) -> np.datetime64:
    """'오후 9:43:32' 같은 시각을 기록 날짜에 맞춰 datetime으로 복원."""
    text = str(text)
    m = re.search(r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    assert m, f"Cannot parse time: {text}"
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    if "오후" in text or "PM" in text:
        h = h % 12 + 12
    elif "오전" in text or "AM" in text:
        h = h % 12
    offset = np.int64(round((h * 3600 + mi * 60 + s) * 1e9)) * NS
    day = np.timedelta64(86_400_000_000_000, "ns")
    cands = np.array([base + k * day + offset for k in range(-1, 4)])
    five = np.timedelta64(300, "s")
    inside = (cands >= start - five) & (cands <= stop + five)
    # 기록 종료 후에도 stage 행이 이어질 수 있습니다. 앞 행과 연속되는 시각이 하나일 때만 복원.
    if not inside.any() and previous is not None:
        cont = (cands >= previous) & (cands < previous + np.timedelta64(12, "h"))
        if cont.sum() == 1:
            return cands[cont][0]
    assert inside.sum() == 1, f"Scoring time outside or ambiguous: {text}"
    return cands[inside][0]


def _read_text(file, encoding) -> str:
    b = Path(file).read_bytes()
    if b[:3] == b"\xef\xbb\xbf":
        return b[3:].decode("utf-8")
    if b[:2] == b"\xff\xfe":
        return b[2:].decode("utf-16-le")
    if b[:2] == b"\xfe\xff":
        return b[2:].decode("utf-16-be")
    return b.decode(encoding, errors="replace")
