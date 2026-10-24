"""PSG 원본(EBM/NDF/EDF) 읽기. Matlab/Function/PSG_Read.m, PSG_readers/*, EbmRead.m 변환.

모든 형식은 같은 Channel 구조로 반환합니다. 원본 이름/단위를 유지하고,
구간(segment)별 실제 시각을 보존하며 빈 구간을 0으로 채우지 않습니다.
"""
from __future__ import annotations

import re
import struct
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

NS = np.timedelta64(1, "ns")


@dataclass
class Channel:
    name: str
    data: np.ndarray            # float64
    time: np.ndarray            # datetime64[ns]
    fs: float
    unit: str
    source_file: str
    segments: pd.DataFrame      # FirstSample(1-based), Samples, StartTime, Fs
    metadata: dict = field(default_factory=dict)


@dataclass
class Recording:
    format: str                 # 'EBM' | 'NDF' | 'EDF'
    source_folder: str
    source_files: list[str]
    channels: list[Channel]
    channel_table: pd.DataFrame
    annotations: pd.DataFrame | None = None
    excluded_channels: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["SourceFile", "Reason"]))
    created_at: datetime = field(default_factory=datetime.now)
    schema_version: str = "PSG-1.0"


def segment_times(start: np.datetime64, n: int, fs: float) -> np.ndarray:
    """start + (0:n-1)/fs 초. MATLAB seconds((0:n-1)'/fs)와 같은 식."""
    return np.datetime64(start, "ns") + np.round(np.arange(n) / fs * 1e9).astype("int64") * NS


def _segments_table(counts, starts, fs) -> pd.DataFrame:
    counts = np.asarray(counts, dtype=np.int64)
    first = np.concatenate([[1], np.cumsum(counts)[:-1] + 1]) if len(counts) else counts
    fs = np.broadcast_to(np.asarray(fs, dtype=float), counts.shape)
    return pd.DataFrame({"FirstSample": first, "Samples": counts,
                         "StartTime": np.asarray(starts, dtype="datetime64[ns]"), "Fs": fs})


# ---------------------------------------------------------------- EBM (Embla)
def read_ebm(file) -> Channel:
    """EbmRead.m + PSG_read_ebm.m. 검증된 little-endian wide-ID 형식만 지원.

    EbmRead는 블록 사이를 0으로 채운 뒤 PSG_read_ebm이 다시 제거하므로,
    결과는 실제 데이터 블록을 이어붙인 값 * unit gain 과 같습니다.
    """
    file = str(file)
    b = Path(file).read_bytes()
    end = b.find(b"\x1a", 0, 80)
    assert end >= 0 and b"Embla" in b[:end], f"EBM 서명 오류: {file}"
    assert b[end + 1] == 0, "현재 검증된 EBM은 little-endian wide-ID 형식입니다."
    offset = end + 2
    assert b[offset:offset + 5] == b"\xff" * 5, "검증되지 않은 EBM short-ID 형식"
    pos = offset + 31

    h = {"physicalunit": "", "outputunit": "EbmRead-scaled (verify unit)"}
    starts, counts, blocks = [], [], []
    stamp, fresh, trailing = None, False, 0
    fs = None
    while pos < len(b):
        rid, ln = struct.unpack_from("<II", b, pos)
        pos += 8
        # 일부 파일은 마지막 센서 블록 뒤를 0xFF로 채웁니다. 남은 영역이 전부 FF일 때만 패딩.
        if rid == 0xFFFFFFFF and ln == 0xFFFFFFFF:
            tail = b[pos:]
            assert tail.count(0xFF) == len(tail), f"손상되었거나 지원하지 않는 EBM 끝 영역: {file}"
            trailing = 8 + len(tail)
            break
        assert pos + ln <= len(b), f"잘린 EBM 블록: {file}"
        blk = b[pos:pos + ln]
        if rid == 0x80:
            minor, major = struct.unpack_from("<bb", blk)
            h["fileversion"] = major + 0.01 * minor
        elif rid == 0xD0:
            h["subjectinfo"] = _deblank(blk)
        elif rid == 0x81:
            h["extra"] = _deblank(blk)
        elif rid == 0x84:
            y = struct.unpack_from("<h", blk)[0]
            mo, d, hh, mi, s, hs = blk[2:8]
            stamp = np.datetime64(datetime(y, mo, d, hh, mi, s), "ns") + np.int64(hs) * 10_000_000 * NS
            fresh = True
        elif rid == 0x85:
            h["channel"] = struct.unpack_from("<h", blk)[0]
        elif rid == 0x86:
            rate = struct.unpack_from("<i", blk)[0] / 1000   # mHz
            assert fs is None or rate == fs, "변동 주파수 EBM은 지원하지 않습니다."
            fs = h["samplingrate"] = rate
        elif rid == 0x87:
            h["rawunitgain"] = struct.unpack_from("<i", blk)[0]
        elif rid == 0x90:
            h["channelname"] = _deblank(blk)
        elif rid == 0x99:
            h["physicalunit"] = blk.replace(b"\x00", b"").decode("latin1").strip()
        elif rid == 0x20:
            assert fresh and ln % 2 == 0, "EBM 시각/데이터 블록 관계를 확인하세요."
            blocks.append(np.frombuffer(blk, "<i2"))
            starts.append(stamp)
            counts.append(ln // 2)
            fresh = False
        pos += ln
    assert counts, f"EBM 센서 데이터 없음: {file}"

    gain = h["rawunitgain"] * 1e-9   # 기존 배율 유지 (검증된 SpO2만 아래에서 예외)
    if h["channelname"].strip().lower().startswith("spo2") and h["physicalunit"] == "%":
        assert h["rawunitgain"] == 1, f"Unverified SpO2 percent gain {h['rawunitgain']} in {file}"
        gain, h["outputunit"] = 1.0, "%"
    elif h["physicalunit"] == "V":
        h["outputunit"] = "V"
    h["unitgain"] = gain
    h["scalingVersion"] = "SpO2-percent-1"
    h["starttime"] = str(starts[0])

    data = np.concatenate(blocks).astype(np.float64) * gain
    time = np.concatenate([segment_times(s, n, fs) for s, n in zip(starts, counts)])
    meta = {"header": h, "unitStatus": h["outputunit"], "scalingVersion": h["scalingVersion"],
            "removedPaddingSamples": _ebm_padding(starts, counts, fs),
            "trailingPaddingBytes": trailing,
            "timePolicy": "Original record timestamps; no zero-filled gap samples"}
    return Channel(h["channelname"], data, time, fs, h["outputunit"], file,
                   _segments_table(counts, starts, fs), meta)


def _deblank(blk: bytes) -> str:
    return blk.decode("latin1").rstrip("\x00 \t\r\n")


def _ebm_padding(starts, counts, fs) -> int:
    """EbmRead가 블록 사이에 넣었을 0 샘플 수 (기록용 메타데이터)."""
    total, prev_len = 0, counts[0]
    for k in range(1, len(starts)):
        gap = (starts[k] - starts[k - 1]) / np.timedelta64(1, "s") - prev_len / fs
        fill = int(round(max(0.0, gap * fs)))
        total += fill
        prev_len = fill + counts[k]
    return total


# ---------------------------------------------------------------- NDF (Nox)
def read_ndf(file) -> Channel:
    """PSG_read_ndf.m. Nox NOX03 NDF. 참고: https://github.com/jussivirkkala/Noxturnal-NDF"""
    file = str(file)
    b = Path(file).read_bytes()
    assert b[:4] == bytes([78, 79, 88, 3]), f"지원하지 않는 NDF 서명: {file}"
    pos = 4
    chunks, times, seg_start, seg_count, seg_fs, unknown = [], [], [], [], [], set()
    stamp, fs, xml = None, np.nan, ""
    dtypes = {"Int16": "<i2", "Int32": "<i4", "UInt32": "<u4", "Double": "<f8", "Byte": "u1"}
    while pos < len(b):
        typ, ln = struct.unpack_from("<HI", b, pos)
        pos += 6
        assert pos + ln <= len(b), f"잘린 NDF 블록: {file}"
        blk = b[pos:pos + ln]
        if typ == 256:      # UTF-16 XML channel header
            xml = blk.decode("utf-16-le").replace("\x00", "")
            label, unit, encoding = _tag(xml, "Label"), _tag(xml, "Unit"), _tag(xml, "Format")
            gain, off = float(_tag(xml, "Scale")), float(_tag(xml, "Offset"))
            nominal_fs = fs = float(_tag(xml, "SamplingRate"))
            assert np.isfinite(gain) and np.isfinite(off) and fs > 0, "잘못된 NDF 환산/주파수"
            assert not re.search(r"<Key>Data_Encryption</Key>.*?<Value>[1-9]\d*</Value>", xml, re.S), \
                "암호화 NDF는 지원하지 않습니다. 제조사에서 EDF로 내보내세요."
        elif typ == 512:    # YYYYMMDDTHHMMSS.ffffff
            text = blk.decode("utf-16-le").replace("\x00", "")
            stamp = np.datetime64(datetime.strptime(text, "%Y%m%dT%H%M%S.%f"), "ns")
        elif typ == 514:    # 구간별 정밀 샘플링 주파수
            assert ln == 8, "잘못된 NDF 주파수 블록"
            fs = struct.unpack("<d", blk)[0]
        elif typ == 513:    # 디지털 신호 블록
            assert xml and stamp is not None and np.isfinite(fs) and fs > 0, "NDF 메타데이터 누락"
            dt = dtypes.get(encoding)
            assert dt, f"지원하지 않는 NDF 샘플 형식: {encoding}"
            width = np.dtype(dt).itemsize
            assert ln % width == 0, "불완전한 NDF 샘플"
            raw = np.frombuffer(blk, dt).astype(np.float64)
            if raw.size:
                chunks.append(raw * gain + off)
                times.append(segment_times(stamp, raw.size, fs))
                seg_start.append(stamp); seg_count.append(raw.size); seg_fs.append(fs)
                stamp = stamp + np.int64(round(raw.size / fs * 1e9)) * NS
        else:
            unknown.add(typ)
        pos += ln
    assert chunks, f"NDF 데이터 블록 없음: {file}"
    meta = {"headerXML": xml, "nominalFs": nominal_fs, "scale": gain, "offset": off,
            "encoding": encoding, "unitStatus": "from NDF XML", "otherBlockTypes": sorted(unknown),
            "timePolicy": "Segment timestamps + precise rate block 514; no invented gap samples",
            "readerReference": "https://github.com/jussivirkkala/Noxturnal-NDF"}
    fs0 = seg_fs[0] if all(abs(f - seg_fs[0]) <= 1e-9 for f in seg_fs) else np.nan
    return Channel(label, np.concatenate(chunks), np.concatenate(times), fs0, unit, file,
                   _segments_table(seg_count, seg_start, seg_fs), meta)


def _tag(xml: str, name: str) -> str:
    m = re.search(f"<{name}>(.*?)</{name}>", xml, re.S)
    v = m.group(1) if m else ""
    return v.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


# ---------------------------------------------------------------- EDF / EDF+
def read_edf(file) -> tuple[list[Channel], pd.DataFrame]:
    """PSG_read_edf.m. EDF/EDF+ (C/D). EDF+D는 record 시작 시각(TAL)으로 단절을 보존.

    ponytail: 현재 Data/에 EDF가 없어 실데이터 검증은 안 됨. EDF가 생기면 MATLAB edfread와 대조할 것.
    """
    file = str(file)
    b = Path(file).read_bytes()
    a = lambda s, e: b[s:e].decode("latin1").strip()
    d, t = a(168, 176), a(176, 184)
    dd, mm, yy = (int(x) for x in d.split("."))
    start = datetime(1900 + yy if yy >= 85 else 2000 + yy, mm, dd, *(int(x) for x in t.split(".")))
    start = np.datetime64(start, "ns")
    header_bytes, nrec = int(a(184, 192)), int(a(236, 244))
    rec_dur, ns = float(a(244, 252)), int(a(252, 256))
    cursor = 256

    def fields(width):   # 신호 헤더는 필드별로 ns개씩 연속 저장됩니다.
        nonlocal cursor
        out = [a(cursor + i * width, cursor + (i + 1) * width) for i in range(ns)]
        cursor += width * ns
        return out
    labels, trans, dim = fields(16), fields(80), fields(8)
    pmin, pmax, dmin, dmax = ([float(x) for x in fields(8)] for _ in range(4))
    pre, nsamp = fields(80), [int(x) for x in fields(8)]
    rec_len = sum(nsamp)
    raw = np.frombuffer(b, "<i2", count=nrec * rec_len, offset=header_bytes).reshape(nrec, rec_len)
    bounds = np.concatenate([[0], np.cumsum(nsamp)])
    is_ann = [lab == "EDF Annotations" for lab in labels]

    rec_start = start + np.round(np.arange(nrec) * rec_dur * 1e9).astype("int64") * NS
    ann_rows = []
    if any(is_ann):
        k = is_ann.index(True)
        for r in range(nrec):
            tals = raw[r, bounds[k]:bounds[k + 1]].tobytes().split(b"\x00")
            first = True
            for tal in tals:
                if not tal:
                    continue
                parts = tal.split(b"\x14")
                head = parts[0].split(b"\x15")
                onset = float(head[0])
                dur = float(head[1]) if len(head) > 1 and head[1] else 0.0
                texts = [p.decode("utf-8", "replace") for p in parts[1:] if p]
                if first:
                    rec_start[r] = start + np.int64(round(onset * 1e9)) * NS   # 시간 유지 TAL
                    first = False
                for txt in texts:
                    ann_rows.append({"Onset": start + np.int64(round(onset * 1e9)) * NS,
                                     "Annotations": txt, "Duration": dur})
    channels = []
    for k in range(ns):
        if is_ann[k]:
            continue
        fs = nsamp[k] / rec_dur
        dig = raw[:, bounds[k]:bounds[k + 1]].astype(np.float64)
        phys = (dig - dmin[k]) * (pmax[k] - pmin[k]) / (dmax[k] - dmin[k]) + pmin[k]
        time = np.concatenate([segment_times(s, nsamp[k], fs) for s in rec_start])
        meta = {"unitStatus": "from EDF header", "transducer": trans[k], "physicalMin": pmin[k],
                "physicalMax": pmax[k], "digitalMin": dmin[k], "digitalMax": dmax[k],
                "prefilter": pre[k], "timePolicy": "EDF record timestamps; original sample rates"}
        channels.append(Channel(labels[k], phys.ravel(), time, fs, dim[k], file,
                                _segments_table([nsamp[k]] * nrec, rec_start, fs), meta))
    return channels, pd.DataFrame(ann_rows, columns=["Onset", "Annotations", "Duration"])


# ---------------------------------------------------------------- 폴더 단위
def read_psg(input_path) -> Recording:
    """PSG_Read.m. 폴더(EBM/NDF 채널 파일들) 또는 EDF 파일 하나를 읽습니다.

    MATLAB 버전의 waitbar/저장 대화상자/MAT 캐시는 옮기지 않았습니다 (저장은 batch.py).
    """
    p = Path(input_path)
    if p.is_dir():
        files = [f for f in p.iterdir() if f.is_file()]
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(f"폴더/파일을 찾을 수 없습니다: {p}")
    files = [f for f in files if f.suffix.lower() in (".ebm", ".ndf", ".edf")]
    formats = {f.suffix.lower() for f in files}
    if not formats:
        raise ValueError("이 폴더에 EBM/NDF/EDF가 없습니다. 기록 폴더를 선택하세요.")
    if len(formats) > 1:
        raise ValueError("서로 다른 형식이 섞여 있습니다. 형식별 폴더 또는 EDF 파일을 선택하세요.")
    fmt = formats.pop()[1:].upper()
    if fmt == "EDF" and len(files) > 1:
        raise ValueError("EDF 파일이 여러 개입니다. 읽을 EDF 파일 경로를 지정하세요.")
    files.sort(key=lambda f: f.name.lower())
    folder = files[0].parent
    rec = Recording(fmt, str(folder), [str(f) for f in files], [], pd.DataFrame())

    if fmt == "EDF":
        rec.channels, rec.annotations = read_edf(files[0])
    else:
        channels = [read_ebm(f) if fmt == "EBM" else read_ndf(f) for f in files]
        # 다른 날 검사의 채널이 섞인 폴더가 있습니다 (현재 P_23: 2018-10-30 기록에 2019-03-21 채널 7개).
        # MATLAB은 옛 번호 'P_43_PSG_E'로 고정해 걸렀지만 번호가 바뀌어 동작하지 않았으므로,
        # 채널 시작 시각이 전체 중앙값과 12시간 넘게 다르면 제외하는 규칙으로 바꿨습니다.
        starts = np.array([c.time[0] for c in channels])
        median = np.sort(starts)[len(starts) // 2]
        excluded = []
        for ch in channels:
            if abs(ch.time[0] - median) > np.timedelta64(12, "h"):
                reason = (f"Different recording date ({ch.time[0].astype('datetime64[D]')}); "
                          f"recording is {median.astype('datetime64[D]')}.")
                excluded.append({"SourceFile": ch.source_file, "Reason": reason})
                warnings.warn(f"{ch.source_file}: {reason}")
                continue
            rec.channels.append(ch)
        rec.excluded_channels = pd.DataFrame(excluded, columns=["SourceFile", "Reason"])

    for ch in rec.channels:
        assert ch.data.size == ch.time.size, f"신호/시간 길이가 다릅니다: {ch.name}"
        assert ch.data.size and not np.isnat(ch.time).any(), f"빈 채널 또는 잘못된 시각: {ch.name}"
    rec.channel_table = pd.DataFrame({
        "Index": np.arange(1, len(rec.channels) + 1),
        "Name": [c.name for c in rec.channels], "Fs": [c.fs for c in rec.channels],
        "Unit": [c.unit for c in rec.channels], "Samples": [c.data.size for c in rec.channels],
        "StartTime": [c.time[0] for c in rec.channels], "EndTime": [c.time[-1] for c in rec.channels]})
    print(f"{fmt}: {len(rec.channels)}개 채널, 총 {int(rec.channel_table.Samples.sum())}개 샘플")
    return rec
