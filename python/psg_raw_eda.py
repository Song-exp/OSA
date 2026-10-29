"""PSG 원신호(EBM/NDF) 기초 EDA. docs/PSG_원신호_기초_EDA_및_파형_이해.md용.

87명 모든 채널을 읽어 구조(채널, 샘플링, 단위, 길이)를 정리하고,
주요 채널의 품질 의심 구간(미기록, 평탄, 범위 끝 고정, 범위 밖 값, 한쪽만 신호 약함)을 셉니다.
값을 지우거나 고치지 않고 '의심'으로 표시만 합니다. 원본과 기존 코드는 읽기만 합니다.

결과 (Result_py/psg_raw_eda/):
  raw_eda_channels.csv     환자 × 모든 채널 현황
  raw_eda_quality.csv      환자 × 주요 채널 품질
  raw_eda_channel_map.csv  장비별 채널 대응표
  raw_eda_time_info.csv    환자별 시간 정보·판독 연결 가능성
  raw_eda_review_list.csv  추가 확인이 필요한 환자 목록
사용: python psg_raw_eda.py [환자번호 ...]   (전체 약 20~30분)
"""
from __future__ import annotations

import re
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from check_missing import scoring_info
from osa.psg_events import DOTNET_EPOCH_TICKS, _stage_level, read_events
from osa.psg_plot import display_unit
from osa.psg_read import read_psg
from psg_waveform_check import nox_sleep_and_events

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py" / "psg_raw_eda"
SEC = np.timedelta64(1, "s")

# 주요 채널: (기능 묶음, 후보 이름들 — 앞에 있는 것 우선). 소문자로 비교, 파일 이름과 채널 이름 모두 봄
KEY = [
    ("SpO2", ["spo2", "spo2_pds"]),
    ("NasalPressure", ["nasal pressure", "nasal_pds"]),
    ("Thermistor", ["thermistor"]),
    ("Thorax", ["thorax", "thorax rip", "thorax_pds"]),
    ("Abdomen", ["abdomen", "abdomen rip", "abdomen_pds"]),
    ("EEG", ["c3", "c4", "eeg"]),
    ("EOG", ["loc", "e1", "eog left", "eog up", "eog r"]),
    ("ChinEMG", ["chin", "emg.subm", "1"]),          # Nox '1'은 기존 check_missing.py 기준 턱 근전도
    ("ECG", ["ekg", "ecg"]),
    ("Position", ["position", "position_dr", "position_cu", "position_pds"]),
    ("Snore", ["snore", "snore_dr", "snore_cu", "snore_pds", "snoring sensor"]),
]
# 같은 값이 이 시간 이상 그대로면 '평탄'. SpO2는 1% 단위라 안정 수면에서도 오래 같을 수 있어 길게 잡음
FLAT_SEC = {"SpO2": 300, "Position": None}
FLAT_DEFAULT = 10
LOWAMP_MIN = 5          # 호흡 채널: 1분 표준편차가 밤 중앙값의 10% 미만인 상태가 5분 이상 → 신호 약함
PAIRS = {"NasalPressure": "Thermistor", "Thermistor": "NasalPressure", "Thorax": "Abdomen", "Abdomen": "Thorax"}


def pick(rec) -> dict:
    """기능 묶음 → 채널. 파일 이름(확장자 제외)과 채널 이름 모두로 찾음."""
    names = {}
    for c in rec.channels:
        for n in {c.name.strip().lower(), Path(c.source_file).stem.strip().lower()}:
            names.setdefault(n, c)
    return {g: next(names[n] for n in cands if n in names) for g, cands in KEY if any(n in names for n in cands)}


def nox_stages(folder: Path) -> tuple[pd.DataFrame, int]:
    """Nox 수면단계. 활성 판독층에는 같은 30초에 Automatic과 Manual 단계가 함께 있어 기사(Manual) 것을 우선함
    (psg_reconcile.nox_counts와 같은 규칙). 반환: (단계표, 두 층의 단계가 서로 다른 30초 칸 수)."""
    db = sqlite3.connect(f"file:{(folder / 'Data.ndb').as_posix()}?mode=ro", uri=True)
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    live = {}
    for pre in ("scoring", "temporary_scoring"):
        if f"{pre}_marker" in tabs:
            m = pd.read_sql(f"select CAST(m.starts_at AS TEXT) a, CAST(m.ends_at AS TEXT) b, m.type, k.type kt "
                            f"from {pre}_marker m join {pre}_key k on m.key_id = k.id where m.is_deleted = 0", db)
            if len(m):
                live[pre] = m.assign(a=m.a.astype("int64"), b=m.b.astype("int64"), type=m.type.astype(str))
    db.close()
    if not live:
        return pd.DataFrame(columns=["StartTime", "EndTime", "Level"]), 0
    m = live[max(live, key=lambda k: (live[k].kt == "Manual").sum())]
    st = m[m.type.str.startswith("sleep-")]
    conflict = int((st.groupby("a").type.nunique() > 1).sum())
    st = st.sort_values("kt").drop_duplicates("a", keep="last").sort_values("a")
    t = lambda v: (v.values - DOTNET_EPOCH_TICKS) * 100
    out = pd.DataFrame({"StartTime": t(st.a).astype("datetime64[ns]"), "EndTime": t(st.b).astype("datetime64[ns]"),
                        "Level": [_stage_level(x) for x in st.type.str.upper()]})
    return out, conflict


def spans_mask(n, t0, spans) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in spans:
        i0 = int((np.datetime64(a, "ns") - t0) // SEC)
        i1 = int((np.datetime64(b, "ns") - t0) // SEC)
        m[max(i0, 0):max(min(i1, n), 0)] = True
    return m


def flat_runs(x: np.ndarray, min_len: int) -> np.ndarray:
    """같은 값이 min_len 샘플 이상 이어지는 샘플 표시."""
    out = np.zeros(x.size, bool)
    if x.size < 2 or not min_len:
        return out
    change = np.flatnonzero(np.diff(x) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, x.size]
    for a, b in zip(starts[(ends - starts) >= min_len], ends[(ends - starts) >= min_len]):
        out[a:b] = True
    return out


def low_amp_seconds(x, sec_idx, n, sleep) -> np.ndarray:
    """1분 창 표준편차가 수면 중 중앙값의 10% 미만인 분이 LOWAMP_MIN분 이상 이어지면 그 시간(초) 표시."""
    minute = sec_idx // 60
    nm = n // 60 + 1
    cnt = np.bincount(minute, minlength=nm)
    s1 = np.bincount(minute, x, minlength=nm)
    s2 = np.bincount(minute, x * x, minlength=nm)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0))
    sleep_min = np.zeros(nm, bool)
    sleep_min[np.flatnonzero(sleep) // 60] = True
    ref = np.nanmedian(sd[sleep_min & (cnt > 0)]) if (sleep_min & (cnt > 0)).any() else np.nan
    low = (sd < 0.1 * ref) & (cnt > 0)
    keep = flat_runs(low.astype(np.int8), LOWAMP_MIN) & low
    out = np.repeat(keep, 60)[:n]
    return np.r_[out, np.zeros(n - out.size, bool)]


def quality(group, ch, t0, n, sleep, info) -> tuple[dict, np.ndarray, np.ndarray]:
    """채널 하나의 품질. 반환: (요약, 초별 '기록 있음', 초별 '의심')."""
    x = ch.data
    # Nox는 구간마다 정밀 주파수가 조금씩 달라 ch.fs가 NaN인 채널이 있음 → 구간 주파수의 중앙값 사용
    fs = ch.fs if np.isfinite(ch.fs) else float(np.median(ch.segments.Fs))
    idx = ((ch.time - t0) // SEC).astype("int64")
    ok = (idx >= 0) & (idx < n)
    x, idx = x[ok], idx[ok]
    cnt = np.bincount(idx, minlength=n)
    covered = cnt > 0
    fin = np.isfinite(x)
    flags = {"NonFinite": ~fin}
    fl = FLAT_SEC.get(group, FLAT_DEFAULT)
    if fl:
        flags["Flat"] = flat_runs(x, int(round(fl * fs)))
    if group == "SpO2":
        flags["Zero"] = fin & (x == 0)
        flags["Below50"] = fin & (x > 0) & (x < 50)
        flags["Above100"] = fin & (x > 100)
    elif group != "Position" and fin.any():
        lo, hi = np.nanmin(x), np.nanmax(x)
        at = fin & ((x == lo) | (x == hi))
        flags["Rail"] = at if at.mean() >= 0.001 else np.zeros(x.size, bool)   # 0.1% 이상 끝값에 붙어 있을 때만
    r = {"Patient": info["Patient"], "Device": info["Device"], "Group": group, "Channel": ch.name,
         "Fs": round(fs, 3), "Unit": display_unit(ch) or "(미확인)"}
    sus = np.zeros(x.size, bool)
    for k, f in flags.items():
        r[f"{k}_Min"] = round(f.sum() / fs / 60, 2)
        if not (group == "SpO2" and k == "Flat"):     # SpO2 평탄은 안정 수면일 수 있어 참고로만 셈
            sus |= f
    sus_sec = np.bincount(idx, sus, minlength=n) >= np.maximum(cnt, 1) * 0.5
    sus_sec &= covered
    if group == "SpO2":
        # 1초 중앙값이 바로 앞 초보다 10%p 이상 변함: 생리적으로 1초 안에 일어나기 어려움 → 횟수만 셈
        med = pd.Series(x).groupby(idx).median()
        d = med.reindex(range(n)).diff().abs()
        r["Jump10_Count"] = int((d >= 10).sum())
        # 50% 미만이 연속 몇 초인지: 짧으면 잡음, 길면 실제 심한 저산소일 수도 → 구분용
        low = np.zeros(n, bool)
        low[np.unique(idx[flags["Below50"]])] = True
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0])).reshape(-1, 2), axis=1).ravel()
        r["Below50_LongestSec"] = int(runs.max()) if runs.size else 0
    if group in PAIRS:
        la = low_amp_seconds(np.where(fin, x, 0), idx, n, sleep)
        r["LowAmp_Min"] = round((la & sleep).sum() / 60, 1)
        info.setdefault("_lowamp", {})[group] = la & sleep
    usable = covered & ~sus_sec
    r["NotRecorded_Min_All"] = round((~covered).sum() / 60, 1)
    r["Usable_Pct_All"] = round(100 * usable.mean(), 1)
    if sleep.any():
        r["NotRecorded_Min_Sleep"] = round((~covered & sleep).sum() / 60, 1)
        r["Suspect_Min_Sleep"] = round((sus_sec & sleep).sum() / 60, 1)
        r["Usable_Pct_Sleep"] = round(100 * (usable & sleep).sum() / sleep.sum(), 1)
        xs = x[fin & sleep[idx]]
        if xs.size:
            r["Sleep_P1"], r["Sleep_Median"], r["Sleep_P99"] = (float(v) for v in np.percentile(xs, [1, 50, 99]))
    return r, covered, sus_sec


def patient(folder: Path) -> tuple[list, list, dict]:
    pid = re.match(r"(P_\d+)", folder.name).group(1)
    dev = "E" if folder.name.endswith("_E") else "N"
    rec = read_psg(folder)
    ct = rec.channel_table
    t0 = ct.StartTime.min().to_datetime64()
    t1 = ct.EndTime.max().to_datetime64()
    n = int((t1 - t0) // SEC) + 1
    # 모든 채널 현황
    inv = []
    for c in rec.channels:
        span = (c.time[-1] - c.time[0]) / SEC
        fs = c.fs if np.isfinite(c.fs) else float(np.median(c.segments.Fs))
        inv.append({"Patient": pid, "Device": dev, "Channel": c.name, "File": Path(c.source_file).name,
                    "Fs": round(fs, 3),
                    "FsVaries": bool(not np.isfinite(c.fs)), "Unit": display_unit(c) or "(미확인)",
                    "SampleFormat": c.metadata.get("encoding", "미기록(EBM 헤더)"),
                    "Samples": int(c.data.size), "Start": str(c.time[0].astype("datetime64[s]")),
                    "End": str(c.time[-1].astype("datetime64[s]")), "DurationMin": round(span / 60, 1),
                    "Segments": len(c.segments),
                    "TimeGapMin": round(max(0.0, span - c.data.size / fs) / 60, 2),
                    "StartOffsetSec": round((c.time[0] - t0) / SEC, 1), "EndOffsetSec": round((t1 - c.time[-1]) / SEC, 1),
                    "NonFinite": int((~np.isfinite(c.data)).sum()), "Status": "읽음"})
    for _, e in rec.excluded_channels.iterrows():
        inv.append({"Patient": pid, "Device": dev, "File": Path(e.SourceFile).name, "Status": f"제외: {e.Reason}"})
    # 수면 구간
    if dev == "N":
        sleep_spans, used = nox_sleep_and_events(folder)
        ev, _, _ = read_events(rec)
        st, conflict = nox_stages(folder)
        rem_spans = list(zip(st[st.Level == 1].StartTime.values, st[st.Level == 1].EndTime.values))
        stage_src = "Data.ndb (같은 30초에 Automatic·Manual 단계가 함께 있으면 Manual 우선)"
    else:
        ev, st, _ = read_events(rec)
        sl = st[st.Level.between(1, 5)]
        sleep_spans = list(zip(sl.StartTime.values, sl.EndTime.values))
        rem_spans = list(zip(st[st.Level == 1].StartTime.values, st[st.Level == 1].EndTime.values))
        stage_src = "EN-Stage Events.txt"
        conflict = 0
        used = ev[ev.Group.isin(["APNEA", "HYPOPNEA"])]
    sleep = spans_mask(n, t0, sleep_spans)
    rem = spans_mask(n, t0, rem_spans) & sleep
    info = {"Patient": pid, "Device": dev}
    qrows, cov = [], {}
    chans = pick(rec)
    for g, _ in KEY:
        if g not in chans:
            qrows.append({"Patient": pid, "Device": dev, "Group": g, "Channel": "(없음)"})
            continue
        r, c, s = quality(g, chans[g], t0, n, sleep, info)
        qrows.append(r)
        cov[g] = c
    # 한쪽만 신호가 약한 시간 (짝 채널은 정상): 센서 문제 후보
    la = info.get("_lowamp", {})
    for r in qrows:
        g = r["Group"]
        if g in la and PAIRS[g] in la:
            r["LowAmpAlone_Min"] = round((la[g] & ~la[PAIRS[g]]).sum() / 60, 1)
    sc = scoring_info(rec)
    resp = ev[ev.Group.isin(["APNEA", "HYPOPNEA"])]
    t = {"Patient": pid, "Device": dev, "RecStart": str(t0.astype("datetime64[s]")),
         "RecEnd": str(t1.astype("datetime64[s]")), "RecMin": round(n / 60, 1), "Channels": len(rec.channels),
         "ExcludedChannels": len(rec.excluded_channels), "StageSource": stage_src, "StageConflictEpochs": conflict,
         "SleepMin": round(sleep.sum() / 60, 1), "REM_Min": round(rem.sum() / 60, 1),
         "NREM_Min": round((sleep & ~rem).sum() / 60, 1),
         "RespEvents_ActiveLayer": len(resp), "RespEvents_Used": len(used), "RespEvents_HaveEnd": bool(len(resp) and (resp.DurationSeconds > 0).all()),
         "ArousalEventsInFile": int((ev.Group == "AROUSAL").sum()),
         "SpontArousal": sc["SpontArousal"], "DesatEvents": sc["DesatEvents"], "PositionInScoring": sc["Position"],
         "PositionChannel": chans["Position"].name if "Position" in chans else "(없음)",
         "AccelChannels": sum(c.name.lower() in ("x axis", "y axis", "z axis", "gravity x", "gravity y") for c in rec.channels)}
    del rec
    return inv, qrows, t


def safe_patient(d: Path):
    """실패해도 전체를 멈추지 않고 사유를 기록."""
    try:
        return patient(d)
    except Exception as e:
        pid = re.match(r"(P_\d+)", d.name).group(1)
        return [{"Patient": pid, "Status": f"읽기 실패: {type(e).__name__}: {e}"}], [], None


def channel_map(inv: pd.DataFrame) -> pd.DataFrame:
    """기능 묶음별로 장비마다 어떤 이름·주파수·단위로 들어 있는지."""
    rows = []
    ok = inv[inv.Status == "읽음"]
    for g, cands in KEY:
        r = {"Group": g}
        for dev, lab in (("E", "Embla"), ("N", "Nox")):
            d = ok[(ok.Device == dev) & ok.Channel.str.strip().str.lower().isin(cands)]
            tot = inv[inv.Device == dev].Patient.nunique()
            names = d.groupby("Channel").Patient.nunique().sort_values(ascending=False)
            r[f"{lab}_Names"] = ", ".join(f"{k}({v})" for k, v in names.items()) or "(없음)"
            r[f"{lab}_Patients"] = f"{d.Patient.nunique()}/{tot}"
            r[f"{lab}_Fs"] = ", ".join(f"{v:g}" for v in sorted(d.Fs.dropna().round(1).unique())) or "-"
            r[f"{lab}_Unit"] = ", ".join(sorted(d.Unit.dropna().unique())) or "-"
        rows.append(r)
    return pd.DataFrame(rows)


# 자동 지표로는 잡히지 않고 그림으로 확인한 것 (psg_raw_eda_figures.py, psg_guide_figures.py 그림)
VISUAL_NOTES = [("P_59", "N", "Thermistor",
                 "그림 확인: 00:30~00:40, 01:55~01:59 구간에서 호흡 박자가 뚜렷하지 않고 느린 오르내림이 주로 보임. "
                 "04:10~04:14에는 최대값(1.024 V)에 붙음. 자동 지표(평탄·약함)로는 잡히지 않음")]


def review_list(q: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """추가 확인 대상. 사유만 모으고 판정은 하지 않음. 기준:
    주요 채널 없음 / 수면 중 사용 가능 < 95% / 짝 채널은 정상인데 혼자 약함 ≥ 10분 /
    SpO2 같은 값 5분 이상 지속의 합 ≥ 60분 (안정 상태인지 센서 고정인지 구분 불가) /
    REM < 30분 / Nox 수면단계 두 층 불일치 / 엑셀 요약과 다른 검사."""
    rows = []
    add = lambda r, g, why: rows.append({"Patient": r.Patient, "Device": r.Device, "Group": g, "Reasons": why})
    for _, r in q.iterrows():
        if r.Channel == "(없음)":
            add(r, r.Group, "채널 없음")
            continue
        why = []
        if r.get("Usable_Pct_Sleep", 100) < 95:
            why.append(f"수면 중 사용 가능 {r.Usable_Pct_Sleep}%")
        if r.get("LowAmpAlone_Min", 0) >= 10:
            why.append(f"짝 채널은 정상인데 혼자 약함 {r.LowAmpAlone_Min}분")
        if r.Group == "SpO2" and r.get("Flat_Min", 0) >= 60:
            why.append(f"SpO2가 같은 값으로 5분 이상 이어진 시간 합 {r.Flat_Min:.0f}분 (안정 상태인지 센서 고정인지 미확인)")
        if why:
            add(r, r.Group, " / ".join(why))
    for pid, dev, g, why in VISUAL_NOTES:
        rows.append({"Patient": pid, "Device": dev, "Group": g, "Reasons": why})
    for _, r in t.iterrows():
        if r.REM_Min < 30:
            add(r, "Stages", f"REM {r.REM_Min}분 (30분 미만)")
        if r.get("StageConflictEpochs", 0) > 0:
            add(r, "Stages", f"Nox 판독층끼리 수면단계가 다른 30초 칸 {r.StageConflictEpochs}개 (Manual 우선 사용)")
        if r.get("ExamStatus") != "엑셀 요약 = 이 원신호 검사":
            add(r, "Exam", r.ExamStatus)
    out = pd.DataFrame(rows, columns=["Patient", "Device", "Group", "Reasons"])
    return out.sort_values("Patient", key=lambda s: s.str[2:].astype(int), kind="stable")


def main(wanted: set[int]):
    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted((p for p in (ROOT / "Data/PSG").iterdir() if p.is_dir() and re.match(r"P_\d+_", p.name)),
                     key=lambda p: int(p.name.split("_")[1]))
    if wanted:
        folders = [p for p in folders if int(p.name.split("_")[1]) in wanted]
    inv, q, t = [], [], []
    with ProcessPoolExecutor(max_workers=2) as ex:      # 환자 하나가 메모리 약 1 GB
        for d, (a, b, c) in zip(folders, ex.map(safe_patient, folders)):
            inv += a; q += b
            if c:
                t.append(c)
            print(d.name, "ok" if c else "fail", flush=True)
    inv, q, t = pd.DataFrame(inv), pd.DataFrame(q), pd.DataFrame(t)
    v = pd.read_csv(ROOT / "Result_py/psg_verified.csv", encoding="utf-8-sig")
    v["ExamStatus"] = np.where(v.SameExam == True, "엑셀 요약 = 이 원신호 검사",
                               np.where(v.SummaryExam.isna(), "엑셀 요약 출처 검사 없음/미확인", "엑셀 요약은 다른 날 검사"))
    t = t.merge(v[["Patient", "RawExam", "SummaryExam", "ExamStatus"]], on="Patient", how="left")
    inv.to_csv(OUT / "raw_eda_channels.csv", index=False, encoding="utf-8-sig")
    q.to_csv(OUT / "raw_eda_quality.csv", index=False, encoding="utf-8-sig")
    t.to_csv(OUT / "raw_eda_time_info.csv", index=False, encoding="utf-8-sig")
    channel_map(inv).to_csv(OUT / "raw_eda_channel_map.csv", index=False, encoding="utf-8-sig")
    review_list(q, t).to_csv(OUT / "raw_eda_review_list.csv", index=False, encoding="utf-8-sig")
    print("saved", OUT)


if __name__ == "__main__":
    main({int(a.replace("P_", "")) for a in sys.argv[1:]})
