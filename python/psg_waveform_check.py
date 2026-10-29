"""PSG 원신호(EBM/NDF 파형) 교차 확인. 엑셀·리포트·판독파일 대조(psg_reconcile.py) 결과를 파형으로 다시 봅니다.

판독을 새로 하지 않습니다. 파형에서 직접 잴 수 있는 것만 잽니다.
  1) 검사 날짜: SpO2 파형 기록 시작 시각 (판독 파일 날짜, 리포트 날짜와 비교)
  2) 기록 길이: SpO2 파형 길이 (리포트 TRT·TST보다 짧으면 이상)
  3) 최저 SpO2: 원 샘플 최솟값, 전체 기록과 수면 중 각각 (리포트·엑셀 Lowest SpO2와 비교)
  4) 산소저하 횟수(ODI 3%): 단순 알고리즘. 장비 알고리즘과 다르므로 참고값
  5) 판독 사건 뒤 산소저하 동반 비율: 판독된 무호흡·저호흡이 파형상 SpO2 하락을 동반하는지 (참고)
원본과 기존 코드는 읽기만 합니다.
결과: Result_py/psg_waveform_check.csv
사용: python psg_waveform_check.py [환자번호 ...]
"""
from __future__ import annotations

import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from osa.psg_events import DOTNET_EPOCH_TICKS, read_events
from osa.psg_read import Recording, read_ebm, read_ndf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "Result_py"
SEC = np.timedelta64(1, "s")
VALID = (50, 100)          # 이 범위 밖 SpO2는 센서 탈락·잡음으로 보고 제외
AH = ("APNEA-OBSTRUCTIVE", "APNEA-CENTRAL", "APNEA-MIXED", "HYPOPNEA")


def spo2_file(folder: Path) -> Path | None:
    return next((f for f in sorted(folder.iterdir()) if re.fullmatch(r"(?i)spo2(_pds)?\.(ebm|ndf)", f.name)), None)


def per_second(ch) -> tuple[np.ndarray, np.ndarray]:
    """1초 단위 중앙값. 시각 배열(초 시작)과 값."""
    sec = ((ch.time - ch.time[0]) // SEC).astype("int64")
    s = pd.Series(ch.data).groupby(sec).median()
    full = np.full(int(sec[-1]) + 1, np.nan)
    full[s.index.values] = s.values
    full[(full < VALID[0]) | (full > VALID[1])] = np.nan
    t = ch.time[0] + np.arange(full.size) * SEC
    return t, full


def ticks(v) -> np.datetime64:
    return np.datetime64((int(v) - DOTNET_EPOCH_TICKS) * 100, "ns")


def nox_sleep_and_events(folder: Path):
    """psg_reconcile.nox_counts의 후보 규칙과 같은 층·구간: Manual 마커, 분석구간, 수면 중."""
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
        return [], pd.DataFrame(columns=["StartTime", "EndTime", "EventType"])
    m = live[max(live, key=lambda k: (live[k].kt == "Manual").sum())]
    man = m[m.kt == "Manual"]
    st = m[m.type.str.startswith("sleep-")].sort_values("kt").drop_duplicates("a", keep="last")
    a0 = man[man.type == "period-analysisstart"].a.min()
    a1 = man[man.type == "period-analysisstop"].a.max()
    if pd.isna(a0) or pd.isna(a1):
        a0, a1 = st.a.min(), st.b.max()
    sl = st[(st.type != "sleep-wake") & (st.a >= a0) & (st.a < a1)]
    sleep = [(ticks(a), ticks(b)) for a, b in zip(sl.a, sl.b)]
    up = man.type.str.upper()
    ev = man[up.isin(AH) & (man.a >= a0) & (man.a <= a1)]
    ev = pd.DataFrame({"StartTime": [ticks(a) for a in ev.a], "EndTime": [ticks(b) for b in ev.b],
                       "EventType": ev.type.str.upper().values})
    return sleep, ev


def embla_sleep_and_events(folder: Path, ch):
    rec = Recording("EBM", str(folder), [], [ch], pd.DataFrame({"StartTime": [ch.time[0]], "EndTime": [ch.time[-1]]}))
    ev, st, _ = read_events(rec)
    st = st[st.Level.between(1, 5)]
    sleep = list(zip(st.StartTime.values, st.EndTime.values))
    return sleep, ev[ev.EventType.isin(AH)][["StartTime", "EndTime", "EventType"]]


def mask_of(t: np.ndarray, spans) -> np.ndarray:
    m = np.zeros(t.size, bool)
    for a, b in spans:
        i0, i1 = np.searchsorted(t, [np.datetime64(a, "ns"), np.datetime64(b, "ns")])
        m[i0:i1] = True
    return m


def odi3(s: np.ndarray, sleep: np.ndarray) -> int:
    """직전 120초 최댓값보다 3%p 이상 낮은 상태가 10초 이상 이어지면 1회. 수면 중 시작한 것만."""
    base = pd.Series(s).rolling(120, min_periods=30).max().shift(1).values
    low = (base - s) >= 3
    n, run, i = 0, 0, 0
    for i in range(s.size):
        if low[i]:
            run += 1
            if run == 10 and sleep[i - 9]:
                n += 1
        else:
            run = 0
    return n


def desat_after(t, s, ev: pd.DataFrame) -> float:
    """사건 시작~끝+30초 안의 최저가 사건 직전 30초 최고보다 3%p 이상 낮은 비율."""
    hit = []
    for a, b in zip(ev.StartTime.values, ev.EndTime.values):
        i0, i1, i2 = np.searchsorted(t, [a - 30 * SEC, a, b + 30 * SEC])
        pre, post = s[i0:i1], s[i1:i2]
        if np.isfinite(pre).any() and np.isfinite(post).any():
            hit.append(np.nanmax(pre) - np.nanmin(post) >= 3)
    return round(100 * np.mean(hit), 1) if hit else np.nan


def check(folder: Path) -> dict:
    out = {"Patient": re.match(r"(P_\d+)", folder.name).group(1), "Device": folder.name[-1]}
    f = spo2_file(folder)
    if f is None:
        return {**out, "WaveNote": "SpO2 파형 없음"}
    ch = read_ebm(f) if f.suffix.lower() == ".ebm" else read_ndf(f)
    t, s = per_second(ch)
    out.update({"WaveFile": f.name, "WaveStart": str(ch.time[0].astype("datetime64[s]")),
                "WaveExam": str((ch.time[0] - np.timedelta64(12, "h")).astype("datetime64[D]")),
                "WaveStartDate": str(ch.time[0].astype("datetime64[D]")),
                "WaveMinutes": round((ch.time[-1] - ch.time[0]) / SEC / 60, 1),
                "WaveValidPct": round(100 * np.isfinite(s).mean(), 1)})
    sleep, ev = (nox_sleep_and_events(folder) if out["Device"] == "N"
                 else embla_sleep_and_events(folder, ch))
    sm = mask_of(t, sleep)
    raw_ok = (ch.data >= VALID[0]) & (ch.data <= VALID[1])
    out["Wave_LSaO2_All"] = float(ch.data[raw_ok].min()) if raw_ok.any() else np.nan
    raw_sleep = raw_ok & mask_of(ch.time, sleep)
    out["Wave_LSaO2_Sleep"] = float(ch.data[raw_sleep].min()) if raw_sleep.any() else np.nan
    out["StageSleepMin"] = round(sum((np.datetime64(b, "ns") - np.datetime64(a, "ns")) / SEC for a, b in sleep) / 60, 1)
    out["SleepOnWaveMin"] = round(sm.sum() / 60, 1)       # 판독 수면 구간 중 파형이 실제로 있는 시간
    ss = np.where(sm, s, np.nan)
    if np.isfinite(ss).any():
        out["Wave_SpO2_P1"] = round(float(np.nanpercentile(ss, 1)), 1)
        out["Wave_SpO2_Median"] = float(np.nanmedian(ss))
        n = odi3(s, sm)
        out["Wave_ODI3_n"] = n
        out["Wave_ODI3"] = round(n / (out["StageSleepMin"] / 60), 1) if out["StageSleepMin"] else np.nan
    in_wave = ev[(ev.StartTime >= ch.time[0]) & (ev.StartTime <= ch.time[-1])]
    out["ScoredAH"] = len(ev)
    out["ScoredAH_OnWavePct"] = round(100 * len(in_wave) / len(ev), 1) if len(ev) else np.nan
    for k, names in (("Apnea", AH[:3]), ("Hypopnea", AH[3:])):
        out[f"{k}_DesatPct"] = desat_after(t, s, in_wave[in_wave.EventType.isin(names)])
    return out


def judge(r: pd.Series) -> pd.Series:
    """파형 확인 결과를 판독 파일·리포트 값과 비교한 플래그."""
    f = {}
    # 자정 넘어 시작한 기록은 '밤 날짜'와 '달력 날짜'가 하루 다름. 둘 중 하나와 같으면 같은 검사
    days = lambda x: {r.WaveExam, r.WaveStartDate} if isinstance(x, str) else set()
    f["Date_vs_RawFile"] = ("일치" if r.RawExam in days(r.RawExam) else
                           "불일치" if isinstance(r.RawExam, str) else "판독파일 없음")
    f["Date_vs_Report"] = ("일치" if r.SummaryExam in days(r.SummaryExam) else
                          "다른 검사" if isinstance(r.SummaryExam, str) else "엑셀과 맞는 리포트 없음")
    f["Length_OK"] = ("TST보다 짧음" if np.isfinite(r.V_TST) and r.WaveMinutes < r.V_TST - 1 else
                      "확인" if np.isfinite(r.V_TST) else "리포트 TST 없음")
    # 리포트 최저 SpO2는 기록 전체 기준인 사람도, 수면 중 기준인 사람도 있음. 둘 중 하나와 맞으면 일치
    if np.isfinite(r.Wave_LSaO2_All) and np.isfinite(r.V_LSaO2):
        if abs(r.Wave_LSaO2_All - r.V_LSaO2) <= 1:
            f["LSaO2_vs_Report"] = "일치(±1, 전체 기록)"
        elif abs(r.Wave_LSaO2_Sleep - r.V_LSaO2) <= 1:
            f["LSaO2_vs_Report"] = "일치(±1, 수면 중)"
        else:
            f["LSaO2_vs_Report"] = "파형이 더 낮음" if r.Wave_LSaO2_All < r.V_LSaO2 else "파형이 더 높음"
    else:
        f["LSaO2_vs_Report"] = "비교 불가"
    return pd.Series(f)


def main(wanted: set[int]):
    folders = sorted((p for p in (ROOT / "Data/PSG").iterdir() if p.is_dir() and re.match(r"P_\d+_", p.name)),
                     key=lambda p: int(p.name.split("_")[1]))
    if wanted:
        folders = [p for p in folders if int(p.name.split("_")[1]) in wanted]
    rows = []
    for d in folders:
        try:
            rows.append(check(d))
        except Exception as e:      # 한 명 실패해도 나머지는 진행, 사유 기록
            rows.append({"Patient": d.name.split("_PSG")[0], "WaveNote": f"읽기 실패: {type(e).__name__}: {e}"})
        print(rows[-1]["Patient"], rows[-1].get("WaveExam"), rows[-1].get("Wave_LSaO2_All"), rows[-1].get("WaveNote", ""))
    w = pd.DataFrame(rows)
    v = pd.read_csv(RES / "psg_verified.csv", encoding="utf-8-sig")
    rep = pd.read_csv(RES / "psg_reports.csv", encoding="utf-8-sig")
    x = pd.read_csv(RES / "psg_reconcile.csv", encoding="utf-8-sig")[["Patient", "x_TST", "x_ODI", "x_LSaO2"]]
    trt = rep.assign(Study=rep.Study.astype(str)).groupby(["Patient", "Study"]).TRT.first().reset_index()
    df = (w.merge(v[["Patient", "SummaryExam", "RawExam", "SameExam", "V_TST", "V_ODI", "V_LSaO2", "Raw_TST",
                     "Causes", "Unresolved"]], on="Patient", how="left")
          .merge(x, on="Patient", how="left")
          .merge(trt.rename(columns={"Study": "SummaryExam", "TRT": "r_TRT"}), on=["Patient", "SummaryExam"],
                 how="left"))
    ok = df.WaveExam.notna()
    df.loc[ok, ["Date_vs_RawFile", "Date_vs_Report", "Length_OK", "LSaO2_vs_Report"]] = df[ok].apply(judge, axis=1)
    df["LSaO2_Excel_vs_Wave"] = df.x_LSaO2 - df.Wave_LSaO2_All
    df["ODI_Excel_vs_Wave"] = df.x_ODI - df.Wave_ODI3
    df.to_csv(RES / "psg_waveform_check.csv", index=False, encoding="utf-8-sig")
    print("saved", RES / "psg_waveform_check.csv")


if __name__ == "__main__":
    main({int(a.replace("P_", "")) for a in sys.argv[1:]})
