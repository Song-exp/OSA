"""눈검증 목록용 계산. docs/PSG_원신호_EDA_결과_및_눈검증_목록.md

1) 수면단계 판독층 확인: 리포트 'Sleep Information' 표의 단계별 분(N1, N2, N3, R)과
   우리가 쓰는 판독(Embla: EN-Stage, Nox: 같은 30초에 두 층이 있으면 Manual 우선, 분석구간 안)을 비교.
   Nox는 Automatic 층만 썼을 때도 함께 계산 (두 층이 다른 것 자체는 오류가 아님).
2) 품질 의심 구간의 정확한 시각: psg_raw_eda.py가 환자×채널 '합계'로 센 것을 구간(시작·끝)으로 풀어 씀.
3) 아직 그림으로 보지 않은 대상의 확인용 그림.
원본과 기존 코드·결과는 읽기만 합니다.
결과: Result_py/psg_visual_check/ (stage_vs_report.csv, intervals.csv, fig/*.png)
사용: python psg_visual_check.py
"""
from __future__ import annotations

import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf

import psg_guide_figures as pgf
from osa.psg_events import DOTNET_EPOCH_TICKS, _stage_level, read_events
from osa.psg_read import Recording, read_ebm, read_ndf
from psg_raw_eda import flat_runs, low_amp_seconds
from psg_waveform_check import nox_sleep_and_events

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PSG = ROOT / "Data/PSG"
OUT = ROOT / "Result_py" / "psg_visual_check"
FIG = OUT / "fig"
pgf.OUT = FIG
SEC = np.timedelta64(1, "s")
STAGE_NAME = {0: "W", 1: "R", 2: "N1", 3: "N2", 4: "N3", 5: "N3"}     # N4는 N3로 합침 (AASM)


def folder(pid: str) -> Path:
    return next(p for p in PSG.iterdir() if p.is_dir() and re.match(rf"{pid}_", p.name))


# ------------------------------------------------------------ 1) 수면단계 vs 리포트
def report_stage_minutes(pdf: Path) -> dict:
    """'Minutes: 83.5 min 176.0 min 23.0 min 82.5 min 39.0 min' (N1 N2 N3 R Wake 순서)."""
    lines = [l for pg in pypdf.PdfReader(pdf).pages for l in (pg.extract_text() or "").splitlines()]
    for i, l in enumerate(lines):
        if re.search(r"N1\s+N2\s+N3\s+R\s+Wake", l):
            for m in lines[i + 1:i + 4]:
                if m.strip().startswith("Minutes"):
                    v = [float(x) for x in re.findall(r"([\d.]+)\s*min", m)]
                    if len(v) >= 4:
                        return dict(zip(["N1", "N2", "N3", "R"], v[:4]))
    return {}


def nox_markers(f: Path) -> pd.DataFrame:
    db = sqlite3.connect(f"file:{(f / 'Data.ndb').as_posix()}?mode=ro", uri=True)
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    live = {}
    for pre in ("scoring", "temporary_scoring"):
        if f"{pre}_marker" in tabs:
            m = pd.read_sql(f"select CAST(m.starts_at AS TEXT) a, CAST(m.ends_at AS TEXT) b, m.type, k.type kt "
                            f"from {pre}_marker m join {pre}_key k on m.key_id = k.id where m.is_deleted = 0", db)
            if len(m):
                live[pre] = m.assign(a=m.a.astype("int64"), b=m.b.astype("int64"), type=m.type.astype(str))
    db.close()
    return live[max(live, key=lambda k: (live[k].kt == "Manual").sum())]   # psg_reconcile과 같은 층 선택


def nox_stage_minutes(f: Path) -> tuple[dict, dict]:
    """분석구간 안 단계별 분. (Manual 우선, Automatic만)."""
    m = nox_markers(f)
    man = m[m.kt == "Manual"]
    a0, a1 = man[man.type == "period-analysisstart"].a.min(), man[man.type == "period-analysisstop"].a.max()
    st = m[m.type.str.startswith("sleep-")]
    if pd.notna(a0) and pd.notna(a1):
        st = st[(st.a >= a0) & (st.a < a1)]
    lev = lambda s: s.type.str.upper().map(_stage_level).map(STAGE_NAME)
    pref = st.sort_values("kt").drop_duplicates("a", keep="last")
    auto = st[st.kt == "Automatic"]
    count = lambda s: {k: round(((s.b - s.a)[lev(s) == k]).sum() / 1e7 / 60, 1) for k in ("N1", "N2", "N3", "R")}
    return count(pref), (count(auto) if len(auto) else {})


def embla_stage_minutes(f: Path) -> dict:
    spo = read_ebm(next(p for p in f.glob("*.ebm") if re.fullmatch(r"(?i)spo2(_pds)?\.ebm", p.name)))
    rec = Recording("EBM", str(f), [], [spo], pd.DataFrame({"StartTime": [spo.time[0]], "EndTime": [spo.time[-1]]}))
    _, st, _ = read_events(rec)
    st = st.assign(S=st.Level.map(STAGE_NAME), D=(st.EndTime - st.StartTime).dt.total_seconds() / 60)
    return {k: round(st.D[st.S == k].sum(), 1) for k in ("N1", "N2", "N3", "R")}


def stage_check() -> pd.DataFrame:
    v = pd.read_csv(ROOT / "Result_py/psg_verified.csv", encoding="utf-8-sig")
    rows = []
    for _, r in v.iterrows():
        f = folder(r.Patient)
        row = {"Patient": r.Patient, "Device": r.Device, "RawExam": r.RawExam, "Report": r.RawExamReport}
        rep = report_stage_minutes(f / r.RawExamReport) if isinstance(r.RawExamReport, str) else {}
        if r.Device == "N":
            ours, auto = nox_stage_minutes(f)
        else:
            ours, auto = embla_stage_minutes(f), {}
        for k in ("N1", "N2", "N3", "R"):
            row[f"Rep_{k}"] = rep.get(k, np.nan)
            row[f"Use_{k}"] = ours.get(k, np.nan)
            row[f"Auto_{k}"] = auto.get(k, np.nan)
        d = [abs(row[f"Use_{k}"] - row[f"Rep_{k}"]) for k in ("N1", "N2", "N3", "R")]
        da = [abs(row[f"Auto_{k}"] - row[f"Rep_{k}"]) for k in ("N1", "N2", "N3", "R")]
        row["Use_MaxDiffMin"] = round(np.nanmax(d), 1) if rep else np.nan
        row["Auto_MaxDiffMin"] = round(np.nanmax(da), 1) if rep and auto else np.nan
        row["Status"] = ("리포트 없음(원신호 검사)" if not isinstance(r.RawExamReport, str) else
                         "리포트에 단계 표 없음" if not rep else
                         "일치(모든 단계 ±1분)" if row["Use_MaxDiffMin"] <= 1 else "불일치")
        rows.append(row)
        print(r.Patient, row["Status"], row.get("Use_MaxDiffMin"), flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 2) 의심 구간
def load(pid: str, names: list[str]):
    """채널 이름(소문자, 확장자 제외)으로 파일을 찾아 읽고, 판독 사건과 단계(Nox는 Manual 우선)를 붙임."""
    f = folder(pid)
    nox = f.name.endswith("_N")
    files = {p.stem.strip().lower(): p for p in f.glob("*.ndf" if nox else "*.ebm")}
    chs = [(read_ndf if nox else read_ebm)(files[n]) for n in names if n in files]
    ct = pd.DataFrame({"StartTime": [c.time[0] for c in chs], "EndTime": [c.time[-1] for c in chs]})
    rec = Recording("NDF" if nox else "EBM", str(f), [], chs, ct)
    ev, st, _ = read_events(rec)
    if nox:
        from psg_raw_eda import nox_stages
        _, ev = nox_sleep_and_events(f)
        st = nox_stages(f)[0]
    return rec, ev, st


def runs(mask: np.ndarray, t: np.ndarray, min_sec: float, fs: float):
    """True가 min_sec 이상 이어진 구간의 (시작, 끝, 초)."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) / fs >= min_sec
    return [(t[a], t[b - 1], round((b - a) / fs, 1)) for a, b in zip(s[keep], e[keep])]


def stage_at(st: pd.DataFrame, t) -> str:
    i = st[(st.StartTime <= t) & (st.EndTime > t)]
    return STAGE_NAME.get(int(i.Level.iloc[-1]), "?") if len(i) and np.isfinite(i.Level.iloc[-1]) else "판독 없음"


def sleep_share(st: pd.DataFrame, a, b) -> float:
    """구간 중 판독 수면(N1~N3, R)인 비율."""
    s = st[(st.EndTime > a) & (st.StartTime < b) & st.Level.between(1, 5)]
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    tot = sum((min(e, b) - max(x, a) for x, e in zip(s.StartTime, s.EndTime)), pd.Timedelta(0))
    return round(100 * (tot / (b - a)), 0) if b > a else np.nan


# (환자, 채널 파일 이름, 무엇을 찾을지, 짝 채널)
TARGETS = [
    ("P_28", "thorax rip", ["flat", "rail", "lowamp"], "abdomen rip"),
    ("P_44", "1", ["flat", "rail"], None), ("P_44", "thorax rip", ["flat", "rail"], "abdomen rip"),
    ("P_50", "1", ["flat", "rail"], None), ("P_50", "spo2", ["spo2flat"], None),
    ("P_59", "thermistor", ["flat", "rail"], None),
    ("P_61", "thorax rip", ["flat", "lowamp"], "abdomen rip"),
    ("P_39", "nasal pressure", ["lowamp"], "thermistor"), ("P_39", "thermistor", ["flat", "rail"], None),
    ("P_71", "nasal pressure", ["lowamp"], "thermistor"),
    ("P_80", "nasal pressure", ["lowamp"], "thermistor"),
    ("P_10", "spo2", ["spo2flat"], None), ("P_25", "spo2", ["spo2flat"], None), ("P_30", "spo2", ["spo2flat"], None),
    ("P_60", "spo2", ["spo2flat"], None), ("P_69", "spo2", ["spo2flat"], None), ("P_84", "spo2", ["spo2flat"], None),
    ("P_43", "spo2", ["zero"], None), ("P_79", "spo2", ["zero", "gap"], None), ("P_35", "spo2", ["zero"], None),
    ("P_66", "spo2", ["zero"], None),
    ("P_82", "spo2", ["gap"], None), ("P_83", "spo2", ["gap"], None), ("P_22", "spo2", ["gap"], None),
    ("P_42", "spo2", ["gap"], None),
    ("P_37", "abdomen rip", ["rail"], "thorax rip"), ("P_57", "chin", ["rail", "flat"], None),
    ("P_57", "abdomen", ["flat"], "thorax"), ("P_18", "thorax", ["rail"], "abdomen"),
    ("P_1", "thermistor", ["flat", "rail"], None), ("P_79", "thermistor", ["flat", "rail"], None),
]
# 1초 중앙값이 앞 초보다 10%p 이상 변했고 0과 관계없는 것 (psg_raw_eda 6-2에서 22명 42회)
JUMP_PATIENTS = ["P_1", "P_2", "P_4", "P_10", "P_13", "P_14", "P_22", "P_29", "P_30", "P_32", "P_33", "P_39", "P_43",
                 "P_48", "P_51", "P_54", "P_55", "P_57", "P_63", "P_81", "P_86", "P_87"]


def intervals() -> pd.DataFrame:
    rows = []
    add = lambda pid, ch, kind, a, b, sec, st, note="": rows.append(
        {"Patient": pid, "Channel": ch, "Type": kind, "Start": str(np.datetime64(a, "s")), "End": str(np.datetime64(b, "s")),
         "DurationSec": sec, "StageAtStart": stage_at(st, a), "SleepPct": sleep_share(st, a, b), "Note": note})
    for pid, name, kinds, partner in TARGETS:
        rec, ev, st = load(pid, [name] + ([partner] if partner else []))
        if not rec.channels or Path(rec.channels[0].source_file).stem.strip().lower() != name:
            print(pid, name, "채널 파일 없음 - 건너뜀")
            continue
        ch = rec.channels[0]
        x, t = ch.data, ch.time
        fs = ch.fs if np.isfinite(ch.fs) else float(np.median(ch.segments.Fs))
        if "flat" in kinds:
            for a, b, s in runs(flat_runs(x, int(10 * fs)), t, 30, fs):
                add(pid, ch.name, "평탄(같은 값 지속)", a, b, s, st, f"값 {x[np.searchsorted(t, a)]:.4g}")
        if "spo2flat" in kinds:
            for a, b, s in runs(flat_runs(x, int(300 * fs)), t, 600, fs):
                add(pid, ch.name, "SpO2 같은 값 10분 이상", a, b, s, st, f"값 {x[np.searchsorted(t, a)]:.0f}%")
        if "rail" in kinds:
            lo, hi = np.nanmin(x), np.nanmax(x)
            for a, b, s in runs((x == lo) | (x == hi), t, 10, fs):
                add(pid, ch.name, "최대·최소값에 붙음", a, b, s, st, f"최소 {lo:.4g} / 최대 {hi:.4g}")
        if "zero" in kinds:
            for a, b, s in runs(x == 0, t, 60, fs):
                add(pid, ch.name, "SpO2 = 0", a, b, s, st)
        if "gap" in kinds:
            gap = np.diff(t) / SEC > 5
            for i in np.flatnonzero(gap):
                if sleep_share(st, t[i], t[i + 1]) > 0:
                    add(pid, ch.name, "기록 공백(수면 판독 구간과 겹침)", t[i], t[i + 1], round((t[i + 1] - t[i]) / SEC, 1), st)
        if "lowamp" in kinds and len(rec.channels) > 1:
            t0 = min(c.time[0] for c in rec.channels)
            n = int((max(c.time[-1] for c in rec.channels) - t0) // SEC) + 1
            sl = np.zeros(n, bool)
            for a, b in zip(st[st.Level.between(1, 5)].StartTime, st[st.Level.between(1, 5)].EndTime):
                sl[max(int((np.datetime64(a, "ns") - t0) // SEC), 0):max(int((np.datetime64(b, "ns") - t0) // SEC), 0)] = True
            la = []
            for c in rec.channels[:2]:
                idx = ((c.time - t0) // SEC).astype("int64")
                la.append(low_amp_seconds(np.where(np.isfinite(c.data), c.data, 0), idx, n, sl) & sl)
            alone = la[0] & ~la[1]
            grid = t0 + np.arange(n) * SEC
            for a, b, s in runs(alone, grid, 300, 1.0):
                add(pid, ch.name, f"혼자 약함(짝 {rec.channels[1].name}은 아님)", a, b, s, st)
        del rec
        print(pid, name, "done", flush=True)
    for pid in JUMP_PATIENTS:
        rec, ev, st = load(pid, ["spo2", "spo2_pds"])
        ch = rec.channels[0]
        sec = ((ch.time - ch.time[0]) // SEC).astype("int64")
        med = pd.Series(ch.data).groupby(sec).median().reindex(range(int(sec[-1]) + 1))
        d = med.diff().abs()
        hit = (d >= 10) & (med != 0) & (med.shift(1) != 0)
        for i in np.flatnonzero(hit.values):
            a = ch.time[0] + i * SEC
            add(pid, ch.name, "SpO2 1초에 10%p 이상 변화(0과 무관)", a, a + SEC, 1, st,
                f"{med.iloc[i - 1]:.0f} → {med.iloc[i]:.0f}%")
        del rec
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 3) 확인용 그림
def figures(iv: pd.DataFrame):
    """대상별로 가장 긴 구간 하나를 골라 앞뒤 여유를 두고 짝 채널과 함께 그림."""
    plan = [
        ("P_44", "1", ["1", "2", "c3", "nasal pressure"], "v1_chin_flat_P44", "턱 근전도 '1'이 같은 값으로 이어짐 / '2'와 비교"),
        ("P_61", "Thorax RIP", ["thorax rip", "abdomen rip", "nasal pressure", "spo2"], "v2_thorax_P61", "흉부 평탄·혼자 약함 / 복부와 비교"),
        ("P_39", "Nasal Pressure", ["nasal pressure", "thermistor", "thorax", "spo2"], "v3_nasal_P39", "비강압력 혼자 약함 / 서미스터와 비교"),
        ("P_80", "Nasal Pressure", ["nasal pressure", "thermistor", "thorax", "spo2"], "v4_nasal_P80", "비강압력 혼자 약함 / 서미스터와 비교"),
        ("P_71", "Nasal Pressure", ["nasal pressure", "thermistor", "thorax rip", "spo2"], "v5_nasal_P71", "비강압력 혼자 약함 / 서미스터와 비교"),
        ("P_30", "SpO2", ["spo2", "pulse", "nasal pressure", "thorax"], "v6_spo2_flat_P30", "SpO2 같은 값 장시간 / 맥박·호흡과 비교"),
        ("P_50", "SpO2", ["spo2", "pulse", "nasal pressure", "thorax rip"], "v7_spo2_flat_P50", "SpO2 98% 장시간 / 맥박·호흡과 비교"),
        ("P_43", "SpO2", ["spo2", "pulse", "nasal pressure"], "v8_spo2_zero_P43", "SpO2 = 0 장시간 / 맥박과 비교"),
        ("P_82", "SpO2", ["spo2", "nasal pressure", "c3"], "v9_gap_P82", "수면 판독 구간 안의 기록 공백"),
        ("P_37", "Abdomen RIP", ["abdomen rip", "thorax rip", "nasal pressure"], "v10_abdomen_rail_P37", "복부 최대·최소값에 붙음 / 흉부와 비교"),
    ]
    for pid, ch, names, fname, what in plan:
        sub = iv[(iv.Patient == pid) & (iv.Channel == ch)]
        if sub.empty:
            print("skip", fname)
            continue
        r = sub.sort_values("DurationSec").iloc[-1]
        a, b = np.datetime64(r.Start), np.datetime64(r.End)
        dur = (b - a) / SEC
        pad = np.timedelta64(int(min(max(dur * 0.3, 60), 300)), "s")
        w1 = min(b + pad, a + np.timedelta64(15, "m"))      # 너무 길면 시작 부분 15분만
        rec, ev, st = load(pid, names)
        pgf.panel(rec, ev, st, [c.name for c in rec.channels], str(a - pad), str(w1),
                  f"{pid} · {what} · 대상 구간 {r.Start[11:]}~{r.End[11:]}", fname)
        del rec


def layer_figure(pid="P_75"):
    """자동(Automatic)과 기사(Manual) 판독층의 무호흡·저호흡을 같은 파형 위에 따로 칠함."""
    m = nox_markers(folder(pid))
    name = {"apnea-obstructive": "APNEA-OBSTRUCTIVE", "apnea-central": "APNEA-CENTRAL", "apnea-mixed": "APNEA-MIXED",
            "hypopnea": "HYPOPNEA"}
    m = m[m.type.isin(name)]
    t = lambda v: pd.to_datetime((v.astype("int64") - DOTNET_EPOCH_TICKS) * 100)
    lay = {k: pd.DataFrame({"StartTime": t(g.a), "EndTime": t(g.b), "EventType": g.type.map(name)})
           for k, g in m.groupby("kt")}
    auto = lay.get("Automatic", pd.DataFrame(columns=["StartTime", "EndTime", "EventType"]))
    man = lay.get("Manual", pd.DataFrame(columns=["StartTime", "EndTime", "EventType"]))
    print(pid, "Automatic", len(auto), "Manual", len(man))
    c = auto.set_index("StartTime").resample("10min").size()
    w0 = c.idxmax()
    rec, _, st = load(pid, ["nasal pressure", "spo2"])
    pgf.panel(rec, None, st, ["Nasal Pressure", "Nasal Pressure", "SpO2"], str(w0), str(w0 + pd.Timedelta(minutes=10)),
              f"{pid} (Nox) · 위: Automatic 층 무호흡·저호흡({len(auto)}개) / 가운데·아래: Manual 층({len(man)}개)",
              f"v11_layers_{pid}", per_axis_events=[auto, man, man])
    del rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if not (OUT / "stage_vs_report.csv").exists() or "--all" in sys.argv:     # 87명 리포트 대조는 오래 걸려 한 번만
        stage_check().to_csv(OUT / "stage_vs_report.csv", index=False, encoding="utf-8-sig")
    iv = intervals()
    iv.to_csv(OUT / "intervals.csv", index=False, encoding="utf-8-sig")
    figures(iv)
    layer_figure("P_75")
    print("saved", OUT)


if __name__ == "__main__":
    main()
