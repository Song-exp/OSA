"""PSG 채널 해석 학습가이드(docs/PSG_채널_해석_학습가이드.md)용 그림.

뷰어와 같은 그리기 함수(osa.psg_plot._draw_channel, _draw_stages)를 쓰되, 사례마다 필요한 채널만 모아
Y축을 자르지 않고(fraction=1.0) 그립니다. 원본과 기존 코드는 읽기만 합니다.
결과: Result_py/psg_guide/*.png, channel_stats_*.csv
사용: python psg_guide_figures.py
"""
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from osa.psg_events import DOTNET_EPOCH_TICKS, read_events
from osa.psg_plot import _draw_channel, _draw_stages, display_unit
from osa.psg_read import read_psg

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py" / "psg_guide"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8})

RESP = ["C3", "Chin", "Nasal Pressure", "Thermistor", "Thorax", "Abdomen", "SpO2"]
AROUSAL = ["C3", "O1", "LOC", "Chin", "Nasal Pressure", "Thorax", "Abdomen", "SpO2"]
# P_6 (Embla) 사례: (파일, 시작, 끝, 채널, 제목)
CASES_P6 = [
    ("01_normal_breathing", "2016-07-28T00:36:20", "2016-07-28T00:37:20", RESP + ["EKG"], "정상 호흡 (N2, 60초)"),
    ("02_obstructive_apnea", "2016-07-27T23:33:30", "2016-07-27T23:35:40", RESP, "폐쇄성 무호흡 (리포트 판독: APNEA-OBSTRUCTIVE)"),
    ("03_central_apnea", "2016-07-27T23:08:30", "2016-07-27T23:10:20", RESP, "중추성 무호흡 (리포트 판독: APNEA-CENTRAL)"),
    ("04_mixed_apnea", "2016-07-28T01:21:00", "2016-07-28T01:23:40", RESP,
     "판독: APNEA-MIXED · 파형상 '초반 무노력' 구간은 뚜렷하지 않음 · 산소포화도 하락·회복 예시"),
    ("06_rera", "2016-07-27T22:48:30", "2016-07-27T22:50:30", AROUSAL, "RERA (판독: AROUSAL-RERA)"),
    ("07_resp_arousal", "2016-07-28T01:30:00", "2016-07-28T01:32:00", AROUSAL, "무호흡 뒤 호흡 관련 각성 (판독: AROUSAL-RESP)"),
]
GALLERY_P6 = {
    "g1_eeg_eog_emg": ["C3", "C4", "F3", "O1", "A1", "LOC", "ROC", "Chin", "Leg-L"],
    "g2_respiration": ["Nasal Pressure", "Thermistor", "Thermo-2", "Thorax", "Abdomen", "XSum_CU", "XFlow_CU",
                       "C-Flow", "Flow_CU", "Snoring Sensor"],
    "g3_oxygen_cardiac": ["SpO2", "SpO2 BB", "SpO2-Quality_CU", "Plethysmogram", "Pleth Amplitude", "Pulse",
                          "EKG", "Heart Rate_CU", "R-R Intervals"],
    "g4_derived_device": ["Position_CU", "Gravity X", "Gravity Y", "Activity_CU", "Light_CU", "Flattening",
                          "RMI_CU", "Phase_CU", "RR_CU", "Tidal Volume_CU"],
}
NOX = ["Nasal Pressure", "Flow", "Thermistor", "Thorax RIP", "Abdomen RIP", "RIP Sum", "cRIP Flow", "SpO2",
       "Pulse", "Position", "Snore"]


def chan(rec, name):
    return next((c for c in rec.channels if c.name == name), None)


def full_range(ax, s):
    """뷰어는 중앙값 ± (자동 범위 × fraction/2)로 Y축을 잡아 치우친 신호가 잘림. 학습용 그림은 실제 최소~최대."""
    f = s[np.isfinite(s)]
    if f.size and f.max() > f.min():
        pad = (f.max() - f.min()) * 0.05
        ax.set_ylim(f.min() - pad, f.max() + pad)


def panel(rec, events, stages, names, w0, w1, title, fname, fraction=1.0, per_axis_events=None):
    w0, w1 = np.datetime64(w0, "ns"), np.datetime64(w1, "ns")
    chs = [chan(rec, n) for n in names]
    chs = [c for c in chs if c is not None]
    fig, axes = plt.subplots(len(chs) + 1, 1, sharex=True, figsize=(13, 1.15 * (len(chs) + 1) + 0.6),
                             constrained_layout=True)
    _draw_stages(axes[0], stages, w0, w1)
    for i, (ax, c) in enumerate(zip(axes[1:], chs)):
        ev = per_axis_events[i] if per_axis_events else events
        _, s, _ = _draw_channel(ax, c, ev, w0, w1, 200_000, fraction, legend=(i == 0))
        full_range(ax, s)
    for ax in axes:
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlim(w0, w1)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("기록 시각 (시:분:초)")
    fig.suptitle(title, fontsize=11)
    fig.savefig(OUT / f"{fname}.png", dpi=90)
    plt.close(fig)
    print("saved", fname)


def channel_stats(rec, stages, tag):
    """수면 중 구간의 채널별 값 범위 (y축 읽기 근거)."""
    sl = stages[stages.Level.between(1, 5)]
    rows = []
    for c in rec.channels:
        m = np.zeros(c.time.size, bool)
        for a, b in zip(sl.StartTime.values, sl.EndTime.values):
            i0, i1 = np.searchsorted(c.time, [a, b])
            m[i0:i1] = True
        x = c.data[m]
        x = x[np.isfinite(x)]
        q = np.percentile(x, [1, 50, 99]) if x.size else [np.nan] * 3
        rows.append({"Channel": c.name, "Unit": display_unit(c) or "(미확인)", "Fs_Hz": round(c.fs, 2),
                     "P1": q[0], "Median": q[1], "P99": q[2], "UniqueValues": int(np.unique(x[:200000]).size)})
    pd.DataFrame(rows).to_csv(OUT / f"channel_stats_{tag}.csv", index=False, encoding="utf-8-sig")


def nox_layer_events(folder: Path) -> dict[str, pd.DataFrame]:
    """Nox Data.ndb 임시 판독본의 Automatic / Manual 호흡 이벤트를 따로 (P_59는 temporary_scoring만 있음)."""
    db = sqlite3.connect(f"file:{(folder / 'Data.ndb').as_posix()}?mode=ro", uri=True)
    m = pd.read_sql("select CAST(m.starts_at AS TEXT) a, CAST(m.ends_at AS TEXT) b, m.type, k.type kt "
                    "from temporary_scoring_marker m join temporary_scoring_key k on m.key_id = k.id "
                    "where m.is_deleted = 0", db)
    db.close()
    name = {"apnea-obstructive": "APNEA-OBSTRUCTIVE", "apnea-central": "APNEA-CENTRAL", "apnea-mixed": "APNEA-MIXED",
            "hypopnea": "HYPOPNEA", "rera": "AROUSAL-RERA", "arousal-respiratory": "AROUSAL-RESP"}
    m = m[m.type.isin(name)]
    t = lambda s: pd.to_datetime((s.astype("int64") - DOTNET_EPOCH_TICKS) * 100)
    out = {}
    for k, g in m.groupby("kt"):
        out[k] = pd.DataFrame({"StartTime": t(g.a), "EndTime": t(g.b), "EventType": g.type.map(name)})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- Embla P_6
    rec = read_psg(next((ROOT / "Data/PSG").glob("P_6_*")))
    ev, st, _ = read_events(rec)
    channel_stats(rec, st, "P6")
    for f, a, b, names, title in CASES_P6:
        panel(rec, ev, st, names, a, b, f"P_6 (Embla) · {title}", f)
    for f, names in GALLERY_P6.items():
        panel(rec, ev, st, names, "2016-07-28T00:36:20", "2016-07-28T00:37:20",
              f"P_6 (Embla) · 채널 모음 · 정상 호흡 N2 60초", f)
    del rec
    # ---- Embla P_81: 중추성 무호흡 두 번째 사례
    rec = read_psg(next((ROOT / "Data/PSG").glob("P_81_*")))
    ev, st, _ = read_events(rec)
    panel(rec, ev, st, RESP, "2024-06-14T02:11:30", "2024-06-14T02:13:30",
          "P_81 (Embla) · 중추성 무호흡 (판독: APNEA-CENTRAL)", "03b_central_apnea_P81")
    panel(rec, ev, st, RESP, "2024-06-14T02:25:20", "2024-06-14T02:27:20",
          "P_81 (Embla) · 저호흡 (판독: HYPOPNEA)", "05_hypopnea_P81")
    del rec
    # ---- Nox P_59: 채널 모음, Y축 잘림 비교, 판독층 중복
    folder = next((ROOT / "Data/PSG").glob("P_59_*"))
    rec = read_psg(folder)
    ev, st, _ = read_events(rec)
    channel_stats(rec, st, "P59")
    panel(rec, ev, st, NOX, "2022-12-13T02:00:00", "2022-12-13T02:05:00",
          "P_59 (Nox) · 주요 채널 · 이벤트 색칠은 뷰어와 같음 (Automatic+Manual 함께)", "n1_nox_channels")
    # 같은 SpO2를 뷰어 기본 탭(30%), 뷰어 확대 탭(100%, 중앙값 중심), 실제 최소~최대로
    w0, w1 = np.datetime64("2022-12-13T02:00:00", "ns"), np.datetime64("2022-12-13T02:05:00", "ns")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 6), constrained_layout=True)
    for ax, fr, lab in zip(axes, (0.30, 1.0, None), ("① 뷰어 기본 탭: 중앙값 ± 자동범위×0.30/2",
                                                    "② 뷰어 채널 확대 탭: 중앙값 ± 자동범위×1.0/2 (치우친 신호는 여전히 잘림)",
                                                    "③ 실제 최소~최대 (학습가이드 그림 방식)")):
        _, s, _ = _draw_channel(ax, chan(rec, "SpO2"), ev, w0, w1, 200_000, fr or 1.0, legend=False)
        if fr is None:
            full_range(ax, s)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.grid(True, alpha=0.4)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.suptitle("P_59 (Nox) · 같은 SpO2, Y축 범위만 다름", fontsize=11)
    fig.savefig(OUT / "n2_ylim_clipping.png", dpi=90)
    plt.close(fig)
    layers = nox_layer_events(folder)
    panel(rec, ev, st, ["Nasal Pressure", "Nasal Pressure", "Nasal Pressure", "SpO2"],
          "2022-12-13T02:00:00", "2022-12-13T02:05:00",
          "P_59 (Nox) · 위: 뷰어 표시(두 층 합침) / 가운데: Automatic만 / 아래 두 칸: Manual(기사)만",
          "n3_nox_layers", per_axis_events=[ev, layers.get("Automatic"), layers.get("Manual"), layers.get("Manual")])
    panel(rec, ev, st, ["Thermistor", "Nasal Pressure"], "2022-12-12T22:40:00", "2022-12-13T05:10:00",
          "P_59 (Nox) · 하룻밤 전체 Thermistor와 Nasal Pressure", "n4_thermistor_night")


if __name__ == "__main__":
    main()
