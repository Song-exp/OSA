"""PSG 채널 + 수면단계 + 이벤트 표시. Matlab/Function/PSG_Plot.m 변환.

MATLAB의 탭 하나 = matplotlib Figure 하나. 파일 저장은 하지 않습니다 (batch.py에서 저장).
화면에서 확대/이동하려면 plt.show()로 띄우면 됩니다.
"""
from __future__ import annotations

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .psg_events import COUNT_NAMES, read_events
from .psg_read import Channel, Recording

EVENT_COLORS = [(0.55, 0.04, 0.07), (0.78, 0.16, 0.19), (0.96, 0.38, 0.40),
                (0.12, 0.30, 0.82), (0.24, 0.58, 0.95), (0.55, 0.80, 0.20)]
SIGNAL_COLOR, STAGE_COLOR = (0.08, 0.25, 0.48), (0.20, 0.15, 0.48)
EVENT_ALPHA = 0.28


def plot_psg(rec: Recording, window=None, channels_per_tab=8, max_plot_points=100_000,
             skip_initial_seconds=180, y_axis_range_fraction=0.30):
    """Returns (figures, events, stages, plot_data, event_counts)."""
    events, stages, counts = read_events(rec)
    start = rec.channel_table.StartTime.min().to_datetime64()
    stop = rec.channel_table.EndTime.max().to_datetime64()
    w0, w1 = (start, stop) if window is None else (np.datetime64(window[0], "ns"), np.datetime64(window[1], "ns"))
    w0 = max(w0, start + np.int64(round(skip_initial_seconds * 1e9)) * np.timedelta64(1, "ns"))
    assert w1 > w0, "종료 시각은 시작 시각보다 뒤여야 합니다."

    subject = rec.source_folder.replace("\\", "/").rstrip("/").split("/")[-1]
    n = len(rec.channels)
    plot_data = {"Window": (w0, w1), "ChannelsPerTab": channels_per_tab, "MaxPlotPoints": max_plot_points,
                 "Channels": [None] * n, "SleepStage": None}
    figures = []
    for tab in range(math.ceil(n / channels_per_tab)):
        idx = list(range(tab * channels_per_tab, min((tab + 1) * channels_per_tab, n)))
        fig, axes = plt.subplots(len(idx) + 1, 1, sharex=True, figsize=(16, 9.5), constrained_layout=True)
        st_time, st_level = _draw_stages(axes[0], stages, w0, w1)
        plot_data["SleepStage"] = {"time": st_time, "level": st_level}
        for row, i in enumerate(idx, start=1):
            ch = rec.channels[i]
            t, s, unit = _draw_channel(axes[row], ch, events, w0, w1, max_plot_points,
                                       y_axis_range_fraction, legend=(row == 1))
            plot_data["Channels"][i] = {"index": i + 1, "name": ch.name, "unit": unit, "time": t, "data": s}
        for ax in axes:
            ax.grid(True, alpha=0.4)
        axes[-1].set_xlim(w0, w1)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
        axes[-1].set_xlabel("Recorded date/time")
        fig.suptitle(f"{subject} | channels {idx[0] + 1}-{idx[-1] + 1} / {n}")
        figures.append(fig)
    print(f"Plot ready: {n} channels in {len(figures)} figures. plot_psg does not save files.")
    return figures, events, stages, plot_data, counts


def _draw_stages(ax, stages, w0, w1):
    st = stages[(stages.EndTime > w0) & (stages.StartTime < w1)].copy()
    if st.empty:
        time, level = np.array([w0, w1]), np.array([np.nan, np.nan])
        ax.plot(time, level)
        ax.text(0.5, 0.5, "Sleep staging unavailable", transform=ax.transAxes, ha="center")
    else:
        st.StartTime = st.StartTime.clip(lower=w0)
        st.EndTime = st.EndTime.clip(upper=w1)
        time = np.append(st.StartTime.to_numpy(), st.EndTime.to_numpy()[-1])
        level = np.append(st.Level.to_numpy(), st.Level.to_numpy()[-1])
        ax.step(time, level, where="post", color=STAGE_COLOR, linewidth=1.3)
    ax.set_yticks(range(6), ["W", "REM", "N1", "N2", "N3", "N4"])
    ax.set_ylim(5.4, -0.4)                        # MATLAB YDir reverse
    ax.set_ylabel("Sleep stage")
    return time, level


def _draw_channel(ax, ch: Channel, events, w0, w1, max_points, fraction, legend):
    t, s = prepare_plot_data(ch, w0, w1, max_points)
    if t.size == 0:
        t, s = np.array([w0, w1]), np.array([np.nan, np.nan])
    ax.plot(t, s, color=SIGNAL_COLOR, linewidth=0.6, zorder=3)
    # 앞부분 제외 후 자동 Y축 폭의 일부만 표시합니다. 데이터 값은 바꾸지 않습니다.
    finite = s[np.isfinite(s)]
    if finite.size:
        lo, hi = ax.get_ylim()
        half = (hi - lo) * fraction / 2
        if np.isfinite(half) and half > 0:
            c = np.median(finite)
            ax.set_ylim(c - half, c + half)
    unit = display_unit(ch)
    ax.set_ylabel(f"{ch.name} ({unit})" if unit else ch.name, fontsize=8)
    _draw_events(ax, events, w0, w1, legend)
    return t, s, unit


def _draw_events(ax, events, w0, w1, legend):
    for name, color in zip(COUNT_NAMES, EVENT_COLORS):
        e = events[(events.EventType == name) & (events.EndTime > w0) & (events.StartTime < w1)]
        if e.empty:
            continue
        a = mdates.date2num(e.StartTime.clip(lower=w0).to_numpy())
        b = mdates.date2num(e.EndTime.clip(upper=w1).to_numpy())
        ax.broken_barh(list(zip(a, b - a)), (0, 1), transform=ax.get_xaxis_transform(),
                       facecolors=color, alpha=EVENT_ALPHA, edgecolor="none", zorder=1)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, alpha=EVENT_ALPHA, label=n) for n, c in zip(COUNT_NAMES, EVENT_COLORS)],
                  loc="upper right", ncol=3, fontsize=7)


def prepare_plot_data(ch: Channel, w0, w1, max_points):
    """창 안 샘플을 구간별 min/max로 줄이고, 시간 단절 위치에 NaT/NaN을 넣습니다."""
    m = (ch.time >= w0) & (ch.time <= w1)
    t, s = ch.time[m], ch.data[m]
    if t.size == 0:
        return t, s
    gaps = np.flatnonzero(np.diff(t) / np.timedelta64(1, "s") > 1.5 / ch.fs) + 1
    if s.size > max_points:
        bins = max_points // 2
        edges = np.round(np.linspace(0, s.size, bins + 1)).astype(int)
        lo, hi = edges[:-1], edges[1:]
        # 구간별 극값 위치. 길이가 다른 구간을 NaN으로 채운 2D 배열로 만들어 한 번에 계산.
        # NaN은 MATLAB min/max처럼 무시하고, 전부 NaN이면 구간 첫 샘플.
        idx = lo[:, None] + np.arange((hi - lo).max())
        vals = np.where(idx < hi[:, None], s[np.minimum(idx, s.size - 1)], np.nan)
        empty = ~np.isfinite(vals).any(axis=1)
        vals[empty, 0] = 0.0
        mins = lo + np.nanargmin(vals, axis=1)
        maxs = lo + np.nanargmax(vals, axis=1)
        missing = np.flatnonzero(~np.isfinite(s))
        miss_starts = missing[np.r_[True, np.diff(missing) > 1]] if missing.size else missing
        keep = np.unique(np.concatenate([[0], mins, maxs, [s.size - 1], gaps,
                                         np.maximum(0, gaps - 1), miss_starts]))
        gaps = np.searchsorted(keep, gaps)
        t, s = t[keep], s[keep]
    t = np.insert(t, gaps, np.datetime64("NaT"))
    s = np.insert(s.astype(float), gaps, np.nan)
    return t, s


def display_unit(ch: Channel) -> str:
    """축에는 확인된 단위만 표시. 현재 리더는 확인된 EBM 단위를 'V'/'%'로 이미 반환합니다."""
    unit = (ch.unit or "").strip()
    return "" if any(k in unit.lower() for k in ("ebmread-scaled", "verify", "unverified", "unknown")) else unit
