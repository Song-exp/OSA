"""품질 의심 3사례를 한 장에: 이상 채널과 비교 채널을 같은 시간축으로.

① P_28 Thorax RIP 0 기록   (비교: Abdomen RIP, Nasal Pressure)   = q4_thorax_zero_P28.png 구간
② P_43 SpO2·Pulse 0 기록   (비교: Nasal Pressure)                = v8_spo2_zero_P43.png 대상 구간
③ P_59 Thermistor 최대값 고정 (비교: Nasal Pressure)               = q1_thermistor_rail_P59.png 구간
음영 = Result_py/psg_visual_check/intervals.csv에서 찾은 해당 구간 (창 밖으로 이어지면 창 끝까지만 칠함).
원본 PSG 파일은 읽기만 합니다.

결과: Result_py/psg_raw_eda/fig/q6_quality_cases_combined.png
사용: python psg_quality_cases_figure.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from osa.psg_read import read_ebm, read_ndf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py" / "psg_raw_eda" / "fig"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8})
BAD, OK, SHADE, INK, MUTED = "#b2432f", "#2a78d6", "#f6d9d3", "#1f2328", "#6b7280"

# (환자, 제목, 창 시작, 창 끝, 음영 시작, 음영 끝, [(파일, 표시 이름, 이상 채널 여부)], 음영 설명)
CASES = [
    ("P_28", "① P_28 (Nox) · Thorax RIP가 0으로 기록 / 복부·기류는 계속 기록",
     "2019-08-21T23:14:00", "2019-08-21T23:20:00", "2019-08-21T23:16:51", "2019-08-21T23:40:38",
     [("thorax rip.ndf", "Thorax RIP", True), ("abdomen rip.ndf", "Abdomen RIP", False),
      ("nasal pressure.ndf", "Nasal Pressure", False)],
     "Thorax RIP = 0 (23:16:51~23:40:38, 약 24분)"),
    ("P_43", "② P_43 (Embla) · SpO₂와 맥박이 함께 0으로 기록 / 비강압력은 계속 기록",
     "2024-05-30T05:14:00", "2024-05-30T05:38:00", "2024-05-30T05:17:37", "2024-05-30T05:34:38",
     [("SpO2.ebm", "SpO₂", True), ("Pulse.ebm", "Pulse", True), ("Nasal Pressure.ebm", "Nasal Pressure", False)],
     "SpO₂·Pulse = 0 (05:17:37~05:34:38, 약 17분, 시작 시 N1)"),
    ("P_59", "③ P_59 (Nox) · Thermistor가 이 채널의 최대값(1.024 V)에 고정 / 비강압력은 계속 기록",
     "2022-12-13T04:08:30", "2022-12-13T04:16:00", "2022-12-13T04:10:11", "2022-12-13T04:14:19",
     [("thermistor.ndf", "Thermistor", True), ("nasal pressure.ndf", "Nasal Pressure", False)],
     "Thermistor 최대값 고정 (04:10:11~04:14:19, 약 4분)"),
]


def window(ch, w0, w1, bins=1500):
    """창 안의 값. 표본이 많으면 구간별 최소·최대 띠로 줄임 (짧은 파형 모양 보존)."""
    i0, i1 = np.searchsorted(ch.time, [w0, w1])
    t, x = ch.time[i0:i1], ch.data[i0:i1]
    if x.size <= bins * 4:
        return t, x, None
    k = x.size // bins
    t = t[: k * bins].reshape(bins, k)[:, 0]
    x = x[: k * bins].reshape(bins, k)
    return t, np.nanmin(x, 1), np.nanmax(x, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [len(c[6]) for c in CASES]
    fig = plt.figure(figsize=(13, 2.0 + 1.05 * sum(rows) + 0.9 * len(CASES)), constrained_layout=True)
    outer = fig.add_gridspec(len(CASES), 1, height_ratios=[r + 0.9 for r in rows], hspace=0.08)
    for g, (pid, title, w0, w1, s0, s1, chans, shade_lab) in zip(outer, CASES):
        folder = next((ROOT / "Data/PSG").glob(f"{pid}_*"))
        reader = read_ndf if folder.name.endswith("_N") else read_ebm
        w0, w1, s0, s1 = (np.datetime64(v, "ns") for v in (w0, w1, s0, s1))
        sub = g.subgridspec(len(chans), 1, hspace=0.05)
        axes = []
        for k, (f, lab, bad) in enumerate(chans):
            ax = fig.add_subplot(sub[k], sharex=axes[0] if axes else None)
            ch = reader(folder / f)
            t, lo, hi = window(ch, w0, w1)
            col = BAD if bad else OK
            ax.axvspan(s0, min(s1, w1), color=SHADE, lw=0, zorder=0)
            if hi is None:
                ax.plot(t, lo, color=col, lw=0.8)
                vals = lo
            else:
                ax.fill_between(t, lo, hi, color=col, lw=0, step="post")
                for y in (lo, hi):      # 값이 계속 같으면 띠 폭이 0이라 선으로도 그림
                    ax.plot(t, y, color=col, lw=0.7, drawstyle="steps-post")
                vals = np.r_[lo, hi]
            f_ = vals[np.isfinite(vals)]
            if f_.size and f_.max() > f_.min():
                pad = (f_.max() - f_.min()) * 0.08
                ax.set_ylim(f_.min() - pad, f_.max() + pad)
            unit = "단위 미확인" if "verify" in ch.unit else ch.unit
            ax.set_ylabel(f"{lab}\n({unit})" if unit else lab, rotation=0, ha="right", va="center",
                          fontsize=8, color=BAD if bad else INK)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.grid(axis="x", color="#e5e7eb", lw=0.6)
            if k < len(chans) - 1:
                ax.tick_params(labelbottom=False)
            axes.append(ax)
            del ch
        axes[0].set_xlim(w0, w1)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[0].set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        axes[0].text(1.0, 1.02, f"음영 = {shade_lab}", transform=axes[0].transAxes, ha="right", va="bottom",
                     fontsize=8, color=BAD)
        print("drew", pid, flush=True)
    fig.suptitle("품질 의심 채널 3사례 · 빨강 = 의심 채널, 파랑 = 같은 시간의 비교 채널 · Y축은 창 안의 실제 최소~최대",
                 fontsize=11, color=INK)
    fig.savefig(OUT / "q6_quality_cases_combined.png", dpi=110)
    plt.close(fig)
    print("saved q6_quality_cases_combined")


if __name__ == "__main__":
    main()
