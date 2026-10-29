"""PSG 원신호 기초 EDA 문서(docs/PSG_원신호_기초_EDA_및_파형_이해.md)의 그림.

사례 파형은 psg_guide_figures.panel(뷰어와 같은 그리기 함수, Y축은 실제 최소~최대)로 그립니다.
Nox 사건 색칠은 기사 판독(Manual)층만 씁니다 (뷰어는 Automatic+Manual을 함께 칠함).
요약 그림은 Result_py/psg_raw_eda/raw_eda_quality.csv (psg_raw_eda.py 결과)를 씁니다.
결과: Result_py/psg_raw_eda/fig/*.png
사용: python psg_raw_eda_figures.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import psg_guide_figures as pgf
from osa.psg_events import read_events
from osa.psg_read import Recording, read_ebm, read_ndf
from psg_raw_eda import nox_stages
from psg_waveform_check import nox_sleep_and_events

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Result_py" / "psg_raw_eda" / "fig"
pgf.OUT = FIG                      # panel()이 저장하는 폴더만 바꿈
BLUE = "#2a78d6"


def load(pid: str, files: list[str]):
    """필요한 채널 파일만 읽어 가벼운 Recording을 만들고 판독(사건, 수면단계)을 붙임."""
    folder = next((ROOT / "Data/PSG").glob(f"{pid}_*"))
    nox = folder.name.endswith("_N")
    chs = [(read_ndf if nox else read_ebm)(folder / f) for f in files]
    ct = pd.DataFrame({"StartTime": [c.time[0] for c in chs], "EndTime": [c.time[-1] for c in chs]})
    rec = Recording("NDF" if nox else "EBM", str(folder), [], chs, ct)
    ev, st, _ = read_events(rec)
    if nox:
        _, ev = nox_sleep_and_events(folder)      # 기사층 무호흡·저호흡만
        st = nox_stages(folder)[0]                # 같은 30초에 두 층 단계가 있으면 Manual 우선
    return rec, ev, st


def cases():
    # ---- P_59 (Nox): 하룻밤 전체, 산소저하 반복, 폐쇄성 무호흡, 서미스터 최대값 고정
    rec, ev, st = load("P_59", ["spo2.ndf", "position.ndf", "nasal pressure.ndf", "thermistor.ndf",
                                "thorax rip.ndf", "abdomen rip.ndf", "pulse.ndf"])
    pgf.panel(rec, ev, st, ["SpO2", "Position", "Nasal Pressure", "Pulse"],
              "2022-12-12T22:28:00", "2022-12-13T05:15:00",
              "P_59 (Nox) · 하룻밤 전체 · 색칠 = 기사 판독 무호흡·저호흡", "c0_overview_P59")
    pgf.panel(rec, ev, st, ["Nasal Pressure", "Thermistor", "Thorax RIP", "Abdomen RIP", "SpO2", "Pulse"],
              "2022-12-13T00:30:00", "2022-12-13T00:40:00",
              "P_59 (Nox) · 10분 · 무호흡·저호흡과 산소저하가 반복되는 구간", "c1_desat_cycles_P59")
    pgf.panel(rec, ev, st, ["Nasal Pressure", "Thermistor", "Thorax RIP", "Abdomen RIP", "SpO2"],
              "2022-12-13T01:55:30", "2022-12-13T01:58:40",
              "P_59 (Nox) · 폐쇄성 무호흡 2회 (기사 판독: APNEA-OBSTRUCTIVE)", "c2_obstructive_nox_P59")
    pgf.panel(rec, ev, st, ["Thermistor", "Nasal Pressure", "Thorax RIP", "SpO2"],
              "2022-12-13T04:08:30", "2022-12-13T04:16:00",
              "P_59 (Nox) · 품질 의심: 서미스터가 이 채널의 최대값(1.024 V)에 붙음, 비강압력은 계속 기록", "q1_thermistor_rail_P59")
    del rec
    # ---- P_6 (Embla): 자세 변화, 수면단계 전환
    rec, ev, st = load("P_6", ["Position_CU.ebm", "Gravity X.ebm", "Gravity Y.ebm", "Thorax.ebm", "Abdomen.ebm",
                               "Chin.ebm", "C3.ebm", "LOC.ebm", "ROC.ebm", "SpO2.ebm", "Nasal Pressure.ebm"])
    pgf.panel(rec, ev, st, ["Position_CU", "Gravity X", "Gravity Y", "Thorax", "Abdomen", "Chin", "C3"],
              "2016-07-27T22:50:00", "2016-07-27T22:53:00",
              "P_6 (Embla) · 자세 변화 (22:50:50 무렵 Position_CU 값 약 51 → 약 -7, 단위 미확인)", "c3_position_change_P6")
    pgf.panel(rec, ev, st, ["C3", "LOC", "ROC", "Chin", "Nasal Pressure", "SpO2"],
              "2016-07-27T23:49:40", "2016-07-27T23:52:40",
              "P_6 (Embla) · 수면단계 전환 N2 → REM (판독 23:51:10)", "c4_n2_to_rem_P6")
    del rec


def nox_stage_layers():
    """뷰어가 그리는 Nox 수면단계(두 판독층 함께)와 Manual 우선 단계 비교."""
    rec, _, st_view = load_raw("P_59", ["spo2.ndf"])
    st_man = nox_stages(Path(rec.source_folder))[0]
    w0, w1 = np.datetime64("2022-12-13T03:55:00", "ns"), np.datetime64("2022-12-13T04:25:00", "ns")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 4.2), constrained_layout=True)
    for ax, st, lab in zip(axes, (st_view, st_man), ("① 뷰어 표시: Automatic·Manual 단계가 함께 그려짐",
                                                     "② Manual(기사) 우선으로 한 단계만")):
        pgf._draw_stages(ax, st, w0, w1)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlim(w0, w1)
    axes[-1].xaxis.set_major_formatter(pgf.mdates.DateFormatter("%H:%M"))
    fig.suptitle("P_59 (Nox) · 같은 30분의 수면단계", fontsize=11)
    fig.savefig(FIG / "q3_nox_stage_layers_P59.png", dpi=90)
    plt.close(fig)
    print("saved q3_nox_stage_layers_P59")


def load_raw(pid, files):
    """뷰어와 같은 판독 읽기 (read_events 그대로)."""
    folder = next((ROOT / "Data/PSG").glob(f"{pid}_*"))
    chs = [(read_ndf if folder.name.endswith("_N") else read_ebm)(folder / f) for f in files]
    ct = pd.DataFrame({"StartTime": [c.time[0] for c in chs], "EndTime": [c.time[-1] for c in chs]})
    rec = Recording("NDF" if folder.name.endswith("_N") else "EBM", str(folder), [], chs, ct)
    ev, st, _ = read_events(rec)
    return rec, ev, st


def p68():
    """P_68 SpO2 기록 공백. 채널 파일 이름은 P_68 폴더에서 확인한 것만 씀."""
    folder = next((ROOT / "Data/PSG").glob("P_68_*"))
    have = {f.stem for f in folder.glob("*.ebm")}
    files = [f"{n}.ebm" for n in ("SpO2", "Nasal Pressure", "Thorax", "Pulse", "C3") if n in have]
    rec, ev, st = load("P_68", files)
    pgf.panel(rec, ev, st, [Path(f).stem for f in files], "2023-06-19T21:00:00", "2023-06-19T23:15:00",
              "P_68 (Embla) · 품질 의심: SpO2·비강압력 기록이 빈 구간", "q2_spo2_gap_P68")
    del rec


def nox_quality():
    """5장 품질 점검에서 걸린 Nox 채널을 같은 시간축의 짝 채널과 비교."""
    rec, ev, st = load("P_28", ["thorax rip.ndf", "inductance thorax.ndf", "abdomen rip.ndf", "nasal pressure.ndf",
                                "spo2.ndf"])
    pgf.panel(rec, ev, st, ["Thorax RIP", "Inductance Thorax", "Abdomen RIP", "Nasal Pressure", "SpO2"],
              "2019-08-21T23:14:00", "2019-08-21T23:20:00",
              "P_28 (Nox) · 품질 의심: Thorax RIP가 0으로 기록 (23:16:51부터 약 24분), 복부·기류는 계속 기록",
              "q4_thorax_zero_P28")
    del rec
    rec, ev, st = load("P_50", ["1.ndf", "2.ndf", "f.ndf", "c3.ndf", "nasal pressure.ndf"])
    pgf.panel(rec, ev, st, ["1", "2", "F", "C3", "Nasal Pressure"],
              "2023-11-07T02:30:00", "2023-11-07T02:36:00",
              "P_50 (Nox) · 품질 의심: 턱 근전도 전극 1·2가 02:33:32부터 0으로 기록 (뇌파·기류는 계속)",
              "q5_chin_zero_P50")
    del rec


def summary():
    q = pd.read_csv(ROOT / "Result_py/psg_raw_eda/raw_eda_quality.csv", encoding="utf-8-sig")
    groups = ["SpO2", "NasalPressure", "Thermistor", "Thorax", "Abdomen", "EEG", "EOG", "ChinEMG", "ECG",
              "Position", "Snore"]
    m = q.pivot_table(index="Patient", columns="Group", values="Usable_Pct_Sleep", aggfunc="first")
    m = m.reindex(columns=groups)
    m = m.loc[sorted(m.index, key=lambda p: int(p[2:]))]
    dev = q.drop_duplicates("Patient").set_index("Patient").Device.reindex(m.index)
    fig, ax = plt.subplots(figsize=(7.5, 15), constrained_layout=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#d9e8f8", BLUE, "#0b3d78"])
    vals = np.clip(m.values, 80, 100)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=80, vmax=100)
    miss = np.isnan(m.values)
    for i, j in zip(*np.nonzero(miss)):
        ax.text(j, i, "없음", ha="center", va="center", fontsize=6, color="#555")
    for i, j in zip(*np.nonzero(~miss & (m.values < 95))):
        ax.text(j, i, f"{m.values[i, j]:.0f}", ha="center", va="center", fontsize=6, color="#000")
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
    ax.set_yticks(range(len(m)), [f"{p} ({d})" for p, d in zip(m.index, dev)], fontsize=6)
    ax.set_title("수면 중 사용 가능 비율 (%) · 95% 미만만 숫자 표시 · 80% 이하는 같은 색", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.3, label="%")
    fig.savefig(FIG / "s1_usable_heatmap.png", dpi=90)
    plt.close(fig)
    print("saved s1_usable_heatmap")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cases()
    nox_stage_layers()
    nox_quality()
    p68()
    summary()


if __name__ == "__main__":
    main()
