"""PSG 환자별 분포 시각화: 엑셀 PSG 요약 지표 5개(TST, AHI, RDI, ODI, Lowest SpO2)의 87명 분포.

docs/PSG Excel EDA.md '주요 지표의 기초통계' 표와 같은 엑셀 값을 씁니다 (원본 엑셀은 읽기만 함).
P_65·P_67·P_69는 엑셀 AHI·RDI·ODI·LSaO2가 리포트와 다른 3명이라 그림에서 따로 표시만 하고 값은 그대로 둡니다.

결과: Result_py/psg_patient_distribution/
  psg_patient_summary.csv   지표별 평균·중앙값·최솟값·최댓값·사분위수 (문서 표와 대조용)
  psg_patient_values.csv    환자별 5개 지표와 AHI 중증도 구간
  d1_indicator_distribution.png   5개 지표 분포 (히스토그램 + 환자 점)
  d2_ahi_by_patient.png           환자별 AHI (작은 값부터 정렬, 중증도 기준선)
  d3_ahi_severity_count.png       AHI 중증도 구간별 환자 수
사용: python psg_patient_distribution.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXCEL = ROOT / "Patient_Info_Data_87명_261019.xlsx"
OUT = ROOT / "Result_py" / "psg_patient_distribution"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8})

# 엑셀 열 번호 (0부터) ↔ 머리글 3행 이름
COLS = {11: ("TST", "Total sleep time", "TST (분)"), 12: ("AHI", "AHI", "AHI (회/시간)"),
        13: ("RDI", "RDI", "RDI (회/시간)"), 14: ("ODI", "ODI", "ODI (회/시간)"),
        15: ("LSaO2", "Lowest\nSpO2", "Lowest SpO₂ (%)")}
UNVERIFIED = ["P_65", "P_67", "P_69"]
BINS = [(0, 5, "정상 범위 (<5)"), (5, 15, "경증 (5~15)"), (15, 30, "중등도 (15~30)"), (30, np.inf, "중증 (≥30)")]
SEV_COLORS = ["#d9e8f8", "#8fbbe9", "#2a78d6", "#0b3d78"]   # 한 색(파랑)의 밝기 단계
BLUE, INK, MUTED, GRID = "#2a78d6", "#1f2328", "#6b7280", "#e5e7eb"


def read_excel() -> pd.DataFrame:
    x = pd.read_excel(EXCEL, header=None)
    for c, (_, name, _) in COLS.items():
        assert str(x.iloc[2, c]).strip() == name, f"엑셀 {c}열 머리글이 예상과 다름: {x.iloc[2, c]!r}"
    r = x.iloc[3:]
    r = r[r[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    df = pd.DataFrame({k: pd.to_numeric(r[c].values) for c, (k, _, _) in COLS.items()},
                      index=[f"P_{int(n)}" for n in r[0]])
    df.index.name = "Patient"
    return df


def severity(ahi: float) -> str:
    return next(lab for lo, hi, lab in BINS if lo <= ahi < hi)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({"N": df.count(), "Mean": df.mean(), "Median": df.median(), "Min": df.min(),
                      "Q1": df.quantile(0.25), "Q3": df.quantile(0.75), "Max": df.max(), "SD": df.std()})
    return s.round(1)


def fig_distribution(df: pd.DataFrame, s: pd.DataFrame):
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (_, (k, _, label)) in zip(axes, COLS.items()):
        v = df[k]
        ax.hist(v, bins=20, color="#d9e8f8", edgecolor="white", lw=2)
        ymax = ax.get_ylim()[1]
        # 환자 한 명 = 점 하나 (히스토그램 위 띠, 겹치지 않게 세로로만 흩뜨림)
        y = ymax * (1.12 + rng.uniform(-0.05, 0.05, len(v)))
        ok = ~v.index.isin(UNVERIFIED)
        ax.scatter(v[ok], y[ok], s=14, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(v[~ok], y[~ok], s=36, marker="D", facecolor="white", edgecolor=INK, lw=1, zorder=4)
        for j, p in enumerate(v.index[~ok]):
            ax.annotate(p, (v[p], y[v.index.get_loc(p)]), xytext=(0, 6 if j % 2 == 0 else -13), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK)
        mean, med = s.loc[k, "Mean"], s.loc[k, "Median"]
        ax.axvline(mean, color=INK, lw=1.2, ls="--")
        ax.axvline(med, color=INK, lw=1.2)
        ax.set_ylim(0, ymax * 1.3)
        ax.set_ylabel("환자 수")
        ax.set_title(f"{label} · 평균 {mean:.1f} (점선) / 중앙값 {med:.1f} (실선) · 범위 {s.loc[k, 'Min']:g}~{s.loc[k, 'Max']:g}",
                     loc="left", fontsize=9, color=INK)
        if k == "AHI":
            for cut in (5, 15, 30):
                ax.axvline(cut, color=MUTED, lw=0.8, ls=":")
                ax.text(cut, ymax * 0.98, f" {cut}", color=MUTED, fontsize=7, va="top")
        style(ax)
    fig.suptitle("PSG 환자별 분포 시각화 · 엑셀 87명 · 위쪽 점 = 환자 1명, 막대 = 구간별 환자 수\n"
                 "◇ = P_65·P_67·P_69 (엑셀 값이 리포트와 다른 3명, 값은 엑셀 그대로)", fontsize=10, color=INK)
    fig.savefig(OUT / "d1_indicator_distribution.png", dpi=110)
    plt.close(fig)


def fig_ahi_by_patient(df: pd.DataFrame, s: pd.DataFrame):
    v = df.AHI.sort_values()
    sev = v.map(severity)
    colors = [SEV_COLORS[[b[2] for b in BINS].index(x)] for x in sev]
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(v))
    ax.bar(x, v.values, width=0.8, color=colors, edgecolor="white", lw=0.5)
    for i, p in enumerate(v.index):
        if p in UNVERIFIED:
            ax.annotate("◇", (i, v[p]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK)
    for cut in (5, 15, 30):
        ax.axhline(cut, color=MUTED, lw=0.8, ls=":")
        ax.text(1.003, cut, f"{cut}", color=MUTED, fontsize=7, va="center", ha="left",
                transform=ax.get_yaxis_transform())
    ax.axhline(s.loc["AHI", "Mean"], color=INK, lw=1.2, ls="--")
    ax.axhline(s.loc["AHI", "Median"], color=INK, lw=1.2)
    ax.text(0, s.loc["AHI", "Mean"], f" 평균 {s.loc['AHI', 'Mean']:.1f}", va="bottom", fontsize=8, color=INK)
    ax.text(0, s.loc["AHI", "Median"], f" 중앙값 {s.loc['AHI', 'Median']:.1f}", va="top", fontsize=8, color=INK)
    ax.set_xticks(x, v.index, rotation=90, fontsize=6)
    ax.set_xlim(-0.6, len(v) - 0.4)
    ax.set_ylabel("AHI (회/시간)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEV_COLORS]
    n = sev.value_counts()
    ax.legend(handles, [f"{lab} {n.get(lab, 0)}명" for _, _, lab in BINS], loc="upper left", frameon=False)
    ax.set_title("환자별 AHI (작은 값부터 정렬) · 점선 5/15/30 = 중증도 기준 · ◇ = P_65·P_67·P_69 (엑셀 값)",
                 loc="left", fontsize=10, color=INK)
    style(ax)
    fig.savefig(OUT / "d2_ahi_by_patient.png", dpi=110)
    plt.close(fig)


def fig_severity(df: pd.DataFrame):
    n = df.AHI.map(severity).value_counts().reindex([b[2] for b in BINS], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    ax.barh(range(len(n))[::-1], n.values, color=SEV_COLORS, height=0.6)
    for i, (lab, c) in enumerate(n.items()):
        ax.text(c + 0.8, len(n) - 1 - i, f"{c}명 ({c / n.sum() * 100:.1f}%)", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(n))[::-1], n.index)
    ax.set_xlim(0, n.max() * 1.3)
    ax.set_xlabel("환자 수")
    ax.set_title(f"AHI 중증도 구간별 환자 수 (엑셀 {n.sum()}명)", loc="left", fontsize=10, color=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.savefig(OUT / "d3_ahi_severity_count.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_excel()
    s = summary(df)
    s.to_csv(OUT / "psg_patient_summary.csv", encoding="utf-8-sig")
    df.assign(AHI_Severity=df.AHI.map(severity), ExcelDiffersFromReport=df.index.isin(UNVERIFIED)) \
      .to_csv(OUT / "psg_patient_values.csv", encoding="utf-8-sig")
    print(s.to_string())
    print(df.AHI.map(severity).value_counts().to_string())
    fig_distribution(df, s)
    fig_ahi_by_patient(df, s)
    fig_severity(df)


if __name__ == "__main__":
    main()
