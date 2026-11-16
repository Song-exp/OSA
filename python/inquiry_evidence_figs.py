"""자료 확인 요청서의 원신호 근거 그림. 기존 psg_visual_check의 읽기·그림 함수를 그대로 쓴다. 원본은 읽기만 한다.

출력: Result_py/patient_matching/inquiry_evidence/figs/*.png (이미 있으면 건너뜀)
사용: python inquiry_evidence_figs.py
"""
from pathlib import Path

import numpy as np

import psg_guide_figures as pgf
import psg_visual_check as pvc

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"Result_py/patient_matching/inquiry_evidence/figs"


def longest_flat(c):
    """같은 값이 가장 길게 이어진 구간 (시작, 끝, 값)."""
    x = c.data; change = np.flatnonzero(np.diff(x) != 0)+1
    s, e = np.r_[0, change], np.r_[change, x.size]; i = np.argmax(e-s)
    return c.time[s[i]], c.time[e[i]-1], float(x[s[i]])


def window(pid, names, w0, w1, title, fname):
    if (FIG/f"{fname}.png").exists(): print("skip", fname); return
    rec, ev, st = pvc.load(pid, names)
    pgf.panel(rec, ev, st, [c.name for c in rec.channels], str(w0), str(w1), title, fname); print("saved", fname)


def main():
    FIG.mkdir(parents=True, exist_ok=True); pgf.OUT = FIG
    m = np.timedelta64
    # P_50 SpO2: 같은 값이 가장 길게 이어진 구간의 시작 15분 (맥박·호흡과 비교)
    if not (FIG/"p50_spo2_flat.png").exists():
        rec, _, _ = pvc.load("P_50", ["spo2"]); a, b, v = longest_flat(rec.channels[0]); del rec
        window("P_50", ["spo2", "pulse", "nasal pressure", "thorax rip"], a-m(2, "m"), min(b, a+m(13, "m")),
               f"P_50 · SpO2 {v:g}% 고정 구간 {str(a)[11:19]}~{str(b)[11:19]} ({(b-a)/m(1, 'm'):.0f}분) / 맥박·호흡과 비교", "p50_spo2_flat")
    # P_50 턱 근전도 1·2: 0 기록 최장 구간 시작부 (기존 점검 문서의 구간)
    window("P_50", ["1", "2", "nasal pressure", "spo2"], "2023-11-07T02:32:30", "2023-11-07T02:44:30",
           "P_50 · 턱 근전도 1·2 = 0 기록 (02:33:32~05:16:16) / 같은 시간 비강압력에는 호흡이 기록됨", "p50_chin_zero")
    # P_59 서미스터: 최대값 고정 구간 (기존 점검 문서의 구간)
    window("P_59", ["thermistor", "nasal pressure", "spo2"], "2022-12-13T04:07:00", "2022-12-13T04:17:30",
           "P_59 · 서미스터가 최대값(1.024 V)에 고정 04:10:11~04:14:19 / 같은 시간 비강압력은 정상", "p59_thermistor_rail")
    # Nox 판독층: 자동(Automatic)과 수동(Manual) 판독의 무호흡·저호흡을 같은 파형에 따로 표시
    for pid in ("P_75", "P_50"):    # P_76은 수동(Manual) 판독층 하나뿐이라 비교 그림 없음
        if (FIG/f"v11_layers_{pid}.png").exists(): print("skip layers", pid); continue
        pvc.layer_figure(pid); print("saved layers", pid)


if __name__ == "__main__":
    main()
