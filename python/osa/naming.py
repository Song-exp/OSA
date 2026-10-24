"""채널 이름 표준화. main_YJ_PSG_All.m의 standardizeSignals 변환."""
from __future__ import annotations

import re

# (표준 이름, EBM 원본 이름들, NDF 원본 이름들)
ALIASES = [
    ("ECG", ["EKG", "ECG"], ["EKG", "ECG"]),
    ("SpO2", ["SpO2"], ["SpO2"]),
    ("SpO2BeatToBeat", ["SpO2 BB"], ["SpO2 B-B"]),
    ("NasalPressure", ["Nasal Pressure"], ["Nasal Pressure"]),
    ("Thermistor", ["Thermistor"], ["Thermistor"]),
    ("Thorax", ["Thorax"], ["Thorax RIP"]),
    ("Abdomen", ["Abdomen"], ["Abdomen RIP"]),
    ("Pleth", ["Plethysmogram"], ["Pleth"]),
    ("Pulse", ["Pulse"], ["Pulse"]),
    ("HeartRate", ["Heart Rate_CU", "Heart Rate_DR", "Heart Rate"], ["Heart Rate"]),
    ("RespRate", ["RR_CU", "RR_DR"], ["Resp Rate"]),
    ("LeftLegEMG", ["Leg-L"], ["Left Leg"]),
    ("RightLegEMG", ["Leg-R"], ["Right Leg"]),
    ("EEG_C3", ["C3"], ["C3"]),
    ("EEG_C4", ["C4"], ["C4"]),
    ("EEG_F3", ["F3"], ["F3"]),
    ("EEG_F4", ["F4"], ["F4"]),
    ("EEG_O1", ["O1"], ["O1"]),
    ("EEG_O2", ["O2"], ["O2"]),
]


def make_valid_name(s: str) -> str:
    """matlab.lang.makeValidName: 공백 제거 후 다음 글자 대문자, 기타 문자는 '_'."""
    s = s.strip()
    s = re.sub(r"\s+(\w)", lambda m: m.group(1).upper(), s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if not s or not s[0].isalpha():
        s = "x" + s
    return s[:63]


def standard_names(fmt: str, names: list[str]) -> tuple[list[str], list[bool], list[str]]:
    """Returns (채널별 표준 이름, 공통 신호 여부, 전체 공통 이름 목록)."""
    col = 1 if fmt == "EBM" else 2
    low = [n.strip().lower() for n in names]
    std, used = [""] * len(names), [False] * len(names)
    for alias in ALIASES:
        match = [i for i, n in enumerate(low) if n in {a.lower() for a in alias[col]}]
        assert len(match) <= 1, f"공통 이름 {alias[0]}에 대응하는 채널이 둘 이상입니다."
        if match:
            std[match[0]], used[match[0]] = alias[0], True
    taken = {a[0] for a in ALIASES}
    for i, n in enumerate(names):
        if used[i]:
            continue
        key = make_valid_name(f"{fmt}_{n}")
        if key in taken:
            key = f"{key}_Channel{i + 1}"
        std[i] = key
        taken.add(key)
    return std, used, [a[0] for a in ALIASES]


if __name__ == "__main__":
    assert make_valid_name("EBM_Heart Rate Variability") == "EBM_HeartRateVariability"
    assert make_valid_name("EBM_R-R Intervals") == "EBM_R_RIntervals"
    assert make_valid_name("EBM_SpO2-Quality_CU") == "EBM_SpO2_Quality_CU"
    assert make_valid_name("NDF_c3 impedance") == "NDF_c3Impedance"
    print("ok")
