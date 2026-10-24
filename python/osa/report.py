"""PSG 리포트 PDF(KHMC 수면센터 양식, Embla/Nox 공통) 파싱. 식별정보(이름, 생년월일, 등록번호)는 반환하지 않습니다.

2012년 이전 Embla 리포트는 'AHI 1 / AHI 2'처럼 저호흡 기준이 다른 값이 두 개 있습니다.
'AHI 2'(A+H 표와 같은 값)를 사용합니다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pypdf


def _num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else np.nan


def _rem_nrem(text):
    """'Number in REM  REM Index  Number in NREM  NREM Index' 표의 Apnea, Hypopnea 행."""
    i = text.find("Number in REM")
    if i < 0:
        return np.nan, np.nan
    rem = nrem = 0.0
    found = 0
    for name in ("Apnea", "Hypopnea"):
        m = re.search(rf"\n\s*{name}(?: \(All\))?:?\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)", text[i:i + 800])
        if m:
            rem += float(m.group(2))
            nrem += float(m.group(4))
            found += 1
    return (rem, nrem) if found == 2 else (np.nan, np.nan)


def parse_report(path) -> dict:
    t = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    # 두 기준이 있으면 '2', 없으면 기본값
    two = lambda name: (_num(rf"{name}\s*2\s*:\s*([\d.]+)", t)
                        if re.search(rf"{name}\s*2\s*:", t) else _num(rf"{name}\s*:\s*([\d.]+)", t))
    oa = _num(r"Obstructive Apnea:\s*\d+\s+([\d.]+)", t)
    ca = _num(r"Central Apnea:\s*\d+\s+([\d.]+)", t)
    ma = _num(r"Mixed Apnea:\s*\d+\s+([\d.]+)", t)
    rem, nrem = _rem_nrem(t)
    study = re.search(r"Study Date:\s*([\d-]+)", t)
    return {
        "Study": study.group(1) if study else None,
        "TST": _num(r"Total Sleep Time:\s*([\d.]+)", t),
        "WASO": _num(r"Wake After Sleep Onset:\s*([\d.]+)", t),
        "AH_count": _num(r"\(A\+H\):\s*(\d+)", t),
        "AHI": _num(r"\(A\+H\):\s*\d+\s+([\d.]+)", t),
        "AHI_SUP": two("Supine AHI"),
        "AI": oa + ca + ma,
        "CAI": ca,
        "HI": _num(r"Hypopnea(?: \(All\))?:?\s+\d+\s+([\d.]+)\s*/\s*h", t),
        "RDI": two(r"(?<!Supine )RDI"),
        "ODI": _num(r"\(OD\):\s*\d+\s+([\d.]+)", t),
        "LSaO2": _num(r"Lowest O2 Sat\.\s*:\s*([\d.]+)", t),
        "ArI": _num(r"Total Arousals\s+\d+\s+([\d.]+)", t),
        "AHI_REM": rem,
        "AHI_NREM": nrem,
        "ESS": _num(r"ESS\s*:\s*(\d+)", t),
    }


def reports_in(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if re.match(r"(?i)report.*\.pdf$", p.name))
