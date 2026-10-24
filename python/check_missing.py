"""원본 검사 데이터(PSG 원신호·판독 파일, CBCT 영상)의 결측 점검. 엑셀/PDF는 보지 않습니다.

PSG : 핵심 채널 유무, 기록 중 신호가 비는 구간 비율, SpO2 이상값 비율,
      판독 파일에 들어 있는 정보(수면단계, 호흡 이벤트, 자발 각성, 산소저하, 자세)
CBCT: 상위 폴더 장수, 간격 규칙성, 촬영 범위(mm)
결과: Result_py/missing_raw.csv
"""
import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from osa.cbct import _read
from osa.naming import standard_names
from osa.psg_events import _read_text
from osa.psg_read import read_psg

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
KEY = {"SpO2": ["SpO2"], "Flow_NasalPressure": ["NasalPressure"], "Flow_Thermistor": ["Thermistor"],
       "Effort_Thorax": ["Thorax"], "Effort_Abdomen": ["Abdomen"], "ECG": ["ECG"],
       "EEG": ["EEG_C3", "EEG_C4", "EEG_F3", "EEG_F4", "EEG_O1", "EEG_O2"], "LegEMG": ["LeftLegEMG", "RightLegEMG"]}


def gap_fraction(ch) -> float:
    span = (ch.time[-1] - ch.time[0]) / np.timedelta64(1, "s")
    return float(max(0.0, 1 - ch.data.size / ch.fs / span)) if span > 0 and np.isfinite(ch.fs) else np.nan


def scoring_info(rec) -> dict:
    folder = Path(rec.source_folder)
    if rec.format == "EBM":
        f, s = folder / "EN-Apnea Events.txt", folder / "EN-Stage Events.txt"
        txt = _read_text(f, "cp949") if f.exists() else ""
        inc = re.search(r"Events Included:\s*(.*?)\n\s*\n", txt, re.S)
        inc = {x.strip() for x in inc.group(1).splitlines()} if inc else set()
        header = next((l for l in txt.splitlines() if "Event" in l and "Time" in l), "")
        return {"Stages": s.exists(), "RespEvents": f.exists(),
                "SpontArousal": any("SPONT" in x or x == "AROUSAL" for x in inc),
                "DesatEvents": any("DESAT" in x or "SPO2" in x for x in inc),
                "Position": "Position" in header, "EventTypesExported": len(inc)}
    db = sqlite3.connect(f"file:{(folder / 'Data.ndb').as_posix()}?mode=ro", uri=True)
    try:
        tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        types = set()
        for t in ("scoring_marker", "temporary_scoring_marker"):
            if t in tabs:
                types |= {r[0] for r in db.execute(f"select distinct type from {t} where is_deleted=0")}
    finally:
        db.close()
    has = lambda p: any(x.startswith(p) for x in types)
    return {"Stages": has("sleep-"), "RespEvents": has("apnea") or has("hypopnea"),
            "SpontArousal": "arousal-spontaneous" in types, "DesatEvents": "oxygensaturation-drop" in types,
            "Position": has("position-"), "EventTypesExported": len(types)}


def main():
    rows = []
    for d in sorted((ROOT / "Data/PSG").iterdir(), key=lambda p: int(re.match(r"P_(\d+)", p.name).group(1))):
        pid = int(re.match(r"P_(\d+)", d.name).group(1))
        print(f"P_{pid}", flush=True)
        rec = read_psg(d)
        std, _, _ = standard_names(rec.format, [c.name for c in rec.channels])
        chs = dict(zip(std, rec.channels))
        row = {"Patient": f"P_{pid}", "PSG": rec.format, "Channels": len(rec.channels),
               "RecordHours": round((rec.channel_table.EndTime.max() - rec.channel_table.StartTime.min())
                                    / pd.Timedelta(hours=1), 2)}
        for k, names in KEY.items():
            row[k] = any(n in chs for n in names)
        row["EOG"] = any(c.name.upper() in ("LOC", "ROC", "E1", "E2") for c in rec.channels)
        row["ChinEMG"] = any("CHIN" in c.name.upper() or c.name.strip() in ("1", "2") for c in rec.channels)
        row["PositionChannel"] = any("POSITION" in c.name.upper() or "AXIS" in c.name.upper() for c in rec.channels)
        if "SpO2" in chs:
            s = chs["SpO2"]
            row["SpO2_GapPct"] = round(100 * gap_fraction(s), 2)
            bad = ~np.isfinite(s.data) | (s.data < 50) | (s.data > 100)
            row["SpO2_InvalidPct"] = round(100 * bad.mean(), 2)
        if "NasalPressure" in chs:
            row["Flow_GapPct"] = round(100 * gap_fraction(chs["NasalPressure"]), 2)
        row.update(scoring_info(rec))
        # CBCT
        c = ROOT / "Data/CBCT" / f"P_{pid}_CBCT"
        fs = sorted(c.glob("*.dcm"))
        h = _read(fs[0], True)
        ps = [float(v) for v in h.PixelSpacing]
        row.update(CBCT_TopSlices=len(fs), CBCT_FOV_mm=f"{h.Rows * ps[0]:.0f}x{h.Columns * ps[1]:.0f}x{len(fs) * ps[0]:.0f}")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "Result_py/missing_raw.csv", index=False, encoding="utf-8-sig")
    print(df.to_string())


if __name__ == "__main__":
    main()
