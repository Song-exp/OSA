"""엑셀 No.와 Data 폴더 P_n이 같은 환자인지 대조합니다. 환자 값은 출력하지 않고 일치 여부만 보고합니다.

대조 항목: CBCT 슬라이스 간격, 이벤트 6종 개수, 총수면시간(TST).
결과: Result_py/linkage_check.csv (git 제외 폴더)
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from osa.cbct import _read
from osa.psg_events import COUNT_NAMES, read_events
from osa.psg_read import read_psg

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None)
names = {11: "TST", 3: "SliceMm", **{16 + i: n for i, n in enumerate(COUNT_NAMES)}}
rows = x.iloc[3:]
rows = rows[rows[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
xl = pd.DataFrame({"No": rows[0].astype(int).to_numpy(),
                   **{n: pd.to_numeric(rows[c], errors="coerce").to_numpy() for c, n in names.items()}})

psg = {int(re.match(r"P_(\d+)_", d.name).group(1)): d for d in (ROOT / "Data/PSG").iterdir()}
cbct = {int(re.match(r"P_(\d+)_", d.name).group(1)): d for d in (ROOT / "Data/CBCT").iterdir()}

out = []
for _, r in xl.iterrows():
    pid = int(r.No)
    row = {"Patient": f"P_{pid}", "HasPSG": pid in psg, "HasCBCT": pid in cbct}
    if pid in cbct:
        f = next(cbct[pid].glob("*.dcm"))
        row["SliceMatch"] = bool(np.isclose(float(_read(f, True).PixelSpacing[0]), r.SliceMm, atol=1e-6))
    if pid in psg:
        rec = read_psg(psg[pid])
        ev, st, ct = read_events(rec)
        diffs = {n: ct.set_index("EventType").Count[n] - r[n] for n in COUNT_NAMES}
        row["EventsMatch"] = all(d == 0 for d in diffs.values())
        row["EventDiff"] = ";".join(f"{n}:{d:+.0f}" for n, d in diffs.items() if d != 0)
        sleep = st[st.Level.between(1, 5)]
        tst = ((sleep.EndTime - sleep.StartTime).dt.total_seconds().sum()) / 60
        row["TSTDiffMin"] = round(tst - r.TST, 1)
    out.append(row)
    print(row, flush=True)

df = pd.DataFrame(out)
df.to_csv(ROOT / "Result_py/linkage_check.csv", index=False, encoding="utf-8-sig")
print("\n엑셀 환자 수", len(df), "| PSG 폴더 있음", int(df.HasPSG.sum()), "| CBCT 폴더 있음", int(df.HasCBCT.sum()))
print("슬라이스 간격 일치", int(df.SliceMatch.sum()), "/", int(df.SliceMatch.notna().sum()))
print("이벤트 6종 개수 완전 일치", int(df.EventsMatch.sum()), "/", int(df.EventsMatch.notna().sum()))
print("TST 차이 |≤1분|", int((df.TSTDiffMin.abs() <= 1).sum()), "/", int(df.TSTDiffMin.notna().sum()))
