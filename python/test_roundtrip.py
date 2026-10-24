"""저장(h5) → 읽기 왕복이 원본 읽기 결과와 같은지 확인. 사용: python test_roundtrip.py [PSG폴더] [CBCT폴더]"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from osa.batch import load_cbct, load_psg, save_cbct, save_psg
from osa.cbct import read_cbct
from osa.naming import standard_names
from osa.psg_events import read_events
from osa.psg_read import read_psg

ROOT = Path(__file__).resolve().parents[1]
psg_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Data/PSG/P_12_PSG_N"
cbct_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "Data/CBCT/P_1_CBCT"

with tempfile.TemporaryDirectory() as tmp:
    rec = read_psg(psg_dir)
    ev, st, ct = read_events(rec)
    save_psg(Path(tmp) / "p.h5", rec, ev, st, ct, {"SavedAt": datetime.now()})
    back = load_psg(Path(tmp) / "p.h5")
    std, _, _ = standard_names(rec.format, [c.name for c in rec.channels])
    for key, ch in zip(std, rec.channels):
        s = back["Signals"][key]
        assert np.array_equal(s["data"], ch.data), key
        assert np.array_equal(s["time"], ch.time), key
        assert s["unit"] == ch.unit and (s["fs"] == ch.fs or np.isnan(ch.fs)), key
    assert back["Events"].equals(ev) and back["SleepStages"].equals(st), "events/stages"
    assert np.array_equal(back["EventCounts"].Count, ct.Count, equal_nan=True)

    d = read_cbct(cbct_dir)
    save_cbct(Path(tmp) / "c.h5", d, {"SavedAt": datetime.now()})
    c = load_cbct(Path(tmp) / "c.h5")
    assert np.array_equal(c["Volume"], d.volume) and np.array_equal(c["RescaledVolume"], d.rescaled)
    assert np.allclose(c["VoxelSizeMm"], d.voxel_size_mm, equal_nan=True)
print("roundtrip ok")
