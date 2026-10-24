"""환자별 데이터 품질 점검. 원본은 읽기만 하고, 식별정보는 결과에 남기지 않습니다 (일치 여부만 기록).

점검 항목
  CBCT : 시리즈 개수/하위 폴더 혼입, 상위 영상이 본인인지(PSG 리포트와 생년월일·성별·등록번호 대조),
         하위 폴더 영상의 실제 주인, CBCT-PSG 검사 간격
  PSG  : 리포트 개수, 원신호가 어느 검사인지, 엑셀 값이 어느 검사인지,
         다른 날 채널 혼입, 판독(TST, 무호흡+저호흡 개수)이 리포트와 맞는지
         - N(Nox)은 현재 규칙과 후보 규칙(Manual 우선)을 함께 계산
  엑셀 : NAV/PAV 결측
결과: Result_py/quality_flags.csv
사용: python check_quality.py [ID ...]
"""
from __future__ import annotations

import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pypdf

from osa.cbct import _read
from osa.psg_events import read_events
from osa.psg_read import read_psg

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RESP = {"APNEA-OBSTRUCTIVE", "APNEA-CENTRAL", "APNEA-MIXED", "HYPOPNEA"}
TST_TOL, AH_TOL = 1.0, 2          # 분, 개수


# ------------------------------------------------------------ 리포트
def read_reports(folder: Path) -> list[dict]:
    out = []
    for f in sorted(p for p in folder.iterdir() if re.match(r"(?i)report.*\.pdf$", p.name)):
        t = " ".join(pg.extract_text() or "" for pg in pypdf.PdfReader(f).pages)
        g = lambda pat: (m.group(1) if (m := re.search(pat, t)) else None)
        num = lambda pat: float(v) if (v := g(pat)) else np.nan
        out.append({"file": f.name, "study": g(r"Study Date:\s*([\d-]+)"),
                    "dob": (g(r"Date of Birth:\s*([\d-]+)") or "").replace("-", ""),
                    "id": g(r"\bID:\s*(\d+)"), "sex": (g(r"Gender:\s*(\w)") or "").upper(),
                    "tst": num(r"Total Sleep Time:\s*([\d.]+)"),
                    "ah": num(r"\(A\+H\):\s*(\d+)")})
    return out


# ------------------------------------------------------------ Nox 후보 규칙
def nox_candidate(db_file: Path) -> tuple[float, int]:
    """후보 규칙: 임시 판독본에 수면단계가 있으면 그것을, 없으면 저장 판독본(revision)을 씀.
    수면단계는 같은 30초 구간이면 Manual이 Automatic을 덮어씀.
    이벤트는 Manual 층에 호흡 이벤트가 있으면 Manual만, 없으면 Automatic.
    ponytail: 이벤트 규칙은 미검증 후보. 리포트와 맞지 않는 환자를 보고 다듬을 것.
    """
    db = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
    try:
        tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}

        def load(pre, keys_sql):
            keys = dict(db.execute(f"select id, type from {pre}_key").fetchall())
            m = pd.DataFrame(db.execute(
                f"select starts_at, ends_at, type, key_id from {pre}_marker "
                f"where is_deleted=0 and key_id in ({keys_sql})").fetchall(), columns=["a", "b", "type", "key"])
            m["manual"] = m.key.map(keys).eq("Manual")
            return m

        layers = []
        if "temporary_scoring_marker" in tabs:
            g = db.execute("select id from temporary_scoring_group where is_deleted=0").fetchall()
            if len(g) == 1:
                layers.append(load("temporary_scoring",
                                   f"select key_id from temporary_scoring_group_to_key where group_id={g[0][0]}"))
        if "scoring_marker" in tabs:
            r = db.execute("select id from scoring_revision where is_deleted=0").fetchall()
            if r:
                layers.append(load("scoring", f"select key_id from scoring_revision_to_key where revision_id={r[-1][0]}"))
    finally:
        db.close()
    m = next((x for x in layers if x.type.str.startswith("sleep-").sum() > 10), layers[0] if layers else None)
    if m is None:
        return np.nan, -1
    s = m[m.type.str.startswith("sleep-")].sort_values("manual").drop_duplicates("a", keep="last")
    tst = (s.b.astype("int64") - s.a.astype("int64"))[s.type != "sleep-wake"].sum() / 1e7 / 60
    ev = m[m.type.str.upper().isin(RESP)]
    return round(tst, 1), int(ev.manual.sum() if ev.manual.any() else len(ev))


def stage_tst(stages: pd.DataFrame) -> float:
    s = stages[stages.Level.between(1, 5)]
    return round((s.EndTime - s.StartTime).dt.total_seconds().sum() / 60, 1)


# ------------------------------------------------------------ CBCT
def dicom_series(folder: Path):
    """폴더(하위 포함)의 시리즈별 (상대경로 폴더, 장수, 헤더 하나)."""
    groups = {}
    for f in folder.rglob("*.dcm"):
        groups.setdefault(f.parent, []).append(f)
    out = []
    for d, fs in groups.items():
        h = _read(fs[0], True)
        out.append({"dir": d, "top": d == folder, "n": len(fs), "dob": str(h.get("PatientBirthDate", "")),
                    "sex": str(h.get("PatientSex", ""))[:1].upper(), "id": str(h.get("PatientID", "")),
                    "date": str(h.get("StudyDate", "")), "uid": str(h.get("SeriesInstanceUID", ""))})
    return out


def identity(h: dict, reports: list[dict]) -> str:
    if any(h["dob"] and h["dob"] == r["dob"] and h["sex"] == r["sex"] for r in reports):
        return "match"
    if any(h["id"] and h["id"] == r["id"] for r in reports):
        return "id_only"          # 등록번호는 같고 생년월일/성별이 다름 → 입력 오류 가능
    return "mismatch"


# ------------------------------------------------------------ 메인
def main(wanted: set[int]):
    pid_of = lambda p: int(re.match(r"P_(\d+)_", p.name).group(1))
    psg_dirs = {pid_of(d): d for d in (ROOT / "Data/PSG").iterdir() if d.is_dir()}
    cbct_dirs = {pid_of(d): d for d in (ROOT / "Data/CBCT").iterdir() if d.is_dir()}
    x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None)
    xr = x.iloc[3:]
    xr = xr[xr[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    excel = {int(n): {"tst": t, "nav": a, "pav": b} for n, t, a, b in zip(
        xr[0], pd.to_numeric(xr[11], errors="coerce"), pd.to_numeric(xr[29], errors="coerce"),
        pd.to_numeric(xr[30], errors="coerce"))}

    reports = {p: read_reports(d) for p, d in psg_dirs.items()}       # 하위 폴더 주인 찾기에 전체 필요
    ids = sorted(i for i in psg_dirs if not wanted or i in wanted)
    rows = []
    for k, pid in enumerate(ids, 1):
        print(f"[{k}/{len(ids)}] P_{pid}", flush=True)
        reps, row = reports[pid], {"Patient": f"P_{pid}"}
        # ---- PSG
        rec = read_psg(psg_dirs[pid])
        events, stages, counts = read_events(rec)
        sig_date = pd.Timestamp(rec.channel_table.StartTime.min()).normalize()
        near = lambda r: r["study"] and abs((pd.Timestamp(r["study"]) - sig_date).days) <= 1
        sig_rep = next((i for i, r in enumerate(reps) if near(r)), None)
        xl_rep = next((i for i, r in enumerate(reps) if abs(r["tst"] - excel[pid]["tst"]) <= 0.6), None)
        row.update(PSG=rec.format, Reports=len(reps), SignalReport=None if sig_rep is None else sig_rep + 1,
                   SignalIsLatest=sig_rep == len(reps) - 1 if sig_rep is not None else False,
                   ExcelReport=None if xl_rep is None else xl_rep + 1,
                   ExcelMatchesSignal=xl_rep is not None and xl_rep == sig_rep,
                   ExcludedChannels=len(rec.excluded_channels))
        ref = reps[sig_rep] if sig_rep is not None else {"tst": np.nan, "ah": np.nan}
        ah_now = int(events.EventType.isin(RESP).sum())
        tst_now = stage_tst(stages)
        row.update(ReportTST=ref["tst"], TST_now=tst_now, ReportAH=ref["ah"], AH_now=ah_now)
        if rec.format == "NDF":
            tst_c, ah_c = nox_candidate(psg_dirs[pid] / "Data.ndb")
            row.update(TST_candidate=tst_c, AH_candidate=ah_c)
        ok = lambda a, b, tol: bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)
        tst_final = row.get("TST_candidate", tst_now)
        ah_final = row.get("AH_candidate", ah_now)
        row.update(TST_OK=ok(tst_final, ref["tst"], TST_TOL), AH_OK=ok(ah_final, ref["ah"], AH_TOL),
                   ReportHasAH=bool(np.isfinite(ref["ah"])))
        # ---- CBCT
        series = dicom_series(cbct_dirs[pid])
        top = next((s for s in series if s["top"]), None)
        subs = [s for s in series if not s["top"]]
        row.update(CBCT_Series=len(series), CBCT_TopSlices=top["n"] if top else 0,
                   CBCT_Identity=identity(top, reps) if top else "no_top",
                   CBCT_PSG_GapDays=(pd.Timestamp(top["date"]) - sig_date).days if top and top["date"] else np.nan)
        owners = []
        for s in subs:
            who = [f"P_{q}" for q, rr in reports.items() if identity(s, rr) != "mismatch"]
            owners.append(f"{s['n']}장 {s['date'][:6]} → {'/'.join(who) or '외부인'}")
        row["CBCT_Subfolders"] = "; ".join(owners)
        # ---- 엑셀
        row.update(Excel_NAV_missing=not np.isfinite(excel[pid]["nav"]),
                   Excel_PAV_missing=not np.isfinite(excel[pid]["pav"]))
        # ---- 플래그
        f = []
        if row["CBCT_Series"] > 1: f.append("CBCT_혼입")
        if row["CBCT_Identity"] == "mismatch": f.append("CBCT_본인아님")
        if row["CBCT_Identity"] == "id_only": f.append("CBCT_생년월일불일치")
        if row["Reports"] > 1: f.append("PSG_복수검사")
        if not row["ExcelMatchesSignal"]: f.append("엑셀≠원신호검사")
        if row["ExcludedChannels"]: f.append("PSG_다른날채널")
        if not row["TST_OK"]: f.append("TST불일치")
        if row["ReportHasAH"] and not row["AH_OK"]: f.append("이벤트불일치")
        if not row["ReportHasAH"]: f.append("리포트AHI없음")
        if abs(row["CBCT_PSG_GapDays"]) > 365: f.append("CBCT-PSG_1년초과")
        row["Flags"] = ", ".join(f)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = ROOT / "Result_py" / "quality_flags.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    flags = df.Flags.str.split(", ").explode()
    print("\n플래그별 환자 수\n" + flags[flags != ""].value_counts().to_string())
    print(f"\n플래그 없음: {(df.Flags == '').sum()} / {len(df)}  → {out}")


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]})
