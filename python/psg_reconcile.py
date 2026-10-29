"""PSG 엑셀 값 대조: 엑셀 ↔ 리포트 PDF ↔ 원본 판독(Embla TXT / Nox Data.ndb 판독층별).

목적: 엑셀 요약값(TST, AHI, RDI, ODI, 최저 SpO2)과 사건 6종 개수가 각각 어느 검사일, 어느 판독본에서
왔는지 환자마다 확인합니다. 원본과 엑셀은 읽기만 합니다. 식별정보(이름, 생년월일, 등록번호)는 남기지 않습니다.

Nox 판독층
  code   : 기존 코드(osa.psg_events)가 읽는 방식. 활성 판독본의 Automatic + Manual 마커를 모두 셈
  auto   : Automatic 키 마커만
  manual : Manual 키(기사 판독) 마커만
  rule   : Manual 마커 중 분석 구간(period-analysisstart~stop) 안에서, 시작 시각이 수면 중(Manual 우선 수면단계)인 것
           TST도 같은 구간의 수면단계로 계산. P_12, P_59 리포트로 맞춰 본 후보 규칙 (리포트로 환자마다 검증)

결과: Result_py/psg_reconcile.csv (환자별 1행), Result_py/psg_reports.csv (리포트별 1행)
사용: python psg_reconcile.py [ID ...]
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf

from osa.psg_events import DOTNET_EPOCH_TICKS, _read_nox_markers, _read_text

ROOT = Path(__file__).resolve().parents[1]
EXCEL = ROOT / "Patient_Info_Data_87명_261019.xlsx"
# 엑셀 열 번호 (0부터). 머리글 3행째 이름으로 확인함
XCOLS = {11: "TST", 12: "AHI", 13: "RDI", 14: "ODI", 15: "LSaO2",
         16: "CA", 17: "MA", 18: "OA", 19: "RERA", 20: "RESP", 21: "H"}
XHEAD = {11: "Total sleep time", 12: "AHI", 13: "RDI", 14: "ODI", 15: "Lowest\nSpO2", 16: "APNEA-CENTRAL",
         17: "APNEA-MIXED", 18: "APNEA-OBSTRUCTIVE", 19: "AROUSAL-RERA", 20: "AROUSAL-RESP", 21: "HYPOPNEA"}
COUNTS = ["OA", "CA", "MA", "H", "RERA", "RESP"]
SUMMARY = ["TST", "AHI", "RDI", "ODI", "LSaO2"]
EVENT = {"APNEA-OBSTRUCTIVE": "OA", "APNEA-CENTRAL": "CA", "APNEA-MIXED": "MA", "HYPOPNEA": "H",
         "AROUSAL-RERA": "RERA", "RERA": "RERA", "AROUSAL-RESP": "RESP", "AROUSAL-RESPIRATORY": "RESP"}


# ------------------------------------------------------------ 엑셀
def read_excel() -> pd.DataFrame:
    x = pd.read_excel(EXCEL, header=None)
    for c, name in XHEAD.items():
        assert str(x.iloc[2, c]).strip() == name.strip(), f"엑셀 {c}열 머리글이 예상과 다름: {x.iloc[2, c]!r}"
    r = x.iloc[3:]
    r = r[r[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    df = pd.DataFrame({k: pd.to_numeric(r[c], errors="coerce").values for c, k in XCOLS.items()},
                      index=r[0].astype(int).values)
    df["ExcelRow"] = r.index.values + 1          # 엑셀에서 보이는 행 번호
    df["Age"] = pd.to_numeric(r[6], errors="coerce").values
    df["SexLetter"] = r[5].astype(str).str.strip().str.upper().values
    df["SexCode"] = pd.to_numeric(r[10], errors="coerce").values      # M:1, F:2
    return df


# ------------------------------------------------------------ 리포트 PDF
def _f(pat, t):
    m = re.search(pat, t)
    return float(m.group(1)) if m else np.nan


def parse_report(path: Path) -> dict:
    t = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    # 2012년 이전 Embla 양식은 'AHI 1 / AHI 2' 두 기준. 'AHI 2'(A+H 표와 같은 값)를 씀 (osa/report.py와 같음)
    two = lambda name: (_f(rf"{name}\s*2\s*:\s*([\d.]+)", t) if re.search(rf"{name}\s*2\s*:", t)
                        else _f(rf"{name}\s*:\s*([\d.]+)", t))
    study = re.search(r"Study Date:\s*([\d-]+)", t)
    # 소견란 비교 문장 'AHI=이전→현재'. 화살표는 깨진 문자 1~3개 (쉼표·공백이면 비교 문장 아님)
    prior = {k: re.search(rf"(?<![A-Za-z]){k}\s*=\s*([\d.]+)[^\d,\s.]{{1,3}}([\d.]+)", t) for k in ("AHI", "RDI", "ODI")}
    pv = lambda k, g: float(prior[k].group(g)) if prior[k] else np.nan
    # 생년월일은 검사일 나이 계산에만 쓰고 반환하지 않음
    dob = re.search(r"Date of Birth:\s*([\d-]+)", t)
    age = ((pd.Timestamp(study.group(1)) - pd.Timestamp(dob.group(1))).days / 365.25
           if study and dob else np.nan)
    sex = re.search(r"Gender:\s*(Male|Female)", t)
    return {
        "Study": study.group(1) if study else None,
        "AgeAtStudy": np.floor(age) if np.isfinite(age) else np.nan,     # 만 나이
        "Gender": sex.group(1)[0] if sex else None,
        "TwoCriteria": bool(re.search(r"AHI\s*2\s*:", t)),
        "TRT": _f(r"Total Recording Time:\s*([\d.]+)\s*minutes", t),
        "TST": _f(r"Total Sleep Time:\s*([\d.]+)", t),
        "AHI": _f(r"\(A\+H\):\s*\d+\s+([\d.]+)", t),
        "AHI_head": two("AHI"),
        "RDI": two(r"(?<!Supine )RDI"),
        "AHI_SUP": two("Supine AHI"),
        "RDI_table": _f(r"(?<!Supine )RDI:\s*([\d.]+)", t),     # 수면요약표 칸 ('RDI :' 머리말과 다를 수 있음)
        "ODI": _f(r"\(OD\):\s*\d+\s+([\d.]+)", t),
        "LSaO2": _f(r"Lowest O2 Sat\.\s*:\s*([\d.]+)", t),
        "AH": _f(r"\(A\+H\):\s*(\d+)", t),
        "OA": _f(r"Obstructive Apnea:\s*(\d+)", t),
        "CA": _f(r"Central Apnea:\s*(\d+)", t),
        "MA": _f(r"Mixed Apnea:\s*(\d+)", t),
        "H": _f(r"Hypopnea(?: \(All\))?:?\s+(\d+)\s+[\d.]+\s*/\s*h", t),
        "RERA": _f(r"\bRERA\s+(\d+)\s+[\d.]+", t),
        "RESP": _f(r"Respiratory Arousals\s+(\d+)", t),
        **{f"Prior{k}": pv(k, 1) for k in prior}, **{f"Prior{k}_to": pv(k, 2) for k in prior},
    }


def reports_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if re.match(r"(?i)report.*\.pdf$", p.name))


# ------------------------------------------------------------ 원본 판독
def embla_counts(folder: Path) -> dict:
    f = folder / "EN-Apnea Events.txt"
    if not f.is_file():
        return {"RawDate": None, "Source": "없음"}
    txt = _read_text(f, "cp949")
    m = re.search(r"Recording Date:\s*(\d{4}-\d{2}-\d{2})", txt)
    lines = txt.splitlines()
    ecol, start = None, None
    for k, line in enumerate(lines):
        low = [c.strip().lower() for c in line.split("\t")]
        if "event" in low and any(c.startswith("time") for c in low):
            ecol, start = low.index("event"), k + 1
            break
    names = [ln.split("\t")[ecol].strip().upper() for ln in lines[start:]
             if ln.strip() and len(ln.split("\t")) > ecol] if ecol is not None else []
    out = {c: 0 for c in COUNTS}
    for n in names:
        if n in EVENT:
            out[EVENT[n]] += 1
    return {"RawDate": m.group(1) if m else None, "Source": f.name, **{f"txt_{k}": v for k, v in out.items()}}


def _ticks(v) -> pd.Timestamp:
    return pd.Timestamp((int(v) - DOTNET_EPOCH_TICKS) * 100)


def _count(types: pd.Series) -> dict:
    up = types.str.upper().map(EVENT)
    return {c: int((up == c).sum()) for c in COUNTS}


def nox_counts(folder: Path) -> dict:
    db = sqlite3.connect(f"file:{(folder / 'Data.ndb').as_posix()}?mode=ro", uri=True)
    try:
        # 기존 코드 방식 그대로
        rows, src = _read_nox_markers(db)
        code = _count(pd.Series([r[2] for r in rows], dtype=str))
        tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        layers = {}
        for pre in ("scoring", "temporary_scoring"):
            if f"{pre}_marker" not in tabs:
                continue
            m = pd.read_sql(f"select CAST(m.starts_at AS TEXT) a, CAST(m.ends_at AS TEXT) b, m.type, m.is_deleted d, "
                            f"k.type kt from {pre}_marker m join {pre}_key k on m.key_id = k.id", db)
            if len(m):
                layers[pre] = m.assign(a=m.a.astype("int64"), b=m.b.astype("int64"), type=m.type.astype(str))
    finally:
        db.close()
    out = {"Source": src, **{f"code_{k}": v for k, v in code.items()}}
    live = {k: v[v.d == 0] for k, v in layers.items()}
    # 기존 코드가 읽은 층의 호흡 이벤트가 Automatic/Manual 중 어디서 왔는지
    cl = live.get("scoring" if src.startswith("Data.ndb: scoring_revision") else "temporary_scoring")
    if cl is not None:
        resp = cl[cl.type.str.upper().isin(EVENT)]
        out["CodeComposition"] = "+".join(k for k in ("Automatic", "Manual") if (resp.kt == k).any()) or "없음"
    # 기사 판독(Manual) 마커가 가장 많은 층을 기준으로 씀
    pick = max(live, key=lambda k: (live[k].kt == "Manual").sum()) if live else None
    if pick is None:
        return out
    m = live[pick]
    man, auto = m[m.kt == "Manual"], m[m.kt == "Automatic"]
    out.update({"Layer": pick, **{f"auto_{k}": v for k, v in _count(auto.type).items()},
                **{f"manual_{k}": v for k, v in _count(man.type).items()}})
    st = m[m.type.str.startswith("sleep-")].sort_values("kt").drop_duplicates("a", keep="last")  # Manual 우선
    a0 = man[man.type == "period-analysisstart"].a.min()
    a1 = man[man.type == "period-analysisstop"].a.max()
    has_period = pd.notna(a0) and pd.notna(a1)
    if not has_period:
        a0, a1 = (st.a.min(), st.b.max()) if len(st) else (np.nan, np.nan)
    first = st.a.min() if len(st) else (m.a.min() if len(m) else np.nan)
    # 검사일 = 자정 넘어 시작해도 전날 밤으로 (12시간 당김)
    out["RawDate"] = str((_ticks(first) - pd.Timedelta(hours=12)).date()) if pd.notna(first) else None
    out["Period"] = "analysis" if has_period else "stages"
    sl = st[(st.type != "sleep-wake") & (st.a >= a0) & (st.a < a1)]
    A, B = sl.a.values, sl.b.values
    ev = man[man.type.str.upper().isin(EVENT) & (man.a >= a0) & (man.a <= a1)]
    inside = np.array([((A <= a) & (a < B)).any() for a in ev.a.values], dtype=bool)
    out.update({f"rule_{k}": v for k, v in _count(ev.type[inside]).items()})
    out["rule_TST"] = round((B - A).sum() / 1e7 / 60, 1)
    return out


# ------------------------------------------------------------ 대조
def same(a, b, tol):
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)


def counts_equal(x: dict, y: dict, keys=COUNTS) -> bool:
    return all(np.isfinite(y.get(k, np.nan)) and x[k] == y[k] for k in keys)


def main(wanted: set[int]):
    xl = read_excel()
    dirs = {int(re.match(r"P_(\d+)_", d.name).group(1)): d for d in (ROOT / "Data/PSG").iterdir() if d.is_dir()}
    ids = sorted(i for i in xl.index if not wanted or i in wanted)
    rows, reps = [], []
    for pid in ids:
        d = dirs.get(pid)
        x = xl.loc[pid]
        row = {"Patient": f"P_{pid}", "ExcelRow": int(x.ExcelRow), **{f"x_{k}": x[k] for k in SUMMARY + COUNTS}}
        if d is None:
            rows.append({**row, "Note": "PSG 폴더 없음"})
            continue
        dev = "E" if (d / "EN-Apnea Events.txt").is_file() or d.name.endswith("_E") else "N"
        raw = embla_counts(d) if dev == "E" else nox_counts(d)
        rp = []
        files = reports_in(d)
        for f in files:
            r = parse_report(f)
            rp.append(r)
            reps.append({"Patient": f"P_{pid}", "File": f.name, **r})
        row.update(Device=dev, Folder=d.name, Reports=len(rp), RawDate=raw.get("RawDate"),
                   RawSource=raw.get("Source"), NoxLayer=raw.get("Layer"), NoxPeriod=raw.get("Period"),
                   NoxRuleTST=raw.get("rule_TST"), CodeComposition=raw.get("CodeComposition"),
                   **{k: v for k, v in raw.items() if re.match(r"(txt|code|auto|manual|rule)_[A-Z]", k)})
        # 기본정보: 엑셀 나이·성별 ↔ 리포트 (리포트 성별은 검사별로 나열)
        row.update(x_Age=x.Age, x_SexLetter=x.SexLetter, x_SexCode=x.SexCode,
                   ReportGenders=", ".join(f"{r['Study']}:{r['Gender']}" for r in rp))

        # (1) 엑셀 요약값과 같은 리포트: TST ±0.6분, AHI ±0.1
        xi = [i for i, r in enumerate(rp) if same(r["TST"], x.TST, 0.6) and same(r["AHI"], x.AHI, 0.1)]
        # (2) 원본 판독과 같은 날 리포트 (±1일)
        near = lambda s: s and row["RawDate"] and abs((pd.Timestamp(s) - pd.Timestamp(row["RawDate"])).days) <= 1
        ri = [i for i, r in enumerate(rp) if near(r["Study"])]
        row["ExcelReport"] = files[xi[0]].name if xi else None
        row["ExcelStudy"] = rp[xi[0]]["Study"] if xi else None
        row["RawReport"] = files[ri[0]].name if ri else None
        row["RawStudy"] = rp[ri[0]]["Study"] if ri else None
        # 엑셀 값이 리포트 소견의 비교 문장 'AHI=이전→현재'의 이전 값과 같은가 (이전 검사 리포트가 없을 때의 단서)
        # '→' 뒤 값이 그 리포트의 AHI와 같아야 비교 문장으로 인정
        prior_hit = [i for i, r in enumerate(rp)
                     if same(r["PriorAHI"], x.AHI, 0.05) and same(r["PriorAHI_to"], r["AHI"], 0.05)]
        row["ExcelIsPriorInComment"] = ", ".join(rp[i]["Study"] or "?" for i in prior_hit)
        # 원본 판독(E: TXT, N: 후보 규칙) ↔ 같은 날 리포트
        tag = "txt" if dev == "E" else "rule"
        if ri:
            ref = rp[ri[0]]
            diffs = {k: raw.get(f"{tag}_{k}", np.nan) - ref[k] for k in COUNTS}
            ah_raw = sum(raw.get(f"{tag}_{k}", np.nan) for k in ("OA", "CA", "MA", "H"))
            row["RawAH"], row["RawReportAH"] = ah_raw, ref["AH"]
            row["RawAH_pct"] = round(100 * (ah_raw - ref["AH"]) / ref["AH"], 1) if ref["AH"] else np.nan
            row["RawCountDiff"] = ", ".join(f"{k}{v:+g}" for k, v in diffs.items() if np.isfinite(v) and v != 0)
            if dev == "N":
                row["RawTST_diff"] = round(raw.get("rule_TST", np.nan) - ref["TST"], 1)
        diff_vars = []
        if xi:
            ref = rp[xi[0]]
            for k in SUMMARY + COUNTS + ["AgeAtStudy"]:
                row[f"r_{k}"] = ref[k]
            diff_vars = [k for k in SUMMARY if not same(ref[k], x[k], 0.15 if k != "TST" else 0.6)]
            row["SummaryMatch"] = not diff_vars
            row["SummaryDiff"] = ", ".join(f"{k} {x[k]:g}≠{ref[k]:g}" for k in diff_vars)
        # (3) 엑셀 사건 수가 어느 출처와 정확히 같은가
        xc = {k: x[k] for k in COUNTS}
        src = []
        for t in ("txt", "code", "auto", "manual", "rule"):
            if counts_equal(xc, {k: raw.get(f"{t}_{k}", np.nan) for k in COUNTS}):
                src.append(t)
        for r in rp:
            if counts_equal(xc, r, ["OA", "CA", "MA", "H"]):
                src.append(f"report:{r['Study']}")
        row["ExcelCountsFrom"] = ", ".join(src) or "일치 출처 없음"
        # 엑셀 값만으로 다시 계산한 AHI (기존 EDA와 같은 방식, 참고용)
        row["x_AHI_recalc"] = round((x.OA + x.CA + x.MA + x.H) / (x.TST / 60), 2)

        # ---- (4) 원인 판정: 파일에서 직접 확인한 근거가 있는 것만
        causes, ev = [], []
        def add(c, e):
            if c not in causes:
                causes.append(c)
            ev.append(e)
        if xi and row["RawDate"] and not near(row["ExcelStudy"]):
            add("서로 다른 검사 날짜",
                f"엑셀 요약=리포트 {row['ExcelStudy']}({row['ExcelReport']}), 원신호·엑셀 사건수=기록일 {row['RawDate']}"
                + ("" if ri else " (이 날짜 리포트 없음)"))
        if not xi:
            hit = [i for i in prior_hit if same(rp[i]["TST"], x.TST, 0.6)
                   and all(not np.isfinite(rp[i][f"Prior{k}"]) or same(rp[i][f"Prior{k}"], x[k], 0.05)
                           for k in ("RDI", "ODI"))]
            if hit:
                r = rp[hit[0]]
                add("엑셀 기록 오류",
                    f"한 행에 두 검사 값: TST·사건수=리포트 {r['Study']}({files[hit[0]].name}), "
                    f"AHI/RDI/ODI=같은 리포트 소견란의 이전 검사 값({r['PriorAHI']:g}/{r['PriorRDI']:g}/{r['PriorODI']:g}), "
                    f"최저 SpO2 {x.LSaO2:g}은 출처 미확인 (이전 검사 리포트 없음)")
            else:
                add("원인 미확인", f"엑셀 TST {x.TST:g}·AHI {x.AHI:g}와 같은 리포트 없음"
                    + (f"; TST·사건수는 리포트 {rp[ri[0]]['Study']}와 같음"
                       if ri and same(rp[ri[0]]["TST"], x.TST, 0.6) else ""))
        for k in diff_vars:
            ref = rp[xi[0]]
            if k == "RDI" and same(ref["RDI_table"], x.RDI, 0.05):
                add("TST 또는 사건 집계 기준 차이",
                    f"리포트 안 RDI 두 값: 머리말 {ref['RDI']:g}(=(A+H+RERA)/TST), 수면요약표 {ref['RDI_table']:g}; 엑셀=요약표 값")
            elif k == "ODI" and same(ref["AHI_SUP"], x.ODI, 0.05):
                add("엑셀 기록 오류", f"엑셀 ODI {x.ODI:g} = 리포트 Supine AHI {ref['AHI_SUP']:g} (리포트 ODI {ref['ODI']:g})")
            else:
                add("원인 미확인", f"엑셀 {k} {x[k]:g} ≠ 리포트 {ref[k]:g}, 출처 못 찾음")
        if dev == "N" and "code" in src:
            comp = raw.get("CodeComposition", "")
            if "Automatic" in comp:
                add("자동/수동 판독 차이", f"엑셀 사건수 = 기존 코드 출력 = {raw['Source']}의 {comp} 마커"
                    + (" 합산" if "+" in comp else "만 (기사 판독 아님)"))
            elif any(raw.get(f"code_{k}") != raw.get(f"rule_{k}") for k in COUNTS):
                add("TST 또는 사건 집계 기준 차이", "엑셀 사건수 = 기사(Manual) 마커 전체, 분석 구간·수면 중 제한 없음")
        if xi:
            ref = rp[xi[0]]
            imp = ref["AH"] / ref["AHI"] * 60 if ref["AHI"] else np.nan
            if not same(ref["AH"] / (ref["TST"] / 60), ref["AHI"], 0.1):
                add("TST 또는 사건 집계 기준 차이",
                    f"리포트 AHI {ref['AHI']:g}의 분모 ≈ {imp:.0f}분 (TST {ref['TST']:g}, 총기록 {ref['TRT']:g})")
        row["Causes"] = " / ".join(causes)
        row["Evidence"] = " | ".join(ev)
        rows.append(row)
        print(f"P_{pid:<3} {dev} raw {row['RawDate']}  엑셀요약=리포트 {row['ExcelStudy']}  "
              f"원본=리포트 {row['RawStudy']}  엑셀개수={row['ExcelCountsFrom']}  원인={row['Causes']}", flush=True)

    out = ROOT / "Result_py"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "psg_reconcile.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reps).to_csv(out / "psg_reports.csv", index=False, encoding="utf-8-sig")
    verified(df).to_csv(out / "psg_verified.csv", index=False, encoding="utf-8-sig")
    print(f"→ {out / 'psg_reconcile.csv'}, psg_reports.csv, psg_verified.csv")


def verified(df: pd.DataFrame) -> pd.DataFrame:
    """검증 데이터셋. 값은 리포트(또는 원본 판독)에서 읽은 것만 넣고, 확인 안 된 칸은 비워 둡니다.

    A 블록(V_): 엑셀 요약값이 나온 검사의 리포트 값. 엑셀과 같은 검사 기준
    B 블록(Raw_): 폴더 원신호 검사의 판독 개수 (E: EN-Apnea Events.txt, N: Data.ndb 기사층 후보 규칙)
    """
    v = pd.DataFrame({"Patient": df.Patient, "Device": df.Device, "ExcelRow": df.ExcelRow})
    v["SummaryExam"] = df.ExcelStudy
    v["SummarySource"] = df.ExcelReport.where(df.ExcelReport.isna(), df.Folder + "/" + df.ExcelReport)
    # 나이: 엑셀 요약 검사 리포트의 검사일 기준 만 나이
    v["Excel_Age"], v["V_Age"] = df.x_Age, df.r_AgeAtStudy
    v["AgeFlag"] = np.select([v.V_Age.isna(), v.Excel_Age.ne(v.V_Age)], ["리포트 없음", "엑셀 나이≠검사일 만나이"], "")
    v["Under19_AtExam"] = v.V_Age.lt(19).where(v.V_Age.notna())
    # 성별: 엑셀 문자가 리포트 중 하나와 같으면 그 값을 씀. 코드(M:1, F:2) 불일치는 표시만
    genders = df.ReportGenders.fillna("").str.findall(r":([MF])")
    code_txt = df.x_SexCode.map({1: "M", 2: "F"})
    v["Excel_Sex"], v["Excel_SexCode"], v["ReportGenders"] = df.x_SexLetter, df.x_SexCode, df.ReportGenders
    v["V_Sex"] = [s if s in g else None for s, g in zip(df.x_SexLetter, genders)]
    v["SexFlag"] = [" / ".join(f for f, c in (("엑셀 코드≠문자", ct != s), ("리포트 간 성별 다름", len(set(g)) > 1),
                                              ("엑셀≠리포트", bool(g) and s not in g)) if c)
                    for s, ct, g in zip(df.x_SexLetter, code_txt, genders)]
    for k in SUMMARY + COUNTS:
        v[f"V_{k}"] = df[f"r_{k}"]
    h = v.V_TST / 60
    v["V_AHI_recalc"] = ((v.V_OA + v.V_CA + v.V_MA + v.V_H) / h).round(2)
    v["V_RDI_recalc"] = ((v.V_OA + v.V_CA + v.V_MA + v.V_H + v.V_RERA) / h).round(2)
    v["V_AHI_consistent"] = (v.V_AHI_recalc - v.V_AHI).abs().le(0.1).where(v.V_AHI.notna())
    v["V_RDI_consistent"] = (v.V_RDI_recalc - v.V_RDI).abs().le(0.15).where(v.V_RDI.notna())
    v["RawExam"] = df.RawDate
    v["RawExamReport"] = df.RawReport
    v["SameExam"] = df.ExcelStudy.notna() & (df.ExcelStudy == df.RawStudy)
    v["RawSource"] = np.where(df.Device == "E", df.Folder + "/EN-Apnea Events.txt",
                              df.Folder + "/Data.ndb (Manual, 분석구간, 수면 중)")
    tag = np.where(df.Device == "E", "txt", "rule")
    for k in COUNTS:
        v[f"Raw_{k}"] = [r.get(f"{t}_{k}") for (_, r), t in zip(df.iterrows(), tag)]
    v["Raw_TST"] = df.NoxRuleTST
    v["RawVsReport_AH_pct"] = df.RawAH_pct
    ok = df.RawAH_pct.abs().le(np.where(df.Device == "E", 2, 5))
    v["RawCountsStatus"] = np.select([df.RawReport.isna(), ok], ["리포트 없음(미검증)", "검증됨"], "리포트와 차이 큼")
    v["SummaryStatus"] = np.select([df.ExcelStudy.isna(), df.SummaryMatch.eq(False)],
                                   ["엑셀 요약 검사 미확인", "검증됨(일부 변수 엑셀≠리포트)"], "검증됨")
    v["Causes"], v["Evidence"] = df.Causes, df.Evidence
    v["Unresolved"] = df.Causes.str.contains("원인 미확인") | v.RawCountsStatus.eq("리포트와 차이 큼")
    return v


if __name__ == "__main__":
    main({int(a) for a in sys.argv[1:]})
