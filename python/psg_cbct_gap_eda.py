"""PSG–CBCT 검사 간격 EDA. 원본 엑셀·PSG·CBCT는 읽기만 합니다.

검사일 정의
  기본 (Summary): PSG = 엑셀 요약 지표(AHI 등)가 나온 리포트의 Study Date (psg_verified.csv SummaryExam)
  보조 (Raw)    : PSG = 폴더 원신호 기록일 (psg_verified.csv RawExam)
  CBCT          : CBCT 상위 폴더 DICOM의 StudyDate (하위 폴더 혼입 영상 제외)
부호: GapDays_Signed = CBCT - PSG (양수 = CBCT가 나중). check_quality.py, basic_info_qc.py와 같은 방향.

CBCT 본인 확인: 상위 폴더 DICOM의 생년월일·성별(또는 등록번호)을 같은 환자 PSG 리포트와 대조.
식별정보는 비교에만 쓰고 결과에는 일치 여부만 남깁니다.

먼저 psg_reconcile.py, basic_info_qc.py를 실행해 두어야 합니다.
결과: Result_py/psg_cbct_gap_patients.csv, psg_cbct_gap_summary.csv, psg_cbct_gap_*.png
사용: python psg_cbct_gap_eda.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import pypdf

from osa.report import reports_in

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py"
GROUPS = [("A", "0~90일", 0, 90), ("B", "91~180일", 91, 180), ("C", "181~365일", 181, 365), ("D", "365일 초과", 366, None)]
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


# ------------------------------------------------------------ 날짜·본인 확인
def cbct_header(pid: int) -> dict:
    d = next((p for p in (ROOT / "Data/CBCT").glob(f"P_{pid}_*") if p.is_dir()), None)
    f = next(iter(sorted(d.glob("*.dcm"))), None) if d else None          # 상위 폴더만 (glob은 하위로 안 내려감)
    if f is None:
        return {"CBCT_Date": None, "CBCT_File": None}
    h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
    s = str(h.get("StudyDate", ""))
    return {"CBCT_Date": f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 else None,
            "CBCT_File": f"{d.name}/{f.name}", "_dob": str(h.get("PatientBirthDate", "")),
            "_sex": str(h.get("PatientSex", ""))[:1].upper(), "_id": str(h.get("PatientID", "")),
            "CBCT_Subfolders": sum(1 for p in d.rglob("*") if p.is_dir())}


def report_ids(folder: Path) -> list[dict]:
    out = []
    for f in reports_in(folder):
        t = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(f).pages)
        g = lambda pat: (m.group(1) if (m := re.search(pat, t)) else "")
        out.append({"dob": g(r"Date of Birth:\s*([\d-]+)").replace("-", ""), "sex": g(r"Gender:\s*(\w)").upper(),
                    "id": g(r"\bID:\s*(\d+)"), "study": g(r"Study Date:\s*([\d-]+)")})
    return out


def identity(h: dict, reps: list[dict]) -> str:
    """check_quality.py identity()와 같은 규칙."""
    if any(h.get("_dob") and h["_dob"] == r["dob"] and h["_sex"] == r["sex"] for r in reps):
        return "match"
    if any(h.get("_id") and h["_id"] == r["id"] for r in reps):
        return "id_only"
    return "mismatch"


def group(days: float) -> str | None:
    if not np.isfinite(days):
        return None
    for code, _, lo, hi in GROUPS:
        if days >= lo and (hi is None or days <= hi):
            return code
    raise ValueError(days)


def stats(s: pd.Series) -> dict:
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return {"N": len(s), "Mean": s.mean(), "SD": s.std(), "Min": s.min(), "Q1": q1, "Median": s.median(),
            "Q3": q3, "Max": s.max(), "IQR": q3 - q1}


# ------------------------------------------------------------ 메인
def main():
    ver = pd.read_csv(OUT / "psg_verified.csv").set_index("Patient")
    bq = pd.read_csv(OUT / "basic_info_qc.csv").set_index("Patient")
    allrep = pd.read_csv(OUT / "psg_reports.csv")
    dirs = {int(re.match(r"P_(\d+)_", d.name).group(1)): d for d in (ROOT / "Data/PSG").iterdir() if d.is_dir()}
    rows = []
    for p, v in ver.iterrows():
        pid = int(p.split("_")[1])
        h = cbct_header(pid)
        reps = report_ids(dirs[pid])
        studies = sorted(allrep.loc[allrep.Patient == p, "Study"].dropna().unique())
        summ = v.SummaryExam if isinstance(v.SummaryExam, str) else None
        raw = v.RawExam if isinstance(v.RawExam, str) else None
        # 잠정 PSG 날짜: 요약 검사를 못 찾은 환자는 키·몸무게·TST가 일치한 폴더 리포트 날짜 (basic_info_qc HW_MatchingReports)
        prov = summ or (bq.loc[p, "HW_MatchingReports"] if isinstance(bq.loc[p, "HW_MatchingReports"], str) else None)
        idt = identity(h, reps) if h["CBCT_Date"] else "no_cbct"
        row = {"Patient": p, "Device": v.Device, "CBCT_Date": h["CBCT_Date"], "CBCT_Source": h["CBCT_File"],
               "CBCT_Identity": idt, "CBCT_Subfolders": h.get("CBCT_Subfolders", 0),
               "PSG_Summary_Date": summ,
               "PSG_Summary_Source": v.SummarySource if summ else None,
               "PSG_Date_Provisional": prov,
               "PSG_Date_Source": ("psg_verified.csv SummaryExam (엑셀 요약값과 일치한 리포트 Study Date)" if summ else
                                   "잠정: 엑셀 TST·키·몸무게와 일치한 폴더 리포트 Study Date (엑셀 AHI 검사 아님)"),
               "PSG_Raw_Date": raw, "PSG_Raw_Source": (f"{v.RawSource} 기록일" if raw else None),
               "PSG_Exams_AllReports": ", ".join(studies), "PSG_ExamCount": len(set(studies) | ({raw} if raw else set())),
               "Adolescent_AtPSG": bq.loc[p, "Adolescent_AtPSG"]}
        for tag, date in (("", prov), ("_Raw", raw)):
            if h["CBCT_Date"] and date:
                g = (pd.Timestamp(h["CBCT_Date"]) - pd.Timestamp(date)).days
                row.update({f"GapDays_Signed{tag}": g, f"GapDays_Abs{tag}": abs(g),
                            f"GapYears{tag}": round(abs(g) / 365.25, 2), f"GapGroup{tag}": group(abs(g))})
        # 검증 상태
        if idt == "mismatch":
            status = "제외: CBCT 본인 영상 아님"
        elif not summ:
            status = "잠정: PSG 검사 선택 미확정"
        else:
            status = "검증됨"
        row["GapStatus"] = status
        notes = []
        if summ and raw and summ != raw and abs((pd.Timestamp(summ) - pd.Timestamp(raw)).days) > 1:
            notes.append(f"원신호 검사일 다름({raw})")
        if row["PSG_ExamCount"] > 1:
            notes.append(f"PSG 검사 {row['PSG_ExamCount']}회")
        if row["Adolescent_AtPSG"] is True or row["Adolescent_AtPSG"] == "True":
            notes.append("청소년")
        if not summ:
            notes.append("PSG 검사 선택 미확정")
        if idt == "mismatch":
            notes.append("CBCT 본인 아님")
        elif idt == "id_only":
            notes.append("CBCT 생년월일 불일치(등록번호 일치)")
        if row["CBCT_Subfolders"]:
            notes.append("CBCT 하위폴더 혼입(상위 폴더 사용)")
        row["Notes"] = "; ".join(notes)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Order"] = np.select([df.GapDays_Signed > 0, df.GapDays_Signed < 0, df.GapDays_Signed == 0],
                            ["PSG → CBCT", "CBCT → PSG", "같은 날"], "")
    # 요약 검사와 원신호 검사가 실제로 다른 검사인 환자만 (±1일은 자정 넘김 차이로 같은 검사)
    other = (pd.to_datetime(df.PSG_Raw_Date) - pd.to_datetime(df.PSG_Summary_Date)).dt.days.abs() > 1
    df["SummaryRaw_DifferentExam"] = other
    df["Gap_Changes_By_Choice"] = (df.GapDays_Abs_Raw - df.GapDays_Abs).where(other)
    df.to_csv(OUT / "psg_cbct_gap_patients.csv", index=False, encoding="utf-8-sig")

    # ---- 요약 통계: 세 집합
    verified = df[df.GapStatus == "검증됨"]
    provisional = df[df.GapStatus != "제외: CBCT 본인 영상 아님"]
    raw_set = df[df.GapStatus != "제외: CBCT 본인 영상 아님"]
    sets = {"기본_검증 (요약검사일, 본인확인)": verified.GapDays_Abs,
            "기본_잠정 (요약검사일 + 미확정 3명 잠정일)": provisional.GapDays_Abs,
            "보조_원신호검사일 (P_39 제외)": raw_set.GapDays_Abs_Raw}
    srows = []
    for name, s in sets.items():
        st = stats(s)
        r = {"Set": name, **{k: round(v, 1) if isinstance(v, float) else v for k, v in st.items()},
             "Median_months": round(st["Median"] / 30.44, 1), "Mean_months": round(st["Mean"] / 30.44, 1)}
        for code, label, lo, hi in GROUPS:
            m = s.dropna().apply(group) == code
            r[f"{code} {label} n"] = int(m.sum())
            r[f"{code} {label} %"] = round(100 * m.mean(), 1)
            r[f"{code} {label} IDs"] = ", ".join(df.loc[s.dropna().index[m], "Patient"])
        for t in (90, 180, 365):
            r[f">{t}일 n"] = int((s.dropna() > t).sum())
        srows.append(r)
    summ_df = pd.DataFrame(srows)
    summ_df.to_csv(OUT / "psg_cbct_gap_summary.csv", index=False, encoding="utf-8-sig")
    print(summ_df.drop(columns=[c for c in summ_df.columns if c.endswith("IDs")]).T.to_string())

    plots(df, verified)


# ------------------------------------------------------------ 그림
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plots(df, verified):
    n = len(verified)
    note = f"기준: PSG 요약 지표 검사일 ↔ CBCT 촬영일 · 검증된 환자 N={n} (CBCT 본인 아님 P_39, 검사 선택 미확정 P_65·67·69 제외)"
    months = verified.GapDays_Abs / 30.44

    # 1. 히스토그램 (개월, 1개월 폭은 너무 잘아서 3개월 폭)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bins = np.arange(0, np.ceil(months.max() / 3) * 3 + 3, 3)
    ax.hist(months, bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=2)
    for d, lab in ((90, "90일"), (180, "180일"), (365, "365일")):
        ax.axvline(d / 30.44, color=MUTED, linestyle="--", linewidth=1)
        ax.text(d / 30.44, ax.get_ylim()[1] * 0.97, f" {lab}", color=MUTED, va="top", fontsize=9)
    ax.set_xlabel("PSG–CBCT 검사 간격 (개월, |CBCT - PSG|, 3개월 구간)")
    ax.set_ylabel("환자 수 (명)")
    ax.set_title(f"PSG–CBCT 검사 간격 분포 (N={n})", loc="left", fontsize=13, color=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=MUTED)
    _style(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "psg_cbct_gap_hist.png", dpi=160)
    plt.close(fig)

    # 2. 구간별 막대
    counts = [int((verified.GapGroup == c).sum()) for c, *_ in GROUPS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar([f"{c}  {lab}" for c, lab, *_ in GROUPS], counts, color=BLUE, width=0.6)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{c}명 ({100 * c / n:.0f}%)",
                ha="center", color=INK, fontsize=10)
    ax.set_ylabel("환자 수 (명)")
    ax.set_xlabel("PSG–CBCT 검사 간격 (|CBCT - PSG|, 일)")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_title(f"검사 간격 구간별 환자 수 (N={n})", loc="left", fontsize=13, color=INK)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)
    _style(ax)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "psg_cbct_gap_groups.png", dpi=160)
    plt.close(fig)

    # 3. 365일 초과 환자 (검증된 환자만, 긴 순서)
    long = verified[verified.GapDays_Abs > 365].sort_values("GapDays_Abs")
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(long) + 1.6))
    ax.barh(long.Patient, long.GapYears, color=BLUE, height=0.6)
    for y, (_, r) in enumerate(long.iterrows()):
        order = "CBCT 먼저" if r.GapDays_Signed < 0 else "PSG 먼저"
        extra = (f"  · 원신호 검사({r.PSG_Raw_Date})로는 {int(r.GapDays_Abs_Raw)}일"
                 if r.SummaryRaw_DifferentExam else "")
        ax.text(r.GapYears + 0.05, y, f"{r.GapYears:.1f}년 ({int(r.GapDays_Abs)}일, {order}){extra}", va="center",
                color=INK, fontsize=9)
    ax.set_xlabel("PSG–CBCT 검사 간격 (년)")
    ax.set_xlim(0, long.GapYears.max() * 1.6)
    ax.set_title(f"검사 간격 365일 초과 환자 ({len(long)}명 / 검증 N={n})", loc="left", fontsize=13, color=INK)
    fig.text(0.01, 0.01, "P_39(CBCT 본인 아님)와 P_65·67·69(PSG 검사 미확정)는 제외. 기준: PSG 요약 지표 검사일",
             fontsize=8, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "psg_cbct_gap_long_cases.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
