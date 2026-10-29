"""기본정보 QC: 엑셀 성별·나이·키·몸무게·BMI ↔ PSG 리포트 대조, BMI 재계산, 측정 시점.

원본 엑셀·PSG·CBCT는 읽기만 합니다. 생년월일은 나이 계산에만 쓰고 결과 파일에 저장하지 않습니다.
먼저 python/psg_reconcile.py를 실행해 Result_py/psg_verified.csv(검증 성별·나이, 분석 검사일)를 만들어 두어야 합니다.

결과
  Result_py/basic_info_qc.csv     환자별 원래 값, 리포트 값, 재계산, 플래그, 정정값(V_*)
  Result_py/basic_info_stats.csv  키·몸무게·BMI·나이 기초통계와 IQR 경계
사용: python basic_info_qc.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pypdf

from osa.report import reports_in

ROOT = Path(__file__).resolve().parents[1]
EXCEL = ROOT / "Patient_Info_Data_87명_261019.xlsx"
# 엑셀 열 번호 (0부터) ↔ 머리글 1행 이름
XCOLS = {4: ("DOB", "생년월일"), 5: ("SexLetter", "성별"), 6: ("Age", "나이"), 7: ("Height", "키"),
         8: ("Weight", "몸무게"), 9: ("BMI", "BMI"), 10: ("SexCode", "성별\nM:1\nF:2")}


def read_excel() -> pd.DataFrame:
    x = pd.read_excel(EXCEL, header=None)
    for c, (_, name) in XCOLS.items():
        assert str(x.iloc[0, c]).strip() == name, f"엑셀 {c}열 머리글이 예상과 다름: {x.iloc[0, c]!r}"
    r = x.iloc[3:]
    r = r[r[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    df = pd.DataFrame({k: r[c].values for c, (k, _) in XCOLS.items()}, index=r[0].astype(int).values)
    df["ExcelRow"] = r.index.values + 1
    return df


def fmt_issue(v, kind) -> str:
    """빈칸·0·문자·비정상 형식."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "결측"
    if kind == "num":
        if not isinstance(v, (int, float)):
            return f"문자({v!r})"
        if v <= 0:
            return "0 이하"
    if kind == "sex" and str(v).strip().upper() not in ("M", "F"):
        return f"형식({v!r})"
    if kind == "code" and v not in (1, 2):
        return f"형식({v!r})"
    return ""


def decimals(v) -> int:
    """엑셀 셀 값의 소수 자릿수 (정수로 저장된 값은 0)."""
    s = repr(float(v)).rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def bmi_bounds(h_cm, w, h_dec, w_dec):
    """키·몸무게가 기록 자릿수로 반올림됐다고 볼 때 가능한 BMI 범위."""
    hh, ww = 0.5 * 10.0 ** -h_dec, 0.5 * 10.0 ** -w_dec
    lo = (w - ww) / ((h_cm + hh) / 100) ** 2
    hi = (w + ww) / ((h_cm - hh) / 100) ** 2
    return lo, hi


def parse_report(path: Path) -> dict:
    t = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    g = lambda pat: (m.group(1) if (m := re.search(pat, t)) else None)
    h = re.search(r"Height:\s*([\d.]+)\s*(c\s*m|m)\b", t)
    hcm = (float(h.group(1)) * (100 if h.group(2) == "m" else 1)) if h else np.nan
    return {"Study": g(r"Study Date:\s*([\d-]+)"), "R_HeightRaw": f"{h.group(1)} {h.group(2).replace(' ', '')}" if h else None,
            "R_Height": round(hcm, 1), "R_Weight": float(g(r"Weight:\s*([\d.]+)\s*kg") or "nan"),
            "R_BMI": float(g(r"BMI:\s*([\d.]+)") or "nan"), "_dob": g(r"Date of Birth:\s*([\d-]+)")}


def cbct_date(pid: int) -> str | None:
    """CBCT 상위 폴더 영상의 촬영일 (하위 폴더 혼입 영상은 제외, 품질점검 문서 4-B)."""
    d = next((p for p in (ROOT / "Data/CBCT").glob(f"P_{pid}_*") if p.is_dir()), None)
    f = next(iter(sorted(d.glob("*.dcm"))), None) if d else None
    if f is None:
        return None
    s = str(pydicom.dcmread(f, stop_before_pixels=True, force=True).get("StudyDate", ""))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 else None


def age_on(dob: str | None, date: str | None) -> float:
    if not dob or not isinstance(date, str):
        return np.nan
    return float(np.floor((pd.Timestamp(date) - pd.Timestamp(dob)).days / 365.25))


def iqr_stats(s: pd.Series) -> dict:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    i = q3 - q1
    return {"N": int(s.notna().sum()), "Mean": s.mean(), "SD": s.std(), "Min": s.min(), "Q1": q1,
            "Median": s.median(), "Q3": q3, "Max": s.max(), "IQR_low": q1 - 1.5 * i, "IQR_high": q3 + 1.5 * i}


def main():
    xl = read_excel()
    ver = pd.read_csv(ROOT / "Result_py/psg_verified.csv").set_index("Patient")
    dirs = {int(re.match(r"P_(\d+)_", d.name).group(1)): d for d in (ROOT / "Data/PSG").iterdir() if d.is_dir()}
    rows = []
    for pid, x in xl.iterrows():
        p = f"P_{pid}"
        v = ver.loc[p]
        row = {"Patient": p, "ExcelRow": int(x.ExcelRow), "x_SexLetter": x.SexLetter, "x_SexCode": x.SexCode,
               "x_Age": x.Age, "x_Height": x.Height, "x_Weight": x.Weight, "x_BMI": x.BMI}
        # ---- 형식
        issues = {k: fmt_issue(x[k], kind) for k, kind in (("SexLetter", "sex"), ("SexCode", "code"), ("Age", "num"),
                                                           ("Height", "num"), ("Weight", "num"), ("BMI", "num"))}
        row["FormatIssues"] = "; ".join(f"{k}:{e}" for k, e in issues.items() if e)
        # ---- BMI 재계산 (키 cm 가정은 값 범위로 확인: 아래 stats의 Min/Max)
        hd, wd, bd = decimals(x.Height), decimals(x.Weight), decimals(x.BMI)
        calc = x.Weight / (x.Height / 100) ** 2
        lo, hi = bmi_bounds(x.Height, x.Weight, hd, wd)
        tol_b = 0.5 * 10.0 ** -bd
        row.update(BMI_calc=round(calc, 3), BMI_diff=round(x.BMI - calc, 3), Height_dec=hd, Weight_dec=wd, BMI_dec=bd,
                   BMI_range=f"{lo:.2f}~{hi:.2f}")
        row["BMI_check"] = ("일치(반올림 범위)" if lo - tol_b <= x.BMI <= hi + tol_b else "불일치")
        # ---- 리포트: 엑셀 요약 검사(SummaryExam)·원신호 검사(RawExam)·키/몸무게가 같은 리포트
        reps = [parse_report(f) for f in reports_in(dirs[pid])] if pid in dirs else []
        dob = next((r["_dob"] for r in reps if r["_dob"]), None)
        summ, raw = v.SummaryExam if isinstance(v.SummaryExam, str) else None, v.RawExam
        ref = next((r for r in reps if r["Study"] == summ), None)
        hw = [r["Study"] for r in reps if abs(r["R_Height"] - x.Height) <= 0.05 and abs(r["R_Weight"] - x.Weight) <= 0.05]
        row.update(Reports=len(reps), SummaryExam=summ, HW_MatchingReports=", ".join(hw))
        if ref:
            row.update(R_HeightRaw=ref["R_HeightRaw"], R_Height=ref["R_Height"], R_Weight=ref["R_Weight"], R_BMI=ref["R_BMI"])
            row["Report_HWB_match"] = bool(abs(ref["R_Height"] - x.Height) <= 0.05 and abs(ref["R_Weight"] - x.Weight) <= 0.05
                                          and abs(ref["R_BMI"] - x.BMI) <= 0.05)
            row["Report_diff"] = ", ".join(f"{k} 엑셀 {x[k]:g}≠리포트 {ref['R_' + k]:g}" for k in ("Height", "Weight", "BMI")
                                           if abs(ref["R_" + k] - x[k]) > 0.05)
        # ---- 나이: 검사일 기준 만 나이 (생년월일은 리포트, 엑셀과 87명 일치 확인됨: PSG_엑셀_불일치_검증.md 8장)
        cb = cbct_date(pid)
        row.update(Age_SummaryExam=age_on(dob, summ), RawExam=raw, Age_RawExam=age_on(dob, raw), CBCT_Date=cb,
                   Age_CBCT=age_on(dob, cb))
        # 요약 검사를 못 찾은 환자(P_65·67·69): 엑셀 나이가 폴더 리포트 검사일 나이와 같은지만 기록
        row["Age_FolderReports"] = ", ".join(f"{r['Study']}:{age_on(dob, r['Study']):.0f}" for r in reps)
        row["PSG_CBCT_GapDays"] = (pd.Timestamp(cb) - pd.Timestamp(summ)).days if cb and summ else np.nan
        # ---- 성별 (검증값은 psg_verified.csv)
        row.update(V_Sex=v.V_Sex, SexFlag=v.SexFlag if isinstance(v.SexFlag, str) else "",
                   ReportGenders=v.ReportGenders)
        rows.append(row)

    q = pd.DataFrame(rows)
    q["V_SexCode"] = q.V_Sex.map({"M": 1, "F": 2})
    # 정정값: 근거가 있는 것만. 원래 값은 x_* 열에 그대로 둠
    q["V_Age"] = q.Age_SummaryExam                      # 분석 검사일 = 엑셀 요약 검사일 (P_65·67·69는 미정 → 빈칸)
    q["AgeFlag"] = np.select([q.V_Age.isna(), q.x_Age.ne(q.V_Age)],
                             ["요약 검사 미확인 → 나이 기준일 미정", "엑셀 나이≠검사일 만나이"], "")
    q["Adolescent_AtPSG"] = q.V_Age.lt(19).where(q.V_Age.notna())
    for k in ("Height", "Weight", "BMI"):
        q[f"V_{k}"] = q[f"x_{k}"].where(q.Report_HWB_match.eq(True))      # 같은 검사 리포트와 일치한 값만
    q["HWB_Source"] = np.select(
        [q.Report_HWB_match.eq(True), q.SummaryExam.isna() & q.HW_MatchingReports.fillna("").ne("")],
        ["엑셀 요약 검사 리포트와 일치 (PSG 검사일 측정)",
         "폴더 리포트(" + q.HW_MatchingReports.fillna("") + ")와 일치. 엑셀 AHI 검사와는 다른 검사"],
        "출처·측정 시점 미확인")
    # BMI: 리포트 기재값과 같아도 재계산과 반올림 범위 밖이면 표시 (값은 바꾸지 않음)
    q["BMI_Flag"] = np.select(
        [q.BMI_check.eq("불일치") & q.Report_HWB_match.eq(True), q.BMI_check.eq("불일치")],
        ["리포트 기재 BMI와 같음, 리포트 BMI가 키·몸무게 재계산과 다름 (원인 미확인)", "재계산과 다름, 출처 미확인"], "")
    q["BMI_AdultCategoryApplicable"] = ~q.Adolescent_AtPSG.eq(True)      # 청소년은 성인 BMI 분류 적용 안 함
    # P_39 CBCT는 본인 영상 아님 (MATLAB_대비_개선점_및_품질점검.md 4-A): CBCT 시점 나이 계산 의미 없음
    q.loc[q.Patient.eq("P_39"), ["Age_CBCT", "PSG_CBCT_GapDays"]] = np.nan
    q["CBCT_Note"] = np.where(q.Patient.eq("P_39"), "CBCT 본인 영상 아님", "")

    # ---- 기초통계·IQR
    num = q.assign(x_Age=pd.to_numeric(q.x_Age), x_Height=pd.to_numeric(q.x_Height),
                   x_Weight=pd.to_numeric(q.x_Weight), x_BMI=pd.to_numeric(q.x_BMI))
    stats = pd.DataFrame({k: iqr_stats(num[f"x_{k}"]) for k in ("Age", "Height", "Weight", "BMI")}).T.round(2)
    for k in ("Height", "Weight", "BMI"):
        lo, hi = stats.loc[k, "IQR_low"], stats.loc[k, "IQR_high"]
        out = num[f"x_{k}"].lt(lo) | num[f"x_{k}"].gt(hi)
        q[f"{k}_IQR_outlier"] = np.where(out, np.where(num[f"x_{k}"].lt(lo), "low", "high"), "")
    out_dir = ROOT / "Result_py"
    q.to_csv(out_dir / "basic_info_qc.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(out_dir / "basic_info_stats.csv", encoding="utf-8-sig")
    print(stats.to_string())
    print(f"→ {out_dir / 'basic_info_qc.csv'}, basic_info_stats.csv")


if __name__ == "__main__":
    main()
