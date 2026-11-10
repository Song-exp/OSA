"""87명 대응표와 환자별 근거 묶음. 원본은 읽기만 하며 자동 제외/값 교체 없음.

먼저 psg_reconcile.py 실행. 결과는 git 제외 Result_py/patient_matching/에 저장.
엑셀 No./폴더, 생년월일·성별을 기준으로 CBCT와 PSG를 독립 대조한다.
CBCT↔PSG 병원 등록번호 비교는 보조 교차확인만 하며 직접 대응 판정에 쓰지 않는다.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pypdf
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from psg_reconcile import COUNTS, EXCEL, ROOT, SUMMARY, parse_report, reports_in
from osa.psg_read import read_ebm, read_ndf

OUT = ROOT / "Result_py/patient_matching"
TOL = {"TST": .6, "AHI": .1, "RDI": .15, "ODI": 0, "LSaO2": 0,
       **{k: 0 for k in COUNTS}}
TAGS = ["PatientID", "PatientBirthDate", "PatientSex", "StudyDate",
        "StudyInstanceUID", "SeriesInstanceUID", "SOPClassUID"]


def rel(p):
    # 모두 ROOT에서 구성한 절대 경로. 수만 개의 파일마다 Windows resolve를
    # 호출하면 네트워크/백신 파일 조회 비용이 커지므로 문자열 상대화만 한다.
    return Path(p).relative_to(ROOT).as_posix()


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def text(v):
    return "" if v is None or pd.isna(v) else str(v)


def norm_id(v):
    # 공백만 제거. 선행 0이나 접두사 제거로 다른 ID를 같게 만들지 않는다.
    return re.sub(r"\s+", "", text(v))


def id_digest(v):
    return hashlib.sha256(v.encode("utf-8")).hexdigest() if v else ""


def dob(v):
    if pd.isna(v):
        return ""
    s = str(v)
    if re.fullmatch(r"\d{1,6}", s.strip()):
        return s.strip().zfill(6)
    if isinstance(v, (int, float, np.integer, np.floating)):
        s = str(int(v))
        if len(s) <= 6:
            return s.zfill(6)
    if re.fullmatch(r"\d{8}", s):
        return s
    return pd.Timestamp(v).strftime("%Y%m%d")


def dob_matches(excel_value, report_dob):
    value = dob(excel_value)
    return bool(value and report_dob and (report_dob.endswith(value) if len(value) == 6 else report_dob == value))


def identity_fields(excel_birth, excel_sex, actual_birth, actual_sex):
    birth = ("일치" if dob_matches(excel_birth, actual_birth) else "불일치") if text(excel_birth) and actual_birth else "확인 불가"
    sex = ("일치" if text(excel_sex).strip().upper() == actual_sex else "불일치") if text(excel_sex) and actual_sex else "확인 불가"
    return birth, sex


def summarize_field(values):
    if "불일치" in values:
        return "불일치"
    return "일치" if values and set(values) == {"일치"} else "확인 불가"


def identity_status(births, sexes):
    birth, sex = summarize_field(births), summarize_field(sexes)
    if birth == "불일치" and sex == "불일치":
        return "생년월일·성별 불일치"
    if birth == "불일치":
        return "생년월일 불일치"
    if sex == "불일치":
        return "성별 불일치"
    return "식별정보 일치" if birth == sex == "일치" else "확인 불가"


def cmp(a, b, tol=0):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "비교 불가", None
    delta = round(float(a) - float(b), 6)
    return ("일치" if abs(delta) <= tol + 1e-8 else "불일치"), delta


def date_str(v):
    return pd.Timestamp(v).strftime("%Y-%m-%d") if text(v) else ""


def link(p, label=None):
    return f'<a href="{html.escape(Path(p).resolve().as_uri(), quote=True)}">{html.escape(label or str(p))}</a>'


def report_identity(path):
    t = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    def find(pattern):
        m = re.search(pattern, t)
        return m.group(1) if m else ""
    return {"id": norm_id(find(r"\bID:\s*(\d+)")),
            "dob": find(r"Date of Birth:\s*([\d-]+)").replace("-", ""),
            "sex": find(r"Gender:\s*(Male|Female)")[:1]}


def files_for(base, pid):
    return sorted(p for p in base.glob(f"P_{pid}_*") if p.is_dir())


def make_book(frames, target):
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for sheet, df in frames.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "B2"
            ws.auto_filter.ref = ws.dimensions
            ws.row_dimensions[1].height = 44
            for c in ws[1]:
                c.font = Font(color="FFFFFF", bold=True)
                c.fill = PatternFill("solid", fgColor="24476A")
                c.alignment = Alignment(wrap_text=True, vertical="center")
            for j, col in enumerate(df.columns, 1):
                width = 19 if "원값" in col or "차이" in col else 25
                if col in ("환자", "변수", "엑셀행"):
                    width = 12
                ws.column_dimensions[get_column_letter(j)].width = width
                for cells in ws.iter_cols(min_col=j, max_col=j, min_row=2):
                    for c in cells:
                        c.alignment = Alignment(vertical="top", wrap_text=True)
                        val = str(c.value or "")
                        if val in ("불일치", "다른 검사", "엑셀 대응 검사 미확인", "생년월일 불일치", "성별 불일치", "생년월일·성별 불일치"):
                            c.fill = PatternFill("solid", fgColor="FCE0DD")
                        elif val in ("확인 불가", "비교 불가", "원신호 대응 리포트 없음", "검토 필요"):
                            c.fill = PatternFill("solid", fgColor="FFF2CC")
                        elif val in ("일치", "동일 검사", "식별정보 일치"):
                            c.fill = PatternFill("solid", fgColor="DFEFDF")
                        if isinstance(c.value, str) and c.value.startswith("Result_py/") and (ROOT / c.value).is_file():
                            c.hyperlink = (ROOT / c.value).as_uri()
                            c.font = Font(color="0563C1", underline="single")


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "Result_py/psg_reconcile.csv"
    if not source.is_file():
        raise SystemExit("먼저 python/psg_reconcile.py를 실행하세요.")
    rec = pd.read_csv(source).set_index("Patient")
    x = pd.read_excel(EXCEL, header=None)
    xr = x.iloc[3:]
    xr = xr[xr[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    assert len(xr) == 87 and set(xr[0].astype(int)) == set(range(1, 88))
    masters, details, reports_table, series_table, all_manifest = [], [], [], [], []
    generated = datetime.now().isoformat(timespec="seconds")
    for _, row in xr.iterrows():
        pid = int(row[0]); patient = f"P_{pid}"
        r = rec.loc[patient]
        case = OUT / "patients" / patient
        case.mkdir(parents=True, exist_ok=True)
        pdirs = files_for(ROOT / "Data/PSG", pid)
        cdirs = files_for(ROOT / "Data/CBCT", pid)
        assert len(pdirs) == 1 and len(cdirs) == 1, f"{patient}: 폴더가 없거나 중복됨"
        ps, cb = pdirs[0], cdirs[0]
        manifest, rp = [], []

        def evidence(path, role, copy_to=None):
            item = {"환자": patient, "역할": role, "원본경로": rel(path),
                    "바이트": path.stat().st_size, "SHA256": "", "검토용사본": ""}
            if copy_to:
                copy_to.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy_to)
                item["SHA256"] = sha(path)
                assert item["SHA256"] == sha(copy_to)
                item["검토용사본"] = rel(copy_to)
            elif path.suffix.lower() not in (".dcm", ".ebm", ".ndf", ".edf"):
                item["SHA256"] = sha(path)
            manifest.append(item)

        for f in reports_in(ps):
            ident = report_identity(f)
            vals = parse_report(f)
            birth_state, sex_state = identity_fields(row[4], row[5], ident["dob"], ident["sex"])
            record = {"file": f, "identity": ident, "excel_birth_state": birth_state,
                      "excel_sex_state": sex_state, **vals}
            rp.append(record)
            target = case / "reports" / f.name
            evidence(f, "PSG 리포트", target)
            reports_table.append({"환자": patient, "파일": rel(f), "검토용사본": rel(target),
                                  "검사일": vals["Study"],
                                  "엑셀생년월일일치": dob_matches(row[4], ident["dob"]),
                                  "엑셀_생년월일대조": birth_state, "엑셀_성별대조": sex_state,
                                  "엑셀_PSG_식별정보대조": identity_status([birth_state], [sex_state]),
                                  "등록번호존재": bool(ident["id"]),
                                  "등록번호_SHA256": id_digest(ident["id"]),
                                  **{k: vals[k] for k in SUMMARY + COUNTS},
                                  "RDI_요약표": vals["RDI_table"], "AHI_머리말": vals["AHI_head"]})
        for f in sorted(ps.rglob("*")):
            if f.is_file() and f not in [q["file"] for q in rp]:
                role = "PSG 원신호" if f.suffix.lower() in (".ebm", ".ndf", ".edf") else "PSG 판독·부속파일"
                evidence(f, role)

        ref = next((q for q in rp if q["file"].name == text(r.get("ExcelReport"))), None)
        candidates = [q for q in rp if dob_matches(row[4], q["identity"]["dob"])]
        ids = {q["identity"]["id"] for q in candidates if q["identity"]["id"]}
        psg_births = [q["excel_birth_state"] for q in rp]
        psg_sexes = [q["excel_sex_state"] for q in rp]
        psg_person = identity_status(psg_births, psg_sexes)
        notes = []
        if len(ids) != 1:
            notes.append("엑셀 생년월일에 대응하는 리포트 등록번호가 없거나 복수임")
        if len(candidates) != len(rp):
            notes.append("일부 리포트의 생년월일이 엑셀과 다르거나 없음")
        if "불일치" in psg_sexes:
            notes.append("엑셀 문자 성별과 다른 PSG 리포트 있음; 환자별·리포트별로 검토 필요")

        # 모든 DICOM의 ID/UID를 읽어 같은 디렉터리 내부 혼입도 놓치지 않는다.
        groups = defaultdict(list)
        cached_headers = {}
        # 명시적으로 요청한 재실행에서만 직전 전수 감사 결과 재사용.
        # 파일 집합/크기/수정시각과 대표 헤더 해시가 그대로인 디렉터리만 허용.
        page = case / "evidence.html"
        old_manifest = case / "source_files.csv"
        if "--reuse-header-audit" in sys.argv and page.is_file() and old_manifest.is_file():
            old = pd.read_csv(old_manifest).fillna("")
            prev = old[old["역할"] == "CBCT 원본 DICOM"]
            current_files = sorted(cb.rglob("*.dcm"))
            sizes = {q["원본경로"]: int(q["바이트"]) for _, q in prev.iterrows()}
            unchanged = (set(sizes) == {rel(f) for f in current_files}
                         and all(f.stat().st_size == sizes[rel(f)] and f.stat().st_mtime <= page.stat().st_mtime for f in current_files))
            if unchanged:
                prior_series = pd.read_html(io.StringIO(page.read_text(encoding="utf-8")),
                                            keep_default_na=False,
                                            converters={"StudyUID": str, "SeriesUID": str,
                                                        "PatientID_SHA256": str, "대표파일_SHA256": str})[2]
                for parent_name, sub in prior_series.groupby("폴더"):
                    if len(sub) != 1:
                        continue
                    q = sub.iloc[0]
                    f = ROOT / q["대표파일"]
                    if sha(f) != q["대표파일_SHA256"]:
                        continue
                    h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS, force=True)
                    if id_digest(norm_id(h.get("PatientID", ""))) == q["PatientID_SHA256"]:
                        cached_headers[ROOT / parent_name] = h
        errors = []
        for f in sorted(cb.rglob("*")):
            if not f.is_file():
                continue
            evidence(f, "CBCT 원본 DICOM" if f.suffix.lower() == ".dcm" else "CBCT 부속파일")
            if f.suffix.lower() != ".dcm":
                continue
            try:
                h = cached_headers.get(f.parent)
                if h is None:
                    h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS, force=True)
                key = (f.parent, norm_id(h.get("PatientID", "")), str(h.get("StudyInstanceUID", "")),
                       str(h.get("SeriesInstanceUID", "")), str(h.get("StudyDate", "")),
                       str(h.get("PatientBirthDate", "")), str(h.get("PatientSex", "")))
                groups[key].append(f)
            except Exception as e:
                errors.append({"파일": rel(f), "오류종류": type(e).__name__})
        top = []
        patient_series = []
        for (parent, cid, study_uid, uid, date, birth, sex), fs in groups.items():
            status = ("일치" if cid in ids else "불일치") if cid and len(ids) == 1 else "확인 불가"
            birth_state, sex_state = identity_fields(row[4], row[5], birth, sex)
            item = {"환자": patient, "범위": "상위(주 검토 영상)" if parent == cb else "하위(별도 검토)",
                    "엑셀_생년월일대조": birth_state, "엑셀_성별대조": sex_state,
                    "엑셀_CBCT_식별정보대조": identity_status([birth_state], [sex_state]),
                    "CBCT_PSG등록번호_보조대조": status, "촬영일": date_str(date), "DICOM수": len(fs),
                    "StudyUID": study_uid, "SeriesUID": uid, "대표파일": rel(fs[0]),
                    "PatientID_SHA256": id_digest(cid), "대표파일_SHA256": sha(fs[0]),
                    "폴더": rel(parent), "리포트생년월일일치": bool(birth and any(birth == q["identity"]["dob"] for q in candidates)),
                    "리포트성별일치": bool(sex and any(sex == q["identity"]["sex"] for q in candidates))}
            patient_series.append(item)
            if parent == cb:
                top.append(item)
        series_table.extend(patient_series)
        if errors:
            notes.append(f"DICOM 헤더 읽기 실패 {len(errors)}개")
        top_states = {q["CBCT_PSG등록번호_보조대조"] for q in top}
        cb_aux = "일치" if top_states == {"일치"} and not errors else (
            "불일치" if "불일치" in top_states else "확인 불가")
        cb_births = [q["엑셀_생년월일대조"] for q in top]
        cb_sexes = [q["엑셀_성별대조"] for q in top]
        cb_status = identity_status(cb_births, cb_sexes) if not errors else "확인 불가"
        if any(q["범위"].startswith("하위") and q["엑셀_CBCT_식별정보대조"] != "식별정보 일치" for q in patient_series):
            notes.append("하위 폴더에 엑셀 식별정보 불일치/미확인 영상 있음; 상위 주 영상과 분리")
        if cb_status != "식별정보 일치":
            notes.append("CBCT와 엑셀의 생년월일·성별 대조 확인 필요; PSG 등록번호로 일치 판정을 덮어쓰지 않음")
        if cb_aux == "불일치":
            notes.append("보조 확인: DICOM PatientID≠PSG 리포트 ID. 엑셀↔CBCT 판정에는 사용하지 않음")

        # SpO2 원신호 한 채널의 실제 시각을 직접 읽는다. 전체 채널 품질 판정과 구분.
        wave = next((f for f in sorted(ps.iterdir()) if re.fullmatch(r"(?i)spo2(_pds)?\.(ebm|ndf)", f.name)), None)
        wave_start, wave_end, wave_error = "", "", ""
        if wave:
            try:
                ch = read_ebm(wave) if wave.suffix.lower() == ".ebm" else read_ndf(wave)
                wave_start, wave_end = str(pd.Timestamp(ch.time[0])), str(pd.Timestamp(ch.time[-1]))
                del ch
            except Exception as e:
                wave_error = type(e).__name__
        if wave_error or not wave:
            notes.append("SpO2 원신호 시각 확인 불가: " + (wave_error or "파일 없음"))
        raw_candidates = []
        if wave_start:
            start = pd.Timestamp(wave_start)
            night = (start - pd.Timedelta(hours=12)).normalize()
            raw_candidates = [q for q in candidates if q["Study"] and pd.Timestamp(q["Study"]) in (start.normalize(), night)]
        raw_ref = raw_candidates[0] if len(raw_candidates) == 1 else None
        raw_status = "동일 검사" if raw_ref else ("원신호 대응 리포트 없음" if not raw_candidates else "확인 불가")
        if len(raw_candidates) > 1:
            notes.append("원신호 시각에 대응하는 리포트 복수; 자동 선택 안 함")
        if not ref:
            exam_status = "엑셀 대응 검사 미확인"
        elif not raw_ref:
            # 리포트가 없어도 판독일과 파형일이 일치하면 다른 검사라는 증거는 남긴다.
            if wave_start and text(r.get("RawDate")) and abs((pd.Timestamp(ref["Study"]) - pd.Timestamp(r.RawDate)).days) > 1:
                exam_status = "다른 검사"
            else:
                exam_status = "확인 불가"
        else:
            exam_status = "동일 검사" if ref["Study"] == raw_ref["Study"] else "다른 검사"

        m = {"환자": patient, "엑셀행": int(r.ExcelRow), "엑셀_CBCT_식별정보대조": cb_status,
             "엑셀_PSG_식별정보대조": psg_person,
             "엑셀_CBCT_연구번호대조": "일치" if re.match(r"P_(\d+)_", cb.name).group(1) == str(pid) else "불일치",
             "엑셀_PSG_연구번호대조": "일치" if re.match(r"P_(\d+)_", ps.name).group(1) == str(pid) else "불일치",
             "엑셀_CBCT_생년월일대조": summarize_field(cb_births), "엑셀_CBCT_성별대조": summarize_field(cb_sexes),
             "엑셀_PSG_생년월일대조": summarize_field(psg_births), "엑셀_PSG_성별대조": summarize_field(psg_sexes),
             "엑셀_성별문자": text(row[5]), "엑셀_성별숫자코드": row[10],
             "CBCT_PSG등록번호_보조대조": cb_aux,
             "엑셀_리포트연결": "요약값 대응 리포트 확인" if ref else "엑셀 대응 검사 미확인",
             "엑셀_원신호검사": exam_status, "원신호_리포트검사": raw_status,
             "PSG_환자확인근거": "엑셀 No.↔PSG 폴더 P_n; 엑셀 생년월일 YYMMDD·문자 성별↔각 리포트. CBCT와 독립 대조",
             "CBCT_환자확인근거": "엑셀 No.↔CBCT 폴더 P_n; 엑셀 생년월일 YYMMDD·문자 성별↔상위 DICOM. PSG와 독립 대조",
             "보조등록번호비교근거": "DICOM PatientID↔PSG 리포트 ID; 두 자료 간 교차확인일 뿐 엑셀 대응 판정에는 미사용",
             "엑셀_직접병원ID대조": "불가: 엑셀에 병원 등록번호 없음. 연구번호·생년월일(세기 미확인)·성별 대조만 수행",
             "PSG등록번호_SHA256": "; ".join(sorted(id_digest(v) for v in ids)),
             "CBCT상위PatientID_SHA256": "; ".join(sorted({q["PatientID_SHA256"] for q in top})),
             "엑셀검사연결근거": "TST±0.6분 및 AHI±0.1의 리포트 후보; 검사일 직접 기재 없음(값 기반 연결)",
             "엑셀대응리포트": rel(ref["file"]) if ref else "",
             "엑셀대응검사일": ref["Study"] if ref else "",
             "원신호대응리포트": rel(raw_ref["file"]) if raw_ref else "",
             "원신호대응검사일": raw_ref["Study"] if raw_ref else "",
             "판독파일기록일": text(r.get("RawDate")),
             "원신호확인파일": rel(wave) if wave else "", "원신호시작": wave_start, "원신호종료": wave_end,
             "원신호환자식별범위": "폴더 P_n 및 검사시각 연결; 원신호 내부 환자ID 별도 미검증",
             "CBCT촬영일": "; ".join(sorted({q["촬영일"] for q in top})),
             "CBCT상위파일수": sum(q["DICOM수"] for q in top), "CBCT전체파일수": sum(len(v) for v in groups.values()),
             "CBCT하위그룹수": sum(q["범위"].startswith("하위") for q in patient_series),
             "CBCT계측값검증": "이번 범위 아님", "리포트수": len(rp),
             "기존판독집계방식": text(r.get("RawSource")),
             "엑셀사건수대응출처": text(r.get("ExcelCountsFrom")),
             "기존원인분류": text(r.get("Causes")), "기존근거": text(r.get("Evidence")),
             "근거문서": rel(case / "evidence.html"), "검토완료": "미검토", "검토자의견": ""}
        # 후보가 중복이면 기존 코드의 첫 번째 선택을 최종 연결로 인정하지 않는다.
        eligible = [q for q in candidates if cmp(r.x_TST, q["TST"], .6)[0] == "일치" and cmp(r.x_AHI, q["AHI"], .1)[0] == "일치"]
        if len(eligible) != 1:
            ref = None
            m.update(엑셀_리포트연결="엑셀 대응 검사 미확인", 엑셀대응리포트="", 엑셀대응검사일="", 엑셀_원신호검사="엑셀 대응 검사 미확인")
            if len(eligible) > 1:
                notes.append("엑셀 요약값에 대응하는 리포트 복수; 자동 선택 안 함")
        statuses = []
        for k in SUMMARY + COUNTS:
            a, b = r.get(f"x_{k}", np.nan), ref[k] if ref else np.nan
            status, delta = cmp(a, b, TOL[k])
            statuses.append(status)
            m.update({f"엑셀_{k}": a, f"리포트_{k}": b, f"차이_{k}": delta, f"판정_{k}": status})
            raw_val = r.get(f"{'txt' if r.Device == 'E' else 'rule'}_{k}", np.nan) if k in COUNTS else (r.get("NoxRuleTST", np.nan) if k == "TST" else np.nan)
            raw_report_val = raw_ref[k] if raw_ref else np.nan
            raw_cmp, raw_delta = cmp(raw_val, raw_report_val, TOL[k])
            detail = {"환자": patient, "변수": k, "엑셀원값": a, "동일검사리포트원값": b,
                      "엑셀-리포트차이": delta, "엑셀-리포트판정": status, "허용차이": TOL[k],
                      "엑셀대응리포트": rel(ref["file"]) if ref else "",
                      "원신호판독값": raw_val, "원신호대응리포트값": raw_report_val,
                      "판독-리포트차이": raw_delta, "판독-리포트판정": raw_cmp,
                      "원신호대응리포트": rel(raw_ref["file"]) if raw_ref else "",
                      "판독값설명": "Embla TXT 전체 사건수" if r.Device == "E" else "Nox Manual 우선 후보 규칙(최종 판독 확정 아님)"}
            details.append(detail)
        m["PSG요약값대조"] = "비교 불가" if "비교 불가" in statuses[:5] else ("일치" if set(statuses[:5]) == {"일치"} else "불일치")
        m["PSG사건수대조"] = "비교 불가" if "비교 불가" in statuses[5:] else ("일치" if set(statuses[5:]) == {"일치"} else "불일치")
        m["검토사항"] = " | ".join(notes)
        # 민감한 식별자 자체는 저장하지 않고 일치 여부와 근거 파일만 저장.
        pd.DataFrame(manifest).to_csv(case / "source_files.csv", index=False, encoding="utf-8-sig")
        all_manifest.extend(manifest)
        (case / "dicom_read_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        patient_details = [q for q in details if q["환자"] == patient]
        numeric_fields = {f"{prefix}_{key}" for prefix in ("엑셀", "리포트", "차이", "판정") for key in SUMMARY + COUNTS}
        display_master = pd.DataFrame([(k, v) for k, v in m.items() if k not in numeric_fields], columns=["항목", "내용"])
        source_links = "<ul>" + "".join(f"<li>{html.escape(q['역할'])}: {link(ROOT / q['원본경로'], q['원본경로'])}</li>" for q in manifest if q["역할"] != "CBCT 원본 DICOM") + "</ul>"
        dicom_links = "<ul>" + "".join(f"<li>{html.escape(q['범위'])}: {link(ROOT / q['대표파일'], q['대표파일'])} — 엑셀 직접 대조: {q['엑셀_CBCT_식별정보대조']}, {q['DICOM수']}개</li>" for q in patient_series) + "</ul>"
        body = (f"<h1>{patient} 환자별 근거</h1><p>생성 {generated}. 자동 대조 결과이며 사람 검토는 미완료. 원본 변경·제외·값 교체 없음.</p>"
                + "<h2>대응 결과</h2>" + display_master.to_html(index=False, na_rep="", escape=True)
                + "<h2>PSG 원값 대조</h2>" + pd.DataFrame(patient_details).to_html(index=False, na_rep="", escape=True)
                + "<h2>CBCT 시리즈·ID</h2>" + pd.DataFrame(patient_series).to_html(index=False, na_rep="", escape=True)
                + dicom_links + "<h2>PSG 근거 원본</h2>" + source_links
                + "<h2>리포트 사본</h2><ul>" + "".join(f"<li>{link(case / 'reports' / q['file'].name, q['file'].name)}</li>" for q in rp) + "</ul>"
                + f"<p>{link(case / 'source_files.csv', '전체 원본 파일 목록(CBCT 전 슬라이스 포함)')}</p>")
        document = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>' + patient + '</title><style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}th{background:#edf2f7}h2{margin-top:28px}</style><body>' + body + '</body></html>'
        (case / "evidence.html").write_text(document, encoding="utf-8")
        (case / "README.md").write_text(f"# {patient}\n\n검토 문서: [evidence.html](evidence.html)\n\n전체 파일: [source_files.csv](source_files.csv)\n\n리포트 사본은 reports/에 보관. 대용량 원신호와 DICOM은 복제하지 않고 원본 경로로 연결합니다.\n", encoding="utf-8")
        masters.append(m)
        print(f"{patient}: CBCT {cb_status}; PSG {m['엑셀_원신호검사']}; DICOM {m['CBCT전체파일수']}", flush=True)

    master = pd.DataFrame(masters)
    detail_df = pd.DataFrame(details)
    assert len(master) == 87 and master.환자.is_unique and len(detail_df) == 87 * 11
    assert all((ROOT / q["원본경로"]).is_file() for q in all_manifest)
    for name, df in [("patient_matching", master), ("psg_values", detail_df),
                     ("reports", pd.DataFrame(reports_table)), ("cbct_series", pd.DataFrame(series_table)),
                     ("source_files", pd.DataFrame(all_manifest))]:
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    rules = pd.DataFrame([
        ("범위", "CBCT↔엑셀↔PSG: 두 독립 연결. CBCT 계측값 대조·표본 제외·검사 선택·값 교체 없음"),
        ("엑셀 ID", "No.는 연구번호이며 PSG/CBCT 열은 O 표시. 병원 등록번호 열 없음"),
        ("엑셀↔PSG", "No.↔PSG 폴더 P_n; 생년월일 YYMMDD·문자 성별↔리포트. CBCT를 판정 근거로 사용하지 않음. 원신호 내부 ID 별도 미검증"),
        ("엑셀↔CBCT", "No.↔CBCT 폴더 P_n; 생년월일 YYMMDD·문자 성별↔상위 DICOM. PSG를 판정 근거로 사용하지 않음"),
        ("CBCT↔PSG 보조 확인", "DICOM PatientID↔PSG 리포트 등록번호. 기존 61일치/26불일치는 이 보조 비교일 뿐 엑셀 대응 결과가 아님"),
        ("직접 대조 한계", "엑셀 병원 등록번호·생년월일 세기 없음. 식별정보 일치는 동일인 절대 확정과 구분. 성별은 문자 열을 사용하고 숫자 코드 오류와 분리"),
        ("CBCT 하위 폴더", "별도 시리즈로 전부 기록. 상위 영상 판정과 분리하며 혼입 경고 표시"),
        ("ID 불일치 해석", "ID 문자열이 다름을 뜻하며 다른 사람임을 단정하지 않음. 기관별 번호 체계 여부는 별도 확인 필요"),
        ("엑셀 검사 연결", "엑셀에 PSG 검사일 없음. TST±0.6분 AND AHI±0.1로 유일 후보 연결; 값 기반 근거임을 명시"),
        ("원신호 검사", "SpO2 실제 시작 시각과 리포트 날짜 비교. 정오 전 시작은 전날 밤 날짜도 허용; 복수 후보 자동 선택 안 함"),
        ("요약값 허용차", "TST±0.6분, AHI±0.1, RDI±0.15; ODI/최저SpO2 정확 일치. 차이는 엑셀-리포트"),
        ("사건수", "6종 각각 정수 정확 일치. 원신호 판독과 해당 검사 리포트도 별도 비교"),
        ("원신호 판독", "Embla TXT 전체 개수; Nox 기존 Manual 우선 후보 규칙. 최종 판독 확정/임상 허용오차 의미 아님"),
        ("AHI", "판독 개수로 재계산한 AHI를 리포트 AHI로 대체하지 않음. 리포트 내부 정의 차이는 기존근거 열 참조"),
        ("원신호 범위", "이번 직접 파형 확인은 SpO2 파일의 시각. 전 채널 품질·AHI 새 판독·ODI 재계산은 수행 안 함"),
        ("근거 파일", "환자별 보고서 사본+SHA256, 원본 전체 경로 목록, HTML 검토문서. 큰 파형·DICOM은 원본 링크"),
        ("비교 불가", "출처 없는 검사·값은 공란/비교 불가. 불일치 또는 제외로 자동 치환 안 함"),
        ("생성시각", generated), ("엑셀 SHA256", sha(EXCEL)), ("reconcile SHA256", sha(source)),
    ], columns=["항목", "기준"])
    # 상태를 먼저 배치하고 상세 수치 열은 뒤에 둔다.
    lead = ["환자", "엑셀행", "엑셀_CBCT_식별정보대조", "엑셀_PSG_식별정보대조", "엑셀_리포트연결", "엑셀_원신호검사",
            "원신호_리포트검사", "PSG요약값대조", "PSG사건수대조", "검토완료", "검토사항", "근거문서"]
    master = master[lead + [c for c in master.columns if c not in lead]]
    master.to_csv(OUT / "patient_matching.csv", index=False, encoding="utf-8-sig")
    rules.to_csv(OUT / "matching_rules.csv", index=False, encoding="utf-8-sig")
    counts = {col: master[col].value_counts().to_dict() for col in ["엑셀_CBCT_식별정보대조", "엑셀_PSG_식별정보대조", "CBCT_PSG등록번호_보조대조", "엑셀_리포트연결", "엑셀_원신호검사", "원신호_리포트검사", "PSG요약값대조", "PSG사건수대조"]}
    counts.update(환자수=87, 리포트수=len(reports_table), DICOM수=sum(q["DICOM수"] for q in series_table), 전체근거파일수=len(all_manifest), 생성시각=generated)
    (OUT / "summary.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    index_rows = []
    for _, m in master.iterrows():
        index_rows.append(f"<tr><td>{link(ROOT / m['근거문서'], m['환자'])}</td>" + "".join(f"<td>{html.escape(text(m[c]))}</td>" for c in lead[2:9]) + "</tr>")
    (OUT / "index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>87명 엑셀 기준 대응 검토</title><style>body{font:14px sans-serif;margin:24px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style><h1>CBCT ↔ 엑셀 ↔ PSG · 87명 대응 검토</h1><p>엑셀 No.·생년월일·성별을 기준으로 CBCT와 PSG를 각각 독립 대조. 두 검사 간 등록번호 비교는 보조 정보입니다. 식별정보 일치와 동일인 절대 확정은 구분하며, 사람 검토/최종 분석 표본 확정 전입니다.</p>' + link(OUT / "환자 일치 검수.xlsx", "엑셀 기준 대응표(산출값 보완 전)") + '<table><tr><th>환자</th>' + ''.join(f'<th>{c}</th>' for c in lead[2:9]) + '</tr>' + ''.join(index_rows) + '</table></html>', encoding="utf-8")
    (OUT / "README.md").write_text("# 환자 대응 검토\n\n- [전체 검토 목차](index.html)\n- [87명 대응표](환자 일치 검수.xlsx)\n- [집계](summary.json)\n\n모든 환자는 미검토 상태로 생성됩니다. 자동 대조는 사람의 최종 검토를 대체하지 않습니다. CBCT 계측값은 이번 범위에 포함하지 않습니다.\n\n재생성: 먼저 `python/psg_reconcile.py`, 이어서 `python/patient_matching.py`. 환자별 검토 의견은 재생성 전에 별도 보존하세요. 원본 파일은 이동/수정하지 않습니다.\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def enrich_source_evidence():
    """공통 엑셀·생성 근거도 환자별 목록에 넣고, 사본/원본 해시를 기록한다."""
    master = pd.read_csv(OUT / "patient_matching.csv")
    common = [(EXCEL, "엑셀 원본"), (ROOT / "Result_py/psg_reconcile.csv", "PSG 재대조 중간 결과"),
              (ROOT / "python/psg_reconcile.py", "PSG 재대조 코드"),
              (ROOT / "python/patient_matching.py", "환자 대응 생성 코드")]
    common_rows = [{"역할": role, "원본경로": rel(p), "바이트": p.stat().st_size,
                    "SHA256": sha(p), "검토용사본": ""} for p, role in common]
    all_rows = []
    for _, row in master.iterrows():
        case = OUT / "patients" / row["환자"]
        f = case / "source_files.csv"
        df = pd.read_csv(f).fillna("")
        df = df[~df.원본경로.isin([q["원본경로"] for q in common_rows])]
        extra = pd.DataFrame([{"환자": row["환자"], **q} for q in common_rows])
        df = pd.concat([extra, df], ignore_index=True)
        df.to_csv(f, index=False, encoding="utf-8-sig")
        all_rows.extend(df.to_dict("records"))
        page = case / "evidence.html"
        content = page.read_text(encoding="utf-8")
        marker = '<section id="excel-source">'
        if marker not in content:
            section = (marker + '<h2>엑셀 및 생성 근거</h2><p>'
                       + link(EXCEL, f"원본 엑셀: {int(row['엑셀행'])}행 (No. {row['환자']})")
                       + '</p><p>엑셀 No.와 P_n 폴더 번호의 연결은 확인됨. 병원 등록번호 직접 대조와는 구분합니다.</p>'
                       + '<ul>' + ''.join('<li>' + link(p, role) + '</li>' for p, role in common[1:])
                       + '</ul></section>')
            page.write_text(content.replace('</body>', section + '</body>'), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUT / "source_files.csv", index=False, encoding="utf-8-sig")
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["전체근거파일수"] = len(all_rows)
    summary["공통입력SHA256"] = {q["역할"]: q["SHA256"] for q in common_rows}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
    enrich_source_evidence()
