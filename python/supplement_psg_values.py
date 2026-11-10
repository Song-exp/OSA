"""기존 환자 대응표에 PSG 판독/파형 산출값을 보완. 원본/초기 엑셀은 보존한다.

먼저 psg_waveform_check.py 실행. 공식 리포트 값과 재계산/파형 참고값을 구분한다.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from patient_matching import OUT, ROOT, cmp, link, rel, sha


def cbct_review(row):
    birth, sex = row["엑셀_CBCT_생년월일대조"], row["엑셀_CBCT_성별대조"]
    aux, psg_birth = row["CBCT_PSG등록번호_보조대조"], row["엑셀_PSG_생년월일대조"]
    encoding_ok = {"M": 1, "F": 2}.get(row["엑셀_성별문자"]) == row["엑셀_성별숫자코드"]
    facts = (f"CBCT↔엑셀: 생년월일 {birth}, 문자 성별 {sex}; "
             f"엑셀 문자·숫자 성별 코드 {'일관' if encoding_ok else '불일치'}; "
             f"엑셀↔PSG 생년월일 {psg_birth}; CBCT↔PSG 등록번호 {aux}(보조 확인)")
    if row["환자"] == "P_39":
        interpretation = "기존 전수 헤더 조사에서 본인 CBCT를 찾지 못한 영상 오배정 사례. 단순 생년월일 차이만으로 내린 판정과 구분"
        followup = "P_39 본인 CBCT 원본과 엑셀 CBCT 계측에 사용한 영상의 출처 확인"
        evidence = "docs/원본데이터_결측_점검.md (3-3), docs/MATLAB_대비_개선점_및_품질점검.md (4-A)"
    elif birth == "불일치" and aux == psg_birth == "일치":
        interpretation = "엑셀 생년월일은 PSG와 일치하고 두 원본의 등록번호도 같아 CBCT 생년월일 입력 오류 가능성. 생년월일 차이만으로 다른 환자로 확정할 수 없음"
        followup = "CBCT 기관의 원등록정보·원자료에서 생년월일 확인. 입력 오류/다른 환자 여부는 확인 전 보류"
        evidence = "환자별 DICOM 대표 파일과 PSG 리포트: 근거문서 참조"
    elif sex == "불일치" and birth == aux == "일치":
        interpretation = ("엑셀 문자·숫자 성별이 일관하므로 정수 인코딩 오류에 따른 불일치가 아님. "
                          "생년월일과 보조 등록번호가 같아 CBCT 성별 입력 오류 가능성; 동일인 여부는 추가 확인 필요"
                          if encoding_ok else "엑셀 문자·숫자 성별 코드도 불일치하므로 인코딩과 DICOM 성별을 각각 재확인 필요")
        followup = "CBCT 기관의 원등록 성별 확인. 직접 대조의 성별 불일치 표시는 유지"
        evidence = "원본 엑셀의 문자·숫자 성별과 DICOM·PSG 리포트: 근거문서 참조"
    elif row["엑셀_CBCT_식별정보대조"] == "식별정보 일치":
        interpretation = "연구번호·생년월일 YYMMDD·문자 성별 대조 일치. 엑셀에 병원 등록번호가 없어 동일인 절대 확정과는 구분"
        followup, evidence = "추가 불일치 근거 발견 시 검토", "환자별 근거문서 참조"
    else:
        interpretation = "직접 대조 불일치 또는 정보 부족. 현재 근거만으로 입력 오류/다른 환자를 확정하지 않음"
        followup, evidence = "원등록정보 확인", "환자별 근거문서 참조"
    return facts, interpretation, followup, evidence


def main():
    source = ROOT / "Result_py/psg_waveform_check.csv"
    wave = pd.read_csv(source).set_index("Patient")
    rec = pd.read_csv(ROOT / "Result_py/psg_reconcile.csv").set_index("Patient")
    master = pd.read_csv(OUT / "patient_matching.csv")
    details = pd.read_csv(OUT / "psg_values.csv")
    assert len(master) == len(wave) == 87
    definitions = {
        "TST": ("수면단계 판독 합계", "수면단계 중 N1/N2/N3/REM 구간 길이 합계(분); 새 수면단계 판독 아님"),
        "AHI": ("기존 판독 기반 재계산값", "(OA+CA+MA+H) / (판독 TST/60). Embla TXT 전체 사건수 또는 Nox 후보 규칙; 공식 리포트 값 아님"),
        "RDI": ("기존 판독 기반 재계산값", "(OA+CA+MA+H+RERA) / (판독 TST/60). 호흡각성 RESP는 중복 합산하지 않음; 공식 리포트 값 아님"),
        "ODI": ("파형 알고리즘 참고값", "1초 SpO2 중앙값; 직전120초 최고값 대비 ≥3%p 하락 ≥10초, 수면 중 시작 사건/TST시간. 장비 알고리즘 미확인"),
        "LSaO2": ("파형 직접 산출값", "판독상 수면 중 유효 SpO2(50~100%) 원샘플 최솟값. 전체 기록 최솟값도 별도 보존"),
    }
    computed = {}
    for patient in master["환자"]:
        w, r = wave.loc[patient], rec.loc[patient]
        hours = w.StageSleepMin / 60
        tag = "txt" if r.Device == "E" else "rule"
        ah = sum(r[f"{tag}_{k}"] for k in ("OA", "CA", "MA", "H"))
        computed[patient] = {
            "TST": w.StageSleepMin,
            "AHI": round(ah / hours, 3) if hours > 0 else np.nan,
            "RDI": round((ah + r[f"{tag}_RERA"]) / hours, 3) if hours > 0 else np.nan,
            "ODI": w.Wave_ODI3, "LSaO2": w.Wave_LSaO2_Sleep,
        }
    for i, row in details.iterrows():
        patient, key = row["환자"], row["변수"]
        if key not in definitions:
            continue
        w = wave.loc[patient]
        val = computed[patient][key]
        state, delta = cmp(val, row["원신호대응리포트값"], row["허용차이"])
        details.loc[i, "원신호판독값"] = val
        details.loc[i, "판독-리포트차이"] = delta
        details.loc[i, "판독-리포트판정"] = ("참고 비교(알고리즘 차이 가능)" if key == "ODI" and state != "비교 불가" else state)
        details.loc[i, "값유형"], details.loc[i, "산출정의"] = definitions[key]
        details.loc[i, "판독값설명"] = definitions[key][1]
        details.loc[i, "산출근거파일"] = rel(source)
        details.loc[i, "LSaO2_전체기록최저"] = w.Wave_LSaO2_All if key == "LSaO2" else np.nan
        details.loc[i, "LSaO2_수면중최저"] = w.Wave_LSaO2_Sleep if key == "LSaO2" else np.nan
    event_rows = ~details["변수"].isin(definitions)
    details.loc[event_rows, "값유형"] = "기존 사건 판독 집계값"
    details.loc[event_rows, "산출정의"] = details.loc[event_rows, "판독값설명"]
    for key in definitions:
        master[f"원신호산출_{key}"] = master["환자"].map({p: v[key] for p, v in computed.items()})
    master["원신호산출값주의"] = "공식 결과값과 구분. TST=판독합계, AHI/RDI=재계산, ODI=참고 알고리즘, LSaO2=수면중 최저"
    review_columns = ["CBCT_확인된사실", "CBCT_검토해석", "CBCT_추가확인사항", "CBCT_해석근거"]
    master[review_columns] = pd.DataFrame([cbct_review(row) for _, row in master.iterrows()], index=master.index)
    details = details.rename(columns={"원신호판독값": "원본판독·파형산출값"})
    assert details[details["변수"].isin(definitions)]["원본판독·파형산출값"].notna().all()
    details.to_csv(OUT / "psg_values_with_raw.csv", index=False, encoding="utf-8-sig")
    master.to_csv(OUT / "patient_matching_with_raw.csv", index=False, encoding="utf-8-sig")
    all_sources = []
    common = [(source, "PSG 판독·파형 산출 근거"), (ROOT / "python/psg_waveform_check.py", "파형 산출 코드"),
              (ROOT / "python/supplement_psg_values.py", "대응표 보완 코드")]
    for patient in master["환자"]:
        case = OUT / "patients" / patient
        page = case / "evidence.html"
        body = page.read_text(encoding="utf-8")
        review = master[master["환자"].eq(patient)][review_columns].T.reset_index()
        review.columns = ["항목", "내용"]
        section = '<section id="cbct-review"><h2>CBCT 대조 사실과 검토 해석</h2><p>해석은 추가 확인을 위한 검토 의견이며 직접 대조 판정이나 분석 포함 여부를 변경하지 않습니다.</p>' + review.to_html(index=False, escape=True) + '</section>'
        if 'id="cbct-review"' in body:
            body = re.sub(r'<section id="cbct-review">.*?</section>', lambda _: section, body, count=1, flags=re.S)
        else:
            body = body.replace('</body>', section + '</body>')
        table = details[details["환자"] == patient].to_html(index=False, na_rep="", escape=True)
        body, n = re.subn(r"(<h2>PSG 원값 대조</h2>)<table.*?</table>", lambda m: m.group(1) + table,
                          body, count=1, flags=re.S)
        assert n == 1
        marker = '<section id="raw-values-note">'
        if marker not in body:
            body = body.replace("</body>", marker + "<h2>원신호 산출값 보완</h2><p>공식 리포트 값과 구분합니다. AHI/RDI는 기존 판독 기반 재계산이며 ODI는 자체 규칙 참고값입니다. LSaO2는 수면 중/전체 기록 최솟값을 모두 보존합니다.</p>"
                                + link(source, "87명 파형 산출 근거 CSV") + "</section></body>")
        page.write_text(body, encoding="utf-8")
        manifest = pd.read_csv(case / "source_files.csv").fillna("")
        for path, role in common:
            manifest = manifest[manifest["원본경로"] != rel(path)]
            manifest = pd.concat([manifest, pd.DataFrame([{"환자": patient, "역할": role,
                "원본경로": rel(path), "바이트": path.stat().st_size, "SHA256": sha(path), "검토용사본": ""}])], ignore_index=True)
        manifest.to_csv(case / "source_files.csv", index=False, encoding="utf-8-sig")
        all_sources.extend(manifest.to_dict("records"))
    pd.DataFrame(all_sources).to_csv(OUT / "source_files.csv", index=False, encoding="utf-8-sig")
    result = {"patients": 87, "supplemented_values": 435,
              "missing_values": int(details[details["변수"].isin(definitions)]["원본판독·파형산출값"].isna().sum()),
              "all_report_values_preserved": bool(details["동일검사리포트원값"].equals(pd.read_csv(OUT / "psg_values.csv")["동일검사리포트원값"])),
              "workbooks": ["환자 일치 검수.xlsx", "PSG 값 일치 검수.xlsx"], "source_sha256": sha(source)}
    assert result["missing_values"] == 0 and result["all_report_values_preserved"]
    (OUT / "psg_computation_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_file = OUT / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["전체근거파일수"] = len(all_sources)
    summary["원신호5종보완"] = result
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    from build_review_workbooks import main as build_reviews
    build_reviews()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
