"""검수 엑셀 2개만 생성. 환자 식별정보 대조와 같은 검사 값 대조를 분리한다."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.worksheet.datavalidation import DataValidation
from patient_matching import ROOT, OUT, EXCEL, dob, dob_matches, report_identity, identity_fields, identity_status, make_book, sha, link

PATIENT_BOOK = OUT / "환자 일치 검수.xlsx"
VALUE_BOOK = OUT / "PSG 값 일치 검수.xlsx"
EDIT = ["검수의견", "최종판단", "검수자", "검수일", "최종사용값", "최종사용출처"]
LEGACY = ["환자별_대응표_엑셀기준_검토해석보완.xlsx", "환자별_대응표_엑셀기준.xlsx",
          "환자별_대응표_원신호보완.xlsx", "환자별_대응표.xlsx"]


def empty_review(df):
    for c in ("검수의견", "최종판단", "검수자", "검수일"):
        df[c] = "미검토" if c == "최종판단" else ""
    return df


def restore(frames, file):
    if not file.is_file():
        return
    old = pd.read_excel(file, sheet_name=None)
    for sheet, df in frames.items():
        if sheet not in old:
            continue
        keys = [c for c in ("환자", "자료키", "검사키", "지표") if c in df and c in old[sheet]]
        prev = old[sheet].set_index(keys)
        if not prev.index.is_unique:
            raise ValueError(f"{file.name}/{sheet}: 중복 검수 키")
        for i, row in df.iterrows():
            key = tuple(row[k] for k in keys) if len(keys)>1 else row[keys[0]]
            if key not in prev.index:
                continue
            for col in EDIT:
                if col in df and col in prev:
                    value = prev.loc[key, col]
                    if pd.notna(value) and str(value).strip():
                        df.at[i, col] = value


def write(frames, file):
    restore(frames, file)
    make_book(frames, file)
    book = load_workbook(file)
    for ws in book:
        for row_index in range(2, ws.max_row+1):
            ws.row_dimensions[row_index].height = 54 if ws.title=="통합·최종판단" else 90 if ws.title=="CBCT" else 66
        cols = {c.value: c.column for c in ws[1]}
        ws.freeze_panes = "B2"
        for name, col in cols.items():
            if name in EDIT:
                ws.cell(1,col).fill = PatternFill("solid",fgColor="386B80")
                for cells in ws.iter_cols(min_col=col,max_col=col,min_row=2):
                    for cell in cells:
                        cell.fill = PatternFill("solid",fgColor="E8F4FA")
            if any(x in str(name) for x in ("해석","확인사항","사실","의견","근거","방법")):
                ws.column_dimensions[ws.cell(1,col).column_letter].width = 46
            if name in ("최종판단","최종사용출처"):
                options = '미검토,확인완료,추가확인필요,자료부족' if name=="최종판단" else '엑셀,보고서,원본산출,보류'
                dv=DataValidation(type="list",formula1='"'+options+'"',allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{ws.cell(1,col).column_letter}2:{ws.cell(1,col).column_letter}{ws.max_row}")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value,str) and cell.value.startswith(("Data/","Result_py/","docs/")) and (ROOT/cell.value).is_file():
                    cell.hyperlink=(ROOT/cell.value).as_uri()
                    cell.font=Font(color="0563C1",underline="single")
    book.calculation.fullCalcOnLoad = True
    book.save(file)


def main():
    m=pd.read_csv(OUT/"patient_matching_with_raw.csv").astype(object).fillna("")
    vals=pd.read_csv(OUT/"psg_values_with_raw.csv")
    series=pd.read_csv(OUT/"cbct_series.csv").fillna("")
    reports=pd.read_csv(OUT/"reports.csv").astype(object).fillna("")
    raw=pd.read_excel(EXCEL,header=None)
    excel={int(r[0]):r for _,r in raw.iloc[3:].iterrows() if isinstance(r[0],(int,float)) and pd.notna(r[0])}
    bypatient=m.set_index("환자")
    cbct=[]
    for _,s in series.iterrows():
        p=s["환자"];r=bypatient.loc[p];x=excel[int(p[2:])];f=ROOT/s["대표파일"]
        assert sha(f)==s["대표파일_SHA256"]
        h=pydicom.dcmread(f,stop_before_pixels=True,specific_tags=["PatientBirthDate","PatientSex"],force=True)
        cbct.append({"환자":p,"자료키":s["대표파일"],"자료범위":s["범위"],"엑셀행":r["엑셀행"],
          "연구번호대조":r["엑셀_CBCT_연구번호대조"],"엑셀_생년월일_YYMMDD":dob(x[4]),
          "DICOM_생년월일_YYYYMMDD":str(h.get("PatientBirthDate","")),"생년월일대조":s["엑셀_생년월일대조"],
          "엑셀_문자성별":x[5],"엑셀_숫자성별":x[10],"DICOM_성별":str(h.get("PatientSex","")),
          "성별대조":s["엑셀_성별대조"],"식별정보대조":s["엑셀_CBCT_식별정보대조"],
          "촬영일":s["촬영일"],"DICOM수":s["DICOM수"],"CBCT_PSG등록번호_보조대조":s["CBCT_PSG등록번호_보조대조"],
          "확인된사실":r["CBCT_확인된사실"] if str(s["범위"]).startswith("상위") else "하위 시리즈는 주 영상과 별도 대조",
          "검토해석":r["CBCT_검토해석"] if str(s["범위"]).startswith("상위") else "다른 환자 영상 혼입 여부를 주 영상과 구분해 검토",
          "추가확인사항":r["CBCT_추가확인사항"] if str(s["범위"]).startswith("상위") else "영상 소유자 및 사용할 시리즈 확인",
          "대표파일":s["대표파일"],"근거문서":r["근거문서"]})
    cbct=empty_review(pd.DataFrame(cbct))
    psg=[]
    for _,s in reports.iterrows():
        p=s["환자"];r=bypatient.loc[p];x=excel[int(p[2:])];ident=report_identity(ROOT/s["파일"])
        b,g=identity_fields(x[4],x[5],ident["dob"],ident["sex"])
        roles=[]
        if s["파일"]==r["엑셀대응리포트"]:roles.append("엑셀 요약값 연결")
        if s["파일"]==r["원신호대응리포트"]:roles.append("원신호 연결")
        role=" / ".join(roles) or "기타 검사 보고서"
        psg.append({"환자":p,"자료키":s["파일"],"자료구분":role,"검사일":s["검사일"],"엑셀행":r["엑셀행"],
          "연구번호대조":r["엑셀_PSG_연구번호대조"],"엑셀_생년월일_YYMMDD":dob(x[4]),"보고서_생년월일_YYYYMMDD":ident["dob"],
          "생년월일대조":b,"엑셀_문자성별":x[5],"엑셀_숫자성별":x[10],"보고서_성별":ident["sex"],
          "성별대조":g,"식별정보대조":identity_status([b],[g]),"보고서":s["파일"],
          "원신호확인파일":r["원신호확인파일"] if "원신호 연결" in roles else "",
          "확인된사실":f"{role}; 생년월일 {b}, 성별 {g}",
          "검토해석":"성별 표기가 다른 보고서. 다른 검사와 구분해 확인" if g=="불일치" else "연구번호·생년월일·성별 기준 대조",
          "추가확인사항":"원등록정보 확인" if g=="불일치" else "",
          "근거문서":r["근거문서"]})
    for _,r in m.iterrows():
        if not r["원신호대응리포트"]:
            psg.append({"환자":r["환자"],"자료키":r["원신호확인파일"],"자료구분":"원신호(대응 보고서 없음)",
              "검사일":r["판독파일기록일"] or str(r["원신호시작"])[:10],"엑셀행":r["엑셀행"],"연구번호대조":r["엑셀_PSG_연구번호대조"],
              "식별정보대조":"확인 불가","원신호확인파일":r["원신호확인파일"],
              "확인된사실":"같은 날짜 보고서 없음. 원신호 내부 환자 ID는 별도 미검증",
              "추가확인사항":"해당 검사 보고서 확보","근거문서":r["근거문서"]})
    psg=empty_review(pd.DataFrame(psg).astype(object).fillna(""))
    overview=m[["환자","엑셀행","엑셀_CBCT_식별정보대조","엑셀_PSG_식별정보대조","엑셀_리포트연결","근거문서"]].copy()
    overview["판정기준요약"]="엑셀 No.·생년월일 YYMMDD·문자 성별을 각각 직접 비교. 병원 등록번호는 엑셀에 없어 보조 확인만 수행"
    for col in ("엑셀요약연결검사_성별대조","원신호연결검사_성별대조"):
        overview[col]="확인 불가"
    for i,r in overview.iterrows():
        pr=psg[psg["환자"].eq(r["환자"])]
        for role,col in [("엑셀 요약값 연결","엑셀요약연결검사_성별대조"),("원신호 연결","원신호연결검사_성별대조")]:
            selected=pr[pr["자료구분"].str.contains(role,regex=False)]
            if len(selected)==1:overview.at[i,col]=selected.iloc[0]["성별대조"]
    overview["주요확인사항"]=m["검토사항"]
    overview=empty_review(overview)
    # 과거 통합 파일의 수기 의견은 출처와 함께 별도 보존. 새 최종판단으로 자동 승계하지 않음.
    overview["이전검수의견"]="";overview["이전검수상태"]=""
    migrated=0
    for name in LEGACY:
        file=OUT/name
        if not file.exists():continue
        old=pd.read_excel(file,sheet_name="87명_대응표").astype(object).fillna("").set_index("환자")
        for i,r in overview.iterrows():
            if r["환자"] not in old.index:continue
            o=old.loc[r["환자"]]
            if str(o.get("검토자의견","")).strip() and not overview.at[i,"이전검수의견"]:
                overview.at[i,"이전검수의견"]=f"{name}: {o['검토자의견']}";migrated+=1
            if o.get("검토완료","") not in ("","미검토") and not overview.at[i,"이전검수상태"]:
                overview.at[i,"이전검수상태"]=f"{name}: {o['검토완료']}"
    write({"통합·최종판단":overview,"CBCT":cbct,"PSG":psg},PATIENT_BOOK)
    details=[]
    for _,r in m.iterrows():
        p=r["환자"];sub=vals[vals["환자"].eq(p)]
        same=r["엑셀_원신호검사"]=="동일 검사"
        exams=[("동일 검사",True,True)] if same else [("엑셀 연결검사" if r["엑셀대응검사일"] else "엑셀 검사 미확인",True,False),("원신호 검사",False,True)]
        for scope,has_excel,has_raw in exams:
            date=r["엑셀대응검사일"] if has_excel else r["원신호대응검사일"] or r["판독파일기록일"] or str(r["원신호시작"])[:10]
            exam_key=f"{p}:{scope}:{date or '미확인'}"
            for _,v in sub.iterrows():
                report_col="동일검사리포트원값" if has_excel else "원신호대응리포트값"
                a=v["엑셀원값"] if has_excel else np.nan
                b=v[report_col]
                c=v["원본판독·파형산출값"] if has_raw else np.nan
                tol=v["허용차이"]
                pairs=[(a,b),(c,b),(a,c)];diffs=[x-y if pd.notna(x) and pd.notna(y) else np.nan for x,y in pairs]
                flags=[abs(x)>tol+1e-8 for x in diffs if pd.notna(x)]
                details.append({"환자":p,"검사키":exam_key,"검사구분":scope,"검사일":date,"지표":v["변수"],
                  "단위":"분" if v["변수"]=="TST" else "%p" if v["변수"]=="LSaO2" else "회/시간" if v["변수"] in ("AHI","RDI","ODI") else "건",
                  "엑셀값":a,"보고서값":b,"원본산출값":c,"엑셀-보고서":diffs[0],"원본-보고서":diffs[1],"엑셀-원본":diffs[2],
                  "허용차후보":tol,"초기기준초과":int(any(flags)),"비교가능쌍수":len(flags),
                  "산출방법":v["산출정의"],"값유형":v["값유형"],
                  "비교주의":"원본은 판독 집계/재계산/참고 알고리즘. 허용차 후보는 임상 오류 기준이 아님. 같은 검사에 속한 값만 비교",
                  "보고서":r["엑셀대응리포트"] if has_excel else r["원신호대응리포트"],
                  "원본파일":r["원신호확인파일"] if has_raw else "","근거문서":r["근거문서"],"최종사용값":"","최종사용출처":""})
    detail=empty_review(pd.DataFrame(details))
    vo=m[["환자","엑셀_원신호검사","엑셀대응검사일","원신호대응검사일","PSG요약값대조","PSG사건수대조","근거문서"]].rename(columns={"엑셀_원신호검사":"PSG_엑셀검사_vs_원신호검사"}).copy()
    vo["환자검수참조"]=m["엑셀_PSG_식별정보대조"]
    vo["환자검수파일"]=str(PATIENT_BOOK.relative_to(ROOT)).replace('\\','/')
    vo["비교주의"]="다른 검사 및 검사 미확인 엑셀값은 별도 행. 사용값·출처와 최종 판단은 검수자가 결정"
    vo=empty_review(vo)
    for col in ("비교항목수","초기기준초과항목수"):
        vo[col]=""
    flag_col=detail.columns.get_loc("초기기준초과")+1
    from openpyxl.utils import get_column_letter
    for i in range(len(vo)):
        excel_row=i+2
        vo.at[i,"비교항목수"]=f'=COUNTIF(PSG!$A$2:$A${len(detail)+1},A{excel_row})'
        vo.at[i,"초기기준초과항목수"]=f'=SUMIF(PSG!$A$2:$A${len(detail)+1},A{excel_row},PSG!${get_column_letter(flag_col)}$2:${get_column_letter(flag_col)}${len(detail)+1})'
    write({"통합·최종판단":vo,"PSG":detail},VALUE_BOOK)
    assert len(overview)==len(vo)==87 and len(detail)==1111
    assert detail["엑셀값"].count()==detail["원본산출값"].count()==957
    assert detail["엑셀-원본"].count()==73*11
    assert not detail.loc[detail["검사구분"].ne("동일 검사"),"엑셀-원본"].notna().any()
    for file,sheets in [(PATIENT_BOOK,["통합·최종판단","CBCT","PSG"]),(VALUE_BOOK,["통합·최종판단","PSG"])]:
        b=load_workbook(file,read_only=True)
        assert b.sheetnames==sheets and b[sheets[0]].max_row==88
        b.close()
    result={"status":"passed","patients":87,"identity_sheets":3,"value_sheets":2,"CBCT_series":len(cbct),
            "PSG_identity_records":len(psg),"PSG_value_records":len(detail),"Excel_values":957,"raw_values":957,
            "same_exam_Excel_raw_comparisons":803,"migrated_legacy_comments":migrated,
            "workbooks":{PATIENT_BOOK.name:sha(PATIENT_BOOK),VALUE_BOOK.name:sha(VALUE_BOOK)}}
    (OUT/"review_workbooks_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    links=link(PATIENT_BOOK,"환자 일치 검수")+" · "+link(VALUE_BOOK,"PSG 값 일치 검수")
    index='<!doctype html><html lang="ko"><meta charset="utf-8"><title>환자·값 검수</title><style>body{font:15px sans-serif;margin:30px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>검수 파일</h1><p>'+links+'</p><p>'+link(OUT/"figures/psg_error_review.html","PSG 11개 지표 시각화")+'</p><h2>환자별 근거</h2><table><tr><th>환자</th><th>CBCT–엑셀</th><th>PSG–엑셀</th></tr>'
    for _,r in m.iterrows():
        index+='<tr><td>'+link(ROOT/r["근거문서"],r["환자"])+'</td><td>'+html.escape(r["엑셀_CBCT_식별정보대조"])+'</td><td>'+html.escape(r["엑셀_PSG_식별정보대조"])+'</td></tr>'
    (OUT/"index.html").write_text(index+'</table></html>',encoding="utf-8")
    (OUT/"README.md").write_text("# 검수 파일\n\n- [환자 일치 검수](환자 일치 검수.xlsx): 통합·최종판단 / CBCT / PSG\n- [PSG 값 일치 검수](PSG 값 일치 검수.xlsx): 통합·최종판단 / PSG\n- [환자별 근거 목차](index.html)\n- [11지표 시각화](figures/psg_error_review.html)\n\n파란 셀은 수기 검수용입니다. 통합은 환자당 한 행, 상세는 자료·검사·지표별 행입니다. 최종판단은 자동으로 확정하지 않습니다. 엑셀 생년월일은 YYMMDD로 세기가 없으며 병원 등록번호도 없습니다. 서로 다른 검사의 엑셀값과 원본값은 오차 계산을 하지 않습니다.\n\n재생성: `python python/build_review_workbooks.py`. 기존 2개 파일의 수기 입력은 환자·자료·검사·지표 키로 보존합니다. 재산출은 patient_matching.py → supplement_psg_values.py 순서로 실행하며 이전 엑셀을 만들지 않습니다. CSV·원본·리포트·현재 근거 문서·시각화는 유지합니다.\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
