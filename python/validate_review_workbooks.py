"""검수 파일 구조·원값 보존·동일 검사 비교·수기 의견 복원을 검증한다."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from lxml import html
from urllib.parse import urlparse
from urllib.request import url2pathname
from build_review_workbooks import PATIENT_BOOK, VALUE_BOOK, OUT, ROOT, restore
from patient_matching import sha


def main():
    frames=pd.read_excel(PATIENT_BOOK,sheet_name=None)
    vf=pd.read_excel(VALUE_BOOK,sheet_name=None)
    assert list(frames)==["통합·최종판단","CBCT","PSG"]
    assert list(vf)==["통합·최종판단","PSG"]
    assert len(frames["통합·최종판단"])==len(vf["통합·최종판단"])==87
    assert len(frames["CBCT"])==91 and len(frames["PSG"])==116
    detail=vf["PSG"]
    source=pd.read_csv(OUT/"psg_values_with_raw.csv").set_index(["환자","변수"])
    assert len(detail)==1111 and not detail.duplicated(["환자","검사키","지표"]).any()
    for col,key in [("엑셀값","엑셀원값"),("원본산출값","원본판독·파형산출값")]:
        available=detail[detail[col].notna()]
        assert len(available)==957 and not available.duplicated(["환자","지표"]).any()
        for _,r in available.iterrows():
            assert abs(r[col]-source.loc[(r["환자"],r["지표"]),key])<1e-9
    assert detail["엑셀-원본"].count()==803
    for _,r in detail.iterrows():
        for diff,a,b in [("엑셀-보고서","엑셀값","보고서값"),("원본-보고서","원본산출값","보고서값"),("엑셀-원본","엑셀값","원본산출값")]:
            if pd.notna(r[a]) and pd.notna(r[b]):
                assert abs(r[diff]-(r[a]-r[b]))<1e-9
            else:
                assert pd.isna(r[diff])
        if r["검사구분"]!="동일 검사":
            assert pd.isna(r["엑셀-원본"])
        if r["검사구분"]=="엑셀 검사 미확인":
            assert pd.isna(r["보고서값"]) and pd.isna(r["원본산출값"])
    psg=frames["PSG"]
    current=psg[psg["환자"].eq("P_85") & psg["자료구분"].str.contains("엑셀 요약값 연결")]
    past=psg[psg["환자"].eq("P_85") & psg["자료구분"].eq("기타 검사 보고서")]
    assert current.iloc[0]["성별대조"]=="일치" and past["성별대조"].eq("불일치").any()
    cbct=frames["CBCT"]
    p59=cbct[cbct["환자"].eq("P_59") & cbct["자료범위"].str.startswith("상위")].iloc[0]
    assert p59["생년월일대조"]=="불일치"
    assert "입력 오류 가능성" in p59["검토해석"]
    # 같은 환자의 다른 검사에 의견이 옮겨 붙지 않는지 검증.
    test=OUT/".review_restore_check.xlsx"
    assert test.resolve().parent==OUT.resolve()
    old=pd.DataFrame({"환자":["P_8","P_8"],"검사키":["A","B"],"지표":["AHI","AHI"],"검수의견":["exam-A","exam-B"]})
    try:
        with pd.ExcelWriter(test) as writer: old.to_excel(writer,sheet_name="PSG",index=False)
        new=old.iloc[::-1].copy();new["검수의견"]=""
        restore({"PSG":new},test)
        assert dict(zip(new["검사키"],new["검수의견"]))=={"A":"exam-A","B":"exam-B"}
    finally:
        if test.exists():test.unlink()
    links=0
    for file in (PATIENT_BOOK,VALUE_BOOK):
        book=load_workbook(file)
        for ws in book:
            assert ws.freeze_panes=="B2" and ws.auto_filter.ref
            for cells in ws.iter_rows():
                for cell in cells:
                    if cell.hyperlink:
                        u=urlparse(cell.hyperlink.target)
                        if u.scheme=="file":
                            assert Path(url2pathname(u.path)).exists();links+=1
        book.close()
    tree=html.fromstring((OUT/"index.html").read_text(encoding="utf-8"))
    for href in tree.xpath('//a/@href'):
        u=urlparse(href)
        if u.scheme=="file": assert Path(url2pathname(u.path)).exists()
    fig=json.loads((OUT/"figures/psg_error_review_summary.json").read_text(encoding="utf-8"))
    assert sha(OUT/"psg_values_with_raw.csv")==fig["원본SHA256"]
    result={"status":"passed","patients":87,"identity_sheets":3,"value_sheets":2,
      "identity_CBCT_records":91,"identity_PSG_records":116,"value_records":1111,"preserved_excel_values":957,
      "preserved_raw_values":957,"same_exam_three_source_comparisons":803,"cross_exam_comparisons":0,
      "working_workbook_links":links,"review_restore_by_exam_key":"passed","visualization_source_unchanged":True,
      "generator_sha256":sha(ROOT/"python/build_review_workbooks.py"),"validation_sha256":sha(Path(__file__)),
      "workbooks":{f.name:sha(f) for f in (PATIENT_BOOK,VALUE_BOOK)}}
    (OUT/"review_workbooks_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
