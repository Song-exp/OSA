"""연구실 담당자용 자료 확인 요청 docx (v5).

표: 번호 | 확인 요청 사항 | 대상 환자 | 현재 처리 상태 | 회신 내용. 내부 분류 코드는 쓰지 않는다.
어투: 문제 요약 제목(명사형) → 확인한 사실(합쇼체) → 의문형 요청. 사용자가 직접 고친 문구를 원본으로 삼는다.
요청 문장을 사실 문단에 붙인 행은 ask를 ""로 둔다. 문자열 안의 \n은 셀 안 줄바꿈.

용어 통일 (TERMS 검사로 강제):
  PSG 원본 파일(그 안의 판독 기록) · 리포트 · 엑셀 PSG 값 · 받은 자료 / 환자 폴더 · 식별정보 / DICOM 정보 · 최종 판독 기록 · CBCT 영상
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/patient_matching/CBCT_PSG_자료확인요청_2026-11-12_v6.docx"
FONT = "Malgun Gothic"
HEAD = ["번호", "확인 요청 사항", "대상 환자", "현재 처리 상태", "회신 내용"]
WIDTHS = [1.0, 8.2, 8.0, 5.4, 4.5]  # cm, 합 27.1 = A4 가로 29.7 - 여백 1.3×2
INTRO = ["환자 번호(P_번호)는 받은 자료의 환자 폴더 번호입니다. 날짜는 PSG 검사일입니다."]
# 통일 전 표현. 본문에 남아 있으면 실패
TERMS = ["원신호", "원등록", "공식 리포트", "폴더 리포트", "검사 리포트", "요약값", "요약검사", "엑셀 요약",
         "판독본", "판독 파일", "받은 데이터", "전달받은", "촬영본", "검사 원본"]

# (제목, 확인한 사실, 요청, [(환자, 근거)], 현재 처리 상태)
ROWS = [
    ("CBCT 원본과 엑셀 CBCT 계측에 사용한 영상 부재",
     "받은 자료 전체에서 환자와 매칭되는 CBCT 영상을 찾지 못했습니다. 환자 폴더의 상위 영상은 식별정보가 다르고, 하위 영상은 P_24의 영상입니다.",
     "해당 환자의 CBCT 원본(DICOM)과 엑셀 CBCT 계측에 사용한 영상을 받을 수 있을까요?",
     [("P_39", "")],
     "CBCT 계측에서 제외했습니다. PSG 자료는 CBCT와 별도로 판단할 예정입니다."),
    ("CBCT 식별정보(성별·생년월일) 불일치",
     "등록번호 등 다른 식별정보는 일치하지만, CBCT DICOM 정보의 성별 또는 생년월일이 엑셀과 다릅니다.",
     "CBCT 촬영 기관에 등록된 성별·생년월일을 확인해 주실 수 있을까요?",
     [("P_36", "DICOM 성별 M, 엑셀 F"),
      ("P_85", "DICOM 성별 M, 엑셀·2025-03-12 리포트 F.\n2019-06-07 리포트 첫 장 머리말에만 성별 Male로 기재 (같은 리포트 상세 장은 F)"),
      ("P_59", "DICOM 생년월일이 엑셀·리포트와 다름")],
     "같은 환자의 영상인지 확인되기 전까지 CBCT 계측을 보류하고 있습니다."),
    ("PSG 원본 파일 검사일의 리포트 부재",
     "PSG 원본 파일은 엑셀 PSG 값과 다른 날 시행된 검사인데, 그날의 리포트가 받은 자료에 없습니다.",
     "아래 날짜의 리포트를 받을 수 있을까요?",
     [("P_9", "2019-03-14"), ("P_26", "2022-05-22"), ("P_41", "2024-06-05"), ("P_42", "2021-08-11"),
      ("P_43", "2024-05-29"), ("P_47", "2022-01-17"), ("P_77", "2024-06-21")],
     "CBCT는 식별정보가 일치해 계측을 마쳤습니다. PSG 원본 파일로 계산하는 지표는 리포트와 대조하기 전까지 보류하고 있습니다."),
    ("엑셀 PSG 값의 출처 검사 불명확",
     "받은 자료에는 2025년 검사의 리포트와 PSG 원본 파일만 있는데, 엑셀의 AHI·RDI·ODI 값이 이 리포트와 다릅니다. P_65·P_69는 2025년 리포트 소견에 적힌 이전 검사 결과와 같아, 엑셀은 이전 검사를 기준으로 작성된 것으로 보입니다.",
     "엑셀 값이 나온 이전 검사의 날짜와 그 검사의 리포트를 받을 수 있을까요?",
     [("P_65", "엑셀 16.1/18.7/7.6, 2025-04-09 리포트 35.4/37.2/21.8.\n2025년 검사는 리포트 소견상 교정치료 후 추적 검사"),
      ("P_67", "엑셀 28.1/29.1/9.6·최저 SpO2 89, 2025-09-29 리포트 9.6/17.2/4.9.\n리포트 소견에 이전 검사 결과가 없어, 같은 검사의 다른 판독 기록인지 다른 검사의 값인지 확인이 필요합니다"),
      ("P_69", "엑셀 42.9/43.5/41.3, 2025-09-01 리포트 13.4/16.8/5.9.\n2025년 검사는 리포트 소견상 악교정 수술(2024-07-18) 후 추적 검사")],
     "CBCT는 계측을 마쳤습니다. 엑셀 값과 리포트 값은 서로 대체하지 않고, 출처가 확인될 때까지 해당 PSG 지표를 보류하고 있습니다."),
    ("PSG 원본 파일과 리포트의 사건 수 불일치",
     "PSG 원본 파일의 판독 기록에서 센 무호흡 및 저호흡 개수가 리포트와 다릅니다. 원본 파일 안에 자동 판독과 수동 판독 기록이 함께 있어, 어느 쪽이 리포트에 쓰인 최종 판독 기록인지 확인이 어렵습니다. 리포트를 만들 때 사용한 최종 판독 기록을 알려 주시거나, 판독 프로그램에서 최종 판독 사건 목록(사건 종류·시작 시각·지속시간)을 내보낸 파일을 받을 수 있을까요?",
     "",
     [("P_35", "2021-07-14: 원본 파일 107개, 리포트 103개. N2 시간도 약 5.5분 차이.\nEmbla 장비라 판독 기록이 하나이며, 리포트 작성 후 판독 기록이 수정되었는지 확인이 필요합니다"),
      ("P_50", "2023-11-06: 원본 파일 153개(수동 판독 우선으로 센 값), 리포트 173개. 수면 중 SpO2 장시간 평탄·턱 근전도 0 구간이 있어 해당 채널의 유효성도 확인이 필요합니다"),
      ("P_75", "2024-01-29: 원본 파일의 수동 판독 기록에는 3개뿐이고, 리포트는 148개. 최종 판독 기록이 원본 파일에 빠져 있을 가능성"),
      ("P_76", "2024-08-12: 원본 파일 77개(수동 판독 우선으로 센 값), 리포트 72개")],
     "CBCT는 계측을 마쳤습니다. PSG 원본 파일의 판독 기록을 이용한 분석은 최종 판독 기록이 확인될 때까지 보류하고 있습니다."),
    ("리포트와 엑셀 PSG 값 불일치",
     "같은 검사에서 리포트 값과 엑셀 값이 다르게 기재되어 있습니다. 정정값과 연구에 사용할 지표 정의를 확인해 주실 수 있을까요?",
     "",
     [("P_22", "2018-07-23: 엑셀 ODI 29.8은 리포트의 앙와위 AHI 값과 같고, 리포트 ODI는 6.1"),
      ("P_23", "2018-10-30: 리포트 RDI가 머리말 33.5, 요약표 23.3으로 두 가지")],
     "CBCT는 계측을 마쳤습니다. 해당 PSG 지표는 엑셀 값을 그대로 쓰지 않고 회신을 기다리고 있습니다."),
    ("CBCT 촬영 범위 부족",
     "촬영 범위가 약 10 cm(다른 환자 대부분 약 20 cm)라 계측 기준점 일부(Po·N·Or)가 영상 밖에 있습니다.",
     "같은 환자의 더 넓은 범위 CBCT 영상이나, 엑셀 CBCT 계측에 사용한 영상을 받을 수 있을까요?",
     [("P_78", "")],
     "현재 영상으로는 골격 지표를 계산할 수 없어 CBCT 계측에서 제외했습니다."),
    ("엑셀 PSG 값과 PSG 원본 파일의 검사 시점 불일치",
     "엑셀 PSG 값과 PSG 원본 파일이 서로 다른 날 시행된 검사이며, 두 검사의 간격은 약 3개월에서 4년 4개월입니다. 두 검사 사이에 치료나 체중 변화가 있었다면 결과가 달라질 수 있어, 한 환자의 값으로 함께 쓰기 어렵습니다.",
     "① 둘 중 연구에 사용할 검사와, 두 검사 사이의 수술·양압기·구강장치 사용이나 체중 변화를 알려 주실 수 있을까요?\n"
     "② 치료 이력이 없다면, 두 검사 간격이 어느 정도까지일 때 같은 상태의 검사로 보아도 되는지 연구실에서 사용하는 기준이 있을까요?",
     [("P_8", "엑셀 2016-07-01, 원본 파일 2018-10-08. 2016년 리포트 첫 장 머리말에만 성별 Male로 기재 (같은 리포트 상세 장과 엑셀·2018년 리포트는 F)"),
      ("P_9", "엑셀 2018-11-29, 원본 파일 2019-03-14"),
      ("P_10", "엑셀 2018-08-17, 원본 파일 2019-01-25. 2018년 리포트 RDI가 머리말 25.7, 요약표 19.5로 두 가지 (엑셀 19.5)"),
      ("P_19", "엑셀 2020-09-29, 원본 파일 2021-02-09"), ("P_26", "엑셀 2019-07-28, 원본 파일 2022-05-22"),
      ("P_41", "엑셀 2021-01-19, 원본 파일 2024-06-05"), ("P_42", "엑셀 2019-09-25, 원본 파일 2021-08-11"),
      ("P_43", "엑셀 2020-01-28, 원본 파일 2024-05-29"), ("P_45", "엑셀 2021-11-26, 원본 파일 2022-02-25"),
      ("P_47", "엑셀 2021-06-29, 원본 파일 2022-01-17"), ("P_77", "엑셀 2020-09-21, 원본 파일 2024-06-21")],
     "CBCT는 계측을 마쳤습니다. 두 검사의 값을 섞지 않도록, 연구에 사용할 검사가 정해질 때까지 해당 PSG 지표를 보류하고 있습니다."),
    ("연구 포함 기준 해당 여부 확인 필요",
     "16세 등 미성년자도 연구에 포함시킬까요?",
     "",
     [("P_24", "2019-07-26 검사 당시 16세"), ("P_58", "2019-08-07 검사 당시 16세")],
     "CBCT는 계측을 마쳤고, 포함 여부가 정해질 때까지 분석에는 넣지 않았습니다."),
    ("엑셀 나이 기재 불일치",
     "엑셀의 나이가 리포트에 기재된 검사 당시 나이와 다릅니다. 리포트 기준 나이로 정정하는 것이 맞는지 확인해 주실 수 있을까요?",
     "",
     [("P_31", "엑셀 18세, 2021-05-11 리포트 22세")],
     "CBCT는 계측을 마쳤고, 나이가 확인될 때까지 분석에는 넣지 않았습니다."),
    ("환자 폴더 내 다른 사람의 CBCT 영상 혼입",
     "환자 폴더 안의 하위 폴더에 다른 사람의 CBCT 영상이 들어 있습니다. 섞이게 된 경위를 확인해 주실 수 있을까요?",
     "",
     [("P_34", "하위 폴더 영상 1개. 받은 자료의 87명 누구와도 식별정보가 일치하지 않음"),
      ("P_40", "하위 폴더 영상 2개는 P_59·P_50의 영상. 2019-03-20 리포트의 AHI가 수면시간 대신 총기록시간 기준으로 계산된 것으로 보여, 지표 정의도 확인이 필요합니다")],
     "상위 폴더의 영상은 해당 환자와 식별정보가 일치해, 이 영상만으로 계측하고 분석에 포함했습니다. 하위 폴더 영상은 사용하지 않습니다."),
    ("PSG 원본 파일의 일부 채널 이상",
     "PSG 원본 파일의 일부 채널에 다른 날 기록이 섞여 있거나, 값이 고정된 구간이 있습니다.",
     "원본 파일을 내보낸 범위나 신호 품질 검수 기록을 받을 수 있을까요?",
     [("P_23", "다른 날 기록이 섞인 채널이 있음"), ("P_59", "서미스터 값이 고정된 구간이 있음")],
     "해당 채널로 계산하는 PSG 지표만 보류하고, 나머지 지표는 따로 판단합니다."),
    ("엑셀 성별 숫자코드 오기",
     "엑셀의 문자 성별은 F, 숫자코드는 1(남)로 서로 다릅니다. 리포트와 CBCT 영상은 모두 F입니다.",
     "숫자코드를 2(여)로 정정하는 것이 맞는지 확인해 주실 수 있을까요?",
     [("P_2", ""), ("P_63", ""), ("P_70", "")],
     "리포트와 CBCT 영상을 근거로 숫자코드를 2(여)로 새로 표기해 분석에 포함했습니다."),
    ("PSG 검사와 CBCT 촬영의 간격 기준",
     "엑셀 PSG 검사일과 CBCT 촬영일의 간격이 1년을 넘는 환자가 15명입니다. 골격 계측은 비교적 안정적이지만, 기도·연조직 지표는 그사이 체중이나 치료에 따라 달라질 수 있습니다.",
     "연구에서 PSG 검사와 CBCT 촬영을 같은 시점의 자료로 보는 간격 기준이 있을까요?",
     [("P_80", "5.7년"), ("P_2", "4.6년"), ("P_32", "4.5년"),
      ("P_43", "4.2년 (원본 파일 검사일 기준 55일)"), ("P_77", "3.3년 (원본 파일 검사일 기준 168일)"),
      ("P_41", "3.3년 (원본 파일 검사일 기준 36일)"), ("P_58", "2.9년"), ("P_56", "2.9년"), ("P_40", "2.6년"),
      ("P_42", "2.1년 (원본 파일 검사일 기준 70일)"), ("P_60", "2.0년"), ("P_49", "1.6년"),
      ("P_26", "1.4년 (원본 파일 검사일 기준 501일)"), ("P_73", "1.3년"), ("P_5", "1.3년 (CBCT를 먼저 촬영)")],
     "간격 기준이 정해지지 않아, 우선 환자별 간격만 계산해 정리해 두었습니다. 기준이 정해지면 기도·연조직 지표 분석에 반영할 예정입니다."),
]


def run(par, text, bold=False, size=8.5):
    r = par.add_run(text); r.bold = bold; r.font.size = Pt(size); r.font.name = FONT
    r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    return r


def para(cell, first=False):
    p = cell.paragraphs[0] if first else cell.add_paragraph()
    p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(0)
    return p


def flag(tr, tag):
    el = OxmlElement(tag); el.set(qn("w:val"), "true"); tr._tr.get_or_add_trPr().append(el)


def main():
    text = " ".join(INTRO + [x for r in ROWS for x in (r[0], r[1], r[2], r[4], *(n for _, n in r[3]))])
    left = [t for t in TERMS if t in text]
    assert not left, f"통일 전 표현이 남음: {left}"

    doc = Document(); s = doc.sections[0]
    s.orientation = WD_ORIENT.LANDSCAPE; s.page_width, s.page_height = Cm(29.7), Cm(21)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(s, side, Cm(1.3))
    st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(8.5)
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)

    patients = {p for row in ROWS for p, _ in row[3]}
    run(doc.add_paragraph(), "CBCT·PSG 자료 확인 요청", True, 15)
    run(doc.add_paragraph(), f"2026-11-12 · 확인 요청 {len(ROWS)}건 · 대상 환자 {len(patients)}명", size=9)
    for line in INTRO: run(doc.add_paragraph(style="List Bullet"), line)

    t = doc.add_table(rows=1, cols=len(HEAD)); t.style = "Table Grid"; t.autofit = False
    for c, h in zip(t.rows[0].cells, HEAD):
        p = para(c, True); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; run(p, h, True, 9)
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E7ECF2")
        c._tc.get_or_add_tcPr().append(shd)
    flag(t.rows[0], "w:tblHeader")
    for i, (title, fact, ask, targets, now) in enumerate(ROWS, 1):
        tr = t.add_row(); flag(tr, "w:cantSplit"); no, req, who, cur, _ = tr.cells
        p = para(no, True); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; run(p, str(i), True)
        run(para(req, True), title, True); run(para(req), fact)
        for line in filter(None, ask.split("\n")): run(para(req), line)
        for j, (pid, note) in enumerate(targets):
            first, *rest = note.split("\n")
            p = para(who, j == 0); run(p, pid, True)
            if first: run(p, "  " + first)
            for line in rest: run(para(who), line)
        run(para(cur, True), now)
    for row in t.rows:
        for c, w in zip(row.cells, WIDTHS): c.width = Cm(w)
    assert len(ROWS) == 14 and len(patients) == 40, (len(ROWS), len(patients))
    assert "Class" not in text, "Class 분류 표현은 쓰지 않는다"
    assert all((ask or fact).endswith("?") for _, fact, ask, _, _ in ROWS)
    tail = doc.add_paragraph(); tail.paragraph_format.space_after = Pt(0); tail.paragraph_format.line_spacing = Pt(1)
    run(tail, " ", size=1)  # 표 뒤 필수 문단이 빈 쪽을 만들지 않게

    compat = doc.settings.element.find(qn("w:compat"))  # 기본 템플릿의 '호환성 모드' 해제
    if compat is None: compat = OxmlElement("w:compat"); doc.settings.element.append(compat)
    cs = OxmlElement("w:compatSetting")
    for k, v in (("name", "compatibilityMode"), ("uri", "http://schemas.microsoft.com/office/word"), ("val", "15")): cs.set(qn("w:" + k), v)
    compat.append(cs)
    doc.save(OUT)
    print(f"saved {OUT.name}: requests {len(ROWS)}, patients {len(patients)}")


if __name__ == "__main__":
    main()
