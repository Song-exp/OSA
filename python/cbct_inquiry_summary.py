"""문의·보류 33명 최종 정리 HTML. 계측값은 selected_measures_diagnostic.csv에서 읽고, 명단은 cohort.csv에서 확인한다.

출력: Result_py/patient_matching/문의사항_최종정리_2026-11-12.html (로컬 전용, 생년월일·등록번호 미포함)
"""
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "Result_py/CBCT/patient_pipeline/v2"
OUT = ROOT / "Result_py/patient_matching/문의사항_최종정리_2026-11-12.html"

# 유형: (코드, 이름, 정의, 환자, 처리 상황)
TYPES = [
    ("A1", "엑셀·원신호 다른 검사 + 원신호 리포트 없음", "엑셀 요약은 이전 검사, 원신호는 수개월~4년 뒤 검사. 원신호 날짜의 공식 리포트가 없어 대조 불가",
     ["P_9", "P_26", "P_41", "P_42", "P_43", "P_47", "P_77"], "measured"),
    ("A2", "엑셀·원신호 다른 검사 (리포트 둘 다 있음)", "어느 검사를 연구에 쓸지 선택만 필요",
     ["P_8", "P_10", "P_19", "P_45"], "measured"),
    ("A3", "엑셀 요약값의 출처 검사 미확인", "엑셀 AHI가 폴더 리포트와 크게 다름. 리포트 소견의 이전 검사 값과 같음",
     ["P_65", "P_67", "P_69"], "measured"),
    ("A4", "원본 판독 ↔ 리포트 집계 차이", "같은 검사인데 무호흡+저호흡 개수가 리포트와 다름. 리포트를 만든 최종 판독본 확인 필요",
     ["P_35", "P_50", "P_75", "P_76"], "measured"),
    ("A5", "CBCT 식별정보 불일치", "DICOM 성별·생년월일이 엑셀과 다르거나(P_36·P_85·P_59) 본인 영상이 없음(P_39)",
     ["P_36", "P_85", "P_59", "P_39"], "hold"),
    ("B1", "엑셀 성별 숫자코드 입력 오류", "문자 성별 F, 숫자코드 1(M). 리포트·CBCT는 F로 일치 → 코드 2로 자체 정정",
     ["P_2", "P_63", "P_70"], "included"),
    ("B2", "선정 기준 해당 여부", "PSG 당시 16세(P_24·P_58), 나이 18↔22세·AHI 3.3·Class III(P_31)",
     ["P_24", "P_58", "P_31"], "measured"),
    ("B3", "PSG 값 리포트 ↔ 엑셀 차이 (정정 검토)", "같은 검사의 리포트 값과 엑셀 값이 다름. 연구실의 실제 검토·정정 회신 필요",
     ["P_22", "P_23"], "measured"),
    ("B4", "CBCT 하위 폴더에 다른 영상 혼입", "상위 폴더는 본인으로 확인. 하위 폴더는 무시(top_only)하고 계측",
     ["P_34", "P_40"], "included"),
    ("C", "CBCT 촬영 범위 부족", "FOV 약 102 mm로 Po·N·Or이 영상 밖 → 골격 4지표 계산 불가",
     ["P_78"], "impossible"),
]
STATUS = {
    "included": ("분석 명단 포함", "ok", "cohort.csv Include=True. 문의는 유지"),
    "measured": ("CBCT 계측 완료 · 명단 미포함", "info", "CBCT 식별정보 일치. 회신 후 Include만 바꾸면 반영"),
    "hold": ("계측 보류", "warn", "CBCT가 본인 영상인지 확정 전"),
    "impossible": ("계측 불가", "bad", "영상 범위 부족"),
}
HOLD_NOTE = {"P_36": "성별 불일치 · 변환·추론 결과 없음", "P_59": "생년월일 불일치 · 변환·추론 결과 없음",
             "P_85": "성별 불일치 · 영상 결과는 있음", "P_39": "본인 CBCT 없음 (하위 영상은 P_24 촬영본) · 계측 불가"}

# 문의: (우선순위, 제목, 대상, 질문)
INQUIRIES = [
    ("높음", "본인 CBCT 원본과 엑셀 CBCT 계측의 출처", "P_39",
     "받은 자료 전체(시리즈 91개)에서 본인 CBCT를 찾지 못했습니다. 상위 영상은 식별정보가 다르고 하위 영상은 P_24 촬영본입니다. 본인 DICOM과 엑셀 계측에 쓴 영상을 알려 주세요."),
    ("높음", "CBCT 원등록 성별·생년월일", "P_36 · P_85 (성별), P_59 (생년월일)",
     "생년월일 또는 등록번호는 일치하지만 DICOM의 성별/생년월일이 엑셀과 다릅니다. 기관 원등록 정보를 확인해 주세요. P_85는 2019년 과거 PSG 리포트에만 성별 M이 있습니다."),
    ("높음", "원신호 검사일의 공식 PSG 리포트", "P_9 · P_26 · P_41 · P_42 · P_43 · P_47 · P_77",
     "폴더에는 엑셀 요약검사의 리포트만 있어 원신호 결과와 공식 판독을 대조할 수 없습니다."),
    ("높음", "엑셀 요약값의 원 검사·리포트", "P_65 · P_67 · P_69",
     "엑셀 AHI(16.1 / 28.1 / 42.9)가 폴더 리포트(35.4 / 9.6 / 13.4)와 다릅니다. 엑셀 값이 나온 검사의 날짜와 공식 리포트를 제공해 주세요."),
    ("높음", "최종 판독층·revision과 원본 내보내기", "P_35 · P_50 · P_75 · P_76",
     "같은 검사에서 원본 무호흡+저호흡 집계와 리포트가 다릅니다(+3.9% / −11.6% / 3 vs 148 / +6.9%). 리포트를 만든 판독본을 확인해 주세요. P_50은 SpO2 평탄·턱 근전도 0 구간의 채널 유효성도 확인이 필요합니다."),
    ("높음", "리포트 ↔ 엑셀 값 정정 검토", "P_22 · P_23",
     "P_22: 엑셀 ODI 29.8은 리포트의 앙와위 AHI 값이고 리포트 ODI는 6.1입니다. P_23: 리포트 RDI가 머리말 33.5, 요약표 23.3 두 개이며 엑셀은 23.3입니다. 정정값과 연구에 쓸 RDI 정의를 회신해 주세요."),
    ("높음", "더 넓은 범위의 CBCT", "P_78",
     "현재 FOV 약 102 mm로 골격 4지표를 계산할 수 없습니다. 같은 환자의 다른 촬영본이 있는지, 엑셀 계측에 쓴 영상과 범위를 알려 주세요 (M10)."),
    ("중간", "복수 PSG 중 연구 검사 선택·치료 이력", "A1 7명 + A2 4명",
     "엑셀 요약검사와 원신호 검사 중 어느 쪽을 연구에 쓸지, 두 검사 사이 수술·양압기·구강장치·체중 변화 여부를 알려 주세요. P_8은 2016년 리포트 성별 M 표기, P_10은 RDI 25.7/19.5 정의도 함께 확인이 필요합니다."),
    ("중간", "성인 포함 기준·선정 기준", "P_24 · P_58 · P_31",
     "P_24·P_58은 PSG 당시 16세입니다. P_31은 엑셀 나이 18세, 검사일 기준 22세이며 AHI 3.3과 Class III가 포함 기준에 맞는지 결정이 필요합니다."),
    ("중간", "하위 폴더 혼입 경위와 리포트 지표 정의", "P_34 · P_40",
     "P_34 하위 영상은 87명 누구와도 식별정보가 맞지 않습니다(코호트 밖 인물). P_40 하위 영상 2개는 P_59·P_50 촬영본입니다. P_40은 리포트 AHI가 TST 대신 총기록시간 분모로 재현되어 지표 정의 확인도 필요합니다."),
    ("중간", "원신호 채널 범위", "P_23 · P_59",
     "P_23은 다른 날 채널이 섞여 기존 리더가 일부 채널을 뺐습니다. P_59는 서미스터 끝값 고정 구간이 있습니다. 원본 내보내기 범위와 최종 원신호 검수 기록을 받을 수 있나요?"),
    ("낮음", "성별 숫자코드 정정본", "P_2 · P_63 · P_70",
     "자체 정정(코드 2)으로 이미 분석에 포함했습니다. 정정된 데이터셋으로 확인만 부탁드립니다."),
]

# 대표 케이스: (유형, 환자, 제목, 사실들, 판단)
CASES = [
    ("A1", "P_9", "검사가 다르고 원신호 리포트도 없음",
     ["엑셀 요약검사 2018년 11월, 원신호 검사 2019년 3월 (약 4개월 차이)", "엑셀 요약값은 엑셀 대응 리포트와 일치",
      "원신호 날짜의 리포트가 없어 원본 집계와 공식 판독 대조 불가"],
     "CBCT는 본인으로 확인되어 계측 완료. PSG는 어느 검사를 쓸지와 원신호 리포트가 있어야 결정"),
    ("A2", "P_8", "검사 선택만 남은 경우",
     ["엑셀 2016년, 원신호 2018년 검사. 리포트는 둘 다 있음", "원신호 리포트와 원본 집계 289 = 289 (차이 0%)",
      "2016년 리포트에만 성별 M 표기 (엑셀·2018년 리포트·CBCT는 F)"],
     "자료는 다 있음. 연구 검사 선택과 치료 이력 회신만 필요"),
    ("A3", "P_67", "엑셀 값이 어디서 왔는지 모름",
     ["엑셀 AHI 28.1 · RDI 29.1 · ODI 9.6", "폴더의 2025년 리포트 AHI 9.6 · RDI 17.2 · ODI 4.9",
      "원신호와 2025년 리포트는 61 = 61로 일치"],
     "엑셀 값과 폴더 리포트 값은 서로 대체하지 않음. 출처 검사 회신 전까지 PSG 요약값 보류"),
    ("A4", "P_75", "판독 집계가 거의 0",
     ["같은 검사, Nox 원본에서 Manual 우선 규칙으로 고른 무호흡+저호흡 3개", "리포트는 148개 (−98%)",
      "수치 차이가 아니라 이벤트 집계의 출처 자체가 불확실"],
     "최종 판독층·revision과 이벤트 내보내기 요청. CBCT는 계측 완료"),
    ("A5", "P_39", "본인 CBCT가 없음",
     ["시리즈 91개(상위 87, 하위 4) 헤더 전수 대조에서 일치하는 시리즈 없음", "상위 영상은 식별정보 다름, 하위 영상은 P_24의 2022년 촬영본",
      "엑셀에는 CBCT 계측값 11개가 모두 있음"],
     "CBCT 분석 불가. PSG 단독 분석에서 제외할지는 따로 결정"),
    ("B1", "P_2", "입력 오류 → 자체 정정 후 포함",
     ["엑셀 문자 성별 F, 숫자코드 1(M)", "리포트와 CBCT 모두 F", "psg_verified.csv V_Sex=F → 분석용 코드 2"],
     "분석 명단 포함·계측 완료. Nperp-Pog 차이 +11.2, MP-FH −6.4로 FH 기준점 확인 필요"),
    ("B2", "P_31", "자료 오류가 아니라 선정 기준 문제",
     ["엑셀 나이 18세, 연결 PSG 검사일 기준 22세", "AHI 3.3, Class III", "엑셀·리포트·원본 집계는 모두 일치"],
     "CBCT 계측 완료. 포함 여부는 연구실 결정"),
    ("B3", "P_22", "엑셀 칸 전사 오류 의심",
     ["엑셀 ODI 29.8 = 리포트의 앙와위 AHI", "리포트 ODI 6.1", "원본 집계 117 = 리포트 117 (판독은 일치)"],
     "정정값 회신 필요. 상기도 분할에 상악동이 포함됐다는 의심은 영상 QC로 따로 확인"),
    ("B4", "P_34", "하위 폴더의 정체 불명 영상",
     ["상위 폴더는 엑셀과 식별정보 일치", "하위 영상(2022년)은 87명의 상위 폴더 ID·생년월일+성별, 엑셀 생년월일 어디와도 맞지 않음",
      "계측은 상위 폴더만 사용 (원본 확인값·NRRD 모두 상위 기준)"],
     "분석 명단 포함. 혼입 경위만 문의"),
    ("C", "P_78", "촬영 범위 부족",
     ["FOV 약 102 mm 정육면체 (85명은 200×200×179 mm)", "Po·N·Or이 영상 밖"],
     "골격 4지표 계산 불가. 더 넓은 촬영본이 있는지 문의"),
]

METRICS = [("ANB", "ANB"), ("Nperp_A", "Nperp-A"), ("Nperp_Pog", "Nperp-Pog"), ("MPFH_GoMe", "MP-FH")]
QC_FLAG = {"P_10": "Pog–Me 거리 이상", "P_24": "Pog–Me 거리 이상", "P_35": "Po·Or 거리 이상", "P_23": "Po·Or 거리 이상"}

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--muted:#5f5f5a;--line:#deded8;--accent:#2f5d8a;
--ok:#2e7d4f;--ok-bg:#e6f3eb;--info:#2f5d8a;--info-bg:#e7eef6;--warn:#8a5a00;--warn-bg:#fbf0d9;--bad:#a33a2c;--bad-bg:#f8e4e0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--card:#1f1f1d;--fg:#ececea;--muted:#a3a39d;--line:#363633;--accent:#8db4dc;
--ok:#7cc79a;--ok-bg:#1c3326;--info:#8db4dc;--info-bg:#1b2836;--warn:#e3b65c;--warn-bg:#3a2e14;--bad:#ec8f80;--bad-bg:#3d1f1a}}
:root[data-theme="dark"]{--bg:#161615;--card:#1f1f1d;--fg:#ececea;--muted:#a3a39d;--line:#363633;--accent:#8db4dc;
--ok:#7cc79a;--ok-bg:#1c3326;--info:#8db4dc;--info-bg:#1b2836;--warn:#e3b65c;--warn-bg:#3a2e14;--bad:#ec8f80;--bad-bg:#3d1f1a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 "Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.25rem;margin:40px 0 12px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:1.05rem;margin:0 0 6px}.sub,.note{color:var(--muted)}.note{font-size:.88rem}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:14px 0 4px}a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}.kpi b{display:block;font-size:1.7rem;line-height:1.2}.kpi span{font-size:.85rem;color:var(--muted)}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--bg);font-weight:700;white-space:nowrap}tr:last-child td{border-bottom:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.8rem;font-weight:700;white-space:nowrap}
.ok{color:var(--ok);background:var(--ok-bg)}.info{color:var(--info);background:var(--info-bg)}.warn{color:var(--warn);background:var(--warn-bg)}.bad{color:var(--bad);background:var(--bad-bg)}
.big{font-weight:700;color:var(--bad)}
.cases{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.case{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}.case ul{margin:6px 0 8px;padding-left:20px}.case p{margin:0}
.case .who{font-size:.82rem;color:var(--muted)}
ol.inq>li{margin:0 0 12px}
"""


def tag(kind, text):
    return f'<span class="tag {kind}">{escape(text)}</span>'


def pid_link(p):
    return f'<a href="patients/{p}/evidence.html" target="_blank" rel="noopener">{p}</a>'


def main():
    cohort = pd.read_csv(V2 / "cohort.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    included = set(cohort.loc[cohort.Include.str.lower().isin(["true", "1"]), "Patient"])
    diag = pd.read_csv(V2 / "selected_measures_diagnostic.csv", encoding="utf-8-sig")
    run = pd.read_csv(V2 / "last_run.csv", encoding="utf-8-sig").drop_duplicates("Patient", keep="last").set_index("Patient")

    members = {p: (code, status) for code, _, _, ps, status in TYPES for p in ps}
    assert len(members) == 33 and len(cohort) == 87
    for p, (_, status) in members.items():  # 명단·계측 상태가 서술과 맞는지 확인
        assert (p in included) == (status == "included"), p
        has = (V2 / p / "result.json").is_file()
        assert has == (status in ("included", "measured")) or p == "P_78", p
    assert len(included) == 59

    n = {s: sum(1 for _, st in members.values() if st == s) for s in STATUS}
    measured = [p for p, (_, st) in members.items() if st in ("included", "measured")]

    kpis = [("87", "등록"), (str(len(included)), "분석 명단 (기존 54 + B1·B4 5명)"), (str(n["measured"]), "CBCT 계측 완료 · 명단 미포함"),
            (str(n["hold"] + n["impossible"]), "계측 보류·불가"), (str(len(INQUIRIES)), "문의 항목")]
    kpi_html = '<div class="kpis">' + "".join(f'<div class="kpi"><b>{v}</b><span>{escape(t)}</span></div>' for v, t in kpis) + "</div>"

    type_rows = []
    for code, name, desc, ps, status in TYPES:
        label, kind, _ = STATUS[status]
        type_rows.append(f'<tr><td><b>{code}</b></td><td>{escape(name)}</td><td>{escape(desc)}</td>'
                         f'<td>{", ".join(pid_link(p) for p in ps)}</td><td class="num">{len(ps)}</td><td>{tag(kind, label)}</td></tr>')
    types_html = ('<div class="tw"><table><tr><th>유형</th><th>문제</th><th>내용</th><th>환자</th><th>수</th><th>현재 상태</th></tr>'
                  + "".join(type_rows) + "</table></div>")

    status_rows = []
    for key, (label, kind, meaning) in STATUS.items():
        ps = [p for p, (_, st) in members.items() if st == key]
        detail = ", ".join(f"{pid_link(p)} <span class='note'>({escape(HOLD_NOTE[p])})</span>" if p in HOLD_NOTE else pid_link(p) for p in ps)
        status_rows.append(f'<tr><td>{tag(kind, label)}</td><td class="num">{len(ps)}</td><td>{escape(meaning)}</td><td>{detail}</td></tr>')
    status_html = ('<div class="tw"><table><tr><th>상태</th><th>수</th><th>의미</th><th>환자</th></tr>' + "".join(status_rows) + "</table></div>")

    qc_review = [p for p in measured if run.loc[p, "QC"] != "qc_pass_pending_visual_review"]
    status_html += (f'<p class="note">계측한 {len(measured)}명 중 {len(measured) - len(qc_review)}명은 거리 QC 통과·육안 확인 전, '
                    f'{len(qc_review)}명({", ".join(f"{p} {QC_FLAG.get(p, "")}" for p in qc_review)})은 랜드마크 거리 QC 검토 필요입니다. '
                    'FH 보정(fh_repair)·11지표·비강 점검·보고서 탭은 아직 54명 고정 버전이며 새 인원은 반영 전입니다.</p>')

    inq = {}
    for prio, title, who, text in INQUIRIES:
        inq.setdefault(prio, []).append(f'<li><b>{escape(title)}</b> — <span class="who">{escape(who)}</span><br>{escape(text)}</li>')
    kinds = {"높음": "bad", "중간": "warn", "낮음": "ok"}
    inq_html = "".join(f'<h3>{tag(kinds[p], "우선순위 " + p)}</h3><ol class="inq">{"".join(items)}</ol>' for p, items in inq.items())

    case_html = '<div class="cases">' + "".join(
        f'<article class="case"><div class="who">{code} · {escape(next(t[1] for t in TYPES if t[0] == code))}</div>'
        f'<h3>{pid_link(p)} — {escape(title)}</h3><ul>{"".join("<li>" + escape(f) + "</li>" for f in facts)}</ul>'
        f'<p><b>판단:</b> {escape(verdict)}</p></article>'
        for code, p, title, facts, verdict in CASES) + "</div>"

    # 새로 계측한 28명의 골격 4지표
    d = diag[diag.Patient.isin(measured)].drop_duplicates(["Patient", "Variable"], keep="last").set_index(["Patient", "Variable"])
    rows = []
    for p in sorted(measured, key=lambda x: int(x[2:])):
        code, status = members[p]
        cells = []
        for var, _ in METRICS:
            r = d.loc[(p, var)]
            big = abs(r.difference) >= 5
            cells.append(f'<td class="num">{r.excel:g} → {r.auto:.1f} <span class="{"big" if big else "note"}">({r.difference:+.1f})</span></td>')
        qc = run.loc[p, "QC"]
        qc_tag = tag("info", "통과·육안 전") if qc == "qc_pass_pending_visual_review" else tag("warn", "검토 필요: " + QC_FLAG.get(p, qc))
        rows.append(f'<tr><td>{pid_link(p)}</td><td>{code}</td><td>{tag(STATUS[status][1], STATUS[status][0])}</td>{"".join(cells)}<td>{qc_tag}</td></tr>')
    measures_html = ('<div class="tw"><table><tr><th>환자</th><th>유형</th><th>명단</th>' + "".join(f"<th>{m}</th>" for _, m in METRICS)
                     + "<th>거리 QC</th></tr>" + "".join(rows) + "</table></div>"
                     + '<p class="note">엑셀 → 자동 (차이). 단위는 ANB·MP-FH 도, Nperp mm. 차이 절댓값 5 이상은 빨간색으로 표시했으며 판정 기준이 아니라 검토 우선순위 표시입니다. FH 보정 전 automatic 값입니다.</p>')

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CBCT 문의사항 최종 정리</title><style>{CSS}</style></head><body><main>
<h1>CBCT 지표 복원 · 문의사항 최종 정리</h1>
<p class="sub">2026-11-12 · 등록 87명 중 기존 보류 33명의 문제 유형, 처리 상황, 문의, 대표 사례. 탐색적 자동 계측, 전문가 확인 전.</p>
<nav><a href="#types">1. 문제 유형</a><a href="#status">2. 현재 처리 상황</a><a href="#inquiries">3. 문의 사항</a><a href="#cases">4. 환자별 상세 사례</a><a href="#measures">5. 새로 계측한 28명 수치</a></nav>
{kpi_html}
<p class="note">33명 중 보류 사유 대부분은 PSG·엑셀 연결 문제이며 CBCT 영상 자체 문제는 아닙니다. CBCT 식별정보가 엑셀과 일치하는 환자는 회신과 관계없이 CBCT 계측을 먼저 진행했습니다. 문의는 아직 회신 전이며, 분석 명단에 넣은 5명도 문의 대상에서 빼지 않았습니다.</p>
<h2 id="types">1. 문제 유형</h2>
<p class="note">A는 PSG·엑셀 자료 연결 또는 CBCT 식별 문제로 회신 없이는 통합 분석에 쓸 수 없는 경우, B는 정정·기준 결정 문제, C는 영상 범위 문제입니다.</p>
{types_html}
<h2 id="status">2. 현재 처리 상황</h2>
{status_html}
<h2 id="inquiries">3. 문의 사항</h2>
<p class="note">기존 문의 문서(환자별_데이터현황_및_문의사항.md)의 질문을 같은 성격끼리 묶었습니다. 발송·회신 여부는 새로 확인하지 않았으며 기존 문서상 모두 미회신입니다.</p>
{inq_html}
<h2 id="cases">4. 환자별 상세 사례 (유형별 대표)</h2>
<p class="note">환자 번호를 누르면 기존 자료 대조 근거 페이지가 열립니다.</p>
{case_html}
<h2 id="measures">5. 새로 계측한 28명의 골격 4지표</h2>
{measures_html}
</main></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"saved {OUT.relative_to(ROOT)}: types {len(TYPES)}, inquiries {len(INQUIRIES)}, cases {len(CASES)}, measured {len(measured)}")


if __name__ == "__main__":
    main()
