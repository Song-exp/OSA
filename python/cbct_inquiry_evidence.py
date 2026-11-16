"""자료 확인 요청서(cbct_inquiry_docx.ROWS)의 항목별 증거 페이지 한 장.

질문지 제목을 누르면 그 항목의 증거 구역으로, 환자 번호를 누르면 그 환자의 증거 카드로 이동한다.
증거: 리포트 PDF의 해당 쪽(#page=N), 환자별 대조 근거 페이지, 원신호 그림(inquiry_evidence_figs.py), CBCT 그림, 대조 표.
생년월일·등록번호·이름은 싣지 않는다(일치 여부만). 원본·기존 결과는 읽기만 한다.
출력: Result_py/patient_matching/자료확인요청_증거.html (로컬 전용)
"""
from html import escape
import os
from pathlib import Path
import re
from urllib.parse import quote

import pandas as pd
import pydicom
import pypdf

from cbct_inquiry_docx import ROWS

ROOT = Path(__file__).resolve().parents[1]
PM = ROOT/"Result_py/patient_matching"
OUT = PM/"자료확인요청_증거.html"
FIG = PM/"inquiry_evidence/figs"
V2 = ROOT/"Result_py/CBCT/patient_pipeline/v2"

V = pd.read_csv(ROOT/"Result_py/psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
M = pd.read_csv(PM/"patient_matching_with_raw.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("환자")
R = pd.read_csv(ROOT/"Result_py/psg_reports.csv", encoding="utf-8-sig")
S = pd.read_csv(PM/"cbct_series.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
_pages = {}


def href(path):
    return quote(os.path.relpath(path, OUT.parent).replace("\\", "/"), safe="/._-#=")


def a(path, label, page=None):
    assert Path(path).exists(), path
    return f'<a href="{href(path)}{f"#page={page}" if page else ""}" target="_blank" rel="noopener">{escape(label)}</a>'


def report(p, study):
    r = R[(R.Patient == p) & (R.Study == study)]; assert len(r) == 1, (p, study)
    return next(ROOT.glob(f"Data/PSG/{p}_*/{r.iloc[0].File}"))


def pages(pdf, pattern):
    """정규식이 나오는 쪽 번호(1부터)."""
    if pdf not in _pages: _pages[pdf] = [pg.extract_text() or "" for pg in pypdf.PdfReader(pdf).pages]
    hits = [i+1 for i, t in enumerate(_pages[pdf]) if re.search(pattern, t)]
    assert hits, (pdf.name, pattern); return hits


def rp(p, study, pattern, label):
    """리포트의 해당 쪽 링크."""
    f = report(p, study); n = pages(f, pattern)[0]
    return a(f, f"{study} 리포트 {n}쪽 — {label}", n)


def ev(p):
    return a(PM/f"patients/{p}/evidence.html", f"{p} 환자별 대조 근거 페이지")


def fig(name, caption):
    f = FIG/f"{name}.png"; assert f.exists(), f
    return f'<figure><a href="{href(f)}" target="_blank"><img src="{href(f)}" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'


def img(path, caption):
    assert Path(path).exists(), path
    return f'<figure><a href="{href(path)}" target="_blank"><img src="{href(path)}" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'


def table(head, rows):
    return ("<table><tr>"+"".join(f"<th>{escape(h)}</th>" for h in head)+"</tr>"
            + "".join("<tr>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>" for r in rows)+"</table>")


def series(p):
    s = S[S.환자 == p]
    return table(["CBCT 폴더 위치", "촬영일", "엑셀과 생년월일", "엑셀과 성별", "PSG 등록번호 보조 대조"],
                 [[escape(r.범위), r.촬영일, r.엑셀_생년월일대조, r.엑셀_성별대조, r.CBCT_PSG등록번호_보조대조] for _, r in s.iterrows()])


def dicom_head(folder):
    for f in sorted(p for p in folder.iterdir() if p.is_file()):
        try: h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception: continue
        if "PatientID" in h: return str(h.PatientID), str(h.get("PatientBirthDate", "")), str(h.get("PatientSex", ""))


def sub_match():
    """하위 폴더 CBCT의 식별정보를 87명 상위 폴더와 대조 (값은 싣지 않고 일치한 환자 번호만)."""
    top = {d.name.replace("_CBCT", ""): dicom_head(d) for d in (ROOT/"Data/CBCT").iterdir() if d.is_dir()}
    out = {}
    for p in ("P_34", "P_39", "P_40"):
        for sub in sorted(x for x in (ROOT/f"Data/CBCT/{p}_CBCT").iterdir() if x.is_dir()):
            i, b, s = dicom_head(sub)
            out.setdefault(p, []).append(([q for q, v in top.items() if v[0] == i], [q for q, v in top.items() if v[1:] == (b, s)]))
    return out


def evidence(sm):
    """{(항목 번호, 환자): [근거 HTML 조각]} — 질문지의 각 주장에 대응하는 증거."""
    E = {}; add = lambda i, p, *x: E.setdefault((i, p), []).extend(x)
    # 1. 본인 CBCT 부재
    ids, dob = sm["P_39"][0]
    add(1, "P_39", series("P_39"),
        f"<p>하위 폴더 영상의 식별정보: 87명 상위 폴더 중 ID 일치 {', '.join(ids) or '없음'}, 생년월일·성별 일치 {', '.join(dob) or '없음'} → P_24의 영상.</p>",
        "<p>받은 CBCT 91개 시리즈(상위 87·하위 4) 전수 대조에서 P_39와 일치하는 시리즈 없음: "+a(ROOT/"docs/원본데이터_결측_점검.md", "원본데이터 결측 점검 3-3")+"</p>", ev("P_39"))
    # 2. CBCT 식별정보 불일치
    for p in ("P_36", "P_59"): add(2, p, series(p), ev(p))
    add(2, "P_85", series("P_85"),
        "<p>"+rp("P_85", "2019-06-07", r"Gender:\s*Male", "첫 장 머리말 Gender: Male")+" · "
        + rp("P_85", "2019-06-07", r"Gender\s*:\s*F\b", "같은 리포트 상세 장 Gender: F")+" · "
        + rp("P_85", "2025-03-12", r"Gender", "2025-03-12 리포트 성별 F")+"</p>", ev("P_85"))
    # 3. 원신호 검사일 리포트 부재
    for p in ("P_9", "P_26", "P_41", "P_42", "P_43", "P_47", "P_77"):
        have = R[R.Patient == p]
        add(3, p, table(["원신호 파일 검사일", "폴더에 있는 리포트의 검사일"],
                        [[V.loc[p, "RawExam"], ", ".join(a(report(p, s), s) for s in have.Study)]]),
            "<p>원신호 검사일과 같은 날짜의 리포트가 폴더에 없음.</p>", ev(p))
    # 4. 엑셀 값의 출처 검사
    for p, study in (("P_65", "2025-04-09"), ("P_67", "2025-09-29"), ("P_69", "2025-09-01")):
        r = R[(R.Patient == p) & (R.Study == study)].iloc[0]
        rows = [[k, M.loc[p, f"엑셀_{k}"], r[k], (r[f"Prior{k}"] if pd.notna(r[f"Prior{k}"]) else "소견에 없음")] for k in ("AHI", "RDI", "ODI")]
        add(4, p, table(["지표", "엑셀", f"{study} 리포트", "리포트 소견의 이전 검사 값"], rows),
            "<p>"+rp(p, study, r"Findings", "소견(Findings & Diagnosis)")+"</p>", ev(p))
    # 5. 판독 파일과 리포트의 사건 수
    for p, study in (("P_35", "2021-07-14"), ("P_50", "2023-11-06"), ("P_75", "2024-01-29"), ("P_76", "2024-08-12")):
        v = V.loc[p]; raw = int(v.Raw_OA+v.Raw_CA+v.Raw_MA+v.Raw_H)
        add(5, p, table(["", "무호흡", "저호흡", "합계"],
                        [["원본 판독 기록", int(v.Raw_OA+v.Raw_CA+v.Raw_MA), int(v.Raw_H), raw],
                         ["리포트", int(R[(R.Patient == p) & (R.Study == study)].iloc[0][["OA", "CA", "MA"]].sum()), int(R[(R.Patient == p) & (R.Study == study)].iloc[0].H),
                          int(R[(R.Patient == p) & (R.Study == study)].iloc[0].AH)]]),
            "<p>"+rp(p, study, r"Apnea \+ Hypopnea", "리포트 무호흡+저호흡 집계")+"</p>")
    add(5, "P_35", "<p>Embla 장비: 판독 기록은 사건 내보내기 파일 하나 ("+a(next(ROOT.glob("Data/PSG/P_35_PSG_E/EN-Apnea Events.txt")), "EN-Apnea Events.txt")
        + "). N2 시간 리포트 160.0분 / 판독 기록 154.5분: "+a(ROOT/"docs/PSG_원신호_EDA_결과_및_눈검증_목록.md", "원신호 점검 문서")+"</p>", ev("P_35"))
    add(5, "P_75", "<p>Nox 원본 파일의 판독층: <b>자동 132개, 수동 3개</b> (무호흡·저호흡). 리포트 148개는 자동 판독층에 가깝습니다.</p>",
        fig("v11_layers_P_75", "P_75 같은 10분 구간: 위는 자동 판독층의 무호흡·저호흡, 아래는 수동 판독층(3개뿐)"), ev("P_75"))
    add(5, "P_50", "<p>Nox 원본 파일의 판독층: <b>자동 159개, 수동 155개</b> (무호흡·저호흡 표시 수).</p>",
        fig("v11_layers_P_50", "P_50 자동 판독층과 수동 판독층 비교"),
        fig("p50_spo2_flat", "P_50 SpO2가 98%에 41분간 고정 — 같은 시간 맥박·호흡 신호는 변함"),
        fig("p50_chin_zero", "P_50 턱 근전도 1·2가 0으로 기록 — 같은 시간 비강압력에는 호흡이 있음"), ev("P_50"))
    add(5, "P_76", "<p class='warn'>Nox 원본 파일의 판독층: <b>수동 판독층 하나뿐</b>(자동 판독층 없음). 질문지 5번의 '자동·수동 판독 기록이 함께 있어'는 P_76에는 해당하지 않으며, 리포트와 현재 파일의 수정 이력(revision) 확인이 핵심입니다.</p>", ev("P_76"))
    # 6. 리포트 vs 엑셀
    add(6, "P_22", table(["", "엑셀", "리포트"], [["ODI", M.loc["P_22", "엑셀_ODI"], R[(R.Patient == "P_22")].iloc[0].ODI],
                                                  ["앙와위 AHI", "—", R[(R.Patient == "P_22")].iloc[0].AHI_SUP]]),
        "<p>"+rp("P_22", "2018-07-23", r"Supine AHI\s*[:=]\s*29\.8", "Supine AHI 29.8")+" · "+rp("P_22", "2018-07-23", r"ODI=6\.1", "소견 ODI 6.1")+"</p>", ev("P_22"))
    add(6, "P_23", "<p>"+rp("P_23", "2018-10-30", r"RDI\s*:\s*33\.5", "머리말 RDI 33.5")+" · "+rp("P_23", "2018-10-30", r"RDI:\s*23\.3", "요약표 RDI 23.3")+"</p>", ev("P_23"))
    # 7. 촬영 범위
    add(7, "P_78", img(V2/"P_78/review/p78_overview.png", "P_78 CBCT 전체 범위"), img(V2/"P_78/review/p78_sag.png", "P_78 정중 시상 단면: 계측 기준점 일부가 영상 밖"),
        "<p>촬영 범위 약 102 mm (87명 중 85명 약 200 mm, P_50 154 mm). "+a(V2/"evidence_review/v1/index.html", "CBCT 영상 검토 1절 (P_78)")+"</p>", ev("P_78"))
    # 8. 엑셀 검사와 원신호 검사의 시점
    for p in ("P_8", "P_9", "P_10", "P_19", "P_26", "P_41", "P_42", "P_43", "P_45", "P_47", "P_77"):
        xs, ws = V.loc[p, "SummaryExam"], V.loc[p, "RawExam"]
        links = [a(report(p, xs), f"엑셀 검사 {xs} 리포트")] + ([a(report(p, ws), f"원신호 검사 {ws} 리포트")] if ((R.Patient == p) & (R.Study == ws)).any() else ["원신호 검사 리포트 없음"])
        add(8, p, table(["엑셀 검사일", "원신호 검사일", "간격"], [[xs, ws, f"{(pd.Timestamp(ws)-pd.Timestamp(xs)).days}일"]]), "<p>"+" · ".join(links)+"</p>")
    add(8, "P_8", "<p>"+rp("P_8", "2016-07-01", r"Gender:\s*Male", "첫 장 머리말 Gender: Male")+" · "+rp("P_8", "2016-07-01", r"Gender\s*:\s*F\b", "같은 리포트 상세 장 Gender: F")
        + " · "+rp("P_8", "2018-10-08", r"Gender", "2018년 리포트 Female")+"</p>")
    add(8, "P_10", "<p>"+rp("P_10", "2018-08-17", r"RDI\s*:\s*25\.7", "머리말 RDI 25.7")+" · "+rp("P_10", "2018-08-17", r"RDI:\s*19\.5", "요약표 RDI 19.5")+"</p>")
    for p in ("P_8", "P_9", "P_10", "P_19", "P_26", "P_41", "P_42", "P_43", "P_45", "P_47", "P_77"): add(8, p, ev(p))
    # 9·10. 나이
    for p, s in (("P_24", "2019-07-26"), ("P_58", "2019-08-07")): add(9, p, "<p>"+rp(p, s, r"Age", f"검사 당시 {int(R[(R.Patient == p) & (R.Study == s)].iloc[0].AgeAtStudy)}세")+"</p>", ev(p))
    add(10, "P_31", table(["엑셀 나이", "리포트 나이 (2021-05-11)"], [[int(V.loc["P_31", "Excel_Age"]), int(R[(R.Patient == "P_31") & (R.Study == "2021-05-11")].iloc[0].AgeAtStudy)]]),
        "<p>"+rp("P_31", "2021-05-11", r"Age", "리포트 나이")+"</p>", ev("P_31"))
    # 11. 하위 폴더 혼입
    for p in ("P_34", "P_40"):
        hits = "; ".join(f"하위 {k+1}: ID 일치 {', '.join(i) or '없음'}, 생년월일·성별 일치 {', '.join(d) or '없음'}" for k, (i, d) in enumerate(sm[p]))
        add(11, p, series(p), f"<p>하위 폴더 영상을 87명 상위 폴더와 대조: {escape(hits)}.</p>")
    r = R[(R.Patient == "P_40") & (R.Study == "2019-03-20")].iloc[0]
    add(11, "P_40", table(["무호흡+저호흡", "총기록시간", "수면시간", "총기록시간 기준", "수면시간 기준", "리포트 AHI"],
                          [[int(r.AH), f"{r.TRT:g}분", f"{r.TST:g}분", f"{r.AH/(r.TRT/60):.1f}", f"{r.AH/(r.TST/60):.1f}", f"{r.AHI:g}"]]),
        "<p>"+rp("P_40", "2019-03-20", r"Total Recording Time", "수면 요약(총기록시간·A+H)")+"</p>")
    for p in ("P_34", "P_40"): add(11, p, ev(p))
    # 12. 원신호 채널
    add(12, "P_23", "<p>원신호 채널 중 7개의 시작 시각이 나머지와 12시간 넘게 달라(다른 날 기록) 리더가 자동 제외: "
        + a(ROOT/"docs/PSG_원신호_EDA_결과_및_눈검증_목록.md", "원신호 점검 문서 C19")+"</p>", ev("P_23"))
    add(12, "P_59", fig("p59_thermistor_rail", "P_59 서미스터가 최대값에 고정된 4분 — 같은 시간 비강압력·SpO2는 호흡·사건을 보임"), ev("P_59"))
    # 13. 성별 숫자코드
    for p in ("P_2", "P_63", "P_70"):
        s = V.loc[p, "SummaryExam"]
        add(13, p, table(["엑셀 문자 성별", "엑셀 숫자코드", "리포트 성별"], [[V.loc[p, "Excel_Sex"], int(V.loc[p, "Excel_SexCode"]), V.loc[p, "V_Sex"]]]),
            "<p>"+rp(p, s, r"Gender", "리포트 성별")+"</p>", ev(p))
    # 14. PSG–CBCT 간격
    for p in ("P_80", "P_2", "P_32", "P_43", "P_77", "P_41", "P_58", "P_56", "P_40", "P_42", "P_60", "P_49", "P_26", "P_73", "P_5"):
        s, c, w = V.loc[p, "SummaryExam"], M.loc[p, "CBCT촬영일"], V.loc[p, "RawExam"]
        add(14, p, table(["엑셀 PSG 검사일", "CBCT 촬영일", "간격", "원신호 검사일 기준 간격"],
                         [[s, c, f"{abs((pd.Timestamp(c)-pd.Timestamp(s)).days)}일", f"{abs((pd.Timestamp(c)-pd.Timestamp(w)).days)}일" if w != s else "같은 검사"]]), ev(p))
    return E


CSS = """:root{--bg:#f6f6f3;--card:#fff;--fg:#1d1d1b;--muted:#5d5d58;--line:#dcdcd5;--accent:#2f5d8a;--warn:#8a4b00;--warnbg:#fbf0d9}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--fg:#ececea;--muted:#a3a39d;--line:#363633;--accent:#8db4dc;--warn:#e3b65c;--warnbg:#3a2e14}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 "Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}h1{font-size:1.5rem;margin:0 0 4px}.sub{color:var(--muted)}
a{color:var(--accent)}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:6px 0;background:var(--card)}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--bg);white-space:nowrap}
.q td:first-child{text-align:center;font-weight:700;width:3em}.q a.t{font-weight:700;text-decoration:none}.q a.t:hover{text-decoration:underline}
.pid{display:inline-block;margin:0 4px 2px 0;font-weight:700}.q .now{font-size:.85rem;color:var(--muted);width:22%}
section{margin-top:34px;padding-top:6px;border-top:2px solid var(--accent)}section h2{font-size:1.15rem;margin:6px 0}
.ask{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 14px;margin:12px 0;scroll-margin-top:12px}
.card h3{margin:0 0 4px;font-size:1rem}.note{color:var(--muted);margin:0 0 6px}
figure{margin:8px 0}figure img{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}figcaption{font-size:.85rem;color:var(--muted)}
.warn{background:var(--warnbg);color:var(--warn);padding:6px 10px;border-radius:4px}.top{font-size:.85rem}
:target{outline:3px solid var(--accent);outline-offset:3px}"""


def main():
    sm = sub_match(); E = evidence(sm)
    qrows, secs = [], []
    for i, (title, fact, ask, targets, now) in enumerate(ROWS, 1):
        pids = " ".join(f'<a class="pid" href="#q{i}-{p}">{p}</a>' for p, _ in targets)
        qrows.append(f'<tr><td>{i}</td><td><a class="t" href="#q{i}">{escape(title)}</a><br>{escape(fact)}'
                     + "".join(f"<br>{escape(x)}" for x in ask.split("\n") if x)+f'</td><td>{pids}</td><td class="now">{escape(now)}</td></tr>')
        cards = []
        for p, note in targets:
            items = E.get((i, p)); assert items, f"증거 없음: {i}번 {p}"
            note_html = "<br>".join(escape(x) for x in note.split("\n") if x)
            cards.append(f'<div class="card" id="q{i}-{p}"><h3>{p}</h3>'+(f'<p class="note">질문지: {note_html}</p>' if note else "")+"".join(items)+"</div>")
        secs.append(f'<section id="q{i}"><h2>{i}. {escape(title)}</h2><div class="ask">{escape(fact)}'
                    + "".join(f"<br>{escape(x)}" for x in ask.split("\n") if x)+f'</div><p class="note">현재 처리 상태: {escape(now)}</p>'
                    + "".join(cards)+'<p class="top"><a href="#questions">↑ 질문지로</a></p></section>')
    used = {k for k in E}; asked = {(i, p) for i, r in enumerate(ROWS, 1) for p, _ in r[3]}
    assert used == asked, f"질문지에 없는 증거: {used-asked}"
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>자료 확인 요청 증거</title><style>{CSS}</style></head><body><main>
<h1>CBCT·PSG 자료 확인 요청 — 항목별 증거</h1>
<p class="sub">질문지 제목을 누르면 그 항목의 증거로, 환자 번호를 누르면 그 환자의 증거로 이동합니다. 리포트 링크는 해당 쪽을 엽니다. 생년월일·등록번호는 싣지 않고 일치 여부만 적었습니다. 내부 검토용(로컬 파일).</p>
<h2 id="questions">질문지 ({len(ROWS)}건)</h2>
<table class="q"><tr><th>번호</th><th>확인 요청 사항</th><th>대상 환자</th><th>현재 처리 상태</th></tr>{"".join(qrows)}</table>
{"".join(secs)}
</main></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"saved {OUT.relative_to(ROOT)}: 항목 {len(ROWS)}, 증거 카드 {len(asked)}")


if __name__ == "__main__":
    main()
