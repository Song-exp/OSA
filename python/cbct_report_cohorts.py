"""Arrange the local CBCT report into inquiry and included-cohort tabs.

The existing cohort.csv is the sole membership source. No patient is newly
included/excluded, and existing matching evidence and measurements are read-only.
"""
from html import escape
from pathlib import Path
from urllib.parse import quote
import os
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Result_py/CBCT/patient_pipeline/v2"
REPORT = BASE / "report/CBCT_보고서_2026-11-10.html"
MATCH = ROOT / "Result_py/patient_matching"
QUESTIONS = MATCH / "환자별_데이터현황_및_문의사항.md"
ANALYSIS_START = "<!-- COHORT_ANALYSIS_START -->"
ANALYSIS_END = "<!-- COHORT_ANALYSIS_END -->"

CSS = '''/* COHORT_TABS_CSS_START */
.cohort-tabs{display:flex;gap:8px;margin:20px 0 16px;padding:6px;background:var(--chip);border-radius:10px;position:sticky;top:0;z-index:10}
.cohort-tabs button{flex:1;padding:13px 12px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg);font:inherit;font-weight:700;cursor:pointer}
.cohort-tabs button[aria-selected="true"]{background:var(--accent);color:white;border-color:var(--accent)}
.cohort-tabs button:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
.cohort-panel[hidden]{display:none}.cohort-panel{scroll-margin-top:90px}
.cohort-panel [id]{scroll-margin-top:95px}.cohort-case{border:1px solid var(--line);border-radius:7px;padding:14px;margin:16px 0;background:var(--card)}
.cohort-case h3{margin-top:0}.cohort-case iframe{width:100%;height:650px;background:white;border:1px solid var(--line);border-radius:4px;margin-top:10px}
.cohort-count{font-size:1.1rem;font-weight:700}.cohort-index td{font-size:.85rem}
@media print{.cohort-tabs{display:none}.cohort-panel[hidden]{display:block}}
/* COHORT_TABS_CSS_END */'''

JS = '''// COHORT_TABS_SCRIPT_START
document.addEventListener('DOMContentLoaded', () => {
  const buttons = [...document.querySelectorAll('[data-cohort-tab]')];
  const panels = [...document.querySelectorAll('[data-cohort-panel]')];
  if (buttons.length !== 2 || panels.length !== 2) return;
  function select(key, focus = false) {
    buttons.forEach(button => {
      const active = button.dataset.cohortTab === key;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      if (active && focus) button.focus();
    });
    panels.forEach(panel => { panel.hidden = panel.dataset.cohortPanel !== key; });
  }
  function followHash() {
    if (!location.hash) return false;
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (_) { return false; }
    const target = document.getElementById(id);
    const panel = target && target.closest('[data-cohort-panel]');
    if (!panel) return false;
    select(panel.dataset.cohortPanel);
    for (let node = target; node; node = node.parentElement) {
      if (node.tagName === 'DETAILS') node.open = true;
    }
    requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
    return true;
  }
  buttons.forEach((button, index) => {
    button.addEventListener('click', () => {
      select(button.dataset.cohortTab);
      history.replaceState(null, '', '#' + button.getAttribute('aria-controls'));
    });
    button.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') next = 1 - index;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = 1;
      if (next === undefined) return;
      event.preventDefault();
      select(buttons[next].dataset.cohortTab, true);
      history.replaceState(null, '', '#' + buttons[next].getAttribute('aria-controls'));
    });
  });
  window.addEventListener('hashchange', followHash);
  if (!followHash()) select('inquiry');
});
// COHORT_TABS_SCRIPT_END'''


def href(path):
    assert path.exists(), f"Evidence missing: {path}"
    relative = os.path.relpath(path, REPORT.parent).replace("\\", "/")
    return quote(relative, safe="/._-#")


def link(path, label):
    return f'<a href="{href(path)}" target="_blank" rel="noopener">{escape(label)}</a>'


def inline_md(text):
    pattern = re.compile(r'\[([^\]]+)\]\(<([^>]+)>\)')
    out, end = [], 0
    for match in pattern.finditer(text):
        out.append(escape(text[end:match.start()]))
        target = (MATCH / match[2]).resolve()
        if target.is_file():
            out.append(link(target, match[1]))
        else:
            out.append(escape(match[1])+" (파일 확인 필요)")
        end = match.end()
    out.append(escape(text[end:]))
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', "".join(out))


def inquiry_content(excluded, matching):
    source = QUESTIONS.read_text(encoding="utf-8")
    sections = dict(re.findall(r'^### (P_\d+)\s*\n(.*?)(?=^### |^## |\Z)', source, re.M | re.S))
    expected = set(excluded.Patient)-{"P_78"}
    assert expected == set(sections), "Detailed inquiry list and excluded cohort differ"
    intro = '''<h2 id="inquiry-overview">문의·검토 대상 환자 33명</h2>
<p class="note">현재 CBCT 계측 명단에서 보류한 상세 문의 32명과 촬영 범위 문제로 추가 보류한 P_78입니다. 기존 명단에 따라 구분했으며 이번 화면 정리로 환자를 새로 제외하거나 포함하지 않았습니다. 일부 사유는 PSG 검사 연결·판독·기본정보 확인에 관한 것이므로 CBCT 영상 자체의 오류와 동일하게 해석하지 않습니다.</p>
<p><strong>33명 모두 기존 자료 대조 근거 HTML이 있습니다.</strong> 이는 환자 식별·검사 연결·엑셀/리포트 대조 근거이며, 54명 자동 CBCT 계측 결과와는 근거의 종류가 다릅니다. 환자별 질문과 해당 리포트를 아래에서 확인할 수 있습니다. 표시된 미회신은 기존 문의 문서의 상태이며 발송 여부를 새로 확인한 것은 아닙니다.</p>'''
    intro += '<p>'+link(QUESTIONS, '환자별 상세 문의 원문')+' · '+link(ROOT / "docs/연구실_문의_통합_2026-11-09.md", '연구실 문의 통합본')+' · '+link(MATCH / "환자 일치 검수.xlsx", '환자 일치 검수')+' · '+link(MATCH / "PSG 값 일치 검수.xlsx", 'PSG 값 일치 검수')+'</p>'
    rows, cards = [], []
    for _, item in excluded.iterrows():
        patient = item.Patient
        record = matching.loc[patient]
        evidence = MATCH / "patients" / patient / "evidence.html"
        assert evidence.is_file(), patient
        body = sections.get(patient, "")
        status = re.search(r'\*\*검토 상태:\*\*\s*([^\n]+)', body)
        status = status[1] if status else '촬영 범위에 따른 추가 보류; 전문가 확인·넓은 범위 영상 문의 필요'
        facts = re.findall(r'^- (.+)', body, re.M)
        questions = re.findall(r'^\d+\. (.+)', body, re.M)
        if patient == "P_78":
            facts = ["현재 촬영 범위 약 102 mm로 Po·N·Or이 영상 밖이라 골격 4지표를 계산하지 못했습니다.",
                     "기존 환자별 자료 대조 근거 HTML과 별도의 촬영 범위 검토 절이 있습니다."]
            questions = ["더 넓은 범위의 같은 환자 CBCT 또는 다른 촬영본이 있나요?",
                         "엑셀 계측에 사용한 영상의 출처와 촬영 범위, 측정할 수 있었던 변수를 확인해 주세요."]
        assert questions, f"No inquiry questions for {patient}"
        cbct = str(record["엑셀_CBCT_식별정보대조"])
        study = str(record["엑셀_원신호검사"])
        rows.append(f'<tr data-inquiry-patient="{patient}"><td><a href="#inquiry-{patient}">{patient}</a></td><td>{escape(cbct)}</td><td>{escape(study)}</td><td>{escape(status)}</td><td>{len(questions)}</td><td>{link(evidence, "기존 자료 대조 근거")}</td></tr>')
        facts_html = '<ul>'+"".join('<li>'+inline_md(fact)+'</li>' for fact in facts)+'</ul>'
        questions_html = '<ol>'+"".join('<li>'+inline_md(question)+'</li>' for question in questions)+'</ol>'
        fov = '<p><a href="../evidence_review/v1/index.html#s1" target="_blank" rel="noopener">P_78 촬영 범위 영상 검토</a></p>' if patient == "P_78" else ""
        cards.append(f'''<article class="cohort-case" id="inquiry-{patient}"><h3>{patient}</h3>
<p><strong>현재 명단 사유:</strong> {escape(item.Reason)}</p><p><strong>기존 문서의 검토 상태:</strong> {escape(status)}</p>
{facts_html}<h4>확인할 질문</h4>{questions_html}
<p><strong>답변·후속 처리:</strong> 기존 문서상 미회신. 사용할 검사·자료 출처·변수별 보류 범위는 회신과 검수 기록에 따라 확인합니다.</p>
<p>{link(evidence, patient+' 기존 자료 대조 근거 리포트 열기')}</p>{fov}
<details><summary>기존 자료 대조 근거 리포트를 이 화면에서 펼치기</summary><iframe loading="lazy" src="{href(evidence)}" title="{patient} 기존 자료 대조 근거 리포트"></iframe></details>
</article>''')
    return intro+'<div class="tw"><table><tr><th>환자</th><th>CBCT 식별정보 대조</th><th>엑셀·원신호 검사 연결</th><th>검토 상태</th><th>질문 수</th><th>기존 근거</th></tr>'+"".join(rows)+'</table></div>'+"\n".join(cards)


def analysis_index(included):
    audit = pd.read_csv(BASE / "nav_review/v2_label_audit/audit.csv").set_index("Patient")
    values = pd.read_csv(BASE / "fh_repair/v2/comparison.csv")
    qc = values.drop_duplicates("Patient").set_index("Patient").qc_after
    outliers = pd.read_csv(BASE / "fh_repair/v2/outliers.csv")
    marks = outliers.groupby("Patient").metric.agg(lambda s: ", ".join(dict.fromkeys(s)))
    rows = []
    for _, item in included.iterrows():
        patient = item.Patient
        assert patient in audit.index and patient in qc.index
        landmark = BASE / f"fh_repair/v2/figs/{patient}_landmarks.png"
        evidence = [f'<a href="#nasal-{patient}">비강 라벨·NCWR 단면</a>']
        if landmark.exists():
            evidence.append(link(landmark, '새 골격 랜드마크 단면'))
        evidence.append(link(MATCH / "patients" / patient / "evidence.html", '기존 자료 대조 근거'))
        status = '거리 QC 통과·전문가 확인 전' if qc.loc[patient] == 'qc_pass_pending_visual_review' else '거리 QC 검토 필요'
        rows.append(f'<tr data-analysis-patient="{patient}"><td>{patient}</td><td>{status}</td><td>{escape(marks.get(patient, "상위 오차 목록 해당 없음"))}</td><td>{" · ".join(evidence)}</td></tr>')
    return '''<!-- COHORT_INDEX_START -->
<h2 id="analysis-cohort-overview">CBCT 분석에 포함된 54명</h2>
<p class="note">현재 자동 계측 명단의 54명입니다. 전체 결과·지표별 산점도·박스플롯과 환자별 근거를 이 탭에 모았습니다. 일부 환자의 개별 지표에는 큰 오차·미계측·추가 문의가 남아 있으며, 분석 명단 포함이 전문가 확인 완료를 뜻하지 않습니다.</p>
<p><a href="#s1">전체 계측 요약</a> · <a href="#scatter">지표별 산점도·박스플롯</a> · <a href="#nasal-audit">54명 비강 라벨 점검</a> · <a href="#error-review">큰 오차 환자 검토</a></p>
<details><summary>54명 명단·QC·환자별 근거 펼치기</summary><div class="tw"><table class="cohort-index"><tr><th>환자</th><th>현재 4지표 QC</th><th>큰 오차 목록의 지표</th><th>환자별 근거</th></tr>'''+"".join(rows)+'''</table></div></details>
<!-- COHORT_INDEX_END -->'''


def build_tabs(html):
    cohort = pd.read_csv(BASE / "cohort.csv", keep_default_na=False)
    assert len(cohort) == 87 and not cohort.Patient.duplicated().any()
    selected = cohort.Include.astype(str).str.lower().isin(["true", "1"])
    included, excluded = cohort[selected], cohort[~selected]
    assert len(included) == 54 and len(excluded) == 33
    assert set(included.Patient).isdisjoint(excluded.Patient)
    matching = pd.read_csv(MATCH / "patient_matching_with_raw.csv", dtype=str, keep_default_na=False).set_index("환자")
    if ANALYSIS_START in html:
        assert html.count(ANALYSIS_START) == html.count(ANALYSIS_END) == 1
        analysis = html.split(ANALYSIS_START, 1)[1].split(ANALYSIS_END, 1)[0]
        analysis = re.sub(r'\s*<!-- COHORT_INDEX_START -->.*?<!-- COHORT_INDEX_END -->\s*', '', analysis, flags=re.S)
    else:
        inner = re.search(r'<main>(.*?)</main>', html, re.S)
        assert inner
        header = re.search(r'<h1>.*?</h1>\s*<p class="sub">.*?</p>', inner[1], re.S)
        assert header
        analysis = inner[1][header.end():]
    analysis = analysis.strip()
    inquiry = inquiry_content(excluded, matching)
    content = f'''<main>
<!-- COHORT_TABS_START -->
<h1>CBCT 작업 보고서 · 2026-11-10</h1>
<p class="sub">등록 87명 · 현재 명단 기준 문의·검토 대상 33명 / CBCT 분석에 포함된 54명 · 탐색적 자동 분석, 전문가 확인 전</p>
<div class="cohort-tabs" role="tablist" aria-label="환자군별 보고서">
<button id="tab-inquiry" type="button" role="tab" data-cohort-tab="inquiry" aria-controls="cohort-inquiry" aria-selected="true" tabindex="0">문의·검토 대상 · 33명</button>
<button id="tab-analysis" type="button" role="tab" data-cohort-tab="analysis" aria-controls="cohort-analysis" aria-selected="false" tabindex="-1">CBCT 분석에 포함 · 54명</button>
</div>
<noscript><style>.cohort-panel[hidden]{{display:block}}</style><p>스크립트가 꺼져 있어 두 환자군을 아래에 순서대로 표시합니다.</p></noscript>
<section id="cohort-inquiry" class="cohort-panel" role="tabpanel" data-cohort-panel="inquiry" aria-labelledby="tab-inquiry" tabindex="0">
{inquiry}
</section>
<section id="cohort-analysis" class="cohort-panel" role="tabpanel" data-cohort-panel="analysis" aria-labelledby="tab-analysis" tabindex="0" hidden>
{ANALYSIS_START}
{analysis_index(included)}
{analysis}
{ANALYSIS_END}
</section>
<!-- COHORT_TABS_END -->
</main>'''
    result, count = re.subn(r'<main>.*?</main>', lambda _: content, html, count=1, flags=re.S)
    assert count == 1
    style = '<style>\n'+CSS+'\n</style>'
    if '/* COHORT_TABS_CSS_START */' in result:
        result = re.sub(r'<style>\s*/\* COHORT_TABS_CSS_START \*/.*?/\* COHORT_TABS_CSS_END \*/\s*</style>', lambda _: style, result, flags=re.S)
    else:
        result = result.replace('</head>', style+'\n</head>', 1)
    script = '<script>\n'+JS+'\n</script>'
    if '// COHORT_TABS_SCRIPT_START' in result:
        result = re.sub(r'<script>\s*// COHORT_TABS_SCRIPT_START.*?// COHORT_TABS_SCRIPT_END\s*</script>', lambda _: script, result, flags=re.S)
    else:
        result = result.replace('</body>', script+'\n</body>', 1)
    result = re.sub(r'<title>.*?</title>', '<title>CBCT 작업 보고서 · 문의·검토 33명 / 분석 54명</title>', result, count=1)
    assert result.count('data-inquiry-patient=') == 33
    assert result.count('data-analysis-patient=') == 54
    return result


def main():
    original = REPORT.read_text(encoding="utf-8")
    output = build_tabs(original)
    plots = re.compile(r'<svg\b.*?</svg>', re.S)
    assert plots.findall(original) == plots.findall(output), 'Original plots changed'
    assert output == build_tabs(output), 'Tab rebuild is not idempotent'
    REPORT.write_text(output, encoding="utf-8")
    print('Two cohort tabs saved: 33 inquiry patients with existing evidence; 54 included patients; original plots preserved')


if __name__ == '__main__':
    main()
