"""Refresh the 2026-11-10 report narrative from completed local evaluations.

Preserves existing per-metric scatter/box plots. No measurement recalculation.
"""
from html import escape
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Result_py/CBCT/patient_pipeline/v2"
REPORT = BASE / "report/CBCT_보고서_2026-11-10.html"


def managed(html, key, content, anchor):
    start, end = f"<!-- DAILY_{key}_START -->", f"<!-- DAILY_{key}_END -->"
    block = start+"\n"+content+"\n"+end
    if start in html:
        return re.sub(re.escape(start)+r".*?"+re.escape(end), lambda _: block, html, flags=re.S)
    assert html.count(anchor) == 1, anchor
    return html.replace(anchor, block+"\n\n"+anchor)


def update_html(html):
    doctype = html.lower().find("<!doctype html>")
    assert doctype >= 0
    if doctype:
        assert html[:doctype].strip().startswith("file:///"), "Unexpected content before HTML doctype"
        html = html[doctype:]
    summary = pd.read_csv(BASE / "fh_repair/v2/summary.csv")
    all_stats = summary[summary.group == "all"].set_index(["metric", "version"])
    changes = pd.read_csv(BASE / "fh_repair/v2/comparison.csv")
    audit = pd.read_csv(BASE / "nav_review/v2_label_audit/audit.csv")
    assert len(changes) == 216 and len(audit) == 54
    assert not audit.visual_review.str.contains("검토 대기").any()
    paired = audit[audit.NAV_excel.notna()]
    n_width = int(audit.NCW_ts_delta_mm.notna().sum())
    n_ratio = int(audit.NCWR_saved.notna().sum())
    native_ratio = paired.native_NAV_ts_mm3/paired.NAV_excel
    new_note = '<div class="note"><strong>현재 결론.</strong> 핵심 CBCT 변수 11개에서 자동 계측값을 확보했습니다. 고정 FH 보정을 적용한 새 버전으로 54명 4지표를 재평가했고, MP-FH MAE는 4.03°→2.24°로 줄었습니다. NAV·NCWR에 쓰인 비강 라벨은 54명 전원의 지정 단면을 AI가 확인했으며 큰 누출·한쪽 전체 누락은 뚜렷하지 않았습니다. NAV 과소 계측과 NCWR·PMCA 차이는 남아 있어 원 측정 좌표·범위·기준면 정의가 필요합니다. 모든 결과는 전문가 확인 전 탐색 결과입니다.</div>'
    html, count = re.subn(r'<div class="note"><strong>(?:한 줄 요약\.|현재 결론\.)</strong>.*?</div>', lambda _: new_note, html, count=1, flags=re.S)
    assert count == 1
    subtitle = '<p class="sub">OSA 연구 · 엑셀(<code>Patient_Info_Data_87명_261019.xlsx</code>) CBCT 계측값의 자동 재현 · 2026-11-08 ~ 2026-11-10 작업, 2026-11-10 추가 평가·전수 점검 반영 · 모든 자동 계측은 탐색용이며 전문가 확인 전입니다.</p>'
    html = re.sub(r'<p class="sub">OSA 연구.*?</p>', lambda _: subtitle, html, count=1, flags=re.S)
    intro = '<p>오차는 자동값 − 엑셀입니다. 4지표는 고정 FH 보정 후 <strong>54명 전체</strong> 기준이며, MAE는 평균 절대오차입니다. 다른 지표는 각 행에 비교 대상 수를 명시했습니다. 평가군은 기존 파일럿 분할을 제외한 집단이지만 개발 후 재평가가 포함돼 독립 검증을 보장하지 않습니다. 상태는 탐색적 작업 분류이며 임상 허용오차나 전문가 승인을 뜻하지 않습니다.</p>'
    html, count = re.subn(r'(<h2 id="s1">1\. 현황 요약</h2>\s*)<p>.*?</p>', lambda m: m[1]+intro, html, count=1, flags=re.S)
    assert count == 1
    first_start, first_end = html.index('<h2 id="s1">'), html.index('<h2 id="s2">')
    first = html[first_start:first_end]
    units = {"ANB": "°", "Nperp-A": "mm", "Nperp-Pog": "mm", "MP-FH": "°"}
    for metric, unit in units.items():
        row = all_stats.loc[(metric, "after")]
        value = f'FH 보정 후 전체 54명 MAE {row.mae:.2f} {unit}, 중앙 절대오차 {row.median_abs:.2f} {unit}, r {row.r:.2f}, CCC {row.ccc:.2f}'
        pattern = r'(<tr><td>'+re.escape(metric)+r'</td><td class="n">54</td><td>).*?(</td><td>)'
        first, count = re.subn(pattern, lambda m: m[1]+value+m[2], first, count=1, flags=re.S)
        assert count == 1, metric
    html = html[:first_start]+first+html[first_end:]
    latest = '''<h3 id="today-update">2026-11-10 추가 완료 작업</h3>
<ul><li><strong>FH 보정 전후:</strong> 54명 216개 비교쌍을 새 버전으로 계산했습니다. P_16·P_20·P_37만 좌표가 바뀌었으며, 기존 result.json은 보존했습니다. P_20은 일부 지표가 악화돼 함께 기록했습니다.</li>
<li><strong>지표별 그림과 오차 검토:</strong> 11지표의 환자별 산점도와 실제값·복원값·오차 박스플롯을 각 지표 절에 배치했습니다. 절대오차 상위 5명씩과 NAV·PAV·PMCA 상대오차 상위 5명씩, 총 70행의 검토 목록을 만들었습니다. 4지표 검토용 환자 15명의 새 랜드마크 단면을 생성했습니다.</li>
<li><strong>비강 라벨 전수 점검:</strong> 54명의 210개 지정 단면을 생성·AI 검토했습니다. 48명 NCW는 기존 값과 일치했고, NAV 원영상 격자 차이는 중앙값 약 1.1%로 과소 계측이 남았습니다.</li></ul>
<p><a href="#fh-comparison">4지표 전후 집계</a> · <a href="#nasal-audit">54명 비강 라벨 점검</a> · <a href="#error-review">큰 오차 환자 검토</a></p>'''
    html = managed(html, "LATEST", latest, '<h2 id="s2">')
    html = html.replace('통합 검토 페이지</a> 1–15절', '통합 검토 페이지</a> 1–16절')
    old = '<li><code>result.json</code>(4지표 정식 결과)은 바꾸지 않았고, 이 보고서의 나머지 값은 모두 별도 검토 폴더의 탐색 결과입니다. 커밋하지 않았습니다.</li>'
    new = '<li><code>result.json</code>(4지표 정식 결과)은 보존했습니다. 이 보고서의 현재 4지표와 그림은 별도 <code>fh_repair/v2</code>의 보정 후 탐색 결과이며, 다른 지표도 각 검토 폴더의 선택 규칙을 사용합니다. 코드·결과를 커밋하지 않았습니다.</li>'
    html = html.replace(old, new)
    html = html.replace('TDI·PHI·PAV에도 적용했고, <code>result.json</code>에는 아직 반영하지 않았습니다.', 'TDI·PHI·PAV에 적용한 규칙을 새 <code>fh_repair/v2</code> 평가 단계에도 사용했습니다. 54명 4지표를 재계산해 이 보고서의 요약·그림에 반영했고, 원래 <code>result.json</code>은 보존했습니다.')
    ceph = '<p>정의는 Roh 2025 표 1·그림 1A, 좌표계는 Jung 2025의 Invivo 방식(FH = RPo·LPo·ROr, MSP는 N–Ba를 지나 FH에 수직), 각도·거리는 MSP 투영, 하악평면은 Go–Me입니다. 고정 FH 보정 후 같은 54명 전체에서 MAE는 ANB 0.87°, Nperp-A 1.77 mm, Nperp-Pog 3.26 mm, MP-FH 2.24°입니다. 보정 전 QC 통과는 48명, 보정 후는 51명이며 아래 통계는 전체·평가군·보정 전 QC 통과군 고정·보정 후 QC 통과군을 구분합니다. 이전의 QC 41명 집계는 필터 출처가 확인되지 않아 현재 성능값으로 재사용하지 않았습니다.</p>'
    html, count = re.subn(r'(<h2 id="s5">5\. 골격 각도·거리 4지표</h2>\s*)<p>.*?</p>', lambda m: m[1]+ceph, html, count=1, flags=re.S)
    assert count == 1
    html = html.replace('비강 라벨은 관상면에서 비강 공기를 따라가고 상악동·사골동 칸을 빼 줍니다(P_44 41.4→19.4 mL, 엑셀 20.2). P_20은 점막 부종으로 공기가 실제로 적고, P_14는 원인을 찾지 못했습니다.', '기존 영상 규칙보다 TotalSegmentator 비강 라벨이 상악동 포함을 줄였습니다(P_44 영상 규칙 41.4→라벨 기반 19.4 mL, 엑셀 20.2). 후속 점검에서 54명 전원의 지정 관상면을 확인했습니다. P_20은 라벨 안 공기 영역이 적었고, P_14는 큰 NAV 차이가 있어도 표시한 단면에서 큰 상악동 누출이나 한쪽 전체 누락은 뚜렷하지 않았습니다. 원인은 확정하지 않았으며 아래에 전수 점검 결과를 정리했습니다.')
    head = f'''<h3 id="nasal-audit">7-1. 비강 라벨 54명 전수 점검</h3>
<p>기존 비강 라벨(14·15)을 바꾸지 않고 PNS→ANS 25·50·75% 관상면을 54명 전원에서 생성했습니다. {n_width}명은 실제 NCWR 대구치 단면도 생성해 총 {54*3+n_width}개 단면을 AI가 확인했습니다. 초록은 비강 라벨, 노랑은 그 안 공기(−1200 &lt; HU &lt; −400), 하늘색은 저장된 NCW 측정선입니다. 엑셀·오차 정보를 함께 볼 수 있어 독립·블라인드 검증은 아닙니다.</p>
<div class="tw"><table><tr><th>점검</th><th>대상</th><th>결과</th></tr>
<tr><td>지정 단면 AI 시각 검토</td><td>54명</td><td>한쪽 비강 전체 누락이나 상악동으로의 큰 누출은 표시한 단면에서 뚜렷하지 않음</td></tr>
<tr><td>좌우 라벨·FOV 기하 점검</td><td>54명</td><td>양측 라벨 존재, 촬영 범위 경계 접촉 없음</td></tr>
<tr><td>실제 대구치 단면 NCW 재계산</td><td>{n_width}명</td><td>기존 폭과 정확히 일치. 최종 NCWR은 분모 등 추가 조건으로 {n_ratio}명</td></tr>
<tr><td>원영상 격자 NAV / 기존 0.4 mm 격자 NAV</td><td>54명</td><td>중앙 변화 +{audit.native_vs_saved_NAV_pct.median():.2f}%, 범위 {audit.native_vs_saved_NAV_pct.min():.2f}~+{audit.native_vs_saved_NAV_pct.max():.2f}%</td></tr>
<tr><td>엑셀 대비 NAV: 기존 → 원영상 격자</td><td>{len(paired)}명</td><td>자동/엑셀 중앙 비율 {paired.NAV_ratio.median():.3f} → {native_ratio.median():.3f}, r {paired.NAV_saved.corr(paired.NAV_excel):.3f} → {paired.native_NAV_ts_mm3.corr(paired.NAV_excel):.3f}</td></tr></table></div>
<div class="note"><strong>해석.</strong> 격자 차이만으로 남은 NAV 과소 계측을 설명하기 어렵습니다. NCWR 큰 오차 사례도 NCW가 재현되므로 MxBW·측정 높이·원래 폭 정의 확인이 남습니다. 미세 경계와 전체 3차원 정확도는 검증하지 않았고, 라벨 안 비공기 영역에는 비갑개·조직도 포함되므로 이를 곧바로 분할 실패로 해석하지 않습니다. 원영상 격자 NAV는 통제 비교값이며 기존 정식값으로 채택하지 않았습니다.</div>
<p><a href="../nav_review/v2_label_audit/index.html">별도 전수 점검 페이지·9개 묶음 그림</a> · <a href="../nav_review/v2_label_audit/audit.csv" download>54명 점검 CSV</a> · <a href="../nav_review/v2_label_audit/visual_review.json" download>AI 시각 검토 기록</a></p>
<details><summary>54명 환자별 단면과 AI 검토 기록 펼치기</summary>'''
    patient_cards = []
    for _, row in audit.iterrows():
        image = '../nav_review/v2_label_audit/'+row.image
        ratio = f'NAV/엑셀 {row.NAV_ratio:.3f}' if pd.notna(row.NAV_ratio) else 'NAV 엑셀값 없음'
        error = f'NCWR 오차 {row.NCWR_error:+.2f}' if pd.notna(row.NCWR_error) else 'NCWR 미계측'
        patient_cards.append(f'<details id="nasal-{row.Patient}"><summary>{row.Patient} · {ratio} · {error}</summary><p>{escape(row.visual_review)}</p><figure><a href="{image}"><img loading="lazy" src="{image}" alt="{row.Patient} 비강 라벨과 공기, 가능한 NCWR 단면"></a><figcaption>지정 단면의 탐색적 AI 검토. 전문가 확인 전.</figcaption></figure></details>')
    html = managed(html, "NASAL", head+"\n".join(patient_cards)+'</details>', '<h2 id="s8">')
    ncwr_note = '<h3 id="ncwr-audit">NCWR 후속 점검</h3><p>비강 라벨 전수 점검에서 가능한 48명의 NCW가 기존 폭과 일치했고, 최종 NCWR이 있는 46명도 포함됐습니다. 절대오차 상위 P_29·P_12·P_53·P_30·P_79는 표시한 단면에서 큰 상악동 침범이 뚜렷하지 않았습니다. MxBW·원 측정 높이 정의를 확인해야 하지만, 분모가 오차의 확정 원인이라는 뜻은 아닙니다. <a href="#nasal-audit">54명 점검 결과와 개별 단면</a>을 참고하세요.</p>'
    html = managed(html, "NCWR", ncwr_note, '<h2 id="s9">')
    html = html.replace('TotalSegmentator는 CT 학습 모델이며 CBCT 라벨은 그림으로 10명가량만 확인했습니다.', 'TotalSegmentator는 CT 학습 모델입니다. 54명 전원의 지정 관상면과 가능한 48명의 NCWR 단면을 AI가 확인했지만 미세 경계·전체 3차원 정확도·전문가 판독은 검증되지 않았습니다.')
    html = html.replace('FH 보정 규칙과 턱끝·Po 후보는 정식 파이프라인(<code>result.json</code>)에 아직 반영하지 않았습니다.', '고정 FH 규칙은 새 버전 <code>fh_repair/v2</code>에 적용해 54명을 재평가했습니다. 원래 <code>result.json</code>과 턱끝·Po 높이 수동 후보는 보존했습니다. P_20처럼 지표별 개선·악화가 섞이므로 보정의 해부학적 정확도는 전문가 확인이 필요합니다.')
    error_review = '''<h3 id="error-review">큰 오차 환자 검토와 남은 확인</h3><p>각 지표의 절대오차 상위 5명, NAV·PAV·PMCA의 상대오차 상위 5명을 별도로 추려 총 70행을 저장했습니다. 환자 수가 70명이라는 뜻은 아니며, 목록은 환자 제외나 엑셀 맞춤 보정 기준으로 사용하지 않았습니다. 4지표는 FH 보정 후 값이고 다른 7개는 기존 선택 규칙입니다.</p><p>15명의 새 랜드마크 단면을 생성했고, NAV·NCWR 상위 사례는 54명 전수 점검의 해당 환자로 연결했습니다. 영상 링크는 원인 확정이나 전문가 승인을 뜻하지 않습니다. <a href="../fh_repair/v2/index.html#outliers">11지표 큰 오차 환자·영상 근거</a> · <a href="../fh_repair/v2/outliers.csv" download>상위 사례 CSV</a></p>'''
    html = managed(html, "ERRORS", error_review, '<h2 id="s11">')
    inquiry = '<h3 id="remaining-questions">오늘 점검 후 문의에서 확인할 항목</h3><ul><li><strong>NAV:</strong> 지정 단면의 큰 라벨 누출·누락이나 격자 차이로는 잔여 과소 계측이 충분히 설명되지 않습니다. 비전정·사골동 포함 범위, 임계값, 원래 마스크·경계 설정을 확인해야 합니다.</li><li><strong>NCWR:</strong> 실제 대구치 단면의 NCW가 재현되므로 원 측정 높이·폭의 정의와 MxBW 기준점을 확인해야 합니다.</li><li><strong>FH:</strong> 54명 전체 오차는 줄었지만 P_20의 일부 지표는 악화됐고 다른 큰 오차도 남았습니다. 원 측정 좌표와 FH·머리 정렬 정의가 필요합니다.</li></ul>'
    html = managed(html, "QUESTIONS", inquiry, '<h2 id="s12">')
    files = '''<h3 id="today-files">오늘 추가 결과와 재현 코드</h3><div class="tw"><table><tr><th>내용</th><th>코드</th><th>결과</th></tr>
<tr><td>FH 54명 재평가·오차 검토 목록</td><td><code>cbct_fh_evaluation.py</code></td><td><a href="../fh_repair/v2/index.html">fh_repair/v2/</a></td></tr>
<tr><td>지표별 산점도·분포 박스플롯</td><td><code>cbct_report_scatter.py</code></td><td><a href="scatter/comparison_pairs.csv">환자별 비교 CSV</a> · <a href="scatter/agreement_summary.csv">집계 CSV</a></td></tr>
<tr><td>54명 비강 라벨 검토</td><td><code>cbct_nasal_label_audit.py</code></td><td><a href="../nav_review/v2_label_audit/index.html">nav_review/v2_label_audit/</a></td></tr>
<tr><td>보고서 본문 최신화</td><td><code>cbct_update_daily_report.py</code></td><td>현재 HTML의 요약·결론·전수 점검·한계·문의 항목</td></tr></table></div>
<p>검증: FH 기하 테스트 3개와 물리 좌표 표본화 테스트 2개 통과. 216개 4지표 비교쌍·48개 NCW 폭 재현·원본 결과와 라벨 보존·영상 링크 연결을 확인했습니다. 이 검증은 측정 정확도의 전문가 검증과 다릅니다.</p>'''
    html = managed(html, "FILES", files, '</main>')
    for anchor, title in [('today-update', '오늘 추가 완료 작업'), ('nasal-audit', '비강 라벨 54명 전수 점검'), ('error-review', '큰 오차 환자 검토')]:
        entry = f'<li><a href="#{anchor}">{title}</a></li>'
        if entry not in html:
            html = html.replace('</ol></nav>', entry+'\n</ol></nav>', 1)
    assert 'CBCT 라벨은 그림으로 10명가량만 확인했습니다' not in html
    assert html.count('id="nasal-P_') == 54
    assert html.startswith('<!doctype html>')
    if '<!-- COHORT_TABS_START -->' in html:
        from cbct_report_cohorts import build_tabs
        html = build_tabs(html)
    return html


def main():
    before = REPORT.read_text(encoding="utf-8")
    after = update_html(before)
    # Existing plotted data and points must remain exactly as they were.
    svg = re.compile(r'<svg\b.*?</svg>', re.S)
    assert svg.findall(before) == svg.findall(after)
    REPORT.write_text(after, encoding="utf-8")
    print('Daily narrative updated; 22 existing SVG plots preserved; 54 patient audits embedded')


if __name__ == '__main__':
    main()
