"""현재 기하 설정의 잔차 이상치와 환자별 모든 계산 후보를 공개한다."""
from html import escape
import numpy as np
import pandas as pd
from cbct_develop85 import DEST


def main():
    rank = pd.read_csv(DEST/'error_ranking.csv')
    threshold, flagged = [], []
    for metric, g in rank.groupby('metric', sort=False):
        center = g.difference.median()
        scale = 1.4826*(g.difference-center).abs().median()
        assert scale > 0
        g = g.copy(); g['robust_z'] = (g.difference-center)/scale
        threshold.append(dict(metric=metric, n=len(g), residual_median=center, scaled_MAD=scale,
                              lower_residual=center-3.5*scale, upper_residual=center+3.5*scale))
        flagged.append(g[g.robust_z.abs() > 3.5])
    flags = pd.concat(flagged, ignore_index=True)
    values = pd.read_csv(DEST/'candidate_values.csv')
    candidates = values.merge(flags[['Patient', 'metric', 'current', 'abs_error']], on=['Patient', 'metric'])
    candidates['difference'] = candidates.automatic-candidates.excel
    candidates['candidate_abs_error'] = candidates.difference.abs()
    candidates['is_current'] = candidates.candidate == candidates.current
    for name, t in [('outlier_thresholds.csv', pd.DataFrame(threshold)), ('outliers_current.csv', flags),
                    ('outlier_candidate_values.csv', candidates)]:
        t.to_csv(DEST/name, index=False, encoding='utf-8-sig')
    def table(t):
        return t.to_html(index=False, float_format=lambda x:f'{x:.3f}', border=0)
    sections = []
    for (p, m), g in candidates.groupby(['Patient', 'metric'], sort=False):
        sections.append(f'<h3>{escape(p)} / {escape(m)}</h3>'+table(g[['candidate', 'automatic', 'excel', 'candidate_abs_error', 'is_current']]))
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>85명 현재 설정 이상치와 계산 후보</title>
<style>body{{font:15px/1.6 "Malgun Gothic",sans-serif;max-width:1300px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:5px 9px}}h2{{margin-top:36px}}</style>
<h1>현재 설정의 오차 이상치: {flags.Patient.nunique()}명 · {len(flags)}건</h1>
<p>현재 선택한 기하 계산값에서 자동값−엑셀값의 중앙값과 1.4826×MAD를 지표별로 계산했습니다. |robust z| &gt; 3.5인 사례입니다. 엑셀 0은 제외하지 않았습니다. 기존 보고서의 기본 설정·제로값 우선 표기와는 기준 입력이 다릅니다.</p>
<p>이상치는 지표 내 잔차 분포에서 동떨어졌다는 뜻이며, 엑셀 오류나 자동점 오류가 확정됐다는 뜻이 아닙니다. 일정한 편향이 있어도 전체에 공통되면 이상치로 잡히지 않을 수 있습니다.</p>
{table(flags[['Patient', 'metric', 'unit', 'automatic', 'excel', 'difference', 'robust_z', 'excel_zero']])}
<h2>환자·지표별 모든 후보</h2><p>환자마다 엑셀에 가장 맞는 후보를 고르는 데 쓰면 안 됩니다. 최종 규칙은 환자 공통이며 성능은 별도 중첩 평가로 판단합니다. NaN은 후보 계산 결측입니다. TTA·직선 환산은 이번 비교에서 제외했습니다.</p>{''.join(sections)}
<h2>자료</h2><p><a href="outliers_current.csv">이상치 CSV</a> · <a href="outlier_candidate_values.csv">후보별 실제 값 CSV</a> · <a href="outlier_thresholds.csv">지표별 경계값</a></p></html>'''
    (DEST/'outliers.html').write_text(html, encoding='utf-8')
    print(flags[['Patient','metric','unit','automatic','excel','difference','robust_z']].round(3).to_string(index=False))
    print('PATIENTS', flags.Patient.nunique(), 'CASES', len(flags))
    print('CANDIDATES:')
    for m, g in candidates.groupby('metric', sort=False):
        print(m)
        print(g.pivot(index='Patient', columns='candidate', values='candidate_abs_error').round(2).to_string())


if __name__ == '__main__':
    main()
