"""Po 반복조건 실험의 반환 계약·평가 분모·기준점별 평가 범위를 점검한다."""
from html import escape
import json
import re

import numpy as np
import pandas as pd

from cbct_po_stability85 import DEST,SOURCE,MODULE
from cbct_po_ensemble import VARIANTS,ACCEPT
from cbct_patient_pipeline import landmarks,digest


def main():
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    assert len(protocol['patients'])==85 and len(VARIANTS)==16
    assert digest(SOURCE/'protocol.json')==protocol['source_protocol_sha256']
    assert digest(MODULE)==protocol['algorithm_sha256']
    cases=[json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8')) for p in protocol['patients']]
    agreement=[]
    for c in cases:
        assert c['fingerprint']['code']==protocol['driver_sha256']
        assert c['fingerprint']['algorithm']==protocol['algorithm_sha256']
        P,files=landmarks(int(c['Patient'][2:]))
        assert {f.name:digest(f) for f in files}==c['fingerprint']['landmarks']
        for side,r in c['points'].items():
            assert len(r['variants'])==16
            assert {v['variant'] for v in r['variants']}=={v['name'] for v in VARIANTS}
            s=r['stability'];center=s['representative_lps']
            distance=float(np.linalg.norm(np.array(center)-P[side])) if center is not None else None
            agreement.append(dict(Patient=c['Patient'],point=side,triggered=r['triggered'],
                                  candidate_stability=s['status'],retained_original=r['route']=='existing_point_retained_not_confirmed',
                                  original_to_stable_candidate_mm=distance,
                                  original_candidate_disagreement=distance is not None and distance>3.,
                                  anatomically_confirmed=False))
    agreement=pd.DataFrame(agreement)
    agreement.to_csv(DEST/'original_candidate_agreement.csv',index=False,encoding='utf-8-sig')
    metric=pd.read_csv(DEST/'metric_restoration.csv');scope=pd.read_csv(DEST/'landmark_scope.csv')
    assert len(metric)==340 and metric.groupby('metric').size().eq(85).all()
    assert len(scope)==85*16 and not scope.anatomically_confirmed.any()
    assert len(agreement)==170
    for _,r in metric.iterrows():
        if r.status=='candidate_stable_metric':
            assert r.adequate and pd.notna(r.value)
            assert r.low-1e-8<=r.value<=r.high+1e-8
            assert r.high-r.low<=ACCEPT['metric_width'][r.metric]+1e-8
        elif r.status=='candidate_interval_only':
            assert r.adequate and pd.isna(r.value)
            assert r.high-r.low>ACCEPT['metric_width'][r.metric]
        elif r.status=='automatic_withhold':assert pd.isna(r.value)
    warnings={'P_1','P_9','P_11','P_27','P_41','P_81'}
    assert metric[metric.Patient.isin(warnings)].status.eq('existing_value_retained_unverified').all()
    point=pd.read_csv(DEST/'point_stability.csv')
    focus=point[point.triggered]
    assert focus.Patient.nunique()==5 and len(focus)==6
    summary=metric.groupby(['metric','status']).size().rename('n').reset_index()
    scope_summary=scope.groupby(['landmark','scope']).size().rename('n').reset_index()
    notes=dict(patients=85,Po_points=170,conditions_per_point=16,total_condition_runs=2720,
               metric_rows=340,landmark_scope_rows=1360,untested_landmark_types=14,
               all_landmarks_confirmed=False,all_candidates_anatomically_confirmed=False,
               all85_retained_in_evaluation=True,withheld_values_are_blank=True,
               disagreement_does_not_auto_replace_original=True,
               original_landmark_hashes_unchanged=True,prior_segmentation_protocol_unchanged=True,
               unit_tests=10,Pog_refinement_equivalence_checked_on='P_16',
               limitations=['variants are correlated; not confidence intervals',
                            'two families share image, ROI and tube-shape prior',
                            'existing unchanged points are not confirmed by this experiment'])
    (DEST/'verification.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    extra=f'''<!-- STABILITY_SCOPE_AUDIT_START --><h2>0. 이번 실험이 확정하는 것과 확정하지 않는 것</h2>
<p>확정한 것은 <b>고정된 자동 반환 절차</b>입니다. 기준점의 해부학적 정답을 확정한 것이 아닙니다. RPo·LPo 170개에서 16조건씩 2720회 후보 생성을 실행했습니다. 나머지 14종 기준점은 이번 실험의 대상이 아닙니다. 아래 기존값 유지도 검증 완료로 해석하면 안 됩니다.</p>
<h3>지표별 실제 반환 종류</h3>{table(summary)}
<h3>복구 분기가 발동한 5명·6개 Po</h3>{table(focus)}
<h3>기준점별 이번 평가 범위</h3>{table(scope_summary)}
<p>기존 점과 안정 후보의 차이가 3 mm를 넘는 경우도 경고로 기록했으며 그것만으로 기존 점을 바꾸지 않았습니다. 단위·부위가 다른 점의 정확도를 같은 문턱으로 보장할 수 없습니다. 이 차이는 점 오류가 확정됐다는 뜻이 아닙니다.</p>
<p><a href="original_candidate_agreement.csv">기존점과 안정 후보의 거리</a> · <a href="landmark_scope.csv">85명×16점 평가 범위</a> · <a href="verification.json">실행 검증</a></p>
<!-- STABILITY_SCOPE_AUDIT_END -->'''
    path=DEST/'report.html';html=path.read_text(encoding='utf-8')
    if '<!-- STABILITY_SCOPE_AUDIT_START -->' in html:
        a,b=html.split('<!-- STABILITY_SCOPE_AUDIT_START -->',1);html=a+b.split('<!-- STABILITY_SCOPE_AUDIT_END -->',1)[1]
    html=html.replace('<h2>1. 고정 절차</h2>',extra+'<h2>1. 고정 절차</h2>')
    path.write_text(html,encoding='utf-8')
    refs=re.findall(r'(?:href|src)="([^"]+)"',html)
    assert all((DEST/x).exists() for x in refs if not x.startswith('#'))
    print(focus.to_string(index=False));print(summary.to_string(index=False))
    print('Verified 85 patients / 170 Po / 2720 conditions; no anatomical confirmation claimed')


if __name__=='__main__':main()
