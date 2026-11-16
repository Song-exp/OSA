"""통합 FH 워크플로의 범위·반환 계약·Po 회귀 재현과 입력 불변을 검증한다."""
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from cbct_fh_joint85 import DEST,SOURCE,KEYS,METRICS,required_points
from cbct_po_ensemble import ACCEPT
from cbct_patient_pipeline import digest
import cbct_po_stability85 as po


def main():
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    assert len(protocol['patients'])==85
    assert digest(Path(__file__).with_name('cbct_fh_joint85.py'))==protocol['driver']
    assert digest(Path(__file__).with_name('cbct_or_ensemble.py'))==protocol['Or_algorithm']
    assert digest(SOURCE/'protocol.json')==protocol['source_protocol']
    points=pd.read_csv(DEST/'point_stability.csv');variants=pd.read_csv(DEST/'variant_candidates.csv')
    metric=pd.read_csv(DEST/'metric_restoration.csv');scope=pd.read_csv(DEST/'landmark_scope.csv')
    assert len(points)==340 and points.Patient.nunique()==85
    assert len(variants)==5440 and variants.groupby(['Patient','point']).size().eq(16).all()
    assert len(metric)==85*3*4 and metric.groupby(['mode','metric']).size().eq(85).all()
    assert len(scope)==1360 and not scope.anatomically_confirmed.any()
    assert scope[scope.scope=='not_assessed_in_this_workflow'].landmark.nunique()==12
    for _,r in metric.iterrows():
        if r.status=='candidate_stable_metric':
            assert r.adequate and pd.notna(r.value)
            assert r.low-1e-8<=r.value<=r.high+1e-8
            assert r.high-r.low<=ACCEPT['metric_width'][r.metric]+1e-8
        elif r.status=='candidate_interval_only':
            assert r.adequate and pd.isna(r.value)
            assert r.high-r.low>ACCEPT['metric_width'][r.metric]
        elif r.status=='automatic_withhold':assert pd.isna(r.value)
        assert set(str(r.active_points).split(';'))-{'nan'}<=set(required_points(r.metric))
    # 기존 Po 단계가 통합 과정에서 달라지지 않았는지 같은 행끼리 대조한다.
    old=pd.read_csv(po.DEST/'metric_restoration.csv').set_index(['Patient','metric']).sort_index()
    new=metric[metric['mode']=='Po_only'].set_index(['Patient','metric']).sort_index()
    assert old.index.equals(new.index) and old.status.equals(new.status)
    for col in ('value','low','high'):
        assert np.allclose(old[col],new[col],atol=1e-8,equal_nan=True),col
    for p in protocol['patients']:
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        oc=json.loads((DEST/'or_cases'/f'{p}.json').read_text(encoding='utf-8'))
        assert c['source_hashes']['Po']==digest(po.DEST/'cases'/f'{p}.json')
        assert c['source_hashes']['Or']==digest(DEST/'or_cases'/f'{p}.json')
        assert oc['fingerprint']['code']==protocol['driver']
        assert oc['fingerprint']['algorithm']==protocol['Or_algorithm']
        assert oc['fingerprint']['Po_case']==c['source_hashes']['Po']
    summary=metric[metric['mode']=='joint'].groupby(['metric','status']).size().rename('n').reset_index()
    scope_summary=scope.groupby(['landmark','scope']).size().rename('n').reset_index()
    stable=points[points.point.str.endswith('Or')].groupby(['point','stability_status']).size().rename('n').reset_index()
    disagreement=points[points.point.str.endswith('Or')].groupby('point').agg(
        mean_mask_success=('mask_success_fraction','mean'),mean_native_success=('native_success_fraction','mean'))
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    extra=f'''<!-- FH_JOINT_AUDIT_START --><h2>0. 실행 검증: Po 재현과 Or 근거를 분리</h2>
<p>Po 170개의 기존 16조건 결과를 입력 해시로 검증해 재사용하고, Or 170개를 16조건에서 평가했습니다. 총 5440조건 기록이 있으며, Po 단독 모드의 값·범위·보류 상태는 이전 Po 실험과 일치했습니다.</p>
<h3>통합 모드 반환 상태</h3>{table(summary)}
<h3>Or 자체의 안정성</h3>{table(stable)}
<p>분할 기반 후보가 반복 조건에서 안정적이어도 원본 영상 후보가 뒷받침하지 않는 경우가 있습니다. 숫자 반환과 Or 위치의 정확도 확인을 같은 것으로 해석하지 않습니다.</p>
<div class="scroll">{table(disagreement.reset_index())}</div>
<h3>기준점별 이번 평가 범위</h3>{table(scope_summary)}
<p><a href="verification.json">실행 검증</a> · <a href="rules_frozen.json">재현용 규칙·코드·출력 해시</a> · <a href="README.md">실행 방법과 입력·출력</a></p><!-- FH_JOINT_AUDIT_END -->'''
    page=DEST/'report.html';html=page.read_text(encoding='utf-8')
    if '<!-- FH_JOINT_AUDIT_START -->' in html:
        a,b=html.split('<!-- FH_JOINT_AUDIT_START -->',1);html=a+b.split('<!-- FH_JOINT_AUDIT_END -->',1)[1]
    html=html.replace('<h2>1. Or 탐색 보완</h2>',extra+'<h2>1. Or 탐색 보완</h2>')
    page.write_text(html,encoding='utf-8')
    notes=dict(patients=85,FH_points=340,condition_records=5440,Po_only_matches_previous_experiment=True,
               landmark_scope_rows=1360,not_independently_assessed_landmark_types=12,
               all_landmarks_anatomically_confirmed=False,withheld_values_are_blank=True,
               dependencies_are_metric_specific=True,tests=16)
    (DEST/'verification.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    frozen=dict(protocol=protocol,runtime={p:version(p) for p in ('numpy','scipy','SimpleITK','scikit-image','matplotlib')},
                source_cases={p:json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))['source_hashes'] for p in protocol['patients']},
                outputs={n:digest(DEST/n) for n in ('point_stability.csv','variant_candidates.csv','metric_restoration.csv','evaluation.csv','landmark_scope.csv','report.html')},
                freeze_meaning='reproducible experimental rules, not anatomical confirmation or clinical validation')
    (DEST/'rules_frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    refs=re.findall(r'(?:href|src)="([^"]+)"',html)
    assert all((DEST/r).exists() for r in refs if not r.startswith('#'))
    print(summary.to_string(index=False));print(stable.to_string(index=False))
    print('Verified 85 patients / 340 FH points / 5440 condition records; Po regression reproduced')


if __name__=='__main__':main()
