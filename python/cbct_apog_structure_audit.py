"""구조 게이트 단계의 후보 부분집합·분기 고정·전수 실행·입력 불변 감사."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import cbct_apog_structure85 as driver
from cbct_patient_pipeline import CB,ROOT,landmarks,digest
from cbct_po_ensemble import ACCEPT

DEST=driver.DEST


def main():
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    assert len(protocol['patients'])==85
    assert protocol['code']==digest(Path(driver.__file__))
    assert protocol['engine']==digest(Path(driver.engine.__file__))
    assert protocol['dependencies']==driver.dependencies()
    assert protocol['previous_protocol']==digest(driver.SOURCE/'protocol.json')
    for name,h in protocol['previous_outputs'].items():assert h==digest(driver.SOURCE/name)
    dataset=ROOT/'models/downloads_teeth/UNIVERSALLAB_MODEL/dataset.json'
    assert protocol['teeth_dataset']==digest(dataset)
    labels=json.loads(dataset.read_text(encoding='utf-8'))['labels']
    assert labels['Mandible']==53 and labels['Maxilla']==54 and labels['Mandibular canal']==55
    points=pd.read_csv(DEST/'point_stability.csv');variants=pd.read_csv(DEST/'variant_candidates.csv')
    metric=pd.read_csv(DEST/'metric_restoration.csv');ev=pd.read_csv(DEST/'structural_evidence.csv')
    assert len(points)==170 and points.Patient.nunique()==85
    assert len(variants)==4080 and variants.groupby(['Patient','point']).size().eq(24).all()
    assert len(metric)==1360 and metric.groupby(['mode','metric']).size().eq(85).all()
    scope=pd.read_csv(DEST/'landmark_scope.csv');assert len(scope)==1360 and not scope.anatomically_confirmed.any()
    count=0;rejected=0;affected=[]
    for p in protocol['patients']:
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        old=json.loads((driver.SOURCE/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        fp=c['structure_fingerprint'];inputs=c['fingerprint']
        assert fp['code']==protocol['code'] and fp['engine']==protocol['engine']
        assert fp['dependencies']==protocol['dependencies']
        assert fp['source_case']==digest(driver.SOURCE/'cases'/f'{p}.json')
        assert fp['teeth']==digest(driver.TEETH/f'{p}.nii.gz')
        assert protocol['FH_cases'][p]==digest(driver.base.FH_SOURCE/'cases'/f'{p}.json')
        assert inputs['image']==digest(CB/f'nrrd/{p}.nrrd')
        assert inputs['landmarks']=={f.name:digest(f) for f in landmarks(int(p[2:]))[1]}
        for s in ('MAX','MAND'):
            assert inputs['segmentations'][s]==digest(CB/f'seg/{p}_seg_{s}.nrrd')
            assert inputs['models'][s]==digest(ROOT/f'Result_py/shape/{s}/points.npz')
        for pid,hashes in inputs['reference_annotations'].items():
            assert hashes=={f.name:digest(f) for f in landmarks(int(pid))[1]}
        for key,r in c['points'].items():
            previous=old['points'][key]
            assert r['triggered']==previous['triggered'] and r['reasons']==previous['reasons']
            if not r['triggered']:assert r['route']==previous['route']
            assert p not in r['transfer']['reference_ids']
            assert fp['source_roi'][key]==digest(driver.SOURCE/'roi'/f'{p}_{key}.npz')
            for run,before in zip(r['variants'],previous['variants']):
                assert run['variant']==before['variant']
                assert len(run['structural_checks'])==len(before['eligible'])
                count+=len(run['structural_checks'])
                expected=[tuple(x['point']) for x in run['structural_checks'] if x['accepted']]
                assert [tuple(x['point']) for x in run['eligible']]==expected
                assert set(expected)<=set(tuple(x['point']) for x in before['eligible'])
                assert run['winner']==(run['eligible'][0]['point'] if run['eligible'] else None)
                for x in run['structural_checks']:
                    assert x['accepted']==(not x['reasons'])
                    rejected+=not x['accepted']
            if r['structural_rejected']:affected.append(dict(Patient=p,point=key,triggered=r['triggered'],
                previous_route=previous['route'],route=r['route'],rejected=r['structural_rejected']))
    assert count==len(ev) and rejected==int((~ev.accepted).sum())
    for _,r in metric.iterrows():
        if r.status=='candidate_stable_metric':
            assert r.adequate and r.low-1e-8<=r.value<=r.high+1e-8
            assert r.high-r.low<=ACCEPT['metric_width'][r.metric]+1e-8
        elif r.status=='candidate_interval_only':assert pd.isna(r.value) and r.adequate
        elif r.status=='automatic_withhold':assert pd.isna(r.value)
    previous=pd.read_csv(driver.SOURCE/'metric_restoration.csv')
    for mode,unaffected in [('A_only',['Nperp-Pog','MP-FH']),('Pog_only',['ANB','Nperp-A','MP-FH']),('APog_only',['MP-FH'])]:
        g=metric[(metric['mode']==mode)&metric.metric.isin(unaffected)]
        assert (g.status=='existing_value_retained_unverified').all() and np.allclose(g.value,g.legacy_value)
    a=metric[metric.metric=='MP-FH'].set_index(['Patient','mode']).sort_index()
    b=previous[previous.metric=='MP-FH'].set_index(['Patient','mode']).sort_index()
    assert a.status.equals(b.status)
    for column in ('value','low','high'):assert np.allclose(a[column],b[column],equal_nan=True)
    affected=pd.DataFrame(affected);affected.to_csv(DEST/'affected_points.csv',index=False,encoding='utf-8-sig')
    reasons=ev[~ev.accepted].assign(reason=lambda x:x.reasons.str.split(';')).explode('reason')
    reason_patients=reasons.groupby(['point','reason']).agg(records=('Patient','size'),patients=('Patient','nunique')).reset_index()
    reason_patients.to_csv(DEST/'rejection_patients.csv',index=False,encoding='utf-8-sig')
    notes=dict(patients=85,points=170,variant_records=4080,candidate_records=count,rejected_records=rejected,
               candidate_subset_only=True,v2_trigger_branches_frozen=True,inputs_unchanged=True,
               previous_outputs_unchanged=True,MPFH_reproduced=True,tests=9,anatomically_confirmed=False)
    (DEST/'verification.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'README.md').write_text('''# A/Pog 구조 게이트 1단계

실행: `.venv\\Scripts\\python.exe -X utf8 python/cbct_apog_structure85.py`
감사: `.venv\\Scripts\\python.exe -X utf8 python/cbct_apog_structure_audit.py`
테스트: `.venv\\Scripts\\python.exe -m unittest discover -s python -p test_cbct_apog_structure.py`

v2 후보·교체 분기를 고정해 구조 게이트만 추가한다. UNIVERSALLAB 라벨 1..52만 치아로 보고 53 하악·54 상악·55 하악관은 제외한다. 원본 CT의 피질 대비, 마스크 안쪽/바깥쪽 관계, Pog 하연 거리와 잘림을 평가한다. 치아 분할의 과거 큰 절단창은 원래 Pog·Me를 사용했으므로 독립적인 정답 근거가 아니다. 후보 위치의 저장 범위를 검사해 바깥은 알 수 없음으로 거부한다.

기존 유지 분기는 미검증 상태 그대로다. 치아가 아닌 조건을 통과했다고 치조부·피질의 올바른 지점이 확정되는 것은 아니다. 이번 단계는 신규 후보나 해부학 의미 분할 모델을 만들지 않았다. 나머지 Po/Or·10종 기준점은 이 단계에서 수정하지 않는다. 85명은 모두 개발 자료이고 엑셀은 선별 후 비교에만 쓴다. 제공률과 같은 환자에서의 오차를 함께 본다. 기존 ALI·운영 계측값과 이전 결과는 보존한다.
''',encoding='utf-8')
    page=DEST/'report.html';html=page.read_text(encoding='utf-8')
    decision='''<h2>판단: 후보 거부 근거 확보, 운영 규칙은 채택하지 않음</h2>
<p>조건별 후보 기록 4199건 중 414건(A 83·Pog 331)을 거부했습니다. 23명의 25개 점에서 하나 이상의 후보가 제거됐으나 재탐색 분기가 활성인 것은 이 중 6개 점·58건입니다. 기존 유지점의 정확도는 이번에 검증한 것이 아닙니다.</p>
<p>FH 결합 숫자 제공 인원은 ANB 79명, MP-FH 83명, Nperp-A 80명, Nperp-Pog 50명으로 v2와 같았습니다. 같은 환자에서 Nperp-Pog MAE는 3.978903→3.977934 mm, Nperp-A는 1.724130→1.724263 mm로 거의 같았습니다. 단순 치아·하연·피질 조건만으로 남은 오차를 충분히 설명하지 못했습니다.</p>
<p>다음 단일 작업은 구조 조건을 통과한 후보의 기준축·계측 정의 차이를 분리 진단하는 것입니다. 치조부 전체 구분이나 기준점 확정을 완료한 결과가 아닙니다. CT 밝기는 보정된 HU가 아니며 기준값을 엑셀 오차에 맞춰 조정하지 않았습니다.</p>'''
    extra=decision+'<h2>전수 실행 검증</h2>'+reason_patients.to_html(index=False,border=0)+affected.to_html(index=False,border=0)
    block='<!-- STRUCTURE_AUDIT_START -->'+extra+'<p><a href="verification.json">실행 검증</a> · <a href="rejection_patients.csv">사유별 환자 수</a> · <a href="affected_points.csv">후보가 제거된 점</a> · <a href="rules_frozen.json">재현 기록</a> · <a href="README.md">실행 방법</a></p><!-- STRUCTURE_AUDIT_END -->'
    html=re.sub(r'<!-- STRUCTURE_AUDIT_START -->.*?<!-- STRUCTURE_AUDIT_END -->','',html,flags=re.S)
    html=html.replace('<h2>점별 상태</h2>',block+'<h2>점별 상태</h2>');page.write_text(html,encoding='utf-8')
    outputs=('point_stability.csv','variant_candidates.csv','metric_restoration.csv','structural_evidence.csv','evaluation.csv',
             'comparison_previous.csv','rejection_reasons.csv','rejection_patients.csv','affected_points.csv','report.html')
    frozen=dict(protocol=protocol,audit_code=digest(__file__),case_hashes={p:digest(DEST/'cases'/f'{p}.json') for p in protocol['patients']},
                outputs={n:digest(DEST/n) for n in outputs})
    (DEST/'rules_frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all((DEST/r).exists() for r in re.findall(r'(?:href|src)="([^"]+)"',html) if not r.startswith('#'))
    print(json.dumps(notes,ensure_ascii=False,indent=2));print(affected.to_string(index=False));print(reason_patients.to_string(index=False))


if __name__=='__main__':main()
