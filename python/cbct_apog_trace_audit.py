"""연속 윤곽 실험의 전수 실행·입력 불변·요인 분리·계측 계약 검증."""
from importlib.metadata import version
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import cbct_apog_trace85 as driver
from cbct_patient_pipeline import CB,ROOT,landmarks,digest
from cbct_po_ensemble import ACCEPT

DEST=driver.DEST


def main():
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    assert len(protocol['patients'])==85
    assert protocol['code']==digest(Path(driver.__file__))
    assert protocol['engine']==digest(Path(driver.engine.__file__))
    assert protocol['dependencies']==driver.dependency_hashes()
    assert protocol['previous_protocol']==digest(driver.SOURCE/'protocol.json')
    assert protocol['FH_protocol']==digest(driver.FH_SOURCE/'protocol.json')
    for name,h in protocol['previous_outputs'].items():assert h==digest(driver.SOURCE/name)
    points=pd.read_csv(DEST/'point_stability.csv');variants=pd.read_csv(DEST/'variant_candidates.csv')
    metric=pd.read_csv(DEST/'metric_restoration.csv');scope=pd.read_csv(DEST/'landmark_scope.csv')
    factors=pd.read_csv(DEST/'factor_sensitivity.csv')
    assert len(points)==170 and points.Patient.nunique()==85
    assert len(variants)==4080 and variants.groupby(['Patient','point']).size().eq(24).all()
    assert len(metric)==1360 and metric.groupby(['mode','metric']).size().eq(85).all()
    assert len(scope)==1360 and not scope.anatomically_confirmed.any()
    assert scope[scope.scope=='not_assessed'].landmark.nunique()==10
    assert len(factors)==2040
    assert factors.groupby(['Patient','point','family','angle_deg','window']).size().eq(1).all()
    for _,r in metric.iterrows():
        allowed=set(driver.base.required_points(r.metric))
        if driver.base.target_for(r.metric):allowed.add(driver.base.target_for(r.metric))
        assert set(str(r.active_points).split(';'))-{'nan'}<=allowed
        if r.status=='candidate_stable_metric':
            assert r.adequate and pd.notna(r.value)
            assert r.low-1e-8<=r.value<=r.high+1e-8
            assert r.high-r.low<=ACCEPT['metric_width'][r.metric]+1e-8
        elif r.status=='candidate_interval_only':
            assert r.adequate and pd.isna(r.value)
            assert r.high-r.low>ACCEPT['metric_width'][r.metric]
        elif r.status=='automatic_withhold':assert pd.isna(r.value)
    for mode,unaffected in [('A_only',['Nperp-Pog','MP-FH']),('Pog_only',['ANB','Nperp-A','MP-FH']),('APog_only',['MP-FH'])]:
        g=metric[(metric['mode']==mode)&metric.metric.isin(unaffected)]
        assert (g.status=='existing_value_retained_unverified').all()
        assert np.allclose(g.value,g.legacy_value,atol=1e-8)
    old=pd.read_csv(driver.FH_SOURCE/'metric_restoration.csv')
    old=old[(old['mode']=='joint')&(old.metric=='MP-FH')].set_index('Patient').sort_index()
    new=metric[(metric['mode']=='joint')&(metric.metric=='MP-FH')].set_index('Patient').sort_index()
    assert old.index.equals(new.index) and old.status.equals(new.status)
    for col in ('value','low','high'):assert np.allclose(old[col],new[col],atol=1e-8,equal_nan=True)
    for p in protocol['patients']:
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'));fp=c['fingerprint']
        assert fp['code']==protocol['code'] and fp['engine']==protocol['engine']
        assert fp['dependencies']==protocol['dependencies']
        assert fp['previous_case']==digest(driver.SOURCE/'cases'/f'{p}.json')
        assert protocol['source_FH_cases'][p]==digest(driver.FH_SOURCE/'cases'/f'{p}.json')
        assert fp['image']==digest(CB/f'nrrd/{p}.nrrd')
        assert fp['landmarks']=={f.name:digest(f) for f in landmarks(int(p[2:]))[1]}
        for s in ('MAX','MAND'):
            assert fp['segmentations'][s]==digest(CB/f'seg/{p}_seg_{s}.nrrd')
            assert fp['models'][s]==digest(ROOT/f'Result_py/shape/{s}/points.npz')
        for pid,hashes in fp['reference_annotations'].items():
            assert hashes=={f.name:digest(f) for f in landmarks(int(pid))[1]}
        for key,r in c['points'].items():
            assert p not in r['transfer']['reference_ids']
            assert len(r['variants'])==24
            by_name={v['variant']:v for v in r['variants']}
            all_candidates=[]
            for spec in protocol['variants']:
                v=by_name[spec['name']]
                assert v['angle_deg']==spec['angle_deg'] and v['window']==spec['window']
                all_candidates.extend(x['point'] for x in v['eligible'])
            actual=np.unique(np.round(np.asarray(all_candidates).reshape(-1,3),6),axis=0) if all_candidates else np.empty((0,3))
            np.testing.assert_allclose(np.asarray(r['stability']['candidates']).reshape(-1,3),actual,atol=1e-6)
            assert (DEST/'roi'/f'{p}_{key}.npz').exists()
            assert (DEST/'figs'/f'{p}_{key}.png').exists()
    contrasts=[]
    for p in protocol['patients']:
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        for key,r in c['points'].items():
            for family in ('AMASSS','native_CT'):
                runs={}
                for v,spec in zip(r['variants'],protocol['variants']):
                    if v['family']==family and (family=='AMASSS' or spec['otsu_delta']==0):
                        runs[(v['angle_deg'],v['window'])]=v
                pairs=[('axis',runs[(0.,window)],runs[(angle,window)]) for window in ('base','expanded') for angle in (-5.,5.)]
                pairs += [('window',runs[(angle,'base')],runs[(angle,'expanded')]) for angle in (-5.,0.,5.)]
                for factor,a,b in pairs:
                    both=a['winner'] is not None and b['winner'] is not None
                    contrasts.append(dict(Patient=p,point=key,family=family,factor=factor,
                                from_angle=a['angle_deg'],to_angle=b['angle_deg'],from_window=a['window'],to_window=b['window'],
                                from_success=a['winner'] is not None,to_success=b['winner'] is not None,
                                paired_distance_mm=float(np.linalg.norm(np.asarray(a['winner'])-b['winner'])) if both else None))
    contrasts=pd.DataFrame(contrasts);assert len(contrasts)==2380
    contrasts.to_csv(DEST/'factor_contrasts.csv',index=False,encoding='utf-8-sig')
    notes=dict(patients=85,points=170,conditions=4080,metric_rows=1360,factor_rows=2040,factor_contrasts=2380,
               all_eligible_candidates_included=True,self_reference_excluded=True,inputs_unchanged=True,
               previous_outputs_unchanged=True,MPFH_reproduced=True,withheld_scalar_values_blank=True,
               all_landmarks_anatomically_confirmed=False,not_assessed_types=10)
    (DEST/'verification.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'README.md').write_text('''# A/Pog 연속 윤곽 v2

실행: `.venv\\Scripts\\python.exe -X utf8 python/cbct_apog_trace85.py`
검증: `.venv\\Scripts\\python.exe -X utf8 python/cbct_apog_trace_audit.py`
단위 테스트: `.venv\\Scripts\\python.exe -m unittest discover -s python -p test_cbct_apog_trace.py`

기록 시작점: `docs/CBCT_계측기준_개선기록.md`. 85명 전원이 개발 자료다. 운영 계측과 원본 ALI는 바꾸지 않는다. 참고 ALI는 정답이 아닌 독립 ROI 안내이며 본인을 참고에서 제외한다. 현재 A/Pog·Me·대칭·엑셀은 후보 생성에 쓰지 않는다. 원래 점은 후보 생성 후 교체 필요 판단에만 쓴다.

최대 연결 단면 구조의 전방 경계를 동적 계획법으로 잇고, 4 mm 양쪽 지지와 0.5 mm 돌출/오목 조건을 적용한다. 축 ±5/0도, 기본/확장 창, AMASSS/원본 Otsu 3조건의 완전 교차 설계다. 모든 경쟁 후보를 반환한다. 기준축·창 폭·추적·지지 길이가 함께 바뀌므로 v1 대비 변화 전체를 추적 단독 효과로 볼 수 없다. 이 실험 내 축/창 민감도는 원본 Otsu 보정량을 0으로 고정해 별도 기록한다. 창이 바뀌면 Otsu 임계값 자체는 바뀔 수 있어 창 효과에는 임계값 변화도 포함된다.

CBCT 원본 임계값은 보정된 HU가 아니다. 치아·금속·부분 FOV·연결 구조 선택의 영향을 받을 수 있다. N/S/Ba/ANS/PNS 기준축 자체는 아직 별도 검증하지 않았다. 반복 안정성은 해부학적 정확도가 아니다. 후보 영역에서 계측값 범위가 좁아 숫자를 내는 것과 단일 좌표 확정은 다르다.

후보와 판단을 정한 뒤에만 엑셀을 비교한다. 제공 인원과 같은 환자에서의 오차를 함께 본다. 값 없는 환자도 전체 85명 분모에 남긴다. 4096개 초과 조합은 임의 축소하지 않고 보류한다. 기존 FH 단계의 후보 불확실성은 joint에만 전파한다. 단독 모드와 무관한 계측값, MP-FH의 이전 결과 재현을 검증한다.

각 환자의 2개 ROI와 실제 단면 그림을 모두 저장했다. 보고서는 변경 분기와 고정 예시를 보여주며 CSV에 전원이 있다. 후보 범위는 민감도 범위로 신뢰구간이 아니다. 해시 고정은 재현 기록이며 측정법 채택이나 임상 검증 선언이 아니다.
''',encoding='utf-8')
    page=DEST/'report.html';html=page.read_text(encoding='utf-8')
    caveat='원본 비교에서는 Otsu 보정량을 0으로 고정했으며, 창이 바뀌면 임계값 자체는 바뀔 수 있습니다.'
    if caveat not in html:html=html.replace('이 실험 내 창·축 요인은 각각 분리해 기록했습니다.',
        '이 실험 내 창·축 요인은 각각 분리해 기록했습니다. '+caveat)
    contrast_summary=contrasts.groupby(['point','family','factor']).agg(
        n=('Patient','size'),from_success=('from_success','sum'),to_success=('to_success','sum'),
        mean_paired_shift_mm=('paired_distance_mm','mean'),max_paired_shift_mm=('paired_distance_mm','max')).reset_index()
    finding='''<!-- TRACE_AUDIT_START --><h2>판단: 탐색은 개선됐으나 전원 적용은 채택하지 않음</h2>
<p>v1 대비 A 보류는 10→3개, Pog 보류는 39→34개로 줄었습니다. A 4개(P_15·23·44·69), Pog 2개(P_29·69)는 이 실험의 단일 좌표 안정성 조건을 만족하는 교체 후보입니다. 나머지 교체 후보 영역은 A 5개·Pog 5개입니다. 이는 정답 확인이 아닙니다.</p>
<p>FH와 결합한 숫자 제공은 v1 대비 ANB 72→79명, Nperp-A 73→80명, Nperp-Pog 46→50명입니다. 두 방식 모두 숫자를 제공하는 환자에서 MAE는 ANB 0.9704→0.9710°, Nperp-A 1.7146→1.7132 mm, Nperp-Pog 4.2540→4.2553 mm로 거의 같았습니다. 전수 적용할 만큼의 일관된 오차 개선을 얻지 못했습니다.</p>
<p>계측 분리 모드에서 P_15 ANB 오차는 0.013→1.028°, P_23 Nperp-Pog는 0.130→4.582 mm로 증가하는 반면 다른 표본은 개선됐습니다. 안정적으로 좁은 후보 범위를 얻어도 엑셀과 일치하는 값이라는 보장은 없습니다. P_35는 Pog 단독으로 숫자를 내도 FH 결합에서는 보류되므로 FH 오차와 대상점 오차를 분리해야 합니다.</p>
<p>AMASSS에서 축 0° A 검출은 기본창 25/85→확장창 39/85였지만 Pog는 21/85→11/85였습니다. 기본창 Pog 검출은 -5° 34/85, 0° 21/85, +5° 13/85로 달라졌습니다. 단순 창 확장이나 임의의 유리한 축 선택을 전원 규칙으로 삼지 않습니다. 아래는 같은 창에서 축만, 같은 축에서 창만 바꾼 비교입니다. 한 환자가 여러 대비에 포함되며 n은 환자 수가 아닌 비교 횟수입니다.</p>
'''+contrast_summary.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'''
<p><a href="factor_contrasts.csv">요인 하나씩 바꾼 2380개 비교</a>. 다음에는 ROI 내부에서 전방 피질과 치조부·하연을 분리하는 구조 조건과 기준축 의존성을 다룹니다. 현재 좌표·운영 계측값은 유지하고 이번 방법은 연구 후보로 보존합니다.</p><!-- TRACE_AUDIT_END -->'''
    html=re.sub(r'<!-- TRACE_AUDIT_START -->.*?<!-- TRACE_AUDIT_END -->','',html,flags=re.S)
    html=html.replace('<h2>점별 반환 상태</h2>',finding+'<h2>점별 반환 상태</h2>')
    links='<p><a href="verification.json">전수 실행 검증</a> · <a href="rules_frozen.json">입력·코드·출력 해시</a> · <a href="README.md">실행 방법과 한계</a></p>'
    if links not in html:html=html.replace('</main>',links+'</main>')
    page.write_text(html,encoding='utf-8')
    frozen=dict(protocol=protocol,runtime={p:version(p) for p in ('numpy','scipy','SimpleITK','scikit-image','matplotlib')},
                case_hashes={p:digest(DEST/'cases'/f'{p}.json') for p in protocol['patients']},
                outputs={n:digest(DEST/n) for n in ('point_stability.csv','variant_candidates.csv','metric_restoration.csv','evaluation.csv',
                         'landmark_scope.csv','factor_sensitivity.csv','factor_contrasts.csv','comparison_previous.csv','failure_modes.csv','report.html')})
    (DEST/'rules_frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all((DEST/r).exists() for r in re.findall(r'(?:href|src)="([^"]+)"',html) if not r.startswith('#'))
    print(json.dumps(notes,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
