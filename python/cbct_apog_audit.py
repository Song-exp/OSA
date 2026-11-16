"""A/Pog 실험의 입력 불변·계측 의존성·출력 계약을 별도로 검증한다."""
from importlib.metadata import version
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from cbct_apog85 import DEST,FH_SOURCE,TARGETS,target_for,required_points
from cbct_patient_pipeline import digest,CB,ROOT,landmarks
from cbct_po_ensemble import ACCEPT


def main():
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    assert len(protocol['patients'])==85
    assert digest(Path(__file__).with_name('cbct_apog85.py'))==protocol['code']
    assert digest(Path(__file__).with_name('cbct_apog_ensemble.py'))==protocol['engine']
    assert digest(FH_SOURCE/'protocol.json')==protocol['FH_source_protocol']
    points=pd.read_csv(DEST/'point_stability.csv');variants=pd.read_csv(DEST/'variant_candidates.csv')
    metric=pd.read_csv(DEST/'metric_restoration.csv');scope=pd.read_csv(DEST/'landmark_scope.csv')
    assert len(points)==170 and points.Patient.nunique()==85
    assert len(variants)==2720 and variants.groupby(['Patient','point']).size().eq(16).all()
    assert len(metric)==1360 and metric.groupby(['mode','metric']).size().eq(85).all()
    assert len(scope)==1360 and not scope.anatomically_confirmed.any()
    assert scope[scope.scope=='not_assessed'].landmark.nunique()==10
    for _,r in metric.iterrows():
        allowed=set(required_points(r.metric))
        if target_for(r.metric):allowed.add(target_for(r.metric))
        assert set(str(r.active_points).split(';'))-{'nan'}<=allowed
        if r.status=='candidate_stable_metric':
            assert r.adequate and pd.notna(r.value)
            assert r.low-1e-8<=r.value<=r.high+1e-8
            assert r.high-r.low<=ACCEPT['metric_width'][r.metric]+1e-8
        elif r.status=='candidate_interval_only':
            assert r.adequate and pd.isna(r.value)
            assert r.high-r.low>ACCEPT['metric_width'][r.metric]
        elif r.status=='automatic_withhold':assert pd.isna(r.value)
    # A/Pog 변경이 무관한 계측에 영향을 주지 않아야 한다.
    for mode,unaffected in [('A_only',['Nperp-Pog','MP-FH']),('Pog_only',['ANB','Nperp-A','MP-FH']),('APog_only',['MP-FH'])]:
        g=metric[(metric['mode']==mode)&metric.metric.isin(unaffected)]
        assert (g.status=='existing_value_retained_unverified').all()
        assert np.allclose(g.value,g.legacy_value,atol=1e-8)
    old=pd.read_csv(FH_SOURCE/'metric_restoration.csv')
    old=old[(old['mode']=='joint')&(old.metric=='MP-FH')].set_index('Patient').sort_index()
    new=metric[(metric['mode']=='joint')&(metric.metric=='MP-FH')].set_index('Patient').sort_index()
    assert old.index.equals(new.index) and old.status.equals(new.status)
    for col in ('value','low','high'):assert np.allclose(old[col],new[col],atol=1e-8,equal_nan=True)
    previous=pd.read_csv(FH_SOURCE/'metric_restoration.csv')
    previous=previous[previous['mode']=='joint'].set_index(['Patient','metric'])
    joint=metric[metric['mode']=='joint'].set_index(['Patient','metric'])
    joined=joint.join(previous[['value','output_error']].rename(columns={'value':'previous_value','output_error':'previous_error'}))
    comparisons=[]
    for m,g in joined.groupby(level='metric'):
        same=g[g.value.notna()&g.previous_value.notna()]
        comparisons.append(dict(metric=m,total_n=len(g),previous_numeric_n=int(g.previous_value.notna().sum()),
                                new_numeric_n=int(g.value.notna().sum()),same_numeric_n=len(same),
                                same_previous_MAE=same.previous_error.mean(),same_new_MAE=same.output_error.mean()))
    comparisons=pd.DataFrame(comparisons)
    comparisons.to_csv(DEST/'comparison_previous_joint.csv',index=False,encoding='utf-8-sig')
    failures=variants.groupby(['point','family','status']).size().rename('n').reset_index()
    failures.to_csv(DEST/'failure_modes.csv',index=False,encoding='utf-8-sig')
    for p in protocol['patients']:
        assert digest(FH_SOURCE/'cases'/f'{p}.json')==protocol['source_FH_cases'][p]
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'));fp=c['fingerprint']
        assert fp['code']==protocol['code'] and fp['engine']==protocol['engine']
        assert fp['image']==digest(CB/f'nrrd/{p}.nrrd')
        assert fp['landmarks']=={f.name:digest(f) for f in landmarks(int(p[2:]))[1]}
        for s in ('MAX','MAND'):
            assert fp['segmentations'][s]==digest(CB/f'seg/{p}_seg_{s}.nrrd')
            assert fp['models'][s]==digest(ROOT/f'Result_py/shape/{s}/points.npz')
        for pid,hashes in fp['reference_annotations'].items():
            assert hashes=={f.name:digest(f) for f in landmarks(int(pid))[1]}
        for key in TARGETS:
            assert p not in c['points'][key]['transfer']['reference_ids']
    notes=dict(patients=85,APog_points=170,condition_records=2720,metric_rows=1360,
               input_hashes_unchanged=True,self_reference_excluded=True,MPFH_matches_previous_joint=True,
               unrelated_metrics_unchanged=True,withheld_scalar_values_blank=True,
               unassessed_landmark_types=10,anatomically_confirmed=False)
    (DEST/'verification.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'README.md').write_text('''# A·Pog 독립 후보 실험 v1

실행: `.venv\\Scripts\\python.exe -X utf8 python/cbct_apog85.py`
검증: `.venv\\Scripts\\python.exe -X utf8 python/cbct_apog_audit.py`
단위 검증: `.venv\\Scripts\\python.exe -m unittest discover -s python -p test_cbct_apog_ensemble.py`

85명 모두 개발 자료다. 참고 환자 5·11·28의 검수되지 않은 ALI 점은 표면 대응 ROI 안내에만 쓴다. 본인이 참고 환자이면 자신을 제외한다. 현재 A/Pog와 Me는 후보 추출에 사용하지 않는다. 대칭 반사도 쓰지 않는다. N·S·Ba·ANS·PNS를 쓰는 공통 정중 좌표계 자체의 오차는 이번 단계에서 별도로 해결하지 않았다.

AMASSS 13조건과 원본 CT 3조건은 같은 ROI·윤곽 정의를 공유한다. 독립적인 정답 검증이 아니며 원본 CT 임계값은 보정된 HU를 의미하지 않는다. 원본 CT에는 치아·금속 등도 포함될 수 있다. 후보 성공률·범위와 해부학적 정확도는 구분해야 한다.

후보 추출과 교체 필요 판단을 마친 뒤에만 엑셀 값을 읽어 비교한다. 같은 숫자 제공 환자끼리의 MAE와 전체 제공률·보류 환자의 기존 오차를 함께 본다. 보류로 인한 MAE 감소를 측정 개선으로 해석하지 않는다. 기존값 유지도 검증 완료를 뜻하지 않는다.

FH 실험의 후보와 함께 오차를 전파하는 joint와 A_only/Pog_only/APog_only를 구분한다. Pog 후보를 직접 쓰면 기존 Pog 윤곽 보정을 중복 적용하지 않는다. 후보 조합 4096개 초과는 임의 추출 없이 보류한다. 출력 범위는 조건별 민감도 범위이며 신뢰구간이 아니다.

이 폴더는 연구용 결과이며 기존 좌표·보고서·운영 계측값을 바꾸지 않는다. 검증 대상은 A/Pog 및 이전 Po/Or 4종이고 나머지 10종은 미평가다. `rules_frozen.json`은 재현용 코드·입력·출력 기록이며 측정법의 정확도 확정 선언이 아니다.
''',encoding='utf-8')
    page=DEST/'report.html';html=page.read_text(encoding='utf-8')
    extra='<p>분할·원본 CT 조건은 같은 ROI와 윤곽 정의를 공유하므로 독립적인 정답 검증이 아닙니다. 원본에는 치아·금속도 포함될 수 있으며 정중 좌표계 N·S·Ba·ANS·PNS의 오차는 이번에 별도 해결하지 않았습니다. <a href="verification.json">실행 검증</a> · <a href="rules_frozen.json">재현 기록</a> · <a href="README.md">실행 방법과 한계</a></p>'
    if extra not in html:html=html.replace('<h2>1.',extra+'<h2>1.',1)
    finding='''<!-- APOG_AUDIT_START --><h2>실험 판단: 전원 적용 방식으로 채택하지 않음</h2>
<p>A 85개 중 기존 유지 73개·후보 영역 2개·보류 10개, Pog는 기존 유지 44개·후보 영역 2개·보류 39개입니다. 기존 유지에는 마스크 불일치로 원본을 우선한 A 3개·Pog 6개가 포함됩니다. 단일 좌표 안정성 기준을 만족하며 교체되는 점은 없습니다. 후보 영역이어도 계측값의 범위가 좁으면 숫자가 나올 수 있으며, 이것이 좌표의 확정을 뜻하지 않습니다.</p>
<p>A 후보는 P_15·P_44, Pog 후보는 P_28·P_69입니다. A/Pog 단독 변경 비교에서 P_44 ANB와 두 Pog 계측은 일부 개선됐으나 두 A의 Nperp-A 오차는 모두 증가했습니다. 다른 환자의 보류가 크게 늘었습니다. 기존 FH 결합 실험과 같은 환자끼리 비교한 결과와 제공 인원을 아래에 함께 표시합니다.</p>
'''+comparisons.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'''
<p>이번 규칙의 핵심 병목은 탐색 영역 안에서 정해 둔 오목함·볼록함 조건을 만족하는 윤곽을 찾지 못하는 경우가 많다는 것입니다. 이는 이 검출 규칙의 실패이며 실제 랜드마크가 없다는 뜻이 아닙니다. <a href="failure_modes.csv">조건별 실패 분류</a> · <a href="comparison_previous_joint.csv">이전 FH 결합 방식과 비교</a></p>
<p>다음 개선은 후보를 강제로 반환하거나 보류 문턱을 낮추기보다, 같은 ROI에서 전방 외측 피질 윤곽을 연속적으로 추적하고 탐색창 경계·좌표축 방향에 대한 민감도를 분리 평가하는 것입니다. 현재 결과만으로 B·Me까지 확정할 수 없습니다.</p><!-- APOG_AUDIT_END -->'''
    html=re.sub(r'<!-- APOG_AUDIT_START -->.*?<!-- APOG_AUDIT_END -->','',html,flags=re.S)
    html=html.replace('<h2>1.',finding+'<h2>1.',1)
    page.write_text(html,encoding='utf-8')
    frozen=dict(protocol=protocol,runtime={p:version(p) for p in ('numpy','scipy','SimpleITK','scikit-image','matplotlib')},
                case_hashes={p:digest(DEST/'cases'/f'{p}.json') for p in protocol['patients']},
                outputs={n:digest(DEST/n) for n in ('point_stability.csv','variant_candidates.csv','metric_restoration.csv','evaluation.csv','landmark_scope.csv','comparison_previous_joint.csv','failure_modes.csv','report.html')})
    (DEST/'rules_frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all((DEST/r).exists() for r in re.findall(r'(?:href|src)="([^"]+)"',html) if not r.startswith('#'))
    print(json.dumps(notes,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
