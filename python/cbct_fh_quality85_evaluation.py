"""FH 품질 규칙의 실제 적용 효과와 85명 전체 분모를 보존한 후속 감사."""
from html import escape
import json

import numpy as np
import pandas as pd

from cbct_fh_quality85 import DEST, KEYS, METRICS
from cbct_patient_pipeline import OUT, landmarks, digest
from osa.ceph import measures


def values(points):
    m3=measures(points,'jung');m4=measures(points,'yao4')
    return {m:float((m3 if fh=='jung' else m4)[key]) for m,(key,fh) in METRICS.items()}


def main():
    protocol=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    assert digest(OUT/'restore_ext/v1/compare_all.csv')==protocol['cohort_manifest_sha256']
    routes=pd.read_csv(DEST/'patient_routes.csv').set_index('Patient')
    prior=pd.read_csv(OUT/'develop85/v1/error_ranking.csv')
    sensitivity=[];searches=[];reasons=[]
    for p in protocol['patients']:
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        assert c['fingerprint']['code']==protocol['code_sha256']
        assert c['fingerprint']['reference']==protocol['reference_sha256']
        points,files=landmarks(int(p[2:]))
        assert {f.name:digest(f) for f in files}==c['fingerprint']['landmarks']
        for key in KEYS:
            for stage in ('original','final'):
                for reason in c[stage][key]['reasons']:
                    reasons.append(dict(Patient=p,point=key,stage=stage,reason=reason))
        for key,a in c['attempts'].items():
            searches.append(dict(Patient=p,point=key,source=a['source'],found=a['found'],
                                 raw_candidates=a['raw_candidates'],reason=a.get('reason','candidate_found')))
        assert c['release']==all(c['final'][k]['passed'] for k in KEYS)
        if not c['release']:assert c['released_measurements'] is None
        base={k:np.array(v,float) for k,v in c['recovered_points'].items()}
        all_values=[values(base)]
        # 한 번에 한 기준점의 한 축만 +/-1 mm 움직이는 24개 가상 위치 + 중심.
        for key in KEYS:
            for axis in range(3):
                for shift in (-1.,1.):
                    q=dict(base);q[key]=base[key].copy();q[key][axis]+=shift
                    all_values.append(values(q))
        for metric in METRICS:
            a=[x[metric] for x in all_values]
            sensitivity.append(dict(Patient=p,metric=metric,status=c['status'],
                                    diagnostic_value=all_values[0][metric], perturbation_min=min(a),
                                    perturbation_max=max(a), perturbation_range=max(a)-min(a),
                                    description='one FH point / one LPS axis +/-1 mm; not CI or measured repeatability'))
    sens=pd.DataFrame(sensitivity);srch=pd.DataFrame(searches);reason=pd.DataFrame(reasons)
    sens.to_csv(DEST/'point_perturbation.csv',index=False,encoding='utf-8-sig')
    srch.to_csv(DEST/'search_failures.csv',index=False,encoding='utf-8-sig')
    reason.to_csv(DEST/'review_reasons.csv',index=False,encoding='utf-8-sig')
    outcomes=[]
    for metric in METRICS:
        t=prior[prior.metric==metric].copy()
        ok=t.Patient.map(routes.release).astype(bool)
        outcomes.append(dict(metric=metric,total_n=len(t),provided_n=int(ok.sum()),
                             withheld_n=int((~ok).sum()), prior_all85_MAE=float(t.abs_error.mean()),
                             prior_same_provided_MAE=float(t.loc[ok,'abs_error'].mean()),
                             prior_withheld_MAE=float(t.loc[~ok,'abs_error'].mean()),
                             accepted_coordinate_changes=int(((routes.status=='image_recovered') & routes.release).sum()),
                             applied_improvements=0 if not (routes.status=='image_recovered').any() else None,
                             applied_worsenings=0 if not (routes.status=='image_recovered').any() else None))
    outcome=pd.DataFrame(outcomes)
    outcome.to_csv(DEST/'coverage_and_actual_changes.csv',index=False,encoding='utf-8-sig')
    verification=dict(patients=len(routes),point_assessments=len(routes)*len(KEYS),
                      legacy_manifest_unchanged=True,source_landmark_hashes_unchanged=True,
                      extraction_code_consistent=True,release_matches_quality_all_points=True,
                      withheld_values_not_released=True,synthetic_test_count=4,
                      independent_validation=False)
    (DEST/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    extra=f'''<!-- FH_ACTUAL_AUDIT_START --><h2>0. 이번 실행의 결론: 오류 감지는 작동했지만 자동 복구는 채택하지 못함</h2>
<p>85명 340개 Po·Or에 같은 품질 규칙을 적용했습니다. 73명은 운영 기준 통과, 12명은 검토 대상입니다. 이 구분은 정상/오류 정답 판정이 아닙니다. 영상 재탐색은 5명의 6개 Po에서 시도했지만 모든 조건을 통과한 후보가 없었습니다. 적용한 자동 좌표 변경은 0건이며, 실제 계측 정확도가 개선됐다고 결론내리지 않습니다.</p>
<p>기존 대칭 보정은 반대편 반사 좌표를 직접 측정에 사용하지만, 이번 탐색에서는 반사를 검색 중심으로만 사용했습니다. 재탐색 실패 시 진단용 값과 신뢰값을 구분하고 제공을 보류했습니다. 아래 2절의 candidate는 이 분리 실험의 진단용 값이며, 기존 4명 대칭 보정을 제거한 영향까지 포함되어 있습니다. 이 수치를 새로운 확정 측정값으로 사용하지 않습니다.</p>
<h3>기존 최종 기하 설정을 기준으로 제공률과 실제 변경을 함께 보기</h3><div class="scroll">{table(outcome)}</div>
<p>오차가 낮은 73명만 보고 성능이 좋아졌다고 말하지 않습니다. 보류 12명의 기존 오차와 전체 85명의 기존 오차도 함께 공개했습니다. Nperp-Pog의 여기 기준값은 기존 윤곽 보정을 포함하므로, FH 변경만 분리한 2절의 기준값과 다릅니다.</p>
<h3>재탐색 실패 내역</h3>{table(srch)}
<p>폐쇄 공기 단면을 고정 밝기 −400으로 찾는 가정은 일부 외이도 영상에서 맞지 않았습니다. 외이도가 열려 있거나 조직·부분용적 때문에 단면이 밝게 보일 수 있고, 다른 공기 구멍과의 구분도 필요합니다. 실패를 줄이려고 엑셀 오차에 맞춰 밝기 문턱을 조정하지 않았습니다.</p>
<h3>검토 유형</h3>{table(reason[reason.stage=='final'].groupby('reason').Patient.nunique().rename('환자 수').reset_index())}
<p>좌우 앞뒤/상하 차이 6 mm는 실험용 경고 기준입니다. 정상 해부학적 비대칭도 포함될 수 있어 이 경고만으로 오류나 제외를 확정하지 않습니다. P_1·9·11·27·41·81은 원래 각 점의 분포·뼈 근접 기준은 통과했지만 이 좌우 경고에 걸렸습니다. P_34는 Or 쪽 점검도 필요합니다.</p>
<p><a href="coverage_and_actual_changes.csv">85명 분모와 실제 변경 효과</a> · <a href="point_perturbation.csv">기준점 +/-1 mm 민감도</a> · <a href="review_reasons.csv">환자·점별 검토 이유</a> · <a href="verification.json">실행 검증</a></p>
<!-- FH_ACTUAL_AUDIT_END -->'''
    path=DEST/'report.html';html=path.read_text(encoding='utf-8')
    if '<!-- FH_ACTUAL_AUDIT_START -->' in html:
        first,rest=html.split('<!-- FH_ACTUAL_AUDIT_START -->',1)
        html=first+rest.split('<!-- FH_ACTUAL_AUDIT_END -->',1)[1]
    html=html.replace('<h2>1. 처리 결과</h2>',extra+'<h2>1. 처리 결과</h2>')
    path.write_text(html,encoding='utf-8')
    print(outcome.to_string(index=False));print('Verification',verification)


if __name__=='__main__':main()
