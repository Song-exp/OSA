"""마스크 불일치와 점 오류를 구분하는 원본 뼈 신호 확인 분기. 전문가 검수 전 제안."""
import json
import numpy as np
import pandas as pd

from cbct_seg_landmarks85 import DEST,TARGETS,CENTRAL,load_mask
from cbct_seg_landmarks_policy85 import decide,POLICY
from cbct_patient_pipeline import CB,OUT,landmarks,digest
from cbct_eb_detect import midline_frame
from osa.ceph import measures


def decide_confirmed(original,record,frame,nearest_dense_mm):
    decision=decide(original,record,frame)
    if (decision['reasons']==['original_far_from_own_segmentation']
            and nearest_dense_mm is not None and nearest_dense_mm<=2.5):
        return dict(route='mask_mismatch_original_retained',
                    reasons=['mask_disagrees_but_original_image_has_dense_support'],
                    proposed_point=np.asarray(original).tolist(),changed=False)
    return decision


def main():
    source=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'));assert len(source['patients'])==85
    policy=dict(parent=POLICY,additional_rule='mask-only distance failure with dense image boundary <=2.5 mm: preserve original and flag mask mismatch',
                code_sha256=digest(__file__),pixel_support_source='fh_quality_v1/cases; image hashes checked',
                symmetry_used=False,Excel_used_in_decisions=False,clinical_validation=False)
    (DEST/'confirmed_policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[];candidate_points={};originals={}
    for p in source['patients']:
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        support_path=OUT/f'develop85/fh_quality_v1/cases/{p}.json'
        support=json.loads(support_path.read_text(encoding='utf-8'))
        assert c['fingerprint']['image']==support['fingerprint']['image']
        assert c['fingerprint']['landmarks']==support['fingerprint']['landmarks']
        P,files=landmarks(int(p[2:]));originals[p]=P
        assert {f.name:digest(f) for f in files}==c['fingerprint']['landmarks']
        o,R,_=midline_frame({k:P[k] for k in CENTRAL});Q=dict(P)
        for key in TARGETS:
            d=decide_confirmed(P[key],c['points'][key],(o,R),support['original'][key]['nearest_dense_mm'])
            if d['proposed_point'] is not None:Q[key]=np.array(d['proposed_point'])
            rows.append(dict(Patient=p,point=key,route=d['route'],reasons=';'.join(d['reasons']),changed=d['changed'],
                             original_x=P[key][0],original_y=P[key][1],original_z=P[key][2],
                             proposed_x=d['proposed_point'][0] if d['proposed_point'] is not None else np.nan,
                             proposed_y=d['proposed_point'][1] if d['proposed_point'] is not None else np.nan,
                             proposed_z=d['proposed_point'][2] if d['proposed_point'] is not None else np.nan))
        candidate_points[p]=Q
    routes=pd.DataFrame(rows);routes.to_csv(DEST/'confirmed_routes.csv',index=False,encoding='utf-8-sig')
    # 전원의 분기 결정이 끝난 뒤에만 비교값을 읽는다.
    prior=pd.read_csv(OUT/'develop85/v1/error_ranking.csv').set_index(['Patient','metric'])
    measurements=[];mask_hashes={}
    for p in source['patients']:
        Q=candidate_points[p];changed=bool(routes[(routes.Patient==p)&routes.changed].shape[0])
        if changed:
            from osa.refine import refine
            mand,dirs,org=load_mask(p,'MAND');maxm,mdirs,morg=load_mask(p,'MAX')
            assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
            refined,_=refine(Q,mand,maxm,dirs,org,'yao4');del mand,maxm
            pog=float(measures(refined,'yao4')['Nperp_Pog'])
            mask_hashes[p]={s:digest(CB/f'seg/{p}_seg_{s}.nrrd') for s in ('MAND','MAX')}
        else:pog=float(prior.loc[(p,'Nperp-Pog'),'automatic'])
        for metric,key,fh in [('ANB','ANB','jung'),('Nperp-A','Nperp_A','yao4'),('Nperp-Pog','Nperp_Pog','yao4'),('MP-FH','MPFH_GoMe','yao4')]:
            y=float(prior.loc[(p,metric),'excel']);old=float(prior.loc[(p,metric),'automatic'])
            proposal=pog if metric=='Nperp-Pog' else float(measures(Q,fh)[key])
            measurements.append(dict(Patient=p,metric=metric,current=old,proposal=proposal,excel=y,
                                     current_error=abs(old-y),proposal_error=abs(proposal-y),exception_triggered=changed,
                                     status='proposal_requires_review' if changed else 'original_retained'))
    comp=pd.DataFrame(measurements);comp.to_csv(DEST/'confirmed_measurements.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for m,g in comp.groupby('metric'):
        summary.append(dict(metric=m,n=len(g),current_MAE=g.current_error.mean(),proposal_MAE=g.proposal_error.mean(),
                            current_p90=g.current_error.quantile(.9),proposal_p90=g.proposal_error.quantile(.9),
                            current_max=g.current_error.max(),proposal_max=g.proposal_error.max(),
                            improved_n=int((g.proposal_error<g.current_error-1e-8).sum()),
                            worsened_n=int((g.proposal_error>g.current_error+1e-8).sum()),
                            unchanged_n=int((abs(g.proposal_error-g.current_error)<=1e-8).sum()),
                            within2_to_over2=int(((g.current_error<=2)&(g.proposal_error>2)).sum())))
    summary=pd.DataFrame(summary);summary.to_csv(DEST/'confirmed_evaluation.csv',index=False,encoding='utf-8-sig')
    assert len(routes)==340 and comp.groupby('metric').size().eq(85).all()
    warnings={'P_1','P_9','P_11','P_27','P_41','P_81'}
    assert not routes[routes.Patient.isin(warnings)].changed.any()
    untouched=comp[~comp.exception_triggered]
    assert np.allclose(untouched.current,untouched.proposal)
    verification=dict(patients=85,points=340,all_patients_retained_in_evaluation=True,
                      changes_are_proposals_only=True,all_six_asymmetry_warning_cases_unchanged=True,
                      image_and_landmark_provenance_matches_cached_support=True,
                      independent_validation=False,mandible_maxilla_hashes=mask_hashes)
    (DEST/'confirmed_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    c=routes.groupby(['point','route']).size().rename('n').reset_index()
    changed_cases=set(routes.loc[routes.changed,'Patient'])
    detail=comp[comp.Patient.isin(changed_cases)]
    supplement=f'''<!-- CONFIRMED_POLICY_START --><h2>우선 검토할 보수적 분기: 분할과 원본 신호를 함께 확인</h2>
<p>분할 마스크에서 멀다는 사실만으로 기존 점을 오류로 단정하지 않았습니다. 마스크에서는 멀어도 원본의 뼈 신호가 2.5 mm 안에 있고 해당 해부학적 ROI 안이면 기존 점을 보존하고 마스크 불일치 경고로 남겼습니다. P_31 LPo가 이 조건에 해당합니다. 원본 신호는 앞선 전원 픽셀 감사의 동일 영상·동일 랜드마크 해시를 확인해 재사용했습니다.</p>
<p>현재 교체 제안은 {len(changed_cases)}명·{int(routes.changed.sum())}개 점입니다. 대칭을 이용하지 않았고 비대칭 경고 6명은 모두 좌표·측정값을 유지했습니다. 85명 전원을 평가에 포함했습니다. 아래 수치는 이전 85명 개발 자료의 탐색 결과이며 독립 검증이 아닙니다. 모든 교체는 검수 전 제안으로만 저장했습니다.</p>
<div class="scroll">{table(summary)}</div>
<p>평균오차가 조금 줄어도 일부 환자에서는 악화됩니다. within2_to_over2는 오차 2 이내에서 2 초과로 바뀐 진단용 운영 지표이며 임상 허용오차가 아닙니다. 전체 방식을 바로 채택하지 않습니다.</p>
<h3>340개 점의 최종 제안 경로</h3>{table(c)}
<h3>교체 제안점과 마스크 불일치</h3><div class="scroll">{table(routes[routes.route!='original_retained'])}</div>
<h3>교체 제안 환자의 실제 전후 오차</h3><div class="scroll">{table(detail)}</div>
<p><a href="confirmed_routes.csv">340개 점 처리 경로</a> · <a href="confirmed_measurements.csv">85명×4지표</a> · <a href="confirmed_evaluation.csv">전체 평가</a> · <a href="confirmed_policy.json">원본 확인 규칙</a> · <a href="confirmed_verification.json">실행 검증</a></p>
<p>아래 단면의 초록 점은 추출된 전체 후보입니다. 실제 교체 제안 여부는 위 경로표에서 확인해야 하며, 초록 점이 있다는 이유만으로 기존 점을 교체하지 않았습니다.</p><!-- CONFIRMED_POLICY_END -->'''
    path=DEST/'report.html';html=path.read_text(encoding='utf-8')
    if '<!-- CONFIRMED_POLICY_START -->' in html:
        a,b=html.split('<!-- CONFIRMED_POLICY_START -->',1);html=a+b.split('<!-- CONFIRMED_POLICY_END -->',1)[1]
    html=html.replace('<!-- SEGMENTATION_POLICY_START -->',supplement+'<!-- SEGMENTATION_POLICY_START -->')
    path.write_text(html,encoding='utf-8')
    print(c.to_string(index=False));print(summary.to_string(index=False))


if __name__=='__main__':main()
