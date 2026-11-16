"""분할 후보를 무조건 교체하지 않는 공통 예외 분기. 대칭·엑셀 미사용.

현재 점은 교체 필요 여부 점검에만 사용하며 재검출 후보의 위치 선택에는 사용하지 않는다.
교체 결과는 전부 전문가 검수 전 제안값이다. 기존 분석 산출물을 덮어쓰지 않는다.
"""
from html import escape
import json

import numpy as np
import pandas as pd

from cbct_seg_landmarks85 import DEST,TARGETS,CENTRAL,load_mask
from cbct_patient_pipeline import OUT,landmarks,digest
from cbct_eb_detect import midline_frame
from osa.ceph import measures

POLICY=dict(version='same-side-segmentation-exception-v1',
            original_max_mask_distance_mm=2.5,roi_half_mm=[12.,15.,15.],
            max_transfer_residual_mm=3.,max_transfer_condition=100.,
            preserve_original_if_no_trigger=True,symmetry_used=False,
            Excel_used_for_decisions=False,expert_validated=False)


def decide(original,record,frame,policy=POLICY):
    o,R=frame
    delta=R@(np.asarray(original)-np.asarray(record['transferred_seed_lps']))
    distance=record['original_to_mask_mm']
    reasons=[]
    if distance is None or distance>policy['original_max_mask_distance_mm']:
        reasons.append('original_far_from_own_segmentation')
    if np.any(abs(delta)>np.asarray(policy['roi_half_mm'])):
        reasons.append('original_outside_transferred_anatomical_roi')
    if not reasons:
        return dict(route='original_retained',reasons=[],proposed_point=np.asarray(original).tolist(),changed=False)
    transfer=record['transfer']
    reliable=(not record['registration_fit_flag']
              and transfer['local_transfer_residual_mm']<=policy['max_transfer_residual_mm']
              and transfer['local_transfer_condition']<=policy['max_transfer_condition'])
    if record['candidate'] is None or not reliable:
        return dict(route='needs_review_no_reliable_candidate',reasons=reasons,
                    proposed_point=None,changed=False)
    return dict(route='segmentation_recovery_proposal',reasons=reasons,
                proposed_point=record['candidate'],changed=True)


def main():
    source=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    patients=source['patients'];assert len(patients)==85
    # 평가값을 읽기 전에 고정 규칙과 코드를 기록한다.
    protocol=dict(policy=POLICY,code_sha256=digest(__file__),source_protocol_sha256=digest(DEST/'protocol.json'))
    (DEST/'exception_policy.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    routed=[];case_points={};case_flags={}
    for p in patients:
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        assert c['fingerprint']['code']==source['code_sha256']
        P,files=landmarks(int(p[2:]))
        assert c['fingerprint']['landmarks']=={f.name:digest(f) for f in files}
        o,R,_=midline_frame({k:P[k] for k in CENTRAL})
        Q=dict(P);unresolved=[]
        for key in TARGETS:
            r=decide(P[key],c['points'][key],(o,R))
            if r['proposed_point'] is not None:Q[key]=np.array(r['proposed_point'])
            else:unresolved.append(key)
            routed.append(dict(Patient=p,point=key,route=r['route'],reasons=';'.join(r['reasons']),
                               changed=r['changed'],original_x=P[key][0],original_y=P[key][1],original_z=P[key][2],
                               proposed_x=r['proposed_point'][0] if r['proposed_point'] is not None else np.nan,
                               proposed_y=r['proposed_point'][1] if r['proposed_point'] is not None else np.nan,
                               proposed_z=r['proposed_point'][2] if r['proposed_point'] is not None else np.nan))
        case_points[p]=Q;case_flags[p]=unresolved
    routes=pd.DataFrame(routed);routes.to_csv(DEST/'exception_routes.csv',index=False,encoding='utf-8-sig')
    # 후보 선택과 분기 결정이 전원에서 끝난 후에만 엑셀을 읽는다.
    xl=pd.read_csv(OUT/'restore_ext/v1/compare_all.csv').set_index(['Patient','metric']).excel
    prior=pd.read_csv(OUT/'develop85/v1/error_ranking.csv').set_index(['Patient','metric']).automatic
    comparisons=[];pipeline=[]
    for p in patients:
        P=landmarks(int(p[2:]))[0];Q=case_points[p]
        affected=bool(routes[(routes.Patient==p)&routes.changed].shape[0])
        # 기존 Nperp-Pog와 같은 윤곽 보정을 적용해 전체 파이프라인끼리도 비교한다.
        if affected:
            from osa.refine import refine
            mand,dirs,org=load_mask(p,'MAND');maxm,mdirs,morg=load_mask(p,'MAX')
            assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
            refined,refine_info=refine(Q,mand,maxm,dirs,org,'yao4')
            final_pog=float(measures(refined,'yao4')['Nperp_Pog'])
            del mand,maxm
        else:
            final_pog=float(prior.get((p,'Nperp-Pog')))
        for m,k,fh in [('ANB','ANB','jung'),('Nperp-A','Nperp_A','yao4'),('Nperp-Pog','Nperp_Pog','yao4'),('MP-FH','MPFH_GoMe','yao4')]:
            raw=measures(P,fh)[k];new=measures(Q,fh)[k];y=xl.get((p,m),np.nan)
            comparisons.append(dict(Patient=p,metric=m,raw=raw,proposal=new,current=prior.get((p,m)),excel=y,
                                    raw_error=abs(raw-y),proposal_error=abs(new-y),current_error=abs(prior.get((p,m))-y),
                                    candidate_complete=not case_flags[p],unresolved=';'.join(case_flags[p]),
                                    changed=abs(raw-new)>1e-6))
            final_value=final_pog if m=='Nperp-Pog' else new
            pipeline.append(dict(Patient=p,metric=m,current=prior.get((p,m)),proposal=final_value,excel=y,
                                 current_error=abs(prior.get((p,m))-y),proposal_error=abs(final_value-y),
                                 exception_triggered=affected,candidate_complete=not case_flags[p],
                                 status='proposal_requires_review' if affected else 'original_method_retained'))
    comp=pd.DataFrame(comparisons);comp.to_csv(DEST/'exception_measurements.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for m,g in comp.groupby('metric'):
        changed=g[g.changed];untouched=g[~g.changed]
        summaries.append(dict(metric=m,total_n=len(g),complete_n=int(g.candidate_complete.sum()),
                              unresolved_n=int((~g.candidate_complete).sum()),raw_MAE=g.raw_error.mean(),
                              proposal_all85_diagnostic_MAE=g.proposal_error.mean(),current_MAE=g.current_error.mean(),
                              raw_p90=g.raw_error.quantile(.9),proposal_p90=g.proposal_error.quantile(.9),
                              raw_max=g.raw_error.max(),proposal_max=g.proposal_error.max(),changed_n=len(changed),
                              improved_vs_raw=int((changed.proposal_error<changed.raw_error).sum()),
                              worsened_vs_raw=int((changed.proposal_error>changed.raw_error).sum()),
                              within2_to_over2=int(((changed.raw_error<=2)&(changed.proposal_error>2)).sum()),
                              untriggered_max_value_change=float(abs(untouched.proposal-untouched.raw).max()) if len(untouched) else 0.))
    summary=pd.DataFrame(summaries);summary.to_csv(DEST/'exception_evaluation.csv',index=False,encoding='utf-8-sig')
    pipe=pd.DataFrame(pipeline);pipe.to_csv(DEST/'full_pipeline_measurements.csv',index=False,encoding='utf-8-sig')
    pipe_summary=[]
    for m,g in pipe.groupby('metric'):
        pipe_summary.append(dict(metric=m,n=len(g),current_MAE=g.current_error.mean(),proposal_MAE=g.proposal_error.mean(),
                                 current_p90=g.current_error.quantile(.9),proposal_p90=g.proposal_error.quantile(.9),
                                 current_max=g.current_error.max(),proposal_max=g.proposal_error.max(),
                                 improved_n=int((g.proposal_error<g.current_error-1e-8).sum()),
                                 worsened_n=int((g.proposal_error>g.current_error+1e-8).sum()),
                                 within2_to_over2=int(((g.current_error<=2)&(g.proposal_error>2)).sum())))
    pipe_summary=pd.DataFrame(pipe_summary)
    pipe_summary.to_csv(DEST/'full_pipeline_evaluation.csv',index=False,encoding='utf-8-sig')
    # 이전 위치 이상 사례와 비대칭 경고 사례를 함께 기록한다.
    focus=['P_16','P_20','P_23','P_35','P_37','P_34','P_1','P_9','P_11','P_27','P_41','P_81']
    detail=comp[comp.Patient.isin(focus)]
    detail.to_csv(DEST/'exception_focus.csv',index=False,encoding='utf-8-sig')
    assert len(routes)==340 and routes.Patient.nunique()==85
    assert comp.groupby('metric').size().eq(85).all()
    assert (summary.untriggered_max_value_change<1e-8).all()
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    counts=routes.groupby(['point','route']).size().rename('n').reset_index()
    intro=f'''<!-- SEGMENTATION_POLICY_START --><h2>0. 전원 공통 예외 분기: 기존 점을 불필요하게 바꾸지 않음</h2>
<p>기존 점이 환자 자신의 분할에서 2.5 mm 넘게 떨어졌거나, 표면 대응으로 전달한 해부학적 탐색 영역을 벗어난 경우만 교체 후보로 삼았습니다. 좌우 비대칭 정도는 조건에 사용하지 않습니다. 후보 위치는 현재 Po/Or를 보지 않고 추출했고, 예외 조건이 없으면 기존 좌표를 유지합니다.</p>
<p>아래는 검수 전 제안값입니다. 신뢰할 후보가 없으면 기존 진단용 값을 남기되 미해결로 기록하며 성공으로 세지 않습니다. 전체 85명 진단용 MAE에는 미해결 사례도 포함했습니다. 임상 확정값이나 독립 검증 결과가 아닙니다.</p>
<h3>현재 방식과 같은 후처리까지 포함한 전체 파이프라인 비교</h3><p>Nperp-Pog에는 기존과 같은 Pog 윤곽 보정을 적용했습니다. 기존 반사 보정은 새 제안에 사용하지 않았습니다. 예외가 발동하지 않은 78명은 기존 측정값을 보존합니다. 모든 좌표 변경은 검수 전 제안입니다.</p><div class="scroll">{table(pipe_summary)}</div>
<p><a href="full_pipeline_measurements.csv">전체 방식의 환자별 전후 값</a> · <a href="full_pipeline_evaluation.csv">전체 방식의 오차·악화 사례</a></p>
<h3>FH 변경 효과만 분리한 비교</h3>
<div class="scroll">{table(summary)}</div><h3>340개 점의 처리 경로</h3>{table(counts)}
<h3>교체 제안·미해결 점</h3><div class="scroll">{table(routes[routes.route!='original_retained'])}</div>
<p><a href="exception_routes.csv">340개 점 처리 경로</a> · <a href="exception_measurements.csv">85명×4지표</a> · <a href="exception_evaluation.csv">전체 오차·악화 사례</a> · <a href="exception_focus.csv">이전 검토 대상 12명</a> · <a href="exception_policy.json">예외 규칙</a></p><!-- SEGMENTATION_POLICY_END -->'''
    path=DEST/'report.html';html=path.read_text(encoding='utf-8')
    if '<!-- SEGMENTATION_POLICY_START -->' in html:
        a,b=html.split('<!-- SEGMENTATION_POLICY_START -->',1);html=a+b.split('<!-- SEGMENTATION_POLICY_END -->',1)[1]
    html=html.replace('<h2>1. 구현한 흐름</h2>',intro+'<h2>1. 구현한 흐름</h2>')
    path.write_text(html,encoding='utf-8')
    print(counts.to_string(index=False));print(summary.to_string(index=False));print('FULL PIPELINE');print(pipe_summary.to_string(index=False))


if __name__=='__main__':main()
