"""Po 후보의 관 모양 지속성·원본 경계를 85명 양측에서 검토한다.

분할의 작은 구멍 상연을 바로 Po로 삼지 않는다. 후보 추출과 허용 분기는 엑셀을
읽기 전에 완료한다. 대칭 추정은 사용하지 않는다. 모든 결과는 검수 전 후보다.
"""
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing,binary_erosion,binary_fill_holes,label

from cbct_patient_pipeline import CB,OUT,landmarks,digest
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd
from cbct_seg_landmarks85 import DEST as PREV,CENTRAL,STEP,load_mask,grid,sample,save
from cbct_seg_landmarks_confirmed85 import decide_confirmed
from osa.ceph import measures

DEST=OUT/'develop85/po_tube_v1'
RULE=dict(version='same-side-segmentation-tube-v1',voxel_step_mm=.5,
          roi_half_mm=[12.,15.,15.],hole_area_mm2=[6.,180.],hole_width_height_mm=[3.,20.],
          mask_closing_radius_mm=1.,adjacent_slice_overlap_min=.20,
          tube_lateral_span_min_mm=4.,tube_max_section_min_mm2=12.,
          tube_volume_min_mm3=40.,native_roof_fraction_min=.30,
          raw_roof_minus_lumen_min=100.,winner_score_ratio_min=1.25,
          candidate_max_distance_from_seed_mm=12.,
          roi_lateral_margin_min_mm=1.,symmetry_used=False,Excel_used_in_detection=False,
          threshold_interpretation='engineering proposal, not validated anatomy or calibrated HU',
          deployment='diagnostic proposals only; no prior outputs modified')


def section_nodes(mask,axes,ct):
    """실제 CB 구멍과 작은 마스크 틈을 닫아 얻은 구멍을 수집하고 원본 상연을 확인."""
    L,A,S=axes;nodes=[]
    yy,zz=np.mgrid[-2:3,-2:3];disk=yy*yy+zz*zz<=4
    for i in range(len(L)):
        bone=mask[i]
        closed=binary_closing(bone,structure=disk,border_value=0)
        void=binary_fill_holes(closed)&~closed
        labs,n=label(void)
        for j in range(1,n+1):
            region=labs==j;aa,ss=np.where(region);area=len(aa)*STEP**2
            if not RULE['hole_area_mm2'][0]<=area<=RULE['hole_area_mm2'][1]:continue
            width=np.ptp(A[aa])+STEP;height=np.ptp(S[ss])+STEP
            if not (3<=width<=20 and 3<=height<=20):continue
            roof=int(ss.max());ai=int(np.rint(np.median(aa[ss>=roof-1])))
            cap=bone[max(0,ai-2):ai+3,roof+1:min(roof+4,len(S))]
            cap_fraction=float(cap.mean()) if cap.size else 0.
            if cap_fraction<RULE['native_roof_fraction_min']:continue
            interior=binary_erosion(region)
            lumen=ct[i][interior] if interior.any() else ct[i][region]
            above=ct[i,max(0,ai-2):ai+3,roof+1:min(roof+4,len(S))]
            roof_values=above[cap] if above.size else np.array([])
            if not len(roof_values) or not len(lumen):continue
            contrast=float(np.nanmedian(roof_values)-np.nanmedian(lumen))
            if not np.isfinite(contrast):continue
            nodes.append(dict(index=i,region=region,area_mm2=float(area),
                              centroid=np.array([np.mean(A[aa]),np.mean(S[ss])]),
                              local=np.array([L[i],A[ai],S[roof]+STEP/2]),
                              native_roof_fraction=cap_fraction,contrast=contrast,
                              lumen_median=float(np.nanmedian(lumen)),roof_median=float(np.nanmedian(roof_values))))
    return nodes


def connected_tracks(nodes):
    """인접한 좌우 위치의 단면에서 같은 공간이 겹칠 때만 연결한다."""
    parent=list(range(len(nodes)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    by_slice={}
    for i,n in enumerate(nodes):by_slice.setdefault(n['index'],[]).append(i)
    for s,left in by_slice.items():
        for a in left:
            for b in by_slice.get(s+1,[]):
                x,y=nodes[a],nodes[b]
                intersection=np.count_nonzero(x['region']&y['region'])
                overlap=intersection/min(x['region'].sum(),y['region'].sum())
                ratio=max(x['area_mm2'],y['area_mm2'])/min(x['area_mm2'],y['area_mm2'])
                if overlap>=RULE['adjacent_slice_overlap_min'] and ratio<=3. and np.linalg.norm(x['centroid']-y['centroid'])<=4.:
                    parent[root(b)]=root(a)
    groups={}
    for i in range(len(nodes)):groups.setdefault(root(i),[]).append(i)
    return list(groups.values())


def choose_tube(mask,axes,ct,seed_local):
    nodes=section_nodes(mask,axes,ct);groups=connected_tracks(nodes)
    tracks=[]
    for ids in groups:
        ns=[nodes[i] for i in ids]
        indices=sorted({n['index'] for n in ns})
        span=(indices[-1]-indices[0]+1)*STEP
        volume=sum(n['area_mm2'] for n in ns)*STEP
        area=max(n['area_mm2'] for n in ns)
        median=np.median(np.array([n['local'] for n in ns]),axis=0)
        distance=float(np.linalg.norm((median-seed_local)[1:]))
        # 여러 단면에서 이어지는 큰 관을 우선하며 반대편은 참조하지 않는다.
        score=volume/(1+(distance/8.)**2)
        reasons=[]
        if span<RULE['tube_lateral_span_min_mm']:reasons.append('insufficient_lateral_extent')
        if volume<RULE['tube_volume_min_mm3']:reasons.append('small_void_volume')
        if area<RULE['tube_max_section_min_mm2']:reasons.append('small_void_sections')
        plausible=[n for n in ns if n['contrast']>=RULE['raw_roof_minus_lumen_min']
                   and np.linalg.norm(n['local']-seed_local)<=12.]
        if not plausible:reasons.append('no_native_contrast_supported_rim')
        record=dict(span_mm=float(span),volume_mm3=float(volume),max_area_mm2=float(area),
                    score=float(score),nodes=len(ids),reasons=reasons,
                    median_contrast=float(np.median([n['contrast'] for n in ns])),candidate=None)
        if not reasons:
            # 가장 외측 폐쇄 단면을 입구 근처 후보로 삼지만 실제 입구는 미확인이다.
            plausible.sort(key=lambda n:(-abs(n['local'][0]),np.linalg.norm((n['local']-seed_local)[1:])))
            point=plausible[0]['local']
            if min(abs(point[0]-axes[0][0]),abs(point[0]-axes[0][-1]))<RULE['roi_lateral_margin_min_mm']:
                record['reasons'].append('candidate_at_roi_lateral_boundary')
            else:record['candidate']=point
        tracks.append(record)
    eligible=sorted([t for t in tracks if t['candidate'] is not None and not t['reasons']],key=lambda t:-t['score'])
    reason='no_tube_passed_shape_and_native_image_checks'
    selected=None;ratio=None
    if eligible:
        ratio=float(eligible[0]['score']/eligible[1]['score']) if len(eligible)>1 else None
        if ratio is not None and ratio<RULE['winner_score_ratio_min']:reason='multiple_competing_tubes'
        else:selected=eligible[0];reason='tube_supported_candidate_requires_review'
    return dict(status=reason,section_nodes=len(nodes),track_count=len(tracks),eligible_tracks=len(eligible),
                winner_ratio=ratio,candidate_local=selected['candidate'] if selected else None,
                selected_track=selected,tracks=tracks)


def review_figure(p,side,case,mask,axes,ct,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    L,A,S=axes;o,R=frame
    points=[('이전 분할 후보',case['previous_candidate'],'#ff9800'),
            ('관 모양 후보',case['candidate'],'#00e676')]
    centers=[(title,point,color) for title,point,color in points if point is not None]
    if not centers:centers=[('표면 대응 탐색 중심',case['seed'],'#ffd400')]
    fig,axs=plt.subplots(len(centers)+1,3,figsize=(16,5*(len(centers)+1)),squeeze=False)
    original=np.array(case['original_point'])
    for row,(title,point,color) in enumerate(centers):
        center=R@(np.array(point)-o)
        for col,axis in enumerate((0,1,2)):
            i=int(np.argmin(abs(axes[axis]-center[axis])))
            image=np.take(ct,i,axis=axis).T;bone=np.take(mask,i,axis=axis).T
            ix,iy=((1,2),(0,2),(0,1))[col];X,Y=axes[ix],axes[iy]
            ext=[X[0]-STEP/2,X[-1]+STEP/2,Y[0]-STEP/2,Y[-1]+STEP/2]
            ax=axs[row,col];ax.imshow(image,origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=ext)
            if bone.any():ax.contour(X,Y,bone.astype(float),[.5],colors=['#00c8ff'],linewidths=.8)
            for name,position,c in [('ALI',original,'#ff4040')]+[(t,np.array(q),cc) for t,q,cc in points if q is not None]:
                local=R@(position-o);off=local[axis]-axes[axis][i]
                ax.plot(local[ix],local[iy],'x' if name!='ALI' else '+',color=c,ms=9,mew=1.8,label=f'{name}: off-plane {off:+.1f} mm')
            ax.set_xlim(ext[:2]);ax.set_ylim(ext[2:]);ax.legend(fontsize=7)
            ax.set_title(f'{title} 위치 / '+('시상','관상','축상')[col],fontsize=11)
    # 외측·중앙·내측 단면을 나란히 놓아 관의 연속성을 확인한다.
    center=R@(np.array(case['candidate'] if case['candidate'] is not None else case['previous_candidate'] if case['previous_candidate'] is not None else case['seed'])-o)
    outer_sign=1 if side=='LPo' else -1
    for ax,offset in zip(axs[-1],(4.,0.,-4.)):
        x=center[0]+outer_sign*offset;i=int(np.argmin(abs(L-x)))
        ext=[A[0]-STEP/2,A[-1]+STEP/2,S[0]-STEP/2,S[-1]+STEP/2]
        ax.imshow(ct[i].T,origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=ext)
        if mask[i].any():ax.contour(A,S,mask[i].T.astype(float),[.5],colors=['#00c8ff'],linewidths=.8)
        ax.set_title(f'관의 연속성: 바깥쪽 {offset:+.0f} mm / L={L[i]:.1f}',fontsize=11)
    fig.suptitle(f'{p} {side}: {case["route"]} / 작은 구멍과 외이도를 구분하기 위한 원본 대조',fontsize=13)
    fig.tight_layout(rect=(0,.02,1,.96));fig.savefig(DEST/'figs'/f'{p}_{side}.png',dpi=110);plt.close(fig)


def process(p):
    c=json.loads((PREV/'cases'/f'{p}.json').read_text(encoding='utf-8'))
    support=json.loads((OUT/f'develop85/fh_quality_v1/cases/{p}.json').read_text(encoding='utf-8'))
    assert support['fingerprint']['image']==c['fingerprint']['image']
    P,files=landmarks(int(p[2:]));assert {f.name:digest(f) for f in files}==c['fingerprint']['landmarks']
    o,R,_=midline_frame({k:P[k] for k in CENTRAL})
    v,dirs,org=read_nrrd(CB/f'nrrd/{p}.nrrd');mask,mdirs,morg=load_mask(p,'CB')
    assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
    result=dict(Patient=p,points={},fingerprint=c['fingerprint'],rule=RULE)
    for side in ('RPo','LPo'):
        old=c['points'][side];seed=np.array(old['transferred_seed_lps'])
        m,axes,inside=grid(mask,mdirs,morg,o,R,seed)
        G=np.stack(np.meshgrid(*axes,indexing='ij'),-1);ct=sample(v,dirs,org,o+G@R,order=1,cval=np.nan)
        tube=choose_tube(m,axes,ct,R@(seed-o))
        candidate=(o+np.array(tube['candidate_local'])@R) if tube['candidate_local'] is not None else None
        quality=decide_confirmed(P[side],old,(o,R),support['original'][side]['nearest_dense_mm'])
        triggered=bool(quality['changed'] or quality['route']=='needs_review_no_reliable_candidate')
        if not triggered:route='original_retained'
        elif candidate is not None and inside.mean()>=.98 and not old['registration_fit_flag'] and old['transfer']['local_transfer_residual_mm']<=3.:
            route='tube_recovery_proposal_requires_review'
        else:route='needs_review_no_confirmed_canal_candidate'
        rec=dict(**tube,candidate=candidate,route=route,triggered=triggered,seed=seed,
                 original_point=P[side],previous_candidate=old['candidate'],roi_inside_fraction=float(inside.mean()),
                 reasons=quality['reasons'],registration_fit_flag=old['registration_fit_flag'])
        result['points'][side]=rec
        if triggered:review_figure(p,side,rec,m,axes,ct,(o,R))
    save(DEST/'cases'/f'{p}.json',result)
    print(p,', '.join(k+':'+r['route'] for k,r in result['points'].items()),flush=True)
    return json_safe_result(result)


def json_safe_result(result):
    from cbct_fh_quality85 import json_safe
    return json_safe(result)


def evaluate(cases):
    routes=[]
    for c in cases:
        for side,r in c['points'].items():
            selected=r['selected_track'] or {}
            routes.append(dict(Patient=c['Patient'],point=side,route=r['route'],triggered=r['triggered'],
                               detection_status=r['status'],section_nodes=r['section_nodes'],tracks=r['track_count'],
                               eligible_tracks=r['eligible_tracks'],winner_ratio=r['winner_ratio'],
                               selected_span_mm=selected.get('span_mm'),selected_volume_mm3=selected.get('volume_mm3'),
                               selected_max_area_mm2=selected.get('max_area_mm2'),selected_contrast=selected.get('median_contrast')))
    routes=pd.DataFrame(routes);routes.to_csv(DEST/'point_routes.csv',index=False,encoding='utf-8-sig')
    # 전원 형태 점검이 끝난 뒤에만 오차 비교값을 읽는다.
    prior=pd.read_csv(OUT/'develop85/v1/error_ranking.csv').set_index(['Patient','metric'])
    previous=pd.read_csv(PREV/'confirmed_measurements.csv').set_index(['Patient','metric'])
    measurements=[]
    for c in cases:
        p=c['Patient'];P=landmarks(int(p[2:]))[0];Q=dict(P);unresolved=[];changed=False
        for side,r in c['points'].items():
            if r['route']=='tube_recovery_proposal_requires_review':Q[side]=np.array(r['candidate']);changed=True
            elif r['triggered']:unresolved.append(side)
        # 반사를 실행하지 않으며 미해결 사례의 진단용 값은 원래 자동점으로 계산한다.
        if changed or unresolved:
            from osa.refine import refine
            mand,dirs,org=load_mask(p,'MAND');maxm,mdirs,morg=load_mask(p,'MAX')
            assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
            qref,_=refine(Q,mand,maxm,dirs,org,'yao4');del mand,maxm
            pog=measures(qref,'yao4')['Nperp_Pog']
        else:pog=float(prior.loc[(p,'Nperp-Pog'),'automatic'])
        for metric,key,fh in [('ANB','ANB','jung'),('Nperp-A','Nperp_A','yao4'),('Nperp-Pog','Nperp_Pog','yao4'),('MP-FH','MPFH_GoMe','yao4')]:
            y=prior.loc[(p,metric),'excel'];old=prior.loc[(p,metric),'automatic']
            value=float(pog if metric=='Nperp-Pog' else measures(Q,fh)[key])
            broad=float(previous.loc[(p,metric),'proposal']) if p!='P_34' else float(old)
            measurements.append(dict(Patient=p,metric=metric,current=old,previous_Po_only_proposal=broad,
                                     tube_diagnostic=value,excel=y,current_error=abs(old-y),
                                     broad_error=abs(broad-y),tube_diagnostic_error=abs(value-y),
                                     point_exception=changed or bool(unresolved),unresolved=';'.join(unresolved),
                                     recovery_complete=not unresolved,
                                     proposed_value=value if not unresolved else np.nan))
    comp=pd.DataFrame(measurements);comp.to_csv(DEST/'measurement_comparison.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for metric,g in comp.groupby('metric'):
        complete=g[g.recovery_complete]
        summary.append(dict(metric=metric,total_n=len(g),complete_n=len(complete),unresolved_n=len(g)-len(complete),
                            current_all85_MAE=g.current_error.mean(),previous_Po_only_MAE=g.broad_error.mean(),
                            tube_all85_diagnostic_MAE=g.tube_diagnostic_error.mean(),
                            same_complete_current_MAE=complete.current_error.mean(),same_complete_tube_MAE=complete.tube_diagnostic_error.mean(),
                            withheld_current_MAE=g[~g.recovery_complete].current_error.mean(),
                            improved_n=int((g.tube_diagnostic_error<g.current_error-1e-8).sum()),
                            worsened_n=int((g.tube_diagnostic_error>g.current_error+1e-8).sum())))
    summary=pd.DataFrame(summary);summary.to_csv(DEST/'evaluation.csv',index=False,encoding='utf-8-sig')
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    cards=[]
    for _,r in routes[routes.triggered].iterrows():
        p,side=r.Patient,r.point
        cards.append(f'<article><h3>{p} {side}</h3>{table(routes[(routes.Patient==p)&(routes.point==side)])}'
                     f'<a href="figs/{p}_{side}.png"><img src="figs/{p}_{side}.png"></a>{table(comp[comp.Patient==p])}'
                     f'<p><a href="cases/{p}.json">분할의 관 모양·후보 근거</a></p></article>')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>85명 Po 외이도 관 모양 점검</title>
<style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}table{{border-collapse:collapse;background:white;font-size:12px}}th,td{{border:1px solid #ddd;padding:5px}}.scroll{{overflow:auto}}img{{max-width:100%}}article{{background:white;padding:16px;margin:24px 0;overflow:auto}}h2{{margin-top:36px}}</style><main>
<h1>Po 복구 허용 조건 검토: 단일 구멍에서 관 모양 지속성으로</h1>
<p>85명 양쪽 Po 170개에 같은 규칙을 적용했습니다. Or은 기존 좌표를 유지합니다. 대칭 추정·환자 번호·엑셀 오차로 후보를 선택하지 않았습니다. 기존 결과는 변경하지 않았고 모든 새 좌표는 검수 전 후보입니다.</p>
<h2>1. 확인된 약점과 새 조건</h2><p>이전 검출은 작은 골내 공간의 상연을 고를 수 있었습니다. 이번에는 인접 시상 단면에서 공간이 겹쳐 이어지는지, 관의 길이·단면 크기가 충분한지, 실제 마스크 상연이 있고 원본의 상연과 내부 밝기가 구분되는지 확인했습니다. 후보 관이 여럿 비슷하면 교체를 보류합니다.</p>
<p>이 조건도 외이도와 다른 골내 공간을 완전히 구별하거나 외측 입구를 확정하지 못합니다. 외측으로 가장 멀리 이어진 폐쇄 단면의 상연이라는 운영 정의입니다. 전문가 정답이 없는 상태에서 실제 Po로 확정하지 않습니다. 밝기 차이는 보정된 HU가 아닙니다.</p>
<h2>2. 전원 처리 결과</h2>{table(routes.groupby(['point','route']).size().rename('n').reset_index())}
<h2>3. 85명 분모를 유지한 평가</h2><p>미해결 사례도 all85 진단용 오차에 포함했습니다. 그 값은 원래 자동점으로 계산한 참고값이며 최종 제공값이 아닙니다. complete는 자동 복구 조건 충족 여부일 뿐 전문가 검증 통과가 아닙니다. 현재 방식은 과거 대칭 보정을 포함하는 비교 기준이고 새 규칙에서는 대칭 보정을 실행하지 않았습니다. Or은 별도 재검출을 보류했으며 기존 Or의 정확도를 보장하지 않습니다.</p><div class="scroll">{table(summary)}</div>
<h2>4. 복구 분기가 발동한 5명·6개 Po의 원본 대조</h2><p>단면은 각 후보 위치에서 실제로 잘랐고, 다른 위치의 점은 off-plane 거리를 표시했습니다. 작은 구멍을 외이도로 확정하지 말고 여러 단면의 연속성을 확인해야 합니다.</p>{''.join(cards)}
<h2>5. 전체 170개 점</h2><div class="scroll">{table(routes)}</div>
<h2>규칙·자료</h2><p><a href="point_routes.csv">전원 처리 경로</a> · <a href="measurement_comparison.csv">환자별 오차·미해결</a> · <a href="evaluation.csv">평가표</a> · <a href="protocol.json">고정 규칙·입력</a></p><pre>{escape(json.dumps(RULE,ensure_ascii=False,indent=2))}</pre></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')
    print(routes.groupby(['point','route']).size().to_string());print(summary.to_string(index=False))


def main():
    for f in (DEST,DEST/'cases',DEST/'figs'):f.mkdir(parents=True,exist_ok=True)
    source=json.loads((PREV/'protocol.json').read_text(encoding='utf-8'));assert len(source['patients'])==85
    save(DEST/'protocol.json',dict(rule=RULE,patients=source['patients'],code_sha256=digest(__file__),
                                 source_protocol_sha256=digest(PREV/'protocol.json')))
    cases=[process(p) for p in source['patients']]
    evaluate(cases);print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
