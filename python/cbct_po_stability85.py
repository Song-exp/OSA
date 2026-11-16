"""85명 Po 후보의 반복 조건 안정성과 지표 민감도를 평가. 모든 ALI 점의 확정이 아님."""
from itertools import product
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cbct_po_ensemble import VARIANTS,ACCEPT,detect_variant,stability,metric_decision
from cbct_seg_landmarks85 import DEST as SOURCE,CENTRAL,load_mask,grid,sample,save
from cbct_seg_landmarks_confirmed85 import decide_confirmed
from cbct_patient_pipeline import CB,OUT,landmarks,digest
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd
from batch_cbct_pipeline import LMS
from osa.ceph import measures,head_frame
from osa.refine import front_contour,STEP,MAX_SHIFT

DEST=OUT/'develop85/po_stability_v1'
MAX_COMBINATIONS=4096
MODULE=Path(__file__).with_name('cbct_po_ensemble.py')


def pog_value(points,mask,dirs,org):
    """기존 refine의 Pog 부분만 동일하게 계산하고 A/B는 보정하지 않는다."""
    o,R=head_frame(points,'yao4');me=R@(points['Me']-o);pog=R@(points['Pog']-o)
    sup=np.arange(me[2]+3,me[2]+20+1e-9,STEP)
    front=front_contour(mask,dirs,org,o,R,sup,(pog[1]-25,pog[1]+25))
    if np.isfinite(front).mean()<.5:return float(pog[1])
    best=np.nanmax(front);ties=np.flatnonzero(abs(front-best)<=STEP/2);i=ties[len(ties)//2]
    new=np.array([0.,front[i],sup[i]])
    return float(new[1]) if np.linalg.norm(new-pog)<=MAX_SHIFT else float(pog[1])


def process(patient):
    source_path=SOURCE/'cases'/f'{patient}.json'
    source=json.loads(source_path.read_text(encoding='utf-8'))
    support_path=OUT/f'develop85/fh_quality_v1/cases/{patient}.json'
    support=json.loads(support_path.read_text(encoding='utf-8'))
    P,files=landmarks(int(patient[2:]));lm={f.name:digest(f) for f in files}
    assert lm==source['fingerprint']['landmarks']==support['fingerprint']['landmarks']
    scan_path=CB/f'nrrd/{patient}.nrrd';seg_path=CB/f'seg/{patient}_seg_CB.nrrd'
    fingerprint=dict(image=digest(scan_path),segmentation=digest(seg_path),landmarks=lm,
                     source=digest(source_path),pixel_support=digest(support_path),
                     code=digest(__file__),algorithm=digest(MODULE))
    assert fingerprint['image']==source['fingerprint']['image']==support['fingerprint']['image']
    assert fingerprint['segmentation']==source['fingerprint']['segmentation']['CB']
    path=DEST/'cases'/f'{patient}.json'
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf-8'))
        if cached['fingerprint']==fingerprint:
            print(patient,'cached',flush=True);return cached
    o,R,_=midline_frame({k:P[k] for k in CENTRAL})
    volume,dirs,org=read_nrrd(scan_path);mask,mdirs,morg=load_mask(patient,'CB')
    assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
    case=dict(Patient=patient,fingerprint=fingerprint,points={})
    for side in ('RPo','LPo'):
        old=source['points'][side];seed=np.array(old['transferred_seed_lps'])
        m,axes,inside=grid(mask,mdirs,morg,o,R,seed,half=(15.,18.,18.))
        G=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
        ct=sample(volume,dirs,org,o+G@R,order=1,cval=np.nan)
        local_seed=R@(seed-o)
        results=[detect_variant(m,ct,axes,local_seed,v) for v in VARIANTS]
        s=stability(results)
        s['candidates_lps']=o+s['candidates']@R if s['candidate_count'] else np.empty((0,3))
        s['representative_lps']=o+s['representative']@R if s['representative'] is not None else None
        decision=decide_confirmed(P[side],old,(o,R),support['original'][side]['nearest_dense_mm'])
        triggered=bool(decision['changed'] or decision['route']=='needs_review_no_reliable_candidate')
        source_ok=(not old['registration_fit_flag'] and old['transfer']['local_transfer_residual_mm']<=3.
                   and old['transfer']['local_transfer_condition']<=100. and inside.mean()>=.98)
        if not source_ok:s['adequate_support']=False;s['status']='unreliable_roi_source'
        if not triggered:route='existing_point_retained_not_confirmed'
        elif s['status']=='stable_single_candidate':route='single_reconstruction_candidate'
        elif s['adequate_support']:route='candidate_region'
        else:route='automatic_withhold'
        case['points'][side]=dict(stability=s,variants=results,triggered=triggered,route=route,
                                  original_point=P[side],source_ok=source_ok,
                                  roi_inside_fraction=float(inside.mean()),exception_reasons=decision['reasons'])
        if triggered:
            np.savez_compressed(DEST/'roi'/f'{patient}_{side}.npz',mask=m,ct=ct,L=axes[0],A=axes[1],S=axes[2],origin=o,R=R)
        a,b=s['families']['AMASSS'],s['families']['native_CT']
        print(patient,side,route,f'A {a["successes"]}/13 B {b["successes"]}/3',flush=True)
    save(path,case)
    return json.loads(path.read_text(encoding='utf-8'))


def metric_values(P,Rpoints,Lpoints,mand,dirs,org):
    n=len(Rpoints)*len(Lpoints)
    if n>MAX_COMBINATIONS:return None,n
    result={m:[] for m in ('ANB','MP-FH','Nperp-A','Nperp-Pog')}
    for right,left in product(Rpoints,Lpoints):
        Q=dict(P);Q['RPo']=np.asarray(right);Q['LPo']=np.asarray(left)
        m3=measures(Q,'jung');m4=measures(Q,'yao4')
        result['ANB'].append(m3['ANB']);result['MP-FH'].append(m4['MPFH_GoMe'])
        result['Nperp-A'].append(m4['Nperp_A']);result['Nperp-Pog'].append(pog_value(Q,mand,dirs,org))
    return result,n


def evaluate(cases):
    point_rows=[];variant_rows=[];scope=[]
    for c in cases:
        for key in LMS:
            scope.append(dict(Patient=c['Patient'],landmark=key,
                              scope='Po_candidate_stability_only' if key in ('RPo','LPo') else 'not_assessed_in_this_experiment',
                              anatomically_confirmed=False))
        for side,r in c['points'].items():
            s=r['stability'];a,b=s['families']['AMASSS'],s['families']['native_CT']
            point_rows.append(dict(Patient=c['Patient'],point=side,triggered=r['triggered'],route=r['route'],
                                   stability_status=s['status'],mask_success_fraction=a['success_fraction'],
                                   native_success_fraction=b['success_fraction'],mask_p90_radius=a['p90_radius_mm'],
                                   native_p90_radius=b['p90_radius_mm'],family_distance_mm=s['family_distance_mm'],
                                   candidate_count=s['candidate_count'],adequate_support=s['adequate_support']))
            for v in r['variants']:
                point=v['winner']
                variant_rows.append(dict(Patient=c['Patient'],point=side,variant=v['variant'],family=v['family'],
                                         status=v['status'],candidate_count=len(v['eligible']),
                                         L=point[0] if point is not None else np.nan,A=point[1] if point is not None else np.nan,
                                         S=point[2] if point is not None else np.nan,native_gray_threshold=v['threshold']))
    points=pd.DataFrame(point_rows);variants=pd.DataFrame(variant_rows)
    points.to_csv(DEST/'point_stability.csv',index=False,encoding='utf-8-sig')
    variants.to_csv(DEST/'variant_candidates.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scope).to_csv(DEST/'landmark_scope.csv',index=False,encoding='utf-8-sig')
    # 후보 선택·안정성 판정을 마친 뒤에만 비교값을 읽는다.
    prior=pd.read_csv(OUT/'develop85/v1/error_ranking.csv').set_index(['Patient','metric'])
    rows=[]
    for c in cases:
        p=c['Patient'];P=landmarks(int(p[2:]))[0]
        triggered=any(r['triggered'] for r in c['points'].values())
        if not triggered:
            for metric in ACCEPT['metric_width']:
                old=float(prior.loc[(p,metric),'automatic'])
                rows.append(dict(Patient=p,metric=metric,status='existing_value_retained_unverified',value=old,
                                 low=old,high=old,range_width=0.,combinations=1,adequate=True,
                                 reference=float(prior.loc[(p,metric),'excel']),legacy_value=old))
            continue
        sets=[];adequate=True
        for side in ('RPo','LPo'):
            r=c['points'][side];s=r['stability']
            sets.append(np.asarray(s['candidates_lps'],float).reshape(-1,3) if r['triggered'] else np.asarray([P[side]]))
            if r['triggered']:adequate=adequate and s['adequate_support']
        mand,dirs,org=load_mask(p,'MAND')
        samples,n=metric_values(P,sets[0],sets[1],mand,dirs,org);del mand
        if samples is None:adequate=False
        for metric in ACCEPT['metric_width']:
            vals=samples[metric] if samples is not None else []
            decision=metric_decision(vals,adequate,metric)
            width=decision['high']-decision['low'] if decision['low'] is not None else None
            rows.append(dict(Patient=p,metric=metric,**decision,range_width=width,combinations=n,adequate=adequate,
                             reference=float(prior.loc[(p,metric),'excel']),legacy_value=float(prior.loc[(p,metric),'automatic'])))
    metric=pd.DataFrame(rows);metric['legacy_error']=(metric.legacy_value-metric.reference).abs()
    metric['output_error']=(metric.value-metric.reference).abs()
    metric.to_csv(DEST/'metric_restoration.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for m,g in metric.groupby('metric'):
        available=g[g.value.notna()]
        summaries.append(dict(metric=m,total_n=len(g),numeric_n=len(available),
                              retained_n=int((g.status=='existing_value_retained_unverified').sum()),
                              reconstructed_n=int((g.status=='candidate_stable_metric').sum()),
                              interval_only_n=int((g.status=='candidate_interval_only').sum()),
                              withheld_n=int((g.status=='automatic_withhold').sum()),
                              legacy_all85_MAE=g.legacy_error.mean(),same_numeric_legacy_MAE=available.legacy_error.mean(),
                              same_numeric_output_MAE=available.output_error.mean(),
                              withheld_legacy_MAE=g[g.value.isna()].legacy_error.mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(DEST/'evaluation.csv',index=False,encoding='utf-8-sig')
    report(cases,points,metric,summary)
    print(points.groupby(['point','route']).size().to_string());print(summary.to_string(index=False))


def figures(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    for c in cases:
        for side,r in c['points'].items():
            if not r['triggered']:continue
            p=c['Patient'];z=np.load(DEST/'roi'/f'{p}_{side}.npz');axes=[z[k] for k in ('L','A','S')]
            s=r['stability'];points=[];colors=[]
            for v in r['variants']:
                if v['winner'] is not None:points.append(v['winner']);colors.append('#2a78d6' if v['family']=='AMASSS' else '#eb6834')
            pts=np.asarray(points,float).reshape(-1,3)
            center=np.median(pts,axis=0) if len(pts) else np.array([a[len(a)//2] for a in axes])
            fig=plt.figure(figsize=(14,11));cloud=fig.add_subplot(221,projection='3d')
            if len(pts):cloud.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors,s=35)
            cloud.set_xlabel('L mm');cloud.set_ylabel('A mm');cloud.set_zlabel('S mm')
            cloud.set_title('조건별 우승 후보: 파랑 AMASSS / 주황 원본 CT')
            for col,axis in enumerate((0,1,2)):
                ax=fig.add_subplot(2,2,col+2);i=int(np.argmin(abs(axes[axis]-center[axis])))
                X,Y=[axes[j] for j in range(3) if j!=axis];image=np.take(z['ct'],i,axis=axis).T;bone=np.take(z['mask'],i,axis=axis).T
                ext=[X[0]-.25,X[-1]+.25,Y[0]-.25,Y[-1]+.25]
                ax.imshow(image,origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=ext)
                if bone.any():ax.contour(X,Y,bone.astype(float),[.5],colors=['#00c8ff'],linewidths=.7)
                # 다른 깊이 후보를 실제 단면 위의 점처럼 표시하지 않는다.
                if len(pts):
                    near=abs(pts[:,axis]-axes[axis][i])<=.75;ij=[j for j in range(3) if j!=axis]
                    ax.scatter(pts[near,ij[0]],pts[near,ij[1]],c=np.asarray(colors)[near],s=22)
                ax.set_title(('시상','관상','축상')[col]+' 실제 단면: 단면에서 0.75 mm 이내 후보만 표시')
            fig.suptitle(f'{p} {side}: {r["route"]} / 조건 변화 안정성은 해부학적 정답 보장이 아님',fontsize=12)
            fig.tight_layout(rect=(0,0,1,.95));fig.savefig(DEST/'figs'/f'{p}_{side}.png',dpi=110);plt.close(fig)


def report(cases,points,metric,summary):
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    cards=[]
    for c in cases:
        for side,r in c['points'].items():
            if not r['triggered']:continue
            p=c['Patient'];cards.append(f'<article><h3>{p} {side}</h3>{table(points[(points.Patient==p)&(points.point==side)])}'
                                       f'<a href="figs/{p}_{side}.png"><img src="figs/{p}_{side}.png"></a>'
                                       f'{table(metric[metric.Patient==p])}<p><a href="cases/{p}.json">전체 후보·근거 JSON</a></p></article>')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Po 후보 안정성과 지표 자동 복원</title>
<style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}table{{border-collapse:collapse;background:white;font-size:12px}}td,th{{border:1px solid #ddd;padding:5px}}.scroll{{overflow:auto}}article{{background:white;padding:16px;margin:24px 0;overflow:auto}}img{{max-width:100%}}h2{{margin-top:36px}}</style><main>
<h1>85명 Po 후보의 조건 변화 안정성·측정 영향</h1>
<p><b>이 실험으로 ALI-CBCT 기준점 전체가 확정되지는 않습니다.</b> 16개 두개 기준점 중 RPo·LPo의 후보 안정성을 평가했습니다. 나머지 14개 점은 이번 실험에서 평가하지 않았습니다. 안정적인 오검출도 가능하며 해부학적 정답을 확인한 것이 아닙니다. 기존 좌표·보고서는 변경하지 않았습니다.</p>
<h2>1. 고정 절차</h2><p>각 Po에서 AMASSS 기반 13조건(탐색 중심, 격자 해상도, 마스크 처리, 원본 대비 조건)과 원본 CT의 적응 밝기 경계 3조건을 실행했습니다. 원본 CT 경계는 고정 HU가 아닌 ROI Otsu 문턱과 ±100 영상값입니다. 두 방법은 같은 영상·ROI·관 모양 규칙을 공유하므로 통계적으로 독립적인 검증이 아닙니다. AI 모델을 재추론하거나 학습하지 않았습니다.</p>
<p>후보 생성·안정성 판정에는 엑셀을 사용하지 않았습니다. 기준은 성능을 본 뒤 환자별로 바꾸지 않았습니다. 성공률은 조건별 후보 생성 비율이며 정확도 확률이 아닙니다. 후보의 90% 반경·범위는 신뢰구간이 아닙니다.</p>
<h2>2. 점과 지표를 분리해 반환</h2><p>점은 단일 안정 후보 / 후보 영역 / 근거 부족으로 나눴습니다. 기존 점에서 예외 조건이 발동하지 않은 경우는 기존값 유지로 따로 표시하며 검증된 점으로 세지 않습니다. 지표는 실제 후보 조합을 모두 계산해 변동폭이 작으면 중앙값, 크면 후보 범위만, 근거가 부족하면 자동 보류를 반환합니다. 후보 조합 4096개 초과는 임의 표본을 뽑지 않고 보류합니다.</p>
<p>변동폭 운영 기준은 ANB·MP-FH 1° 이내, Nperp 2 mm 이내입니다. 임상 허용오차가 아닙니다. Nperp-Pog는 각 후보 FH에 맞춰 기존과 동일한 Pog 윤곽 보정을 계산했습니다. 모든 규칙은 연구용 제안이며 자동 복원값도 해부학적 정답 보장이 아닙니다.</p>
<h2>3. 전체 제공률과 비교</h2><div class="scroll">{table(summary)}</div>
<p>legacy_all85_MAE는 전체 기준 오차이고 same_numeric_*는 같은 숫자 제공 환자끼리의 비교입니다. 보류 환자의 기준 오차도 공개했습니다. 표본 제외로 좋아진 평균을 개선으로 해석하지 않습니다. 이번 85명은 이미 개발에 사용한 자료로 독립 검증이 아닙니다. 과거 대칭 보정값은 비교 기준일 뿐 새 복구에는 사용하지 않았습니다.</p>
<h2>4. 전체 170개 Po</h2><div class="scroll">{table(points)}</div>
<h2>5. 복구 분기 사례</h2>{''.join(cards)}
<h2>6. 전체 지표 반환 결과</h2><div class="scroll">{table(metric)}</div>
<h2>자료</h2><p><a href="point_stability.csv">Po 안정성</a> · <a href="variant_candidates.csv">조건별 전체 결과</a> · <a href="metric_restoration.csv">지표별 값·범위·보류</a> · <a href="landmark_scope.csv">85명×16개 점의 평가 범위</a> · <a href="evaluation.csv">평가표</a> · <a href="protocol.json">사전 고정 규칙</a></p><pre>{escape(json.dumps(ACCEPT,ensure_ascii=False,indent=2))}</pre></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--patients',nargs='*',type=int);args=ap.parse_args()
    for p in (DEST,DEST/'cases',DEST/'roi',DEST/'figs'):p.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    patients=source['patients']
    if args.patients:patients=[p for p in patients if int(p[2:]) in args.patients]
    save(DEST/'protocol.json',dict(patients=patients,variants=VARIANTS,acceptance=ACCEPT,
          max_combinations=MAX_COMBINATIONS,driver_sha256=digest(__file__),algorithm_sha256=digest(MODULE),
          source_protocol_sha256=digest(SOURCE/'protocol.json'),no_all_landmark_confirmation=True))
    cases=[process(p) for p in patients]
    figures(cases);evaluate(cases)
    print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
