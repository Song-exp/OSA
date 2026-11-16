"""한 번에 실행하는 Po 검증 캐시 재사용 + Or 후보 안정성 + FH 지표 통합 반환.

python/cbct_fh_joint85.py
이전 보고서·좌표는 변경하지 않는다. 모든 새 값은 자동 규칙 기반 연구용 복원값이다.
"""
from itertools import product
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cbct_po_stability85 as po
import cbct_or_ensemble as orbit
from cbct_po_ensemble import stability,metric_decision,ACCEPT
from cbct_seg_landmarks85 import DEST as SOURCE,CENTRAL,load_mask,grid,sample,save
from cbct_seg_landmarks_confirmed85 import decide_confirmed
from cbct_patient_pipeline import CB,OUT,landmarks,digest
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd
from osa.ceph import measures
from batch_cbct_pipeline import LMS

DEST=OUT/'develop85/fh_joint_v1'
METRICS=('ANB','MP-FH','Nperp-A','Nperp-Pog')
KEYS=('RPo','LPo','ROr','LOr')
MAX_COMBINATIONS=4096


def required_points(metric):
    return ('RPo','LPo','ROr') if metric=='ANB' else KEYS


def process_or(patient,po_case):
    source_path=SOURCE/'cases'/f'{patient}.json'
    source=json.loads(source_path.read_text(encoding='utf-8'))
    support_path=OUT/f'develop85/fh_quality_v1/cases/{patient}.json'
    support=json.loads(support_path.read_text(encoding='utf-8'))
    P,files=landmarks(int(patient[2:]));lm={f.name:digest(f) for f in files}
    assert lm==source['fingerprint']['landmarks']==po_case['fingerprint']['landmarks']
    assert lm==support['fingerprint']['landmarks']
    scan=CB/f'nrrd/{patient}.nrrd';seg=CB/f'seg/{patient}_seg_MAX.nrrd'
    fingerprint=dict(code=digest(__file__),algorithm=digest(Path(orbit.__file__)),
                     common=digest(Path(orbit.__file__).with_name('cbct_po_ensemble.py')),
                     image=digest(scan),MAX=digest(seg),source=digest(source_path),support=digest(support_path),
                     landmarks=lm,Po_case=digest(po.DEST/'cases'/f'{patient}.json'))
    assert fingerprint['image']==po_case['fingerprint']['image']==support['fingerprint']['image']
    assert fingerprint['MAX']==source['fingerprint']['segmentation']['MAX']
    path=DEST/'or_cases'/f'{patient}.json'
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf-8'))
        if cached['fingerprint']==fingerprint:
            print(patient,'Or cached',flush=True);return cached
    volume,dirs,org=read_nrrd(scan);mask,mdirs,morg=load_mask(patient,'MAX')
    assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
    o,R,_=midline_frame({k:P[k] for k in CENTRAL})
    case=dict(Patient=patient,fingerprint=fingerprint,points={})
    for key in ('ROr','LOr'):
        old=source['points'][key];seed=np.array(old['transferred_seed_lps']);local_seed=R@(seed-o)
        m,axes,inside=grid(mask,mdirs,morg,o,R,seed,half=(15.,18.,18.))
        G=np.stack(np.meshgrid(*axes,indexing='ij'),-1);ct=sample(volume,dirs,org,o+G@R,order=1,cval=np.nan)
        variants=[orbit.detect_variant(m,ct,axes,local_seed,v) for v in orbit.VARIANTS]
        s=stability(variants,orbit.VARIANTS)
        s['candidates_lps']=o+s['candidates']@R if s['candidate_count'] else np.empty((0,3))
        s['representative_lps']=o+s['representative']@R if s['representative'] is not None else None
        d=decide_confirmed(P[key],old,(o,R),support['original'][key]['nearest_dense_mm'])
        triggered=bool(d['changed'] or d['route']=='needs_review_no_reliable_candidate')
        source_ok=(not old['registration_fit_flag'] and old['transfer']['local_transfer_residual_mm']<=3.
                   and old['transfer']['local_transfer_condition']<=100. and inside.mean()>=.98)
        if not source_ok:s['adequate_support']=False;s['status']='unreliable_roi_source'
        route=('existing_point_retained_not_confirmed' if not triggered else
               'single_reconstruction_candidate' if s['status']=='stable_single_candidate' else
               'candidate_region' if s['adequate_support'] else 'automatic_withhold')
        case['points'][key]=dict(stability=s,variants=variants,triggered=triggered,route=route,
                                 source_ok=source_ok,original_point=P[key],roi_inside_fraction=float(inside.mean()),
                                 exception_reasons=d['reasons'])
        if triggered or patient=='P_5':
            np.savez_compressed(DEST/'roi'/f'{patient}_{key}.npz',mask=m,ct=ct,L=axes[0],A=axes[1],S=axes[2],origin=o,R=R)
        a,b=s['families']['AMASSS'],s['families']['native_CT']
        print(patient,key,route,f'A {a["successes"]}/13 B {b["successes"]}/3',flush=True)
    save(path,case)
    return json.loads(path.read_text(encoding='utf-8'))


def variants_for_metric(points,P,metric,mode):
    arrays=[];active=[];adequate=True
    for key in required_points(metric):
        r=points[key]
        enabled=(mode=='joint' or (mode=='Po_only' and key.endswith('Po')) or (mode=='Or_only' and key.endswith('Or')))
        if enabled and r['triggered']:
            active.append(key);arrays.append(np.asarray(r['stability']['candidates_lps'],float).reshape(-1,3))
            adequate=adequate and r['stability']['adequate_support']
        else:arrays.append(np.asarray([P[key]]))
    return arrays,active,adequate


def calculate_metric(P,metric,arrays,mand,dirs,org):
    n=int(np.prod([len(x) for x in arrays]))
    if n>MAX_COMBINATIONS:return None,n
    keys=required_points(metric);values=[]
    for combo in product(*arrays):
        Q=dict(P)
        for key,x in zip(keys,combo):Q[key]=np.asarray(x)
        if metric=='Nperp-Pog':value=po.pog_value(Q,mand,dirs,org)
        else:
            source=measures(Q,'jung' if metric=='ANB' else 'yao4')
            value=source[{'ANB':'ANB','MP-FH':'MPFH_GoMe','Nperp-A':'Nperp_A'}[metric]]
        values.append(float(value))
    return values,n


def evaluate(cases):
    point_rows=[];variant_rows=[];scopes=[]
    for c in cases:
        for key,r in c['points'].items():
            s=r['stability'];a,b=s['families']['AMASSS'],s['families']['native_CT']
            point_rows.append(dict(Patient=c['Patient'],point=key,triggered=r['triggered'],route=r['route'],
                                   stability_status=s['status'],mask_success_fraction=a['success_fraction'],
                                   native_success_fraction=b['success_fraction'],family_distance_mm=s['family_distance_mm'],
                                   candidate_count=s['candidate_count'],adequate_support=s['adequate_support']))
            for v in r['variants']:
                variant_rows.append(dict(Patient=c['Patient'],point=key,variant=v['variant'],family=v['family'],
                                         status=v['status'],eligible_candidates=len(v['eligible'])))
        for key in LMS:
            scopes.append(dict(Patient=c['Patient'],landmark=key,anatomically_confirmed=False,
                               scope='FH_point_candidate_stability' if key in KEYS else 'not_assessed_in_this_workflow'))
    points=pd.DataFrame(point_rows);points.to_csv(DEST/'point_stability.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(variant_rows).to_csv(DEST/'variant_candidates.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scopes).to_csv(DEST/'landmark_scope.csv',index=False,encoding='utf-8-sig')
    # 전원 후보 선택을 마친 뒤에만 비교 기준을 읽는다.
    prior=pd.read_csv(OUT/'develop85/v1/error_ranking.csv').set_index(['Patient','metric'])
    rows=[]
    for c in cases:
        p=c['Patient'];P=landmarks(int(p[2:]))[0]
        needs_mask=any(r['triggered'] for r in c['points'].values())
        if needs_mask:mand,dirs,org=load_mask(p,'MAND')
        else:mand=dirs=org=None
        for mode in ('Po_only','Or_only','joint'):
            for metric in METRICS:
                arrays,active,adequate=variants_for_metric(c['points'],P,metric,mode)
                old=float(prior.loc[(p,metric),'automatic'])
                if not active:
                    decision=metric_decision([old],True,metric,legacy=True);n=1
                else:
                    values,n=calculate_metric(P,metric,arrays,mand,dirs,org)
                    if values is None:adequate=False;values=[]
                    decision=metric_decision(values,adequate,metric)
                rows.append(dict(Patient=p,mode=mode,metric=metric,**decision,
                                 active_points=';'.join(active),adequate=adequate,combinations=n,
                                 range_width=decision['high']-decision['low'] if decision['low'] is not None else None,
                                 reference=float(prior.loc[(p,metric),'excel']),legacy_value=old))
        if needs_mask:del mand
    metric=pd.DataFrame(rows);metric['legacy_error']=(metric.legacy_value-metric.reference).abs()
    metric['output_error']=(metric.value-metric.reference).abs()
    metric.to_csv(DEST/'metric_restoration.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for (mode,m),g in metric.groupby(['mode','metric']):
        valid=g[g.value.notna()]
        summary.append(dict(mode=mode,metric=m,total_n=len(g),numeric_n=len(valid),
                            retained_n=int((g.status=='existing_value_retained_unverified').sum()),
                            reconstructed_n=int((g.status=='candidate_stable_metric').sum()),
                            interval_only_n=int((g.status=='candidate_interval_only').sum()),
                            withheld_n=int((g.status=='automatic_withhold').sum()),
                            legacy_all85_MAE=g.legacy_error.mean(),same_numeric_legacy_MAE=valid.legacy_error.mean(),
                            same_numeric_output_MAE=valid.output_error.mean(),withheld_legacy_MAE=g[g.value.isna()].legacy_error.mean()))
    summary=pd.DataFrame(summary);summary.to_csv(DEST/'evaluation.csv',index=False,encoding='utf-8-sig')
    report(cases,points,metric,summary)
    print('OR ROUTES');print(points[points.point.str.endswith('Or')].groupby(['point','route']).size().to_string())
    print('JOINT');print(summary[summary['mode']=='joint'].to_string(index=False))
    return points,metric,summary


def figures(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    for c in cases:
        for key,r in c['points'].items():
            if not key.endswith('Or') or (not r['triggered'] and c['Patient']!='P_5'):continue
            p=c['Patient'];z=np.load(DEST/'roi'/f'{p}_{key}.npz');axes=[z[k] for k in ('L','A','S')]
            points=[];colors=[]
            for v in r['variants']:
                if v['winner'] is not None:points.append(v['winner']);colors.append('#2a78d6' if v['family']=='AMASSS' else '#eb6834')
            pts=np.asarray(points,float).reshape(-1,3)
            center=np.median(pts,axis=0) if len(pts) else np.array([x[len(x)//2] for x in axes])
            original=z['R']@(np.asarray(r['original_point'])-z['origin'])
            fig=plt.figure(figsize=(14,11));ax=fig.add_subplot(221,projection='3d')
            if len(pts):ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors,s=30)
            ax.scatter(*original,c='red',marker='+',s=100);ax.set_xlabel('L');ax.set_ylabel('A');ax.set_zlabel('S')
            ax.set_title('조건별 Or 후보: 파랑 분할 / 주황 원본 / 빨강 기존점')
            for col,axis in enumerate((0,1,2)):
                ax=fig.add_subplot(2,2,col+2);i=int(np.argmin(abs(axes[axis]-center[axis])))
                X,Y=[axes[j] for j in range(3) if j!=axis]
                image=np.take(z['ct'],i,axis=axis).T;bone=np.take(z['mask'],i,axis=axis).T
                ax.imshow(image,origin='lower',cmap='gray',vmin=-700,vmax=2000,
                          extent=[X[0]-.25,X[-1]+.25,Y[0]-.25,Y[-1]+.25])
                if bone.any():ax.contour(X,Y,bone.astype(float),[.5],colors=['#00c8ff'],linewidths=.7)
                ij=[j for j in range(3) if j!=axis]
                if len(pts):
                    near=abs(pts[:,axis]-axes[axis][i])<=.75
                    ax.scatter(pts[near,ij[0]],pts[near,ij[1]],c=np.asarray(colors)[near],s=20)
                if abs(original[axis]-axes[axis][i])<=.75:ax.plot(original[ij[0]],original[ij[1]],'r+',ms=12)
                ax.set_title(('시상','관상','축상')[col]+' 실제 단면 / 0.75 mm 이내 점만 표시')
            fig.suptitle(f'{p} {key}: {r["route"]} / 앞쪽 하연 전환부 후보, 해부학적 정답 미확정',fontsize=12)
            fig.tight_layout(rect=(0,0,1,.95));fig.savefig(DEST/'figs'/f'{p}_{key}.png',dpi=110);plt.close(fig)


def report(cases,points,metric,summary):
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    cards=[]
    for c in cases:
        p=c['Patient']
        or_active=[k for k in ('ROr','LOr') if c['points'][k]['triggered']]
        if not or_active and p!='P_5':continue
        images=''.join(f'<a href="figs/{p}_{k}.png"><img src="figs/{p}_{k}.png"></a>' for k in ('ROr','LOr') if (DEST/'figs'/f'{p}_{k}.png').exists())
        cards.append(f'<article><h3>{p}: Or 국소 경계·후보</h3>{table(points[(points.Patient==p)&points.point.str.endswith("Or")])}{images}'
                     f'{table(metric[(metric.Patient==p)&(metric["mode"]=="joint")])}</article>')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Po·Or 자동 복원 통합 워크플로</title>
<style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}table{{border-collapse:collapse;background:white;font-size:12px}}td,th{{border:1px solid #ddd;padding:5px}}img{{max-width:100%}}.scroll{{overflow:auto}}article{{background:white;margin:24px 0;padding:16px;overflow:auto}}h2{{margin-top:36px}}</style><main>
<h1>Po·Or 자동 규칙 기반 복원: 한 번에 실행하는 85명 워크플로</h1>
<p>실행: <code>.venv\\Scripts\\python.exe python/cbct_fh_joint85.py</code>. Po의 검증 가능한 계산 캐시를 재사용하거나 재계산하고, Or 16조건 후보 생성·안정성·Po/Or 결합 지표 반환·평가를 실행합니다. 기존 좌표와 보고서는 변경하지 않습니다.</p>
<p><b>기준점 전체가 확정된 것은 아닙니다.</b> Po·Or 4종만 후보 안정성을 평가했고 나머지 12종은 이번 워크플로에서 평가하지 않았습니다. 기존값 유지도 정확도 확인을 뜻하지 않습니다. 안정적인 오검출이 가능하며 모든 새 값은 연구용 자동 복원 제안입니다.</p>
<h2>1. Or 탐색 보완</h2><p>안와저 전체에서 낮은 뼈 점을 고르는 대신, 분할 상연이 앞쪽 외면으로 내려가는 전환부를 먼저 제한하고 그 가장자리의 좌우 방향 저점을 후보로 삼았습니다. MAX와 원본 CT 경계를 각각 사용해 총 16조건으로 평가했습니다. 이 전환부는 안와 하연의 운영상 대리 조건이며 실제 안와 하연 전용 분할이 아니므로 정답 보장이 아닙니다.</p>
<p>후보 선택에 좌우 대칭·환자 번호·엑셀 오차를 쓰지 않았습니다. 두 방법은 동일 영상·ROI·형태 규칙을 공유해 독립 검증으로 볼 수 없습니다. 원본 밝기 문턱은 ROI 적응값이며 보정된 HU가 아닙니다.</p>
<h2>2. Po·Or 결합 반환 결과</h2><div class="scroll">{table(summary[summary['mode']=='joint'])}</div>
<p>numeric_n은 숫자 제공 인원이고, retained는 기존값 유지, reconstructed는 후보 조합의 변동폭이 기준 이내인 복원값, interval은 범위만 제공, withheld는 근거 부족입니다. 85명 전원을 분모에 남겼고 보류 환자의 이전 오차도 공개했습니다. 같은 숫자 제공 환자끼리의 오차만 직접 비교해야 합니다.</p>
<h2>3. 단계별 비교</h2><div class="scroll">{table(summary)}</div>
<p>ANB는 기존 정의대로 RPo·LPo·ROr만 사용하므로 LOr만 불확실하다는 이유로 보류하지 않습니다. 나머지 세 지표는 양쪽 Po·Or의 모든 유효 후보 조합을 계산합니다. 조합 4096개 초과는 임의 표본으로 범위를 좁히지 않고 보류합니다. Nperp-Pog에는 후보별 FH에 맞춘 기존 Pog 윤곽 보정을 적용합니다.</p>
<h2>4. 전원 340개 FH 기준점</h2><div class="scroll">{table(points)}</div>
<h2>5. Or 실제 단면과 후보</h2>{''.join(cards)}
<h2>6. 통합 지표 반환</h2><div class="scroll">{table(metric[metric['mode']=='joint'])}</div>
<h2>자료</h2><p><a href="point_stability.csv">Po·Or 안정성</a> · <a href="variant_candidates.csv">조건별 후보</a> · <a href="metric_restoration.csv">단계별 지표 반환</a> · <a href="landmark_scope.csv">평가한 점과 미평가 점</a> · <a href="evaluation.csv">제공률·오차</a> · <a href="protocol.json">고정 규칙과 입력 해시</a></p>
<p>점 반경과 후보 지표 범위는 통계적 신뢰구간이 아니며 운영 기준은 임상 허용오차가 아닙니다. 85명은 이미 개발에 사용한 자료로 독립 검증이 아닙니다. 과거 대칭 보정값은 비교 기준으로만 남겼으며 새 복구에서는 대칭 좌표를 생성하지 않았습니다.</p></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--patients',nargs='*',type=int);args=ap.parse_args()
    for p in (DEST,DEST/'or_cases',DEST/'cases',DEST/'roi',DEST/'figs'):p.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'));patients=source['patients']
    if args.patients:patients=[p for p in patients if int(p[2:]) in args.patients]
    save(DEST/'protocol.json',dict(version='fh-joint-experimental-v1',patients=patients,driver=digest(__file__),
          Or_algorithm=digest(Path(orbit.__file__)),Po_driver=digest(Path(po.__file__)),
          Po_algorithm=digest(po.MODULE),source_protocol=digest(SOURCE/'protocol.json'),
          Or_rule=orbit.RULE,Or_variants=orbit.VARIANTS,acceptance=ACCEPT,
          max_combinations=MAX_COMBINATIONS,all_landmarks_confirmed=False,
          Po_cache='validated against live image, CB and landmark hashes by Po process'))
    cases=[]
    for p in patients:
        pc=po.process(p);oc=process_or(p,pc)
        case=dict(Patient=p,points={**pc['points'],**oc['points']},
                  source_hashes=dict(Po=digest(po.DEST/'cases'/f'{p}.json'),Or=digest(DEST/'or_cases'/f'{p}.json')))
        save(DEST/'cases'/f'{p}.json',case);cases.append(case)
    figures(cases);evaluate(cases)
    print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
