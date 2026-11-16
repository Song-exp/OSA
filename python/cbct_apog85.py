"""85명 A/Pog 독립 후보·선택적 보정과 FH 불확실성 결합 평가."""
from itertools import product
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cbct_apog_ensemble as engine
from cbct_po_ensemble import stability,metric_decision,ACCEPT
from cbct_seg_landmarks85 import physical_correspondences,local_transfer,load_mask,sample,save,CENTRAL
from cbct_patient_pipeline import CB,OUT,ROOT,landmarks,digest
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd
from cbct_po_stability85 import pog_value
from cbct_fh_joint85 import DEST as FH_SOURCE,required_points
from batch_cbct_pipeline import LMS
from osa.ceph import measures

DEST=OUT/'develop85/apog_v1'
TARGETS={'A':'MAX','Pog':'MAND'}
METRICS=('ANB','Nperp-A','Nperp-Pog','MP-FH')
MAX_COMBINATIONS=4096


def reference_seed(patient,key,z,annotations):
    used=[i for i in engine.RULE['reference_patients'] if f'P_{i}'!=patient]
    target=physical_correspondences(z,patient);seeds=[];records=[]
    for pid in used:
        ref=physical_correspondences(z,f'P_{pid}')
        center,info=local_transfer(ref,target,annotations[pid][key])
        seeds.append(center);records.append(dict(patient=f'P_{pid}',**info))
    return np.median(seeds,axis=0),dict(reference_ids=[f'P_{i}' for i in used],
                                       reference_seed_spread_mm=float(np.max(np.linalg.norm(np.asarray(seeds)-np.median(seeds,axis=0),axis=1))),
                                       max_transfer_residual_mm=max(r['local_transfer_residual_mm'] for r in records),
                                       max_condition=max(r['local_transfer_condition'] for r in records),references=records)


def make_grid(mask,dirs,org,o,R,seed):
    center=R@(seed-o)
    axes=[np.arange(-3.,3.001,.25),np.arange(center[1]-18,center[1]+18.001,.25),
          np.arange(center[2]-16,center[2]+16.001,.25)]
    G=np.stack(np.meshgrid(*axes,indexing='ij'),-1);q=o+G@R
    return sample(mask,dirs,org,q)>0,axes,q


def process(patient,models,annotations):
    P,files=landmarks(int(patient[2:]))
    scan=CB/f'nrrd/{patient}.nrrd'
    fp=dict(code=digest(__file__),engine=digest(Path(engine.__file__)),
            image=digest(scan),landmarks={f.name:digest(f) for f in files},
            segmentations={s:digest(CB/f'seg/{patient}_seg_{s}.nrrd') for s in ('MAX','MAND')},
            models={s:digest(ROOT/f'Result_py/shape/{s}/points.npz') for s in models},
            reference_annotations={str(i):{f.name:digest(f) for f in landmarks(i)[1]} for i in annotations})
    path=DEST/'cases'/f'{patient}.json'
    if path.exists():
        c=json.loads(path.read_text(encoding='utf-8'))
        if c['fingerprint']==fp:print(patient,'cached',flush=True);return c
    v,dirs,org=read_nrrd(scan);o,R,residual=midline_frame({k:P[k] for k in CENTRAL})
    c=dict(Patient=patient,fingerprint=fp,frame_residual_mm=residual,points={})
    for key,struct in TARGETS.items():
        seed,transfer=reference_seed(patient,key,models[struct],annotations)
        mask,mdirs,morg=load_mask(patient,struct)
        assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
        m,axes,q=make_grid(mask,mdirs,morg,o,R,seed)
        ct=sample(v,dirs,org,q,order=1,cval=np.nan);local=R@(seed-o)
        ans=float((R@(P['ANS']-o))[2])
        results=[engine.detect_variant(m,ct,axes,local,ans,key,var) for var in engine.VARIANTS]
        s=stability(results,engine.VARIANTS)
        s['candidates_lps']=o+s['candidates']@R if s['candidate_count'] else np.empty((0,3))
        s['representative_lps']=o+s['representative']@R if s['representative'] is not None else None
        quality=engine.decide(R@(P[key]-o),local,results,s)
        source_ok=(transfer['max_transfer_residual_mm']<=3. and transfer['max_condition']<=100.)
        if not source_ok:
            s['adequate_support']=False;s['status']='unreliable_reference_transfer'
            if quality['triggered']:quality['route']='automatic_withhold'
        c['points'][key]=dict(stability=s,variants=results,**quality,source_ok=source_ok,
                              original_point=P[key],seed_lps=seed,transfer=transfer,struct=struct)
        if quality['triggered'] or patient in ('P_5','P_10','P_35','P_62','P_67'):
            np.savez_compressed(DEST/'roi'/f'{patient}_{key}.npz',mask=m,ct=ct,L=axes[0],A=axes[1],S=axes[2],origin=o,R=R)
        print(patient,key,quality['route'],f'A {s["families"]["AMASSS"]["successes"]}/13 B {s["families"]["native_CT"]["successes"]}/3',flush=True)
        del mask
    save(path,c)
    return json.loads(path.read_text(encoding='utf-8'))


def target_for(metric):
    return 'A' if metric in ('ANB','Nperp-A') else 'Pog' if metric=='Nperp-Pog' else None


def configurations(P,ap,fh,metric,mode):
    keys=list(required_points(metric));target=target_for(metric)
    if target is not None:keys.append(target)
    arrays=[];active=[];adequate=True
    for key in keys:
        if key in TARGETS:
            r=ap[key];enabled=mode in ('APog_only','joint') or (mode=='A_only' and key=='A') or (mode=='Pog_only' and key=='Pog')
        else:r=fh[key];enabled=mode=='joint'
        if enabled and r['triggered']:
            active.append(key);arrays.append(np.asarray(r['stability']['candidates_lps'],float).reshape(-1,3))
            adequate=adequate and r['stability']['adequate_support']
        else:arrays.append(np.asarray([P[key]]))
    return keys,arrays,active,adequate


def calculate(P,metric,keys,arrays,active,mand,dirs,org):
    n=int(np.prod([len(a) for a in arrays]))
    if n>MAX_COMBINATIONS:return None,n
    values=[]
    for combo in product(*arrays):
        Q=dict(P)
        for key,x in zip(keys,combo):Q[key]=np.asarray(x)
        if metric=='Nperp-Pog' and 'Pog' not in active:value=pog_value(Q,mand,dirs,org)
        else:value=measures(Q,'jung' if metric=='ANB' else 'yao4')[{'ANB':'ANB','Nperp-A':'Nperp_A','Nperp-Pog':'Nperp_Pog','MP-FH':'MPFH_GoMe'}[metric]]
        values.append(float(value))
    return values,n


def evaluate(cases):
    rows=[];variant_rows=[];scopes=[]
    for c in cases:
        p=c['Patient']
        for key,r in c['points'].items():
            s=r['stability'];a,b=s['families']['AMASSS'],s['families']['native_CT']
            rows.append(dict(Patient=p,point=key,route=r['route'],triggered=r['triggered'],stability=s['status'],
                             mask_success_fraction=a['success_fraction'],native_success_fraction=b['success_fraction'],
                             family_distance_mm=s['family_distance_mm'],candidate_count=s['candidate_count'],
                             adequate_support=s['adequate_support'],mask_original_boundary_deviation=r['original_boundary_deviation']['mask'],
                             native_original_boundary_deviation=r['original_boundary_deviation']['native'],
                             reasons=';'.join(r['reasons']),references=';'.join(r['transfer']['reference_ids'])))
            for v in r['variants']:
                variant_rows.append(dict(Patient=p,point=key,variant=v['variant'],family=v['family'],status=v['status'],
                                         eligible_candidates=len(v['eligible']),winner=v['winner']))
        for k in LMS:
            scopes.append(dict(Patient=p,landmark=k,scope='APog_stability' if k in TARGETS else
                               'FH_stability_from_previous_stage' if k in ('RPo','LPo','ROr','LOr') else 'not_assessed',anatomically_confirmed=False))
    points=pd.DataFrame(rows);points.to_csv(DEST/'point_stability.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(variant_rows).to_csv(DEST/'variant_candidates.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scopes).to_csv(DEST/'landmark_scope.csv',index=False,encoding='utf-8-sig')
    # 후보·교체 판단을 모두 마친 뒤에만 엑셀 비교값을 읽는다.
    prior=pd.read_csv(OUT/'develop85/v1/error_ranking.csv').set_index(['Patient','metric'])
    measurements=[]
    for c in cases:
        p=c['Patient'];P=landmarks(int(p[2:]))[0]
        fh=json.loads((FH_SOURCE/'cases'/f'{p}.json').read_text(encoding='utf-8'))['points']
        needs_mask=any(r['triggered'] for r in c['points'].values()) or any(r['triggered'] for r in fh.values())
        if needs_mask:mand,dirs,org=load_mask(p,'MAND')
        else:mand=dirs=org=None
        for mode in ('A_only','Pog_only','APog_only','joint'):
            for metric in METRICS:
                keys,arrays,active,adequate=configurations(P,c['points'],fh,metric,mode)
                old=float(prior.loc[(p,metric),'automatic'])
                if not active:result=metric_decision([old],True,metric,legacy=True);n=1
                else:
                    vals,n=calculate(P,metric,keys,arrays,active,mand,dirs,org)
                    if vals is None:adequate=False;vals=[]
                    result=metric_decision(vals,adequate,metric)
                measurements.append(dict(Patient=p,mode=mode,metric=metric,**result,active_points=';'.join(active),
                                         combinations=n,adequate=adequate,
                                         width=result['high']-result['low'] if result['low'] is not None else None,
                                         reference=float(prior.loc[(p,metric),'excel']),legacy_value=old))
        if needs_mask:del mand
    metric=pd.DataFrame(measurements);metric['legacy_error']=(metric.legacy_value-metric.reference).abs()
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
                            same_numeric_output_MAE=valid.output_error.mean(),withheld_legacy_MAE=g[g.value.isna()].legacy_error.mean(),
                            improved_n=int((valid.output_error<valid.legacy_error-1e-8).sum()),
                            worsened_n=int((valid.output_error>valid.legacy_error+1e-8).sum())))
    summary=pd.DataFrame(summary);summary.to_csv(DEST/'evaluation.csv',index=False,encoding='utf-8-sig')
    report(cases,points,metric,summary)
    print(points.groupby(['point','route']).size().to_string());print(summary.to_string(index=False))


def figures(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    for c in cases:
        for key,r in c['points'].items():
            p=c['Patient'];path=DEST/'roi'/f'{p}_{key}.npz'
            if not path.exists():continue
            z=np.load(path);axes=[z[k] for k in ('L','A','S')];o,R=z['origin'],z['R']
            points=[];colors=[]
            for v in r['variants']:
                if v['winner'] is not None:points.append(v['winner']);colors.append('#2a78d6' if v['family']=='AMASSS' else '#eb6834')
            pts=np.asarray(points,float).reshape(-1,3);original=R@(np.asarray(r['original_point'])-o)
            center=np.median(pts,axis=0) if len(pts) else R@(np.asarray(r['seed_lps'])-o)
            fig,axs=plt.subplots(1,3,figsize=(17,6))
            i=int(np.argmin(abs(axes[0]-center[0])))
            axs[0].imshow(z['ct'][i].T,origin='lower',cmap='gray',vmin=-700,vmax=2000,
                          extent=[axes[1][0]-.125,axes[1][-1]+.125,axes[2][0]-.125,axes[2][-1]+.125])
            if z['mask'][i].any():axs[0].contour(axes[1],axes[2],z['mask'][i].T.astype(float),[.5],colors=['#00c8ff'],linewidths=.7)
            if len(pts):
                near=abs(pts[:,0]-axes[0][i])<=.75;axs[0].scatter(pts[near,1],pts[near,2],c=np.asarray(colors)[near],s=25)
            if abs(original[0]-axes[0][i])<=.75:axs[0].plot(original[1],original[2],'r+',ms=12)
            axs[0].set_title('실제 시상 단면 / 0.75 mm 이내 점만 표시')
            for family,color in (('AMASSS','#2a78d6'),('native_CT','#eb6834')):
                for v in r['variants']:
                    if v['family']!=family:continue
                    pr=v['profile'];axs[1].plot(pr.get('front',[]),pr.get('sup',[]),color=color,alpha=.25)
            axs[1].plot(original[1],original[2],'r+',ms=12);axs[1].set_title('분할/원본 조건별 앞쪽 윤곽: 파랑/주황')
            axs[2].scatter(pts[:,1],pts[:,2],c=colors,s=30) if len(pts) else None
            axs[2].plot(original[1],original[2],'r+',ms=12);axs[2].set_title('조건별 후보 분포, 빨강=기존 ALI')
            for ax in axs:ax.set_xlabel('앞 A mm');ax.set_ylabel('위 S mm')
            fig.suptitle(f'{p} {key}: {r["route"]} / 현재 점·Me·좌우 대칭으로 후보를 선택하지 않음',fontsize=12)
            fig.tight_layout(rect=(0,0,1,.94));fig.savefig(DEST/'figs'/f'{p}_{key}.png',dpi=110);plt.close(fig)


def report(cases,points,metric,summary):
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    cards=[]
    for c in cases:
        p=c['Patient'];imgs=''.join(f'<a href="figs/{p}_{k}.png"><img src="figs/{p}_{k}.png"></a>' for k in TARGETS if (DEST/'figs'/f'{p}_{k}.png').exists())
        if not imgs:continue
        cards.append(f'<article><h3>{p}</h3>{table(points[points.Patient==p])}{imgs}{table(metric[(metric.Patient==p)&(metric["mode"]=="joint")])}</article>')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A·Pog 독립 자동 복원</title>
<style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}table{{border-collapse:collapse;background:white;font-size:12px}}td,th{{border:1px solid #ddd;padding:5px}}.scroll{{overflow:auto}}img{{max-width:100%}}article{{background:white;padding:16px;margin:24px 0;overflow:auto}}</style><main>
<h1>A·Pog 독립 후보·선택적 보정: 85명 평가</h1>
<p>현재 환자의 A/Pog를 중심으로 영역을 자르지 않았습니다. 고정 참고 환자 5·11·28의 표면 대응에서 탐색 위치를 전달하되 본인 참고는 제외했습니다. 참고 ALI 좌표는 검수된 정답이 아니며 ROI 안내에만 사용했습니다. 환자 자신의 MAX/MAND 윤곽과 원본 영상에서 후보를 다시 찾았습니다.</p>
<p>A는 ANS 아래 제한 범위의 앞쪽 오목점, Pog는 전달된 턱끝 영역의 앞쪽 극점을 각각 독립적으로 찾았습니다. Pog 창을 현재 Me나 현재 Pog 높이로 정하지 않았고, B·Me를 수정하지 않았습니다. 좌우 대칭과 엑셀 오차로 후보를 선택하지 않았습니다.</p>
<h2>1. 점별 처리 결과</h2>{table(points.groupby(['point','route']).size().rename('n').reset_index())}
<p>분할·원본의 16조건을 비교했습니다. 원래 점이 양쪽 영상 근거의 앞 윤곽에서 같은 방향으로 벗어나거나, 독립 ROI 밖이거나, 양쪽 근거에서 안정된 정의 후보와 크게 다르면 보정 분기를 열었습니다. 마스크만 다르고 원본 윤곽과 맞으면 기존점을 유지했습니다. 안정성은 해부학적 정확도 보장이 아닙니다.</p>
<h2>2. 기존 FH 실험과 결합</h2><div class="scroll">{table(summary[summary['mode']=='joint'])}</div>
<p>A만 ANB·Nperp-A에, Pog만 Nperp-Pog에 영향을 주도록 분리했습니다. Pog 후보를 직접 사용하는 경우 기존 Pog 보정을 다시 중복 적용하지 않습니다. MP-FH는 A/Pog 보정 때문에 바뀌지 않습니다. 후보 조합 4096개 초과는 임의 표본을 뽑지 않고 보류합니다.</p>
<h2>3. 점별 분리 비교</h2><div class="scroll">{table(summary)}</div>
<p>A_only·Pog_only·APog_only의 활성 보정은 원래 자동 FH를 고정해 점 변경 효과를 분리했고, joint는 앞 단계 FH 후보도 함께 반영했습니다. 과거 대칭 보정이 포함된 기존값은 비교 기준일 뿐 새 좌표를 반사해 만들지 않았습니다. 표본 제외에 의한 평균오차 감소를 피하려고 전체 인원·숫자 제공률·같은 제공 인원의 오차·보류 인원의 기존 오차를 함께 표시했습니다. 이 85명은 개발 자료이며 독립 검증이 아닙니다.</p>
<h2>4. 전체 170개 A/Pog</h2><div class="scroll">{table(points)}</div>
<h2>5. 실제 영상·윤곽·후보</h2>{''.join(cards)}
<h2>자료</h2><p><a href="point_stability.csv">점별 처리 경로</a> · <a href="variant_candidates.csv">조건별 후보</a> · <a href="metric_restoration.csv">값·범위·보류</a> · <a href="evaluation.csv">오차·제공률·악화 사례</a> · <a href="landmark_scope.csv">평가한 점과 미평가 점</a> · <a href="protocol.json">고정 규칙</a></p>
<p>이번 A/Pog 2종과 이전 FH 4종의 안정성을 다뤘으며 나머지 10종의 독립 평가가 끝난 것은 아닙니다. 기존값 유지도 검증 완료가 아닙니다. 모든 새 결과는 자동 복원 제안이며 기존 좌표·보고서는 변경하지 않았습니다. 후보 범위와 조건별 반경은 신뢰구간이 아닙니다.</p><pre>{escape(json.dumps(engine.RULE,ensure_ascii=False,indent=2))}</pre></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--patients',nargs='*',type=int);args=ap.parse_args()
    for path in (DEST,DEST/'cases',DEST/'roi',DEST/'figs'):path.mkdir(parents=True,exist_ok=True)
    source=json.loads((FH_SOURCE/'protocol.json').read_text(encoding='utf-8'));patients=source['patients']
    if args.patients:patients=[p for p in patients if int(p[2:]) in args.patients]
    models={s:np.load(ROOT/f'Result_py/shape/{s}/points.npz') for s in ('MAX','MAND')}
    annotations={i:landmarks(i)[0] for i in engine.RULE['reference_patients']}
    save(DEST/'protocol.json',dict(patients=patients,rule=engine.RULE,variants=engine.VARIANTS,acceptance=ACCEPT,
          code=digest(__file__),engine=digest(Path(engine.__file__)),FH_source_protocol=digest(FH_SOURCE/'protocol.json'),
          source_FH_cases={p:digest(FH_SOURCE/'cases'/f'{p}.json') for p in patients},
          max_combinations=MAX_COMBINATIONS,anatomically_confirmed=False))
    cases=[process(p,models,annotations) for p in patients]
    figures(cases);evaluate(cases);print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
