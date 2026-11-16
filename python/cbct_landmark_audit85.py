"""2단계: N·ANS·PNS·B·Me·Gn·RGo·LGo 독립 윤곽 후보와 기존점 지지 여부를 85명 전수 평가한다.

이전 산출물은 읽기만 한다. 엑셀은 이 스크립트에서 읽지 않는다.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cbct_landmark_audit as engine
import cbct_apog85 as base
from cbct_po_ensemble import stability,ACCEPT
from cbct_patient_pipeline import CB,OUT,ROOT,digest,landmarks
from cbct_seg_landmarks85 import load_mask,sample,save,CENTRAL
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd

DEST=OUT/'develop85/landmark_audit_v1'
KEYS=tuple(engine.SPECS)


def process(patient,models,annotations):
    P,files=landmarks(int(patient[2:]));scan=CB/f'nrrd/{patient}.nrrd'
    fp=dict(code=digest(__file__),engine=digest(Path(engine.__file__)),image=digest(scan),
            landmarks={f.name:digest(f) for f in files},
            segmentations={s:digest(CB/f'seg/{patient}_seg_{s}.nrrd') for s in ('CB','MAX','MAND')})
    path=DEST/'cases'/f'{patient}.json'
    if path.exists():
        c=json.loads(path.read_text(encoding='utf-8'))
        if c['fingerprint']==fp:print(patient,'cached',flush=True);return c
    vol,dirs,org=read_nrrd(scan);o,R,residual=midline_frame({k:P[k] for k in CENTRAL})
    masks={}
    for s in ('CB','MAX','MAND'):
        m,md,mo=load_mask(patient,s);assert np.allclose(md,dirs) and np.allclose(mo,org);masks[s]=m
    c=dict(Patient=patient,fingerprint=fp,frame_residual_mm=residual,points={})
    for key,spec in engine.SPECS.items():
        seed_lps,transfer=base.reference_seed(patient,key,models[spec['model']],annotations)
        seed=R@(seed_lps-o)
        def sampler(G,kind):
            q=o+G@R
            if kind=='ct':return sample(vol,dirs,org,q,order=1,cval=np.nan)
            return np.any([sample(masks[s],dirs,org,q)>0 for s in spec['structs']],axis=0)
        results=[engine.detect(sampler,seed,key,v) for v in engine.VARIANTS]
        s=stability(results,engine.VARIANTS)
        s['candidates_lps']=o+s['candidates']@R if s['candidate_count'] else np.empty((0,3))
        s['representative_lps']=o+s['representative']@R if s['representative'] is not None else None
        original=R@(P[key]-o)
        source_ok=transfer['max_transfer_residual_mm']<=3. and transfer['max_condition']<=100.
        if not source_ok:s['adequate_support']=False;s['status']='unreliable_reference_transfer'
        c['points'][key]=dict(stability=s,variants=results,source_ok=source_ok,original_point=P[key],
                              raw_deviation=engine.raw_deviation(original,seed,results),
                              original_local=original,seed_local=seed,seed_lps=seed_lps,transfer=transfer,origin=o,R=R)
        print(patient,key,s['status'],flush=True)
    del masks,vol
    save(path,c);return json.loads(path.read_text(encoding='utf-8'))


def offsets(cases):
    """기준점·원천별 ALI 표기 관습 오프셋 = 85명 원 편차의 중앙값."""
    out={}
    for key in KEYS:
        out[key]={f:float(np.median(v)) if len(v:=[c['points'][key]['raw_deviation'][f] for c in cases
                                                   if c['points'][key]['raw_deviation'][f] is not None]) else 0.
                  for f in ('mask','native')}
        d=[np.asarray(c['points'][key]['stability']['representative'])-np.asarray(c['points'][key]['original_local'])
           for c in cases if c['points'][key]['stability']['representative'] is not None]
        out[key]['displacement_n']=len(d)
        out[key]['displacement']=np.median(d,axis=0).tolist() if len(d)>=10 else [0.,0.,0.]
    return out


def finalize(cases,off):
    for c in cases:
        for key,r in c['points'].items():
            s=r['stability']
            for name in ('representative','candidates'):
                if s[name] is not None:s[name]=np.asarray(s[name],float)
            for v in r['variants']:
                v['basis']=[np.asarray(x) for x in v['basis']]
            q=engine.decide(np.asarray(r['original_local']),np.asarray(r['seed_local']),r['variants'],s,key,off[key])
            if not r['source_ok']:
                q['independent_support']=False
                q['route']='automatic_withhold' if q['triggered'] else 'existing_point_retained_not_confirmed'
            r.update(q)
            # 교체 후보를 ALI 관습 좌표로 옮긴다(정의 관습 변위 제거).
            o,R=np.asarray(r['origin']),np.asarray(r['R']);disp=np.asarray(off[key]['displacement'])
            s['candidates_lps']=o+(s['candidates']-disp)@R if s['candidate_count'] else np.empty((0,3))
            s['representative_lps']=o+(s['representative']-disp)@R if s['representative'] is not None else None
    save(DEST/'decisions.json',{c['Patient']:{k:dict(route=r['route'],triggered=r['triggered'],reasons=r['reasons'],
            independent_support=r['independent_support'],deviation=r['original_boundary_deviation'],raw_deviation=r['raw_deviation'],
            candidates_lps=r['stability']['candidates_lps'],adequate_support=r['stability']['adequate_support'],
            status=r['stability']['status'],representative_lps=r['stability']['representative_lps'])
            for k,r in c['points'].items()} for c in cases})


def table(cases):
    rows=[]
    for c in cases:
        for key,r in c['points'].items():
            s=r['stability'];a,b=s['families']['AMASSS'],s['families']['native_CT']
            rows.append(dict(Patient=c['Patient'],point=key,route=r['route'],triggered=r['triggered'],
                             independent_support=r['independent_support'],stability=s['status'],
                             mask_success=a['success_fraction'],native_success=b['success_fraction'],
                             family_distance_mm=s['family_distance_mm'],candidate_count=s['candidate_count'],
                             dev_mask_mm=r['original_boundary_deviation']['mask'],dev_native_mm=r['original_boundary_deviation']['native'],
                             representative_shift_mm=(float(np.linalg.norm(np.asarray(s['representative'])-np.asarray(r['original_local'])))
                                                      if s['representative'] is not None else None),
                             reasons=';'.join(r['reasons']),source_ok=r['source_ok']))
    t=pd.DataFrame(rows);t.to_csv(DEST/'point_status.csv',index=False,encoding='utf-8-sig')
    return t


def figures(cases,t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    for f in (DEST/'figs').glob('*.png'):f.unlink()
    pick=t[t.triggered|(~t.independent_support)]
    for c in cases:
        p=c['Patient'];keys=list(pick[pick.Patient==p].point)
        if not keys:continue
        P=landmarks(int(p[2:]))[0];vol,dirs,org=read_nrrd(CB/f'nrrd/{p}.nrrd')
        fig,axs=plt.subplots(1,len(keys),figsize=(5.2*len(keys),5.2),squeeze=False)
        for ax,key in zip(axs[0],keys):
            r=c['points'][key];o,R=np.asarray(r['origin']),np.asarray(r['R']);seed=np.asarray(r['seed_local'])
            ori=np.asarray(r['original_local']);half=22
            A=np.arange(seed[1]-half,seed[1]+half,.4);S=np.arange(seed[2]-half,seed[2]+half,.4)
            G=np.stack(np.meshgrid([ori[0]],A,S,indexing='ij'),-1)[0]
            img=sample(vol,dirs,org,o+G@R,order=1,cval=np.nan)
            ax.imshow(img.T,origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=[A[0],A[-1],S[0],S[-1]])
            for v in r['variants']:
                if v['variant'] in ('contour_base','native_otsu_+0') and v['profile'].get('u'):
                    eu,ev=(np.asarray(x) for x in v['basis']);u=np.asarray(v['profile']['u'],float);vv=np.asarray(v['profile']['v'],float)
                    xy=seed[1:]+u[:,None]*eu+vv[:,None]*ev
                    ax.plot(xy[:,0],xy[:,1],color='#00c8ff' if v['family']=='AMASSS' else '#eb6834',lw=.9)
            cand=np.asarray(r['stability']['candidates'],float).reshape(-1,3)
            if len(cand):ax.scatter(cand[:,1],cand[:,2],s=10,c='#28ad49',alpha=.6)
            ax.plot(ori[1],ori[2],'m+',ms=14,mew=2)
            ax.set_title(f'{key}: {r["route"]}\n지지={r["independent_support"]}',fontsize=9)
            ax.set_xlabel('앞 (mm)');ax.set_ylabel('위 (mm)')
        fig.suptitle(f'{p}: 기존 좌표의 좌우 위치 단면. 분홍 +=기존점, 초록=독립 후보, 청록/주황=마스크/원본 윤곽',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.93));fig.savefig(DEST/'figs'/f'{p}.png',dpi=95);plt.close(fig)
        del vol


def report(t):
    def tb(x):return x.to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}',na_rep='')
    summary=t.groupby(['point','route']).size().rename('n').reset_index()
    support=t.groupby('point').agg(n=('Patient','size'),independent_support=('independent_support','sum'),
                                  triggered=('triggered','sum'),stable_single=('stability',lambda s:(s=='stable_single_candidate').sum()),
                                  transfer_failed=('source_ok',lambda s:(~s).sum())).reset_index()
    cards=''.join(f'<article><h3>{p}</h3>{tb(t[(t.Patient==p)&(t.triggered|~t.independent_support)])}<img loading="lazy" src="figs/{p}.png"></article>'
                  for p in sorted(t.Patient.unique(),key=lambda x:int(x[2:])) if (DEST/'figs'/f'{p}.png').exists())
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>남은 기준점 독립 평가</title>
<style>body{{font:15px/1.6 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:16px}}table{{border-collapse:collapse;background:white;font-size:12px}}td,th{{border:1px solid #ddd;padding:4px 6px}}img{{max-width:100%}}article{{background:white;padding:12px;margin:16px 0;overflow:auto}}</style><main>
<h1>2단계: N·ANS·PNS·B·Me·Gn·Go 독립 윤곽 평가 (85명)</h1>
<p>고정 참고 환자 5·11·28의 표면 대응으로 탐색 위치를 전달하고(본인 제외), 환자 자신의 AMASSS 분할과 원본 CT에서 정의 방향의 윤곽 극점을 24조건(축 −5/0/+5° × 창 2 × 원천 4)으로 찾았습니다. 현재 기준점·엑셀·대칭으로 ROI나 후보를 고르지 않았습니다. 기존점은 후보를 만든 뒤 교체 필요 판단에만 썼습니다.</p>
<p><b>독립 지지</b>: 기존점이 마스크 또는 원본 윤곽에서 법선 방향 1 mm 이내이고 두 원천이 같은 방향으로 1 mm 넘게 벗어나지 않으며 교체 조건이 없을 때. 윤곽 정의는 공학적 대리 조건입니다(N은 봉합선 대신 전방 최심점, Go는 접선 이등분 대신 후하방 45° 최외곽점). 해부학적 확정이 아닙니다.</p>
<h2>점별 요약</h2>{tb(support)}{tb(summary)}
<h2>교체 조건 또는 지지 부족 환자의 실제 단면</h2>{cards}
<p>ALI 표기 관습 오프셋(85명 기준점·원천별 중앙 편차, 양수=기존점이 윤곽 바깥): <code>{json.dumps(json.loads((DEST/'convention_offsets.json').read_text(encoding='utf-8')))}</code>. 판정은 이 오프셋을 뺀 편차로 했습니다. 기존점이 체계적으로 뼈 표면 밖에 찍히는 관습을 환자별 오류로 보지 않기 위해서이며 엑셀은 쓰지 않았습니다.</p>
<p><a href="point_status.csv">전체 680점 상태</a> · <a href="protocol.json">규칙</a></p></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--patients',nargs='*',type=int);args=ap.parse_args()
    for path in (DEST,DEST/'cases',DEST/'figs'):path.mkdir(parents=True,exist_ok=True)
    patients=json.loads((OUT/'develop85/apog_structure_v3/protocol.json').read_text(encoding='utf-8'))['patients']
    if args.patients:patients=[p for p in patients if int(p[2:]) in args.patients]
    models={s:np.load(ROOT/f'Result_py/shape/{s}/points.npz') for s in ('CB','MAX','MAND')}
    annotations={i:landmarks(i)[0] for i in engine.RULE['reference_patients']}
    save(DEST/'protocol.json',dict(patients=patients,rule=engine.RULE,variants=engine.VARIANTS,acceptance=ACCEPT,
         code=digest(__file__),engine=digest(Path(engine.__file__)),
         models={s:digest(ROOT/f'Result_py/shape/{s}/points.npz') for s in models},anatomically_confirmed=False))
    cases=[process(p,models,annotations) for p in patients]
    off=offsets(cases);save(DEST/'convention_offsets.json',off);finalize(cases,off)
    t=table(cases)
    if not args.patients:figures(cases,t)
    report(t);print(t.groupby(['point','route']).size().to_string())


if __name__=='__main__':main()
