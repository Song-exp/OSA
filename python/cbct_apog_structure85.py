"""첫 구조 단계: v2 후보 생성·분기를 고정하고 해부학 구조 게이트만 추가."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cbct_apog85 as base
import cbct_apog_structure as engine
from cbct_patient_pipeline import CB,OUT,ROOT,digest
from cbct_seg_landmarks85 import sample,save
from cbct_po_ensemble import stability,ACCEPT
from run_ali import ascii_path

DEST=OUT/'develop85/apog_structure_v3'
SOURCE=OUT/'develop85/apog_trace_v2'
TEETH=CB/'seg_teeth/UNIVERSALLAB_MODEL'


def dependencies():
    return {n:digest(Path(__file__).parent/n) for n in ('cbct_apog85.py','cbct_apog_ensemble.py',
            'cbct_po_ensemble.py','cbct_po_stability85.py','cbct_seg_landmarks85.py','osa/ceph.py')}


def process(patient,variants):
    source=SOURCE/'cases'/f'{patient}.json';tooth=TEETH/f'{patient}.nii.gz'
    fp=dict(code=digest(__file__),engine=digest(Path(engine.__file__)),dependencies=dependencies(),
            source_case=digest(source),source_roi={k:digest(SOURCE/'roi'/f'{patient}_{k}.npz') for k in base.TARGETS},
            teeth=digest(tooth))
    path=DEST/'cases'/f'{patient}.json'
    if path.exists():
        c=json.loads(path.read_text(encoding='utf-8'))
        if c['structure_fingerprint']==fp:print(patient,'cached',flush=True);return c
    c=json.loads(source.read_text(encoding='utf-8'));c['structure_fingerprint']=fp
    im=sitk.ReadImage(str(ascii_path(tooth)));labels=sitk.GetArrayFromImage(im)
    dirs=(np.array(im.GetDirection()).reshape(3,3)@np.diag(im.GetSpacing())).T
    org=np.array(im.GetOrigin())
    for key,r in c['points'].items():
        with np.load(SOURCE/'roi'/f'{patient}_{key}.npz') as z:
            axes=[z[k] for k in ('L','A','S')];o,R=z['origin'],z['R'];mask=z['mask'];ct=z['ct']
            G=np.stack(np.meshgrid(*axes,indexing='ij'),-1);q=o+G@R
            tooth_roi=sample(engine.tooth_binary(labels),dirs,org,q,order=0)>0
            ijk=np.linalg.solve(dirs.T,(q.reshape(-1,3)-org).T)
            domain=((ijk>=0)&(ijk<=np.array(labels.shape[::-1])[:,None]-1)).all(0).reshape(mask.shape)
            data=engine.prepare(mask,ct,tooth_roi,domain,axes)
            runs=[engine.filter_variant(v,key,data) for v in r['variants']]
            s=stability(runs,variants)
            s['candidates_lps']=o+s['candidates']@R if s['candidate_count'] else np.empty((0,3))
            s['representative_lps']=o+s['representative']@R if s['representative'] is not None else None
            if not r['source_ok']:s['adequate_support']=False;s['status']='unreliable_reference_transfer'
            old_route=r['route'];r.update(stability=s,variants=runs,previous_route=old_route)
            if r['triggered']:
                r['route']='single_reconstruction_candidate' if s['status']=='stable_single_candidate' else 'candidate_region' if s['adequate_support'] else 'automatic_withhold'
            # Original retained branches remain unverified; this step does not alter their routing.
            r['structural_rejected']=sum(not x['accepted'] for v in runs for x in v['structural_checks'])
            r['structural_accepted']=sum(x['accepted'] for v in runs for x in v['structural_checks'])
            np.savez_compressed(DEST/'roi'/f'{patient}_{key}.npz',mask=mask,ct=ct,teeth=tooth_roi,domain=domain,
                                L=axes[0],A=axes[1],S=axes[2],origin=o,R=R)
            print(patient,key,r['route'],f"candidates {r['structural_accepted']} passed / {r['structural_rejected']} rejected",flush=True)
    save(path,c);return json.loads(path.read_text(encoding='utf-8'))


def evidence(cases):
    rows=[]
    for c in cases:
        for key,r in c['points'].items():
            for v in r['variants']:
                for x in v['structural_checks']:
                    rows.append(dict(Patient=c['Patient'],point=key,variant=v['variant'],family=v['family'],
                                     accepted=x['accepted'],reasons=';'.join(x['reasons']),
                                     tooth_distance_mm=x['tooth_distance_mm'],bone_boundary_distance_mm=x['bone_boundary_distance_mm'],
                                     CT_contrast=x.get('CT_contrast'),inferior_clearance_mm=x.get('inferior_clearance_mm')))
    pd.DataFrame(rows).to_csv(DEST/'structural_evidence.csv',index=False,encoding='utf-8-sig')


def figures(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    for c in cases:
        for key,r in c['points'].items():
            if not r['triggered'] and c['Patient'] not in ('P_5','P_15','P_35','P_44','P_69'):continue
            p=c['Patient']
            with np.load(DEST/'roi'/f'{p}_{key}.npz') as z:
                axes=[z[k] for k in ('L','A','S')];o,R=z['origin'],z['R']
                good=np.array([x['point'] for v in r['variants'] for x in v['structural_checks'] if x['accepted']]).reshape(-1,3)
                bad=np.array([x['point'] for v in r['variants'] for x in v['structural_checks'] if not x['accepted']]).reshape(-1,3)
                pts=np.vstack([good,bad])
                center=np.median(pts,axis=0) if len(pts) else R@(np.asarray(r['seed_lps'])-o)
                i=int(np.argmin(abs(axes[0]-center[0])))
                fig,axs=plt.subplots(1,2,figsize=(12,6))
                axs[0].imshow(z['ct'][i].T,origin='lower',cmap='gray',vmin=-700,vmax=2000,
                              extent=[axes[1][0]-.125,axes[1][-1]+.125,axes[2][0]-.125,axes[2][-1]+.125])
                for name,color in [('mask','#00c8ff'),('teeth','#ffcc00')]:
                    if z[name][i].any():axs[0].contour(axes[1],axes[2],z[name][i].T.astype(float),[.5],colors=[color],linewidths=.8)
                for pts,color,marker in [(good,'#28ad49','o'),(bad,'#e15353','x')]:
                    near=abs(pts[:,0]-axes[0][i])<=.75
                    axs[0].scatter(pts[near,1],pts[near,2],c=color,marker=marker,s=25)
                    axs[1].scatter(pts[:,1],pts[:,2],c=color,marker=marker,s=25,alpha=.4)
                original=R@(np.asarray(r['original_point'])-o)
                if abs(original[0]-axes[0][i])<=.75:axs[0].plot(original[1],original[2],'m+',ms=12)
                axs[1].plot(original[1],original[2],'m+',ms=12)
                axs[0].set_title('실제 단면 ±0.75 mm / 노랑=치아, 청록=뼈')
                axs[1].set_title('후보의 앞·위 좌표 투영: 초록=통과, 빨강=거부')
                for ax in axs:ax.set_xlabel('앞 A mm');ax.set_ylabel('위 S mm')
                fig.suptitle(f'{p} {key}: {r["previous_route"]} → {r["route"]}')
                fig.tight_layout();fig.savefig(DEST/'figs'/f'{p}_{key}.png',dpi=110);plt.close(fig)


def report(cases,points,metric,summary):
    def table(x):return x.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')
    old=pd.read_csv(SOURCE/'metric_restoration.csv');old=old[old['mode']=='joint'].set_index(['Patient','metric'])
    new=metric[metric['mode']=='joint'].set_index(['Patient','metric'])
    paired=new.join(old[['value','output_error']].rename(columns={'value':'previous_value','output_error':'previous_error'}))
    rows=[]
    for m,g in paired.groupby(level='metric'):
        same=g[g.value.notna()&g.previous_value.notna()]
        rows.append(dict(metric=m,total_n=len(g),previous_numeric=int(g.previous_value.notna().sum()),new_numeric=int(g.value.notna().sum()),
                         paired_n=len(same),paired_previous_MAE=same.previous_error.mean(),paired_new_MAE=same.output_error.mean(),
                         improved=int((same.output_error<same.previous_error-1e-8).sum()),worsened=int((same.output_error>same.previous_error+1e-8).sum())))
    comparison=pd.DataFrame(rows);comparison.to_csv(DEST/'comparison_previous.csv',index=False,encoding='utf-8-sig')
    ev=pd.read_csv(DEST/'structural_evidence.csv');rejects=ev[~ev.accepted].assign(reason=lambda x:x.reasons.str.split(';')).explode('reason')
    counts=rejects.groupby(['point','reason']).size().rename('n').reset_index();counts.to_csv(DEST/'rejection_reasons.csv',index=False,encoding='utf-8-sig')
    cards=[]
    for c in cases:
        p=c['Patient'];imgs=''.join(f'<a href="figs/{p}_{k}.png"><img loading="lazy" src="figs/{p}_{k}.png"></a>' for k in base.TARGETS if (DEST/'figs'/f'{p}_{k}.png').exists())
        if imgs:cards.append(f'<article><h3>{p}</h3>{table(points[points.Patient==p])}{imgs}</article>')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>A·Pog 구조 조건 1단계</title>
<style>body{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5}main{max-width:1500px;margin:auto;padding:20px}table{border-collapse:collapse;background:white;font-size:12px}td,th{border:1px solid #ddd;padding:5px}img{max-width:100%}.scroll{overflow:auto}article{background:white;padding:16px;margin:20px 0}</style><main>
<h1>A·Pog 구조 구분 1단계: 치아·피질·하연 게이트</h1>
<p>v2의 85명·170점·24조건 후보 생성과 교체 필요 분기를 그대로 고정하고 구조 게이트만 추가했습니다. 현재 대상점·Me·대칭·엑셀로 후보를 새로 찾거나 임계값을 조정하지 않습니다. 기존 유지점은 검증되지 않은 상태로 유지합니다.</p>
<p>자동 치아 라벨과 0.75 mm 이내인 후보, AMASSS 뼈 경계에서 먼 후보, 원본의 안쪽→바깥쪽 밝기 감소가 75 미만인 후보, 전방 외측 피질 조건에 맞지 않는 후보를 거릅니다. Pog는 ROI에서 측정한 하연보다 최소 3 mm 위여야 하며 하연이 ROI 끝에 걸리면 알 수 없음으로 거부합니다. 수치는 이번 엔지니어링 실험 조건으로 임상 허용오차가 아닙니다.</p>
<p>치아 분할은 과거 원래 Pog·Me를 포함한 큰 절단창에서 생성됐습니다. 따라서 완전히 독립적인 근거가 아닙니다. 후보 생성 ROI는 v2와 같고, 치아 분할은 후처리 거부용으로만 쓰며 후보가 치아 영상의 저장 범위 안에 있는지 확인했습니다. 0 라벨은 치아가 없다는 정답 확인이 아닙니다. 이번에는 치아 자체·하연에 가까운 후보를 구분할 뿐, 치조부 전체의 의미 분할을 완성하지 않았습니다.</p>
<h2>점별 상태</h2>'''+table(points.groupby(['point','route']).size().rename('n').reset_index())+'''
<h2>v2 대비 제공률과 같은 환자의 오차</h2>'''+table(comparison)+'''
<p>보류 증가로 인한 평균오차 감소를 개선으로 해석하지 않습니다. 85명 전원이 개발 자료이며 해부학적 정답 검증이 아닙니다. 안정 후보와 계측 숫자 반환도 구분합니다. 기존 운영값·ALI 좌표는 변경하지 않았습니다.</p>
<h2>구조 조건별 거부 이유</h2>'''+table(counts)+'''
<p>이 표는 조건별 후보 기록이며 한 후보에 여러 사유가 있을 수 있습니다. 환자 수로 해석하지 않습니다.</p>
<h2>계측값 반환</h2><div class="scroll">'''+table(summary[summary['mode']=='joint'])+'''</div>
<h2>실제 단면과 후보</h2>'''+''.join(cards)+'''
<p><a href="structural_evidence.csv">후보별 구조 근거</a> · <a href="rejection_reasons.csv">거부 사유</a> · <a href="point_stability.csv">170점 상태</a> · <a href="metric_restoration.csv">4개 모드 계측</a> · <a href="comparison_previous.csv">v2 비교</a> · <a href="protocol.json">사전 규칙</a></p></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--patients',nargs='*',type=int);args=parser.parse_args()
    for path in (DEST,DEST/'cases',DEST/'roi',DEST/'figs'):path.mkdir(parents=True,exist_ok=True)
    previous=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'));patients=previous['patients']
    if args.patients:patients=[p for p in patients if int(p[2:]) in args.patients]
    save(DEST/'protocol.json',dict(patients=patients,rule=engine.RULE,variants=previous['variants'],acceptance=ACCEPT,
        code=digest(__file__),engine=digest(Path(engine.__file__)),dependencies=dependencies(),
        previous_protocol=digest(SOURCE/'protocol.json'),previous_outputs={n:digest(SOURCE/n) for n in ('metric_restoration.csv','point_stability.csv','rules_frozen.json')},
        teeth_dataset=digest(ROOT/'models/downloads_teeth/UNIVERSALLAB_MODEL/dataset.json'),
        FH_cases={p:digest(base.FH_SOURCE/'cases'/f'{p}.json') for p in patients},anatomically_confirmed=False))
    cases=[process(p,previous['variants']) for p in patients]
    evidence(cases);figures(cases)
    base.DEST=DEST;base.report=report;base.evaluate(cases)
    print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
