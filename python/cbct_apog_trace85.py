"""85명 연속 윤곽 추적 실험. 이전 산출물은 읽기만 한다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import cbct_apog_trace as engine
import cbct_apog85 as base
from cbct_po_ensemble import stability,ACCEPT
from cbct_patient_pipeline import ROOT,CB,OUT,digest,landmarks
from cbct_seg_landmarks85 import load_mask,sample,save,CENTRAL
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd

DEST=OUT/'develop85/apog_trace_v2'
SOURCE=OUT/'develop85/apog_v1'
FH_SOURCE=base.FH_SOURCE


def dependency_hashes():
    names=('cbct_apog85.py','cbct_apog_ensemble.py','cbct_po_ensemble.py','cbct_po_stability85.py',
           'cbct_seg_landmarks85.py','cbct_eb_detect.py','cbct_fh_joint85.py','osa/ceph.py')
    return {n:digest(Path(__file__).parent/n) for n in names}


def process(patient,models,annotations):
    P,files=landmarks(int(patient[2:]));scan=CB/f'nrrd/{patient}.nrrd'
    fp=dict(code=digest(__file__),engine=digest(Path(engine.__file__)),shared_driver=digest(Path(base.__file__)),
            dependencies=dependency_hashes(),
            image=digest(scan),landmarks={f.name:digest(f) for f in files},
            segmentations={s:digest(CB/f'seg/{patient}_seg_{s}.nrrd') for s in ('MAX','MAND')},
            models={s:digest(ROOT/f'Result_py/shape/{s}/points.npz') for s in models},
            reference_annotations={str(i):{f.name:digest(f) for f in landmarks(i)[1]} for i in annotations},
            previous_case=digest(SOURCE/'cases'/f'{patient}.json'))
    path=DEST/'cases'/f'{patient}.json'
    if path.exists():
        c=json.loads(path.read_text(encoding='utf-8'))
        if c['fingerprint']==fp:print(patient,'cached',flush=True);return c
    v,dirs,org=read_nrrd(scan);o,R,residual=midline_frame({k:P[k] for k in CENTRAL})
    c=dict(Patient=patient,fingerprint=fp,frame_residual_mm=residual,points={})
    for key,struct in base.TARGETS.items():
        seed,transfer=base.reference_seed(patient,key,models[struct],annotations)
        mask,mdirs,morg=load_mask(patient,struct)
        assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
        m,axes,q=base.make_grid(mask,dirs,org,o,R,seed)
        ct=sample(v,dirs,org,q,order=1,cval=np.nan);local=R@(seed-o);ans=float((R@(P['ANS']-o))[2])
        results=[engine.detect_variant(m,ct,axes,local,ans,key,var) for var in engine.VARIANTS]
        s=stability(results,engine.VARIANTS)
        s['candidates_lps']=o+s['candidates']@R if s['candidate_count'] else np.empty((0,3))
        s['representative_lps']=o+s['representative']@R if s['representative'] is not None else None
        quality=engine.decide(R@(P[key]-o),local,results,s)
        source_ok=transfer['max_transfer_residual_mm']<=3. and transfer['max_condition']<=100.
        if not source_ok:
            s['adequate_support']=False;s['status']='unreliable_reference_transfer'
            if quality['triggered']:quality['route']='automatic_withhold'
        c['points'][key]=dict(stability=s,variants=results,**quality,source_ok=source_ok,
                              original_point=P[key],seed_lps=seed,transfer=transfer,struct=struct)
        # Every ROI is kept so any case can be inspected without changing selection.
        np.savez_compressed(DEST/'roi'/f'{patient}_{key}.npz',mask=m,ct=ct,L=axes[0],A=axes[1],S=axes[2],origin=o,R=R)
        print(patient,key,quality['route'],s['status'],flush=True)
        del mask
    save(path,c);return json.loads(path.read_text(encoding='utf-8'))


def factor_analysis(cases):
    rows=[]
    for c in cases:
        for key,r in c['points'].items():
            for family in ('AMASSS','native_CT'):
                # Native Otsu +0 is held fixed when isolating the two factors.
                runs=[v for v in r['variants'] if v['family']==family and
                      (family=='AMASSS' or next(x for x in engine.VARIANTS if x['name']==v['variant'])['otsu_delta']==0)]
                baseline=next(v for v in runs if v['angle_deg']==0 and v['window']=='base')
                for v in runs:
                    distance=float(np.linalg.norm(np.asarray(v['winner'])-baseline['winner'])) if v['winner'] is not None and baseline['winner'] is not None else None
                    rows.append(dict(Patient=c['Patient'],point=key,family=family,angle_deg=v['angle_deg'],window=v['window'],
                                     status=v['status'],candidate_count=len(v['eligible']),distance_from_base_mm=distance,
                                     base_has_candidate=baseline['winner'] is not None,has_candidate=v['winner'] is not None))
    pd.DataFrame(rows).to_csv(DEST/'factor_sensitivity.csv',index=False,encoding='utf-8-sig')


def report(cases,points,metric,summary):
    def table(x):return x.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    variants=pd.read_csv(DEST/'variant_candidates.csv')
    stats=variants.groupby(['point','family','status']).size().rename('n').reset_index()
    factors=pd.read_csv(DEST/'factor_sensitivity.csv')
    factors=factors.groupby(['point','family','angle_deg','window']).agg(n=('Patient','size'),success=('has_candidate','sum'),
                mean_shift_mm=('distance_from_base_mm','mean'),max_shift_mm=('distance_from_base_mm','max')).reset_index()
    comparisons=[]
    for stage,path in [('A/Pog v1',SOURCE),('FH joint',FH_SOURCE)]:
        old=pd.read_csv(path/'metric_restoration.csv');old=old[old['mode']=='joint'].set_index(['Patient','metric'])
        new=metric[metric['mode']=='joint'].set_index(['Patient','metric'])
        paired=new.join(old[['value','output_error']].rename(columns={'value':'previous_value','output_error':'previous_error'}))
        for m,g in paired.groupby(level='metric'):
            same=g[g.value.notna()&g.previous_value.notna()]
            comparisons.append(dict(previous=stage,metric=m,total_n=len(g),previous_numeric=int(g.previous_value.notna().sum()),
                                    new_numeric=int(g.value.notna().sum()),paired_n=len(same),
                                    paired_previous_MAE=same.previous_error.mean(),paired_new_MAE=same.output_error.mean(),
                                    previous_only_n=int((g.previous_value.notna()&g.value.isna()).sum()),
                                    new_only_n=int((g.previous_value.isna()&g.value.notna()).sum())))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(DEST/'comparison_previous.csv',index=False,encoding='utf-8-sig')
    stats.to_csv(DEST/'failure_modes.csv',index=False,encoding='utf-8-sig')
    cards=[]
    for c in cases:
        p=c['Patient']
        # All patients remain in CSVs; images focus on changed branches plus fixed examples.
        if not any(r['triggered'] for r in c['points'].values()) and p not in ('P_5','P_10','P_35','P_44','P_69'):continue
        imgs=''.join(f'<a href="figs/{p}_{k}.png"><img loading="lazy" src="figs/{p}_{k}.png"></a>' for k in base.TARGETS)
        cards.append(f'<article><h3>{p}</h3>{table(points[points.Patient==p])}{imgs}</article>')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>A·Pog 연속 윤곽 추적 v2</title>
<style>body{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5}main{max-width:1500px;margin:auto;padding:20px}table{border-collapse:collapse;font-size:12px;background:white}td,th{border:1px solid #ddd;padding:5px}.scroll{overflow:auto}img{max-width:100%}article{background:white;padding:15px;margin:20px 0}</style><main>
<h1>A·Pog 연속 윤곽 추적: 창·축 방향 민감도</h1>
<p>현재 A/Pog·Me·대칭·엑셀로 후보를 선택하지 않습니다. 고정 참고 환자의 표면 대응으로 ROI를 안내하고 자기 참조를 제외했습니다. 분할 6조건과 원본 18조건, 총 24조건입니다. 원본 CT 임계값은 보정된 HU가 아니며 두 방법은 같은 ROI와 정의를 공유합니다.</p>
<p>기존 v1의 높이별 독립 앞쪽 픽셀 대신 최대 연결 구조의 전방 경계 후보를 연속 경로로 잇습니다. 양쪽 윤곽 지지는 4 mm에서 보고, 축 방향 ±5°와 기본/확장 탐색창을 완전 교차 비교합니다. 기본 축은 N·S·Ba·ANS·PNS 기반으로 그 기준점 자체의 오차는 별도 해결하지 않았습니다. 각도는 개인별로 엑셀에 맞춰 고르지 않고 모든 후보를 포함합니다.</p>
<p>안정성 기준·교체 필요 판단·계측 범위 기준은 v1과 동일합니다. 여러 구현 변경을 함께 비교하므로 v1 대비 변화 전체를 연속 추적 단독 효과로 해석할 수 없습니다. 이 실험 내 창·축 요인은 각각 분리해 기록했습니다. 모든 표본은 개발 자료이며 아래 값은 정답 검증이나 임상 확정이 아닙니다.</p>
<h2>점별 반환 상태</h2>'''+table(points.groupby(['point','route']).size().rename('n').reset_index())+'''
<h2>계측값 제공률과 오차</h2><div class="scroll">'''+table(summary[summary['mode']=='joint'])+'''</div>
<h2>이전 방식과 같은 환자끼리 비교</h2><div class="scroll">'''+table(comparison)+'''</div>
<p>보류로 평균오차가 줄어드는 효과와 실제 같은 환자의 개선을 구분해야 합니다. 숫자 반환과 단일 점의 좌표 확정도 다릅니다. 기존점 유지는 검증 완료가 아닙니다. 운영 좌표·계측값은 변경하지 않았습니다.</p>
<h2>창과 축 방향 요인</h2><div class="scroll">'''+table(factors)+'''</div><h2>후보 실패 원인</h2>'''+table(stats)+'''
<h2>환자별 실제 영상</h2>'''+''.join(cards)+'''
<p><a href="point_stability.csv">170개 점</a> · <a href="variant_candidates.csv">조건별 후보</a> · <a href="factor_sensitivity.csv">창·축 분리 결과</a> · <a href="metric_restoration.csv">4개 모드 계측값</a> · <a href="comparison_previous.csv">이전 방식 비교</a> · <a href="protocol.json">사전 고정 규칙</a></p>
<p>A/Pog 및 이전 Po/Or 4종 이외 10종의 독립 평가가 끝난 상태가 아닙니다. 후보 범위는 조건별 민감도이며 신뢰구간이 아닙니다.</p></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--patients',nargs='*',type=int);args=parser.parse_args()
    for path in (DEST,DEST/'cases',DEST/'roi',DEST/'figs'):path.mkdir(parents=True,exist_ok=True)
    source=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'));patients=source['patients']
    if args.patients:patients=[p for p in patients if int(p[2:]) in args.patients]
    save(DEST/'protocol.json',dict(patients=patients,rule=engine.RULE,variants=engine.VARIANTS,acceptance=ACCEPT,
          code=digest(__file__),engine=digest(Path(engine.__file__)),shared_driver=digest(Path(base.__file__)),
          dependencies=dependency_hashes(),
          previous_protocol=digest(SOURCE/'protocol.json'),FH_protocol=digest(FH_SOURCE/'protocol.json'),
          previous_outputs={n:digest(SOURCE/n) for n in ('point_stability.csv','metric_restoration.csv','rules_frozen.json')},
          source_FH_cases={p:digest(FH_SOURCE/'cases'/f'{p}.json') for p in patients},anatomically_confirmed=False))
    models={s:np.load(ROOT/f'Result_py/shape/{s}/points.npz') for s in ('MAX','MAND')}
    annotations={i:landmarks(i)[0] for i in engine.RULE['reference_patients']}
    cases=[process(p,models,annotations) for p in patients]
    factor_analysis(cases)
    # Reuse established propagation, metric-specific dependencies and exact-slice rendering.
    base.DEST=DEST;base.report=report
    base.figures(cases);base.evaluate(cases)
    print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
