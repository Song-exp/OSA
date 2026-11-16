"""최종 통합(잠정): FH·A/Pog·남은 기준점 평가를 결합해 85명×4계측의 값·범위·보류와 확정 등급을 만든다.

1) 등급은 기준점 근거만으로 정하고, 2) 엑셀은 등급·값이 모두 정해진 뒤 사후 비교에만 쓴다.
운영 채택이 아니며 이전 산출물은 읽기만 한다.
"""
from itertools import product
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from cbct_patient_pipeline import OUT,digest,landmarks
from cbct_po_ensemble import metric_decision,ACCEPT
from cbct_po_stability85 import pog_value
from cbct_seg_landmarks85 import load_mask,save
from osa.ceph import measures
import cbct_apog_axis_diag as axdiag

D85=OUT/'develop85'
DEST=D85/'final_criteria_v2'
FH_KEYS={'jung':('RPo','LPo','ROr'),'yao4':('RPo','LPo','ROr','LOr')}
FH_PRIMARY='yao4'   # 좌우 Po·Or 4점 최소제곱: 단일 점 오차에 덜 민감하고 좌우를 대칭 취급, 기존 운영 정의와 연속(v2 결정)
SENSITIVITY='jung'
METRICS=None


def metrics(fh):
    F=FH_KEYS[fh]
    return {'ANB':('jung','ANB',('N','A','B','RPo','LPo','ROr','Ba')),
            'Nperp-A':(fh,'Nperp_A',('N','A',*F,'Ba')),
            'Nperp-Pog':(fh,'Nperp_Pog',('N','Me',*F,'Ba')),
            'MP-FH':(fh,'MPFH_GoMe',('N','Me','RGo','LGo',*F,'Ba'))}
RULE=dict(version='final-criteria-v2',max_candidates_per_point=6,max_combinations=4096,
          metric_width=ACCEPT['metric_width'],Ba_perturbation_mm=2.,
          Ba_sensitivity_limit='Ba ±2 mm(3축) 변동폭 ≤ 계측 허용폭의 1/2',
          tiers={'T1':'모든 관련 점 독립 지지 + 교체 없음 + Ba 민감도 이내 → 잠정 확정',
                 'T2':'교체 후보 조합의 계측 변동폭이 허용폭 이내 + 나머지 점 지지 → 잠정 확정(재계산값)',
                 'T3':'숫자는 있으나 지지 근거가 없는 점 포함 → 미확정 숫자',
                 'T4':'후보 범위만 → 범위 제공',
                 'T5':'근거 부족·조합 초과 → 보류'},
          excel_review_threshold={'ANB':2.,'Nperp-A':3.,'Nperp-Pog':3.,'MP-FH':3.},
          FH_definition='yao4(RPo·LPo·ROr·LOr 최소제곱면)를 4계측 공통 정의로 사용. 근거: Po·Or 개별 재현성 낮음(Dot 2022) → 4점 평균이 단일 점 오차에 강건, 좌우 대칭 취급, 기존 운영 정의와 연속. jung 3점(계획 문서 Invivo AFOR)은 민감도로 제공. 참고: v1에서 yao4가 엑셀 MAE로 선택된 이력이 있음',
          final_value_rule={'A':'T1: 기존 좌표 값, 관련 점 모두 독립 지지','B':'T2: 교체 후보 조합 중앙값, 변동폭 허용 이내',
                            'C':'T3: 기존 좌표 값, 일부 점 지지 근거 없음','D':'교체 후보 조합 중앙값이나 변동폭 초과 또는 후보 근거 부족(범위 함께 제공)',
                            'E':'교체가 필요한데 후보 없음/조합 초과 → 기존 좌표 값(신뢰 낮음, 분석 제외 권장)'},
          Nperp_Pog_definition='모든 환자에서 FH 좌표계 기준 하악 마스크 최전방점(기존 refine: Me+3~20 mm 창). ALI Pog는 refine이 ALI로 되돌아갈 때만 쓰며 그때 2단계 Pog 지지 필요. 정중 좌표계 최전방점(2단계 Pog 후보)은 측정 좌표계와 정의가 달라 Nperp-Pog 값에 쓰지 않음(엑셀 비교에서 발견, 결정 근거는 정의)',
          not_assessed=('S','Ba','Gn 외 계측 미사용점'),Excel_used_for_values=False,production_adoption=False)


def fps(points,k):
    """대표점을 포함한 최원점 표본으로 후보 조합 수를 제한한다(변동 범위 보존 목적)."""
    pts=np.asarray(points,float).reshape(-1,3)
    if len(pts)<=k:return pts
    D=cdist(pts,pts);chosen=[int(D.sum(1).argmin())]
    while len(chosen)<k:chosen.append(int(D[:,chosen].min(1).argmax()))
    return pts[chosen]


def support_from_deviation(dev,off):
    a,b=(None if dev.get(f) is None else dev[f]-off[f] for f in ('mask','native'))
    both=a is not None and b is not None and abs(a)>1 and abs(b)>1 and np.sign(a)==np.sign(b)
    return bool(any(x is not None and abs(x)<=1 for x in (a,b)) and not both)


def collect(patients):
    """점별 {triggered, candidates_lps, adequate, support, route}."""
    audit=json.loads((D85/'landmark_audit_v1/decisions.json').read_text(encoding='utf-8'))
    quality=pd.read_csv(D85/'fh_quality_v1/point_quality.csv');quality=quality[quality.stage=='original'].set_index(['Patient','point']).passed
    apog={p:json.loads((D85/f'apog_structure_v3/cases/{p}.json').read_text(encoding='utf-8'))['points'] for p in patients}
    devs={k:[apog[p][k]['original_boundary_deviation'] for p in patients] for k in ('A','Pog')}
    off={k:{f:float(np.median([d[f] for d in v if d[f] is not None])) for f in ('mask','native')} for k,v in devs.items()}
    out={}
    for p in patients:
        fh=json.loads((D85/f'fh_joint_v1/cases/{p}.json').read_text(encoding='utf-8'))['points'];pts={}
        for k,r in fh.items():
            pts[k]=dict(triggered=r['triggered'],candidates_lps=r['stability']['candidates_lps'],adequate=r['stability']['adequate_support'],
                        support=bool(not r['triggered'] and quality.get((p,k),False)),route=r['route'],stage='FH(fh_joint_v1+fh_quality_v1)')
        for k,r in apog[p].items():
            pts[k]=dict(triggered=r['triggered'],candidates_lps=r['stability']['candidates_lps'],adequate=r['stability']['adequate_support'],
                        support=bool(not r['triggered'] and support_from_deviation(r['original_boundary_deviation'],off[k])),
                        route=r['route'],stage='A/Pog(apog_structure_v3)')
        for k,r in audit[p].items():
            pts[k]=dict(triggered=r['triggered'],candidates_lps=r['candidates_lps'],adequate=r['adequate_support'],
                        support=r['independent_support'],route=r['route'],stage='2단계(landmark_audit_v1)')
        for k in ('S','Ba'):pts[k]=dict(triggered=False,candidates_lps=[],adequate=True,support=None,route='not_assessed',stage='미평가')
        out[p]=pts
    return out,off


def value(Q,metric,mand,dirs,org,pog_triggered=False):
    fh,name,_=METRICS[metric]
    # 기존 refine Pog 규칙을 해당 FH 좌표계에서 실행(yao4에서는 pog_value와 동일, 1-2에서 재현 확인)
    if metric=='Nperp-Pog':return axdiag.legacy_pog(Q,mand,dirs,org,*axdiag.frame(Q,fh))['value']
    return float(measures(Q,fh)[name])


def evaluate(p,pts,mand,dirs,org):
    P=landmarks(int(p[2:]))[0];rows=[]
    for metric,(_,_,keys) in METRICS.items():
        if metric!='Nperp-Pog':keys=tuple(k for k in keys if k!='Me' or metric=='MP-FH')
        active=[k for k in keys if pts[k]['triggered']]
        adequate=all(pts[k]['adequate'] and len(pts[k]['candidates_lps']) for k in active)
        # 후보가 없는 교체 대상점은 보류 사유가 되며, 민감도 계산의 중심에만 기존점을 쓴다.
        arrays=[fps(pts[k]['candidates_lps'],RULE['max_candidates_per_point']) if k in active and len(pts[k]['candidates_lps'])
                else np.asarray([P[k]]) for k in keys]
        n=int(np.prod([len(a) for a in arrays]))
        vals=[]
        if n<=RULE['max_combinations']:
            for combo in product(*arrays):
                Q=dict(P);Q.update({k:np.asarray(x) for k,x in zip(keys,combo)})
                vals.append(value(Q,metric,mand,dirs,org,pts['Pog']['triggered']))
        else:adequate=False
        result=metric_decision(vals,adequate,metric) if active else dict(status='existing_point_value',value=vals[0],low=vals[0],high=vals[0])
        # 미평가 Ba의 영향: 중심 조합에서 Ba를 3축 ±2 mm 이동
        Q=dict(P);Q.update({k:np.asarray(a[0]) for k,a in zip(keys,arrays)});center=value(Q,metric,mand,dirs,org,pts['Pog']['triggered'])
        ba=[]
        for axis in range(3):
            for sgn in (-1,1):
                Q2=dict(Q);Q2['Ba']=np.asarray(P['Ba'],float)+sgn*RULE['Ba_perturbation_mm']*np.eye(3)[axis]
                ba.append(value(Q2,metric,mand,dirs,org,pts['Pog']['triggered']))
        ba_width=float(max(ba+[center])-min(ba+[center]))
        fh_def={f:float(measures(Q,f)[METRICS[metric][1]]) for f in ('jung','yao4','lor')} if metric!='Nperp-Pog' else {}
        assessed=[k for k in keys if pts[k]['support'] is not None]
        unsupported=[k for k in assessed if k not in active and not pts[k]['support']]
        pog_status=None
        if metric=='Nperp-Pog':
            o,R=axdiag.frame(Q,METRICS[metric][0]);pr=axdiag.legacy_pog(Q,mand,dirs,org,o,R);pog_status=pr['status']+(f":edge_{pr['edge']}" if pr['edge'] else '')
            if pr['status']!='refined' and not pts['Pog']['support']:unsupported.append('Pog(ALI fallback)')
        width=ACCEPT['metric_width'][metric];ba_ok=ba_width<=width/2
        if result['status']=='automatic_withhold':tier='T5'
        elif result['status']=='candidate_interval_only':tier='T4'
        elif unsupported or not ba_ok:tier='T3'
        elif result['status']=='candidate_stable_metric':tier='T2'
        else:tier='T1'
        no_cand=[k for k in active if not len(pts[k]['candidates_lps'])]
        if not active:grade={'T1':'A','T3':'C'}[tier];final=vals[0]
        elif no_cand or not vals or n>RULE['max_combinations']:grade='E';final=center
        elif tier=='T2':grade='B';final=result['value']
        elif tier=='T3' and result['value'] is not None:grade='C';final=result['value']
        else:grade='D';final=float(np.median(vals))
        rows.append(dict(Patient=p,metric=metric,final_value=final,grade=grade,
                         range_low=min(vals) if vals else None,range_high=max(vals) if vals else None,
                         no_candidate_points=';'.join(no_cand),tier=tier,status=result['status'],value=result['value'],low=result['low'],high=result['high'],
                         replaced_points=';'.join(active),unsupported_points=';'.join(unsupported),combinations=n,
                         Pog_definition_status=pog_status,Ba_sensitivity=ba_width,Ba_ok=ba_ok,FH_definition_range=(max(fh_def.values())-min(fh_def.values())) if fh_def else None))
    return rows


def excel_compare(t):
    legacy=pd.read_csv(D85/'v1/error_ranking.csv').set_index(['Patient','metric'])
    # legacy_value는 v1 운영값(Nperp·MP-FH는 yao4 정의)이며 정의가 다를 수 있음
    t['legacy_value']=[legacy.loc[(p,m),'automatic'] for p,m in zip(t.Patient,t.metric)]
    t['excel']=[legacy.loc[(p,m),'excel'] for p,m in zip(t.Patient,t.metric)]
    t['excel_zero_suspect']=[bool(legacy.loc[(p,m),'excel_zero']) for p,m in zip(t.Patient,t.metric)]
    t['abs_diff_excel']=(t.final_value-t.excel).abs()
    t.loc[t.excel_zero_suspect,'abs_diff_excel']=np.nan
    t['legacy_abs_diff_excel']=(t.legacy_value-t.excel).abs();t.loc[t.excel_zero_suspect,'legacy_abs_diff_excel']=np.nan
    t['excel_review']=[bool(d>RULE['excel_review_threshold'][m]) if pd.notna(d) else False for d,m in zip(t.abs_diff_excel,t.metric)]
    return t


def zeros_analysis(t):
    z=t[t.excel_zero_suspect]
    return z.groupby('metric').agg(n=('Patient','size'),auto_abs_median=('value',lambda s:s.abs().median()),
                                   auto_abs_over2=('value',lambda s:int((s.abs()>2).sum()))).reset_index()


def airway_inventory():
    b=pd.read_csv(D85/'v1/metric_baseline.csv')
    b=b[~b.metric.isin(list(METRICS))]
    return b[['metric','unit','n','missing_auto','missing_excel','MAE','median_AE','bias']]


def report(t,pts_table,off,zeros,airway,sens):
    def tb(x,f='{:.3f}'):return x.to_html(index=False,border=0,float_format=lambda v:f.format(v),na_rep='')
    tiers=t.pivot_table(index='metric',columns='tier',values='Patient',aggfunc='count',fill_value=0).reset_index()
    t['confirmed']=t.tier.isin(['T1','T2'])
    agree=[]
    for (m,tier),g in t.groupby(['metric','tier']):
        v=g[g.abs_diff_excel.notna()]
        agree.append(dict(metric=m,tier=tier,n=len(g),excel_pairs=len(v),MAE=v.abs_diff_excel.mean(),median_AE=v.abs_diff_excel.median(),
                          excel_review=int(g.excel_review.sum())))
    agree=pd.DataFrame(agree)
    conf=t.groupby('metric').agg(total=('Patient','size'),confirmed_T1_T2=('confirmed','sum'),numeric=('value',lambda s:s.notna().sum())).reset_index()
    pts=pts_table.groupby(['point','stage']).agg(n=('Patient','size'),support=('support',lambda s:int((s==True).sum())),
                                                 replaced=('triggered','sum')).reset_index()
    review=t[t.excel_review].sort_values(['metric','abs_diff_excel'],ascending=[True,False])
    wide=t.pivot(index='Patient',columns='metric',values='tier')
    blockers=pts_table[(pts_table.support==False)&(~pts_table.triggered)].groupby('point').size().rename('unsupported_retained').reset_index()
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>85명 최종 계측값</title>
<style>body{{font:15px/1.6 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:16px}}table{{border-collapse:collapse;background:white;font-size:12px}}td,th{{border:1px solid #ddd;padding:4px 6px}}.scroll{{overflow:auto;max-height:640px}}.warn{{background:#fff4e5;padding:10px;border-left:4px solid #eb6834}}</style><main>
<h1>85명 골격 계측 최종값 v2 (단일 규칙)</h1>
<p class="warn">85명 전원에 같은 규칙을 적용해 계측마다 값 하나와 신뢰 등급(A~E)을 냈습니다. 엑셀은 값·등급을 정한 뒤 비교에만 썼습니다. 기존 ALI 좌표·운영값·이전 보고서는 변경하지 않았습니다. 해부학적 정답 검증이 아닙니다.</p>
<h2>0. 최종값 규칙과 등급 분포</h2>{tb(pd.DataFrame(dict(grade=list(RULE['final_value_rule']),rule=list(RULE['final_value_rule'].values()))))}{tb(t.pivot_table(index='metric',columns='grade',values='Patient',aggfunc='count',fill_value=0).reset_index())}
<p><a href="final_measurements_85.csv">85명 최종값 표(CSV)</a> · <a href="final_measurements_85.xlsx">엑셀 파일</a></p>
<h2>1. 잠정 확정 기준 (엑셀을 쓰지 않는 근거 등급)</h2>{tb(pd.DataFrame(dict(tier=list(RULE['tiers']),meaning=list(RULE['tiers'].values()))))}
<p>독립 지지 = 고정 참고 환자 표면 대응으로 만든 ROI에서 분할·원본 CT 윤곽 극점을 24조건으로 찾고, 기존점이 ALI 관습 오프셋을 뺀 뒤 윤곽 1 mm 이내이며 교체 조건이 없는 경우. FH 4점은 FH 품질 단계 원점 통과 + 교체 없음. S·Ba는 미평가이며 Ba는 ±2 mm 민감도로만 확인했습니다. 계측 허용폭(T2 판정)은 ANB·MP-FH 1°, Nperp 2 mm로 기존 고정값입니다.</p>
<h2>2. 결과: 계측별 등급</h2>{tb(conf)}{tb(tiers)}
<h2>3. 기준점별 근거 (2단계 포함)</h2>{tb(pts)}<p>지지 근거 없이 유지된 점(T3의 원인):</p>{tb(blockers)}
<p>A/Pog 정상 방향 관습 오프셋(85명 중앙값): <code>{json.dumps(off)}</code></p>
<h2>4. 사후 엑셀 비교 (등급별)</h2>{tb(agree)}
<p>엑셀은 값·등급이 정해진 뒤에만 읽었습니다. 엑셀 0은 결측 의심으로 비교에서 뺐습니다. 차이가 임계값(ANB 2°, 그 외 3 mm/°)을 넘는 경우는 값을 바꾸지 않고 검토 목록으로만 둡니다.</p>
<h3>엑셀 0 의심값 점검 (5단계)</h3>{tb(zeros)}
<p>엑셀 0 위치에서 자동값의 절댓값이 2를 넘는 경우가 많으면 0은 실제 측정이 아니라 미입력일 가능성이 큽니다.</p>
<h2>5. 계측 정의 (5단계)</h2>
<ul><li>ANB: 정중면(MSP) 투영 N-A와 N-B 방향각의 차. FH 앞뒤 기울기에 불변(1-2에서 수치 확인)이므로 FH 정의 선택과 무관.</li>
<li>FH: <b>yao4(좌우 Po·Or 4점 최소제곱면)를 4계측 공통으로 사용</b>. 근거: Po·Or은 사람 측정 재현성도 2 mm 이상 흔들림(Dot 2022) → 4점 평균이 단일 점 오차에 강건하고 좌우를 대칭 취급, 기존 운영 정의와 연속. 계획 문서의 jung 3점(Invivo AFOR)은 민감도로 함께 제공. 참고: v1에서 yao4가 엑셀 MAE로 선택된 이력이 있음.</li>
<li>Nperp-A/Pog: N을 지나 FH에 수직인 선에서 A/Pog의 앞뒤 거리(McNamara).</li>
<li>Nperp-Pog의 Pog: 기존 규칙(Me+3~20 mm 창의 하악 최전방점, 10 mm 넘게 이동하면 ALI 유지). 1-2에서 축 의존성이 순수 회전 효과뿐임을 확인.</li>
<li>MP-FH: Go(좌우 중점)-Me 선과 FH의 각.</li>
<li>주의: 과거 v1에서 FH 변형(jung/yao4/lor)과 refine 여부를 엑셀 MAE로 비교·선택한 이력이 있음. 위 근거는 문헌 정의로 다시 정당화한 것이며, 표의 FH_definition_range로 정의 민감도를 함께 표시함.</li></ul>
<h3>FH 정의 민감도 (jung 주 정의 vs yao4)</h3>{tb(sens)}
<h2>6. 기도 계측 현황 (범위 결정 필요)</h2>{tb(airway)}
<p>이 계측들은 이번 골격 기준점 작업 범위 밖입니다. 85명 확정 대상에 포함할지 결정이 필요합니다.</p>
<h2>7. 엑셀 검토 목록 ({len(review)}건)</h2><div class="scroll">{tb(review[['Patient','metric','tier','value','low','high','excel','abs_diff_excel','legacy_value','replaced_points','unsupported_points']])}</div>
<h2>8. 85명 × 4계측 등급표</h2><div class="scroll">{tb(wide.reset_index())}</div>
<h2>자료</h2><p><a href="final_values.csv">최종 값·범위·등급(340행)</a> · <a href="point_evidence.csv">기준점 근거</a> · <a href="protocol.json">규칙</a> ·
단계별 보고서: <a href="../fh_joint_v1/report.html">FH</a> · <a href="../apog_structure_v3/report.html">A/Pog v3</a> · <a href="../apog_axis_diag_v1/report.html">1-2 진단</a> · <a href="../landmark_audit_v1/report.html">2단계 기준점(실제 단면)</a></p></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def final_table(t,patients):
    """85명 1행: 골격 4계측 최종값·등급·범위 + 기도 계측 기존 자동값(이번 기준 재검증 안 함)."""
    w=t.pivot(index='Patient',columns='metric',values='final_value')
    g=t.pivot(index='Patient',columns='metric',values='grade').add_suffix('_grade')
    lo=t.pivot(index='Patient',columns='metric',values='range_low').add_suffix('_low')
    hi=t.pivot(index='Patient',columns='metric',values='range_high').add_suffix('_high')
    order=['ANB','Nperp-A','Nperp-Pog','MP-FH']
    out=pd.concat([w[order],g[[m+'_grade' for m in order]],lo[[m+'_low' for m in order]],hi[[m+'_high' for m in order]]],axis=1)
    legacy=pd.read_csv(D85/'v1/error_ranking.csv')
    air=legacy[~legacy.metric.isin(order)].pivot(index='Patient',columns='metric',values='automatic').add_suffix('_auto_unaudited')
    out=out.join(air).loc[sorted(patients,key=lambda x:int(x[2:]))].reset_index()
    out.to_csv(DEST/'final_measurements_85.csv',index=False,encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(DEST/'final_measurements_85.xlsx') as x:
            out.to_excel(x,sheet_name='values',index=False)
            pd.DataFrame(dict(grade=list(RULE['final_value_rule']),rule=list(RULE['final_value_rule'].values()))).to_excel(x,sheet_name='grades',index=False)
            pd.DataFrame([dict(item=k,value=str(v)) for k,v in RULE.items()]).to_excel(x,sheet_name='rules',index=False)
    except ImportError:pass
    counts=t.pivot_table(index='metric',columns='grade',values='Patient',aggfunc='count',fill_value=0)
    counts.to_csv(DEST/'grade_counts.csv',encoding='utf-8-sig');print(counts.to_string())


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    patients=json.loads((D85/'apog_structure_v3/protocol.json').read_text(encoding='utf-8'))['patients']
    global METRICS
    pts,off=collect(patients)
    save(DEST/'protocol.json',dict(patients=patients,rule=RULE,code=digest(__file__),
         sources={n:digest(D85/n) for n in ('landmark_audit_v1/decisions.json','fh_quality_v1/point_quality.csv','v1/error_ranking.csv')},
         anatomically_confirmed=False))
    tables={}
    for fh in (FH_PRIMARY,SENSITIVITY):
        METRICS=metrics(fh);rows=[]
        for p in patients:
            mand,dirs,org=load_mask(p,'MAND');rows+=evaluate(p,pts[p],mand,dirs,org);del mand
        tables[fh]=pd.DataFrame(rows).assign(FH=fh);print(fh,'done',flush=True)
    METRICS=metrics(FH_PRIMARY);t=tables[FH_PRIMARY]
    pts_table=pd.DataFrame([dict(Patient=p,point=k,stage=r['stage'],route=r['route'],triggered=r['triggered'],support=r['support'])
                            for p in patients for k,r in pts[p].items()])
    pts_table.to_csv(DEST/'point_evidence.csv',index=False,encoding='utf-8-sig')
    # 여기서부터 엑셀 사용(사후 비교)
    t=excel_compare(t);t.to_csv(DEST/'final_values.csv',index=False,encoding='utf-8-sig')
    alt=excel_compare(tables[SENSITIVITY]);alt.to_csv(DEST/f'final_values_{SENSITIVITY}_sensitivity.csv',index=False,encoding='utf-8-sig')
    sens=pd.DataFrame([dict(metric=m,FH=f,numeric=int(x.value.notna().sum()),confirmed=int(x.tier.isin(['T1','T2']).sum()))
                       for f,tt in ((FH_PRIMARY,t),(SENSITIVITY,alt)) for m,x in tt.groupby('metric')])
    both=t.merge(alt,on=['Patient','metric'],suffixes=('','_alt'))
    diff=both.assign(d=(both.final_value-both.final_value_alt).abs()).groupby('metric').d.agg(['median','max']).reset_index()
    sens=sens.merge(diff,on='metric',how='left').rename(columns={'median':'jung_vs_yao4_median_abs','max':'jung_vs_yao4_max_abs'})
    sens.to_csv(DEST/'fh_definition_sensitivity.csv',index=False,encoding='utf-8-sig')
    zeros=zeros_analysis(t);airway=airway_inventory()
    report(t,pts_table,off,zeros,airway,sens)
    final_table(t,patients)
    print(t.groupby(['metric','tier']).size().to_string());print(zeros.to_string())


if __name__=='__main__':main()
