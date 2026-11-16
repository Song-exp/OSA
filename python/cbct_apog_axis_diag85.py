"""1-2: v3 구조 게이트 통과 후보·기존 좌표·같은 FH 계측을 분리하고 FH 기울기 의존성을 진단한다.

이전 산출물은 읽기만 한다. 새 후보를 만들거나 엑셀로 축·점을 고르지 않는다.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cbct_apog_axis_diag as engine
from cbct_patient_pipeline import OUT,CB,digest,landmarks
from cbct_seg_landmarks85 import load_mask,save

DEST=OUT/'develop85/apog_axis_diag_v1'
SOURCE=OUT/'develop85/apog_structure_v3'
LEGACY=OUT/'develop85/v1/error_ranking.csv'
FH_ROUTES=OUT/'develop85/fh_quality_v1/patient_routes.csv'
METRICS=('ANB','Nperp-A','Nperp-Pog','MP-FH')
FOCUS=12   # 그림을 남길 Nperp-Pog 오차 상위 환자 수(오차는 그림 선택에만 사용)


def process(patient):
    P=landmarks(int(patient[2:]))[0]
    case=json.loads((SOURCE/'cases'/f'{patient}.json').read_text(encoding='utf-8'))
    mask,dirs,org=load_mask(patient,'MAND')
    cand={k:np.asarray(case['points'][k]['stability']['candidates_lps'],float).reshape(-1,3) for k in ('A','Pog')}
    route={k:case['points'][k]['route'] for k in ('A','Pog')}
    sweep=[]
    for d in engine.RULE['pitch_grid_deg']:
        oy,Ry=engine.frame(P,'yao4',d);oj,Rj=engine.frame(P,'jung',d)
        pog=engine.legacy_pog(P,mask,dirs,org,oy,Ry)
        sweep.append(dict(Patient=patient,delta_deg=d,Nperp_A=engine.nperp(P,'A',oy,Ry),
                          Nperp_Pog_ALI=engine.nperp(P,'Pog',oy,Ry),Nperp_Pog_legacy=pog['value'],
                          Pog_status=pog['status'],Pog_edge=pog['edge'],Pog_height_above_Me=pog['height'],
                          ANB=engine.anb(P,oj,Rj),MP_FH=engine.mpfh(P,oy,Ry)))
    del mask
    oy,Ry=engine.frame(P,'yao4');oj,Rj=engine.frame(P,'jung')
    # 같은 FH(기존 ALI Po/Or)에서 후보 점만 바꾼 계측
    fixed=dict(A=[engine.nperp({'A':x},'A',oy,Ry) for x in cand['A']],
               Pog=[engine.nperp({'Pog':x},'Pog',oy,Ry) for x in cand['Pog']],
               ANB=[engine.anb({'A':x,'B':P['B']},oj,Rj) for x in cand['A']])
    return dict(Patient=patient,route=route,counts={k:len(v) for k,v in cand.items()},fixed=fixed,sweep=sweep)


def analyse(results):
    legacy=pd.read_csv(LEGACY).set_index(['Patient','metric'])
    joint=pd.read_csv(SOURCE/'metric_restoration.csv');joint=joint[joint['mode']=='joint'].set_index(['Patient','metric'])
    sweeps=pd.DataFrame([s for r in results for s in r['sweep']]);sweeps.to_csv(DEST/'pitch_sweep.csv',index=False,encoding='utf-8-sig')
    fh_release=pd.read_csv(FH_ROUTES).set_index('Patient').release.astype(bool)
    rows=[]
    for r in results:
        p=r['Patient'];s=sweeps[sweeps.Patient==p].sort_values('delta_deg');zero=s[s.delta_deg==0].iloc[0]
        auto={'ANB':zero.ANB,'Nperp-A':zero.Nperp_A,'Nperp-Pog':zero.Nperp_Pog_legacy,'MP-FH':zero.MP_FH}
        slopes={'ANB':engine.slope(s.delta_deg,s.ANB),'Nperp-A':engine.slope(s.delta_deg,s.Nperp_A),
                'Nperp-Pog':engine.slope(s.delta_deg,s.Nperp_Pog_legacy),'MP-FH':engine.slope(s.delta_deg,s.MP_FH)}
        rigid=engine.slope(s.delta_deg,s.Nperp_Pog_ALI)
        # 후보·분기·축 진단을 모두 계산한 뒤 엑셀을 읽는다.
        excel={m:float(legacy.loc[(p,m),'excel']) for m in METRICS}
        zero_ref={m:bool(legacy.loc[(p,m),'excel_zero']) for m in METRICS}
        # 기존 자동값이 원 ALI FH로 재현되지 않으면(반사 보정 Po) FH 불확실로 분리한다.
        repair='' if all(abs(auto[m]-float(legacy.loc[(p,m),'automatic']))<1e-6 for m in METRICS) else str(legacy.loc[(p,'ANB'),'repairs'])
        err={m:excel[m]-auto[m] for m in METRICS}
        use=[m for m in ('Nperp-A','Nperp-Pog') if not zero_ref[m]]
        delta,res=engine.common_pitch([err[m] for m in use],[slopes[m] for m in use])
        resid=dict(zip(use,res))
        pitch_ok=(np.isfinite(delta) and abs(delta)<=engine.RULE['pitch_explained']['max_abs_delta_deg'] and len(use)==2
                  and all(abs(x)<=engine.RULE['pitch_explained']['max_residual_mm'] for x in res))
        cand={'Nperp-A':r['fixed']['A'],'Nperp-Pog':r['fixed']['Pog'],'ANB':r['fixed']['ANB'],'MP-FH':[]}
        for m in METRICS:
            v=np.asarray(cand[m],float);gap=engine.reach(v,excel[m]) if not zero_ref[m] else np.nan
            jr=joint.loc[(p,m)] if (p,m) in joint.index else None
            jgap=engine.reach([jr.low,jr.high],excel[m]) if jr is not None and pd.notna(jr.low) and not zero_ref[m] else np.nan
            implied=err[m]/slopes[m] if abs(slopes[m])>1e-6 else np.nan
            rows.append(dict(Patient=p,metric=m,point='A' if m in ('ANB','Nperp-A') else 'Pog' if m=='Nperp-Pog' else None,
                             route=r['route'].get('A' if m in ('ANB','Nperp-A') else 'Pog'),
                             legacy_repair=repair,FH_released=bool(fh_release[p]),
                             legacy_automatic=float(legacy.loc[(p,m),'automatic']),
                             automatic=auto[m],excel=excel[m],excel_zero=zero_ref[m],error_excel_minus_auto=err[m],
                             passed_candidates=len(v),candidate_min=v.min() if len(v) else np.nan,
                             candidate_median=np.median(v) if len(v) else np.nan,candidate_max=v.max() if len(v) else np.nan,
                             candidate_gap_to_excel=gap,joint_low=None if jr is None else jr.low,joint_high=None if jr is None else jr.high,
                             joint_status=None if jr is None else jr.status,joint_gap_to_excel=jgap,
                             pitch_slope_per_deg=slopes[m],implied_pitch_deg=implied,
                             common_pitch_deg=delta,common_pitch_residual=resid.get(m,np.nan),
                             Pog_rigid_slope=rigid if m=='Nperp-Pog' else np.nan,
                             Pog_definition_slope=slopes[m]-rigid if m=='Nperp-Pog' else np.nan,
                             Pog_status=zero.Pog_status if m=='Nperp-Pog' else None,
                             Pog_edge=zero.Pog_edge if m=='Nperp-Pog' else None,
                             Pog_edge_fraction=float(s.Pog_edge.notna().mean()) if m=='Nperp-Pog' else np.nan,
                             diagnosis='FH_legacy_repaired_excluded' if repair else engine.classify(np.nan if zero_ref[m] else err[m],engine.RULE['small_error'][m],gap,
                                                       pitch_ok and m in ('Nperp-A','Nperp-Pog'))))
    t=pd.DataFrame(rows);t.to_csv(DEST/'decomposition.csv',index=False,encoding='utf-8-sig')
    return t,sweeps


def population(t,subset):
    valid=t[~t.excel_zero&(t.legacy_repair.fillna('')=='')]
    if subset=='FH_released':valid=valid[valid.FH_released]
    w=valid.pivot(index='Patient',columns='metric',values='error_excel_minus_auto')
    s=valid.pivot(index='Patient',columns='metric',values='pitch_slope_per_deg')
    d=valid.pivot(index='Patient',columns='metric',values='implied_pitch_deg')
    both=w[['Nperp-A','Nperp-Pog']].dropna()
    ratio=float(np.polyfit(both['Nperp-A'],both['Nperp-Pog'],1)[0])
    expected=float((s['Nperp-Pog']/s['Nperp-A']).median())
    mp=d[['Nperp-Pog','MP-FH']].dropna();na=d[['Nperp-A','MP-FH']].dropna()
    b=pd.concat([w[['Nperp-A','Nperp-Pog']],s[['Nperp-A','Nperp-Pog']].add_prefix('s_')],axis=1).dropna()
    eA,eP,sA,sP=b['Nperp-A'],b['Nperp-Pog'],b['s_Nperp-A'],b['s_Nperp-Pog']
    dl=(sA*eA+sP*eP)/(sA**2+sP**2);c=(eA+eP)/2
    rms=dict(no_model=np.sqrt((eA**2+eP**2)/2),FH_pitch=np.sqrt(((eA-sA*dl)**2+(eP-sP*dl)**2)/2),N_AP_shift=np.sqrt(((eA-c)**2+(eP-c)**2)/2))
    big=(eA.abs()>1)|(eP.abs()>1)
    models=dict(n=len(b),mean_RMS={k:float(v.mean()) for k,v in rms.items()},
                pitch_better_n=int((rms['FH_pitch']<rms['N_AP_shift']).sum()),
                same_sign_when_any_error_over_1=f"{int((np.sign(eA[big])==np.sign(eP[big])).sum())}/{int(big.sum())}",
                note='두 모델 모두 환자당 자유도 1. 2개 계측으로는 FH 기울기와 N 앞뒤 위치를 완전히 분리할 수 없음')
    dm=w['MP-FH']/s['MP-FH'];cf={}
    for k in ('Nperp-A','Nperp-Pog'):
        e=w[k].dropna();r=(w[k]-s[k]*dm).dropna();e=e.loc[r.index]
        cf[k]=dict(n=len(r),MAE=float(e.abs().mean()),MAE_after_MPFH_implied_pitch=float(r.abs().mean()))
    out=dict(subset=subset,patients=int(valid.Patient.nunique()),MPFH_pitch_counterfactual=cf,common_component_models=models,
             MPFH_implied_pitch_mean=float(dm.mean()),MPFH_implied_pitch_sd=float(dm.std()),n_A_Pog_pairs=len(both),corr_error_A_Pog=float(both.corr().iloc[0,1]),
             fitted_Pog_per_A_error=ratio,expected_Pog_per_A_if_pitch=expected,
             corr_implied_pitch_Pog_vs_MPFH=float(mp.corr().iloc[0,1]),n_Pog_MPFH=len(mp),
             corr_implied_pitch_A_vs_MPFH=float(na.corr().iloc[0,1]),n_A_MPFH=len(na),
             ANB_max_abs_pitch_slope=float(t[t.metric=='ANB'].pitch_slope_per_deg.abs().max()),
             median_slope={m:float(s[m].median()) for m in METRICS},
             median_implied_pitch={m:float(d[m].median()) for m in METRICS},
             common_pitch_median=float(valid.drop_duplicates('Patient').common_pitch_deg.median()))
    return out


def counts_of(t):
    counts=t.groupby(['metric','diagnosis']).size().rename('n').reset_index()
    counts.to_csv(DEST/'diagnosis_counts.csv',index=False,encoding='utf-8-sig')
    return counts


def figures(t,sweeps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    valid=t[~t.excel_zero&(t.legacy_repair.fillna('')=='')];w=valid.pivot(index='Patient',columns='metric',values='error_excel_minus_auto')
    s=valid.pivot(index='Patient',columns='metric',values='pitch_slope_per_deg')
    d=valid.pivot(index='Patient',columns='metric',values='implied_pitch_deg')
    fig,ax=plt.subplots(1,2,figsize=(13,5.5))
    both=w[['Nperp-A','Nperp-Pog']].dropna();k=float((s['Nperp-Pog']/s['Nperp-A']).median())
    ax[0].scatter(both['Nperp-A'],both['Nperp-Pog'],s=18,color='#2a78d6')
    x=np.linspace(both['Nperp-A'].min(),both['Nperp-A'].max(),2);ax[0].plot(x,k*x,color='#eb6834',label=f'FH 기울기만 다를 때 예상 (기울기 {k:.2f})')
    for p,(a,b) in both.iterrows():
        if abs(b)>6:ax[0].annotate(p,(a,b),fontsize=8)
    ax[0].axhline(0,color='#999',lw=.6);ax[0].axvline(0,color='#999',lw=.6);ax[0].legend()
    ax[0].set_xlabel('Nperp-A 엑셀 - 자동 (mm)');ax[0].set_ylabel('Nperp-Pog 엑셀 - 자동 (mm)')
    ax[0].set_title('A·Pog 오차가 같은 FH 기울기로 설명되는가')
    m=d[['Nperp-Pog','MP-FH']].dropna()
    ax[1].scatter(m['MP-FH'],m['Nperp-Pog'],s=18,color='#2a78d6');lim=8
    ax[1].plot([-lim,lim],[-lim,lim],color='#eb6834',label='같은 기울기');ax[1].set_xlim(-lim,lim);ax[1].set_ylim(-lim,lim)
    ax[1].set_xlabel('MP-FH 오차가 뜻하는 FH 기울기 (°)');ax[1].set_ylabel('Nperp-Pog 오차가 뜻하는 FH 기울기 (°)')
    ax[1].set_title('독립 계측(MP-FH)과의 일치');ax[1].legend()
    fig.tight_layout();fig.savefig(DEST/'figs/population.png',dpi=110);plt.close(fig)
    pog=valid[valid.metric=='Nperp-Pog'].assign(a=lambda x:x.error_excel_minus_auto.abs()).nlargest(FOCUS,'a')
    for _,row in pog.iterrows():
        p=row.Patient;g=sweeps[sweeps.Patient==p]
        fig,ax=plt.subplots(1,2,figsize=(13,5))
        ax[0].plot(g.delta_deg,g.Nperp_Pog_legacy,'-o',ms=3,label='기존 Pog 규칙(축 따라 재탐색)')
        ax[0].plot(g.delta_deg,g.Nperp_Pog_ALI,'--',label='ALI Pog 고정(좌표계 회전만)')
        ax[0].plot(g.delta_deg,g.Nperp_A,'-',color='#28ad49',label='Nperp-A')
        edge=g[g.Pog_edge.notna()];ax[0].scatter(edge.delta_deg,edge.Nperp_Pog_legacy,marker='x',color='r',zorder=3,label='창 경계 선택')
        ax[0].axhline(row.excel,color='#eb6834',lw=1,label='엑셀 Nperp-Pog')
        a=t[(t.Patient==p)&(t.metric=='Nperp-A')].iloc[0]
        if not a.excel_zero:ax[0].axhline(a.excel,color='#28ad49',lw=1,ls=':',label='엑셀 Nperp-A')
        if len(row) and pd.notna(row.candidate_min):ax[0].axhspan(row.candidate_min,row.candidate_max,color='#2a78d6',alpha=.12,label='통과 후보 범위(δ=0)')
        ax[0].set_xlabel('FH 앞뒤 기울기 δ (°)');ax[0].set_ylabel('Nperp (mm)');ax[0].legend(fontsize=8)
        ax[0].set_title(f'{p}: 기울기 의존성 / 공통 δ*={row.common_pitch_deg:.2f}°')
        ax[1].plot(g.delta_deg,g.Pog_height_above_Me,'-o',ms=3);ax[1].axhline(3,color='r',lw=.6);ax[1].axhline(20,color='r',lw=.6)
        ax[1].set_xlabel('δ (°)');ax[1].set_ylabel('선택된 Pog 높이: Me 위 (mm)');ax[1].set_title('기존 규칙 창(3~20 mm) 안에서 선택 높이')
        fig.tight_layout();fig.savefig(DEST/'figs'/f'{p}_pitch.png',dpi=105);plt.close(fig)
    return list(pog.Patient)


def report(t,pop,counts,focus):
    def table(x,f='{:.3f}'):return x.to_html(index=False,border=0,float_format=lambda v:f.format(v),na_rep='')
    cols=['Patient','metric','route','automatic','excel','error_excel_minus_auto','passed_candidates','candidate_min','candidate_max',
          'candidate_gap_to_excel','joint_status','joint_gap_to_excel','pitch_slope_per_deg','implied_pitch_deg','common_pitch_deg',
          'common_pitch_residual','Pog_status','Pog_edge','Pog_edge_fraction','diagnosis']
    big=t[~t.excel_zero&(t.legacy_repair.fillna('')=='')].assign(a=lambda x:x.error_excel_minus_auto.abs()).sort_values(['metric','a'],ascending=[True,False]).groupby('metric').head(10)
    def flat(x):return {k:(f'{v:.3f}' if isinstance(v,float) else v) for k,v in x.items() if not isinstance(v,dict)}
    sens=pop['FH_released_sensitivity']
    popt=pd.DataFrame(dict(item=list(flat(pop)),no_legacy_repair=list(flat(pop).values()),FH_released=[flat(sens).get(k) for k in flat(pop)]))
    cf=pd.DataFrame([dict(subset=x['subset'],metric=k,**v) for x in (pop,sens) for k,v in x['MPFH_pitch_counterfactual'].items()])
    med=pd.DataFrame(dict(slope=pop['median_slope'],implied_pitch=pop['median_implied_pitch'])).reset_index().rename(columns={'index':'metric'})
    cards=''.join(f'<figure><img loading="lazy" src="figs/{p}_pitch.png"><figcaption>{p}</figcaption></figure>' for p in focus)
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A·Pog 축 진단</title>
<style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:16px}}table{{border-collapse:collapse;background:white;font-size:12px}}td,th{{border:1px solid #ddd;padding:4px 6px}}.scroll{{overflow:auto}}img{{max-width:100%}}figure{{background:white;margin:16px 0;padding:8px}}</style><main>
<h1>1-2. A·Pog 남은 오차의 원인 분리: 점 위치 vs FH 기준축</h1>
<p>v3 구조 게이트 통과 후보와 분기를 그대로 읽어, 기존 ALI Po/Or로 만든 같은 FH에서 점만 바꾼 계측값과 FH 앞뒤 기울기(δ, −6~+6°)만 바꾼 계측값을 따로 계산했습니다. 엑셀은 모든 후보·기울기 계산이 끝난 뒤 오차 분해에만 사용했고 축·후보 선택에 쓰지 않았습니다. 엑셀 값 0은 결측 의심으로 분리했습니다. 85명 모두 개발 자료이며, 여기의 분류는 원인 가설의 진단이지 보정 규칙이 아닙니다.</p>
<p>ANB는 정중면 투영 각도 차이라 FH 앞뒤 기울기에 영향을 받지 않습니다(최대 기울기 민감도 {pop['ANB_max_abs_pitch_slope']:.2e}°/°). 따라서 ANB 오차는 축이 아니라 A·B·N 위치와 정중면 정의 쪽 문제입니다. Nperp-A·Nperp-Pog·MP-FH는 δ에 민감하며, FH가 다르면 같은 δ가 세 계측에 정해진 비율로 나타나야 합니다.</p>
<p>기존 자동값을 원 ALI 좌표로 재현했습니다(반사 보정이 없는 81명, 최대 차이 1e-6 미만). P_16·20·23·37은 기존값에 반대편 반사 Po가 들어가 있어 원 ALI FH로 재현되지 않습니다. 대칭 보정을 새로 쓰지 않으므로 이 4명은 FH 불확실로 분리하고 집단 통계에서 제외했습니다. FH 품질 단계 통과 환자만 쓴 민감도 결과를 함께 표시합니다.</p>
<h2>1. 집단 수준 진단</h2>{table(popt)}{table(med)}
<h3>반사실 점검: MP-FH 오차가 뜻하는 기울기로 Nperp 오차를 빼면</h3>{table(cf)}
<h3>A·Pog 공통 오차 성분: FH 기울기 모델 vs N 앞뒤 이동 모델</h3><pre>{json.dumps(pop['common_component_models'],ensure_ascii=False,indent=1)}</pre>
<p>MP-FH는 Go·Me만 쓰는 독립 계측입니다. 엑셀과 CBCT의 FH 기울기 차이가 공통 원인이라면 이 보정 후 Nperp 오차가 줄어야 합니다. 진단용 계산이며 보정 규칙이 아닙니다.</p>
<img src="figs/population.png">
<h2>2. 계측별 진단 분류</h2>{table(counts)}
<p>분류 순서: 오차 ≤1(mm 또는 °) → 같은 FH의 통과 후보 범위가 엑셀을 포함 → A·Pog 두 오차를 하나의 δ(|δ|≤3°)로 잔차 ≤1 mm 이내 설명 → 미설명. 임계값은 엔지니어링 진단 기준이며 임상 허용오차가 아닙니다.</p>
<h2>3. 계측별 오차 상위 10명</h2><div class="scroll">{table(big[cols])}</div>
<h2>4. Nperp-Pog 오차 상위 {len(focus)}명의 기울기 의존성</h2>
<p>왼쪽: 기존 Pog 규칙을 기울어진 축으로 다시 실행한 값(정의 효과 포함)과 ALI Pog를 고정한 채 좌표계만 돌린 값(순수 회전 효과)의 비교. 빨간 x는 Pog가 탐색창 경계(Me+3 또는 Me+20 mm)에서 선택된 경우입니다. 오른쪽: 선택 높이.</p>{cards}
<h2>자료</h2><p><a href="decomposition.csv">환자×계측 분해</a> · <a href="pitch_sweep.csv">δ 스윕</a> · <a href="diagnosis_counts.csv">분류 집계</a> · <a href="population.json">집단 지표</a> · <a href="protocol.json">규칙</a></p>
<p>기존 ALI 좌표, 운영 계측, v1~v3 산출물은 변경하지 않았습니다. 해부학적 정답 확인이 아닙니다.</p></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')


def main():
    for path in (DEST,DEST/'figs'):path.mkdir(parents=True,exist_ok=True)
    for f in (DEST/'figs').glob('*_pitch.png'):f.unlink()   # 이전 실행의 환자 그림이 남지 않게
    protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'));patients=protocol['patients']
    save(DEST/'protocol.json',dict(patients=patients,rule=engine.RULE,code=digest(__file__),engine=digest(Path(engine.__file__)),
         source_protocol=digest(SOURCE/'protocol.json'),source_cases={p:digest(SOURCE/'cases'/f'{p}.json') for p in patients},
         legacy=digest(LEGACY),source_metric=digest(SOURCE/'metric_restoration.csv'),anatomically_confirmed=False))
    results=[]
    for p in patients:
        results.append(process(p));print(p,'done',flush=True)
    t,sweeps=analyse(results);counts=counts_of(t)
    pop=population(t,'no_legacy_repair');pop['FH_released_sensitivity']=population(t,'FH_released')
    clean=t[t.legacy_repair.fillna('')=='']
    verification=dict(patients=len(results),rows=len(t),
        max_abs_diff_vs_legacy_unrepaired=float((clean.automatic-clean.legacy_automatic).abs().max()),
        repaired_patients=sorted(t[t.legacy_repair.fillna('')!=''].Patient.unique()),
        ANB_pitch_invariant=bool(pop['ANB_max_abs_pitch_slope']<1e-9),anatomically_confirmed=False)
    assert verification['max_abs_diff_vs_legacy_unrepaired']<1e-6,verification
    save(DEST/'verification.json',verification);save(DEST/'population.json',pop);focus=figures(t,sweeps);report(t,pop,counts,focus)
    print(json.dumps(pop,ensure_ascii=False,indent=1));print(counts.to_string(index=False));print('saved',DEST/'report.html')


if __name__=='__main__':main()
