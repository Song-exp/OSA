"""AMASSS 표면 대응으로 ROI를 전달하고 환자 자신의 마스크에서 Po/Or 후보 추출.

현재 환자의 Po/Or, 반대편 반사, 좌우 차이, 엑셀은 후보 선택에 사용하지 않는다.
SSM PCA 점수가 아닌 원래 3D 표면 대응점을 사용하며 저장 좌표 변환을 역변환한다.
현재 CB/MAX는 외이도·안와 전용 분할이 아니므로 후보는 연구용이며 검수 전이다.
"""
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, label, map_coordinates, minimum_filter1d
from scipy.spatial import cKDTree

from cbct_patient_pipeline import CB, OUT, ROOT, landmarks, digest
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd
from cbct_fh_quality85 import json_safe
from osa.ceph import measures
from run_ali import ascii_path

DEST=OUT/'develop85/segmentation_landmarks_v1'
TARGETS={'RPo':'CB','LPo':'CB','ROr':'MAX','LOr':'MAX'}
CENTRAL=('N','S','Ba','ANS','PNS')
STEP=.5
RULE=dict(version='segmentation-roi-native-boundary-v1',step_mm=STEP,
          roi_half_mm=[12.,15.,15.],template_neighbors=16,
          template_anchor='ALI point of fixed existing SSM template, provisional not expert annotation',
          roi_transfer='local affine fit on existing CPD surface correspondences; no PCA reconstruction',
          Po='upper rim of enclosed CB-mask void; outermost stable side-specific section',
          Or='local minimum of superior MAX boundary near transferred infraorbital patch',
          symmetry=False,patient_Po_Or_used_for_detection=False,Excel_used_for_detection=False,
          production_replacement=False)


def save(path,obj):
    path.write_text(json.dumps(json_safe(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sample(array,dirs,org,q,order=0,cval=0):
    ijk=np.linalg.solve(dirs.T,(q.reshape(-1,3)-org).T)
    return map_coordinates(array,ijk[::-1],order=order,output=np.float32,
                           cval=cval,mode='constant',prefilter=False).reshape(q.shape[:-1])


def load_mask(p,struct):
    path=CB/f'seg/{p}_seg_{struct}.nrrd'
    im=sitk.ReadImage(str(ascii_path(path)))
    array=sitk.GetArrayFromImage(im)
    dirs=np.array(im.GetDirection()).reshape(3,3)@np.diag(im.GetSpacing())
    return array,dirs.T,np.array(im.GetOrigin())


def physical_correspondences(z,patient):
    i=list(z['ids']).index(int(patient[2:]))
    # frame was stored in per-patient FH coordinates; invert exactly to original LPS.
    return z['frame'][i]@z['head_R'][i]+z['head_origin'][i]


def local_transfer(template,patient,anchor,k=16):
    dist,ids=cKDTree(template).query(anchor,k=min(k,len(template)))
    d=template[ids]-anchor
    weights=1/np.maximum(dist,1.)**2
    design=np.c_[np.ones(len(ids)),d]
    solution=np.linalg.lstsq(design*np.sqrt(weights[:,None]),
                             patient[ids]*np.sqrt(weights[:,None]),rcond=None)[0]
    prediction=solution[0]
    residual=np.linalg.norm(design@solution-patient[ids],axis=1)
    return prediction,dict(template_surface_gap_mm=float(dist[0]),
                           patch_radius_mm=float(dist[-1]),local_transfer_residual_mm=float(np.median(residual)),
                           local_transfer_condition=float(np.linalg.cond(design)),neighbor_ids=ids.tolist())


def grid(mask,dirs,org,origin,R,center,half=(12.,15.,15.)):
    local=R@(center-origin)
    axes=[np.arange(local[i]-half[i],local[i]+half[i]+STEP/2,STEP) for i in range(3)]
    G=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    physical=origin+G@R
    sampled=sample(mask,dirs,org,physical)>0
    bounds=np.linalg.solve(dirs.T,(physical.reshape(-1,3)-org).T).T
    inside=((bounds>=0)&(bounds<=np.array(mask.shape[::-1])-1)).all(1).reshape(sampled.shape)
    return sampled,axes,inside


def po_sections(mask,axes,side):
    """CB 마스크의 구멍을 찾는다. 원본 밝기의 고정 공기 문턱은 쓰지 않는다."""
    L,A,S=axes;candidates=[]
    for i in range(len(L)):
        bone=mask[i]
        holes=binary_fill_holes(bone)&~bone
        labs,n=label(holes)
        for j in range(1,n+1):
            aa,ss=np.where(labs==j);area=len(aa)*STEP**2
            if not 3<=area<=180:continue
            width=np.ptp(A[aa])+STEP;height=np.ptp(S[ss])+STEP
            if not (2<=width<=20 and 2<=height<=20):continue
            top=ss.max();near=aa[ss>=top-1]
            a=float(np.median(A[near]));s=float(S[top]+STEP/2)
            # 상연이 실제 분할 마스크에 접하는지 확인한다.
            ai=int(np.argmin(abs(A-a)))
            cap=bone[max(ai-2,0):ai+3,top+1:min(top+4,len(S))]
            if not cap.size or np.mean(cap)<.3:continue
            candidates.append(dict(index=i,local=np.array([L[i],a,s]),area_mm2=area,
                                   cap_mask_fraction=float(np.mean(cap))))
    stable=[]
    for c in candidates:
        neigh={x['index'] for x in candidates if abs(x['index']-c['index'])<=2
               and np.linalg.norm((x['local']-c['local'])[1:])<=2.}
        if len(neigh)>=3:stable.append(c)
    return stable,len(candidates)


def or_sections(mask,axes,center_local):
    """MAX 상연에서 좌우 방향의 국소 최저점 추출. 임의의 최하단 뼈는 사용하지 않음."""
    L,A,S=axes;candidates=[]
    for j in range(len(A)):
        if abs(A[j]-center_local[1])>6:continue
        has=mask[:,j,:].any(1)
        top=np.where(has,len(S)-1-np.argmax(mask[:,j,::-1],axis=1),0)
        heights=np.where(has,S[top],np.inf)
        minima=minimum_filter1d(heights,size=13,mode='constant',cval=np.inf)
        for i in np.flatnonzero(has & (heights<=minima+STEP/4)):
            if i<3 or i>=len(L)-3 or top[i]<2 or top[i]>=len(S)-2:continue
            if abs(L[i]-center_local[0])>8 or abs(heights[i]-center_local[2])>8:continue
            if has[max(0,i-3):i+4].sum()<5:continue
            candidates.append(dict(index=j,local=np.array([L[i],A[j],heights[i]+STEP/2])))
    stable=[]
    for c in candidates:
        neigh={x['index'] for x in candidates if abs(x['index']-c['index'])<=2
               and np.linalg.norm((x['local']-c['local'])[[0,2]])<=2.}
        if len(neigh)>=3:stable.append(c)
    return stable,len(candidates)


def detect(mask,dirs,org,frame,center,key):
    o,R=frame
    m,axes,inside=grid(mask,dirs,org,o,R,center)
    local=R@(center-o)
    stable,raw=(po_sections(m,axes,key[0]) if key.endswith('Po') else or_sections(m,axes,local))
    result=dict(roi_mask_volume_mm3=float(m.sum()*STEP**3),roi_inside_fraction=float(inside.mean()),
                raw_candidates=raw,stable_candidates=len(stable),
                transferred_seed_lps= center,roi_center_local=local,
                candidate=None,status='no_anatomical_boundary_candidate')
    if inside.mean()<.98:
        result['status']='roi_truncated';return result,m,axes
    if not m.any():
        result['status']='empty_segmentation_roi';return result,m,axes
    if not stable:return result,m,axes
    if key.endswith('Po'):
        # 같은 쪽 경계만 사용하며 반대편 비교나 대칭 강제를 하지 않는다.
        eligible=[x for x in stable if np.linalg.norm(x['local']-local)<=12.]
        eligible.sort(key=lambda c:(-abs(c['local'][0]),np.linalg.norm((c['local']-local)[1:])))
    else:
        eligible=[x for x in stable if np.linalg.norm(x['local']-local)<=10.]
        eligible.sort(key=lambda c:np.linalg.norm(c['local']-local))
    if not eligible:
        result['status']='candidate_outside_transferred_patch';return result,m,axes
    selected=eligible[0]
    result.update(candidate=o+selected['local']@R,candidate_local=selected['local'],
                  seed_to_candidate_mm=float(np.linalg.norm(selected['local']-local)),
                  status='segmentation_candidate_requires_review')
    return result,m,axes


def point_mask_distance(mask,dirs,org,point,radius=8.):
    index=np.linalg.solve(dirs.T,point-org)
    r=np.ceil(radius/np.linalg.norm(dirs,axis=1)).astype(int)
    lo=np.maximum(np.floor(index).astype(int)-r,0)
    hi=np.minimum(np.ceil(index).astype(int)+r+1,np.array(mask.shape[::-1]))
    if np.any(hi<=lo):return None
    z,y,x=np.where(mask[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]>0)
    if not len(x):return None
    world=np.c_[x+lo[0],y+lo[1],z+lo[2]]@dirs+org
    d=float(np.linalg.norm(world-point,axis=1).min())
    return d if d<=radius else None


def process(p,models):
    points,lm_files=landmarks(int(p[2:]))
    central={k:points[k] for k in CENTRAL}
    o,R,residual=midline_frame(central)
    result=dict(Patient=p,frame_residual_mm=residual,points={},
                fingerprint=dict(code=digest(Path(__file__)),
                                 landmarks={f.name:digest(f) for f in lm_files},
                                 image=digest(CB/f'nrrd/{p}.nrrd'),
                                 segmentation={s:digest(CB/f'seg/{p}_seg_{s}.nrrd') for s in ('CB','MAX')}))
    figdata=[]
    for struct in ('CB','MAX'):
        mask,dirs,org=load_mask(p,struct)
        z=models[struct];patient=physical_correspondences(z,p)
        template_id=int(z['template_id']);template=physical_correspondences(z,f'P_{template_id}')
        template_points=landmarks(template_id)[0]
        qc=pd.read_csv(ROOT/f'Result_py/shape/{struct}/qc.csv').set_index('Patient').loc[p]
        for key,s in TARGETS.items():
            if s!=struct:continue
            center,transfer=local_transfer(template,patient,template_points[key])
            detected,m,axes=detect(mask,dirs,org,(o,R),center,key)
            # 기존 Po/Or는 후보 검출 완료 후 감사에만 사용한다.
            detected.update(struct=struct,template_id=template_id,transfer=transfer,
                            registration_fit_mean_mm=float(qc.fit_mean_mm),registration_fit_flag=bool(qc.fit_flag),
                            original_point=points[key],
                            original_to_mask_mm=point_mask_distance(mask,dirs,org,points[key]),
                            seed_to_original_mm=float(np.linalg.norm(center-points[key])),
                            candidate_to_original_mm=(float(np.linalg.norm(detected['candidate']-points[key]))
                                                      if detected['candidate'] is not None else None))
            if qc.fit_flag and detected['candidate'] is not None:
                detected['status']='registration_flag_candidate_requires_review'
            result['points'][key]=detected
            figdata.append((key,m,axes))
        del mask
    save(DEST/'cases'/f'{p}.json',result)
    print(p,', '.join(k+':'+v['status'].split('_')[0] for k,v in result['points'].items()),flush=True)
    return json_safe(result),figdata,(o,R)


def figure(p,result,data,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    volume,dirs,org=read_nrrd(CB/f'nrrd/{p}.nrrd');o,R=frame
    fig,axs=plt.subplots(4,3,figsize=(16,19))
    for row,(key,mask,axes) in enumerate(data):
        rec=result['points'][key];L,A,S=axes
        center=np.array(rec['candidate'] if rec['candidate'] is not None else rec['transferred_seed_lps'])
        h=R@(center-o)
        for col,(plane,axis) in enumerate((('시상',0),('관상',1),('축상',2))):
            j=int(np.argmin(abs(axes[axis]-h[axis])))
            if axis==0:
                u,w=np.meshgrid(A,S,indexing='ij');local=np.stack([np.full_like(u,L[j]),u,w],-1);X,Y=A,S;ix,iy=1,2
            elif axis==1:
                u,w=np.meshgrid(L,S,indexing='ij');local=np.stack([u,np.full_like(u,A[j]),w],-1);X,Y=L,S;ix,iy=0,2
            else:
                u,w=np.meshgrid(L,A,indexing='ij');local=np.stack([u,w,np.full_like(u,S[j])],-1);X,Y=L,A;ix,iy=0,1
            ct=sample(volume,dirs,org,o+local@R,order=1,cval=-1000)
            extent=[X[0]-STEP/2,X[-1]+STEP/2,Y[0]-STEP/2,Y[-1]+STEP/2]
            ax=axs[row,col];ax.imshow(ct.T,origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=extent)
            section=np.take(mask,j,axis=axis)
            if section.any():ax.contour(X,Y,section.T.astype(float),[.5],colors=['#00c8ff'],linewidths=.7)
            marks=[('ALI',rec['original_point'],'#ff4040','+'),('ROI seed',rec['transferred_seed_lps'],'#ffd400','x')]
            if rec['candidate'] is not None:marks.append(('surface candidate',rec['candidate'],'#00e676','x'))
            for name,point,color,marker in marks:
                q=R@(np.array(point)-o);off=q[axis]-axes[axis][j]
                ax.plot(q[ix],q[iy],marker,color=color,ms=9,mew=1.7,label=f'{name}: off-plane {off:+.1f} mm')
            ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.legend(fontsize=7,loc='upper right')
            ax.set_title(f'{key} / {rec["struct"]} / 실제 {plane} 단면',fontsize=11)
            ax.set_xlabel('환자 정중 좌표계 mm');ax.set_ylabel('mm')
    fig.suptitle(f'{p}: 분할로 자른 국소 영역과 점 후보 — 하늘색=실제 분할 경계, 빨강=기존 ALI, 초록=새 후보',fontsize=13)
    fig.text(.5,.012,'노랑 ROI seed는 표면 대응으로 옮긴 탐색 중심. 반대편 반사·엑셀·현재 Po/Or로 후보를 선택하지 않음. 전문가 검수 전.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.025,1,.96));fig.savefig(DEST/'figs'/f'{p}.png',dpi=110);plt.close(fig)


def evaluate(cases):
    rows=[]
    for c in cases:
        for key,r in c['points'].items():
            rows.append(dict(Patient=c['Patient'],point=key,struct=r['struct'],status=r['status'],
                             template_id=r['template_id'],roi_volume_mm3=r['roi_mask_volume_mm3'],
                             roi_inside_fraction=r['roi_inside_fraction'],raw_candidates=r['raw_candidates'],
                             stable_candidates=r['stable_candidates'],original_to_mask_mm=r['original_to_mask_mm'],
                             seed_to_original_mm=r['seed_to_original_mm'],candidate_to_original_mm=r['candidate_to_original_mm'],
                             template_surface_gap_mm=r['transfer']['template_surface_gap_mm'],
                             transfer_residual_mm=r['transfer']['local_transfer_residual_mm'],
                             registration_fit_flag=r['registration_fit_flag'],
                             candidate_found=r['candidate'] is not None))
    audit=pd.DataFrame(rows);audit.to_csv(DEST/'point_audit.csv',index=False,encoding='utf-8-sig')
    # 모든 85명 추출을 마친 후에만 수기값을 불러서 비교한다.
    reference=pd.read_csv(OUT/'restore_ext/v1/compare_all.csv').set_index(['Patient','metric']).excel
    current=pd.read_csv(OUT/'develop85/v1/error_ranking.csv').set_index(['Patient','metric']).automatic
    comparisons=[]
    for c in cases:
        p=c['Patient'];P=landmarks(int(p[2:]))[0]
        for mode in ('Po_only','Or_only','combined'):
            required=[k for k in TARGETS if mode=='combined' or (k.endswith('Po')==(mode=='Po_only'))]
            available=all(c['points'][k]['candidate'] is not None and not c['points'][k]['registration_fit_flag'] for k in required)
            if not available:continue
            Q=dict(P)
            for k in required:Q[k]=np.array(c['points'][k]['candidate'])
            for m,key,fh in [('ANB','ANB','jung'),('Nperp-A','Nperp_A','yao4'),('Nperp-Pog','Nperp_Pog','yao4'),('MP-FH','MPFH_GoMe','yao4')]:
                y=reference.get((p,m),np.nan);raw=measures(P,fh)[key];new=measures(Q,fh)[key]
                comparisons.append(dict(Patient=p,mode=mode,metric=m,raw=raw,candidate=new,current=current.get((p,m)),
                                        excel=y,raw_error=abs(raw-y),candidate_error=abs(new-y),
                                        current_error=abs(current.get((p,m))-y)))
    comp=pd.DataFrame(comparisons)
    comp.to_csv(DEST/'candidate_measurements.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    if len(comp):
        for (mode,m),g in comp.groupby(['mode','metric']):
            summaries.append(dict(mode=mode,metric=m,total_n=85,paired_n=len(g),
                                  same_patient_raw_MAE=g.raw_error.mean(),candidate_MAE=g.candidate_error.mean(),
                                  current_MAE=g.current_error.mean(),raw_p90=g.raw_error.quantile(.9),candidate_p90=g.candidate_error.quantile(.9),
                                  improved_vs_raw=int((g.candidate_error<g.raw_error).sum()),
                                  worsened_vs_raw=int((g.candidate_error>g.raw_error).sum()),
                                  raw_within2_to_over2=int(((g.raw_error<=2)&(g.candidate_error>2)).sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(DEST/'evaluation.csv',index=False,encoding='utf-8-sig')
    counts=audit.groupby(['point','status']).size().rename('n').reset_index()
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    cards=[]
    for c in cases:
        p=c['Patient'];path=DEST/'figs'/f'{p}.png'
        if not path.exists():continue
        cards.append(f'<article id="{p}"><h3>{p}</h3>{table(audit[audit.Patient==p])}<a href="figs/{p}.png"><img src="figs/{p}.png" alt="{p} 분할 국소 단면"></a><p><a href="cases/{p}.json">추출 근거 JSON</a></p></article>')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>분할 기반 Po·Or 후보: 85명</title>
<style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}table{{border-collapse:collapse;font-size:12px;background:white}}td,th{{border:1px solid #ddd;padding:5px}}.scroll{{overflow:auto}}img{{max-width:100%}}article{{margin:24px 0;background:white;padding:16px;overflow:auto}}h2{{margin-top:40px}}</style><main>
<h1>AMASSS 분할에서 국소 영역을 자른 뒤 Po·Or 후보 찾기</h1>
<p>85명 전체, 좌우 독립 적용. 반사 대체·좌우 6 mm 제외 규칙·현재 환자의 Po/Or·엑셀을 후보 선택에 사용하지 않았습니다. 기존 측정값·보고서는 변경하지 않았습니다.</p>
<h2>1. 구현한 흐름</h2><p>기존 CB/MAX 표면 대응점 → 고정 템플릿의 해당 부위 전달 → 현재 환자의 실제 분할을 24×30×30 mm 영역으로 자름 → CB 구멍의 상연 / MAX 상연의 국소 저점 후보 → 같은 위치의 원본 단면으로 대조.</p>
<p>PCA 벡터나 평균 모양을 복원한 것이 아닙니다. 저장된 원래 표면 대응점을 환자 LPS 좌표로 역변환했습니다. 템플릿 기준점은 기존 ALI 점을 사용하므로 전문가 정답은 아닙니다. ROI 전달 오차·마스크 경계 누락·다른 구멍 선택·안와저와 안와 테두리 혼동이 남을 수 있습니다. 후보를 실제 Po/Or로 확정하거나 자동 배포하지 않았습니다.</p>
<h2>2. 340개 점 추출 현황</h2>{table(counts)}
<h2>3. 계산 영향: 같은 환자끼리 비교</h2><p>후보가 양쪽 모두 있고 기존 정합 실패 표시에 걸리지 않은 환자만 비교했습니다. paired_n을 반드시 확인해야 합니다. Nperp-Pog는 FH 변화 효과만 분리하려고 후보 측에 Pog 윤곽 보정을 넣지 않았습니다. raw는 대칭 보정 전, current는 이전 선택 방식이므로 차이가 있습니다. 검토한 85명에서의 탐색 비교이며 독립 검증이 아닙니다.</p><div class="scroll">{table(summary)}</div>
<h2>4. 전체 점의 분할 포함·거리·후보 기록</h2><p>original_to_mask_mm는 기존 ALI 점에서 해당 마스크까지의 거리이며, 잘못된 ALI 점이 멀다는 사실만으로 분할 누락이라고 판정하지 않습니다. 국소 마스크가 있어도 외이도·안와 테두리가 보존됐다는 보장은 없습니다.</p><div class="scroll">{table(audit)}</div>
<h2>5. 사례별 원본 단면과 분할 경계</h2>{''.join(cards)}
<h2>자료·고정 규칙</h2><p><a href="point_audit.csv">340개 점 감사</a> · <a href="candidate_measurements.csv">후보 측정값</a> · <a href="evaluation.csv">같은 환자 비교</a> · <a href="protocol.json">입력·규칙</a></p><pre>{escape(json.dumps(RULE,ensure_ascii=False,indent=2))}</pre></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')
    print(counts.to_string(index=False));print(summary.to_string(index=False))


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--patients',nargs='*',type=int);args=ap.parse_args()
    for path in (DEST,DEST/'cases',DEST/'figs'):path.mkdir(parents=True,exist_ok=True)
    models={s:np.load(ROOT/f'Result_py/shape/{s}/points.npz') for s in ('CB','MAX')}
    patients=sorted(set(models['CB']['ids'])&set(models['MAX']['ids']))
    if args.patients:patients=[p for p in patients if p in args.patients]
    save(DEST/'protocol.json',dict(rule=RULE,patients=[f'P_{p}' for p in patients],
                                  code_sha256=digest(Path(__file__)),
                                  correspondence_hashes={s:digest(ROOT/f'Result_py/shape/{s}/points.npz') for s in models},
                                  template_ids={s:int(z['template_id']) for s,z in models.items()}))
    cases=[]
    figures={1,2,5,9,11,16,20,23,27,34,35,37,41,81}
    for pid in patients:
        p=f'P_{pid}'
        result,data,frame=process(p,models);cases.append(result)
        if pid in figures:figure(p,result,data,frame)
    evaluate(cases)
    print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
