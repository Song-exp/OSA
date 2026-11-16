"""남은 기준점의 독립 윤곽 후보 엔진 (2단계). A/Pog v2 추적기를 방향 일반화해 재사용한다.

현재 대상점·엑셀·대칭으로 ROI나 후보를 고르지 않는다. 기존점은 후보 생성 뒤 교체 필요 판단에만 쓴다.
"""
import numpy as np
from skimage.filters import threshold_multiotsu
from scipy.ndimage import label
from cbct_apog_trace import trace_front,contour_candidates

# 방향 d는 정중 좌표계(앞 A, 위 S) 평면의 단위벡터. 'min'은 오목점, 'max'는 그 방향 최외곽점.
SPECS={
    'N':  dict(structs=('CB','MAX'),model='CB',  d=(1,0),  mode='min',window=(10,14),definition='전두비골 전방 윤곽의 가장 깊은 오목점'),
    'ANS':dict(structs=('MAX',),     model='MAX', d=(1,0),  mode='max',window=(6,9),  definition='상악 전방 돌출(전비극) 최전방점'),
    'PNS':dict(structs=('MAX',),     model='MAX', d=(-1,0), mode='max',window=(6,9),  definition='경구개 후방 최후방점'),
    'B':  dict(structs=('MAND',),    model='MAND',d=(1,0),  mode='min',window=(8,12), definition='하악 전방 윤곽 Pog 위 가장 깊은 오목점'),
    'Pog':dict(structs=('MAND',),    model='MAND',d=(1,0),  mode='max',method='support',window=(10,14),definition='턱끝 결합부 최전방점(v2/v3 연속 추적 실패를 지지점 방식으로 재평가)'),
    'Me': dict(structs=('MAND',),    model='MAND',d=(0,-1), mode='max',method='support',window=(8,12), definition='하악 결합부 최하방점'),
    'Gn': dict(structs=('MAND',),    model='MAND',d=(.7071,-.7071),mode='max',method='support',window=(8,12),definition='결합부 전하방 45° 최외곽점'),
    'RGo':dict(structs=('MAND',),    model='MAND',d=(-.7071,-.7071),mode='max',method='support',window=(10,14),definition='하악각 후하방 45° 최외곽점'),
    'LGo':dict(structs=('MAND',),    model='MAND',d=(-.7071,-.7071),mode='max',method='support',window=(10,14),definition='하악각 후하방 45° 최외곽점'),
}
RULE=dict(version='landmark-contour-audit-v1',reference_patients=[5,11,28],exclude_self_reference=True,
          axis_angles_deg=[-5.,0.,5.],windows=['base','expanded'],profile_half_mm=17.,step_mm=.5,slab_mm=1.,
          families='AMASSS mask + native CT 3-class Otsu upper threshold -100/0/+100 (air/soft/bone)',
          displacement_policy='대표 후보-기존점 변위의 85명 중앙값(대표 후보 10명 이상일 때)을 정의 관습 변위로 보고 판정 전 빼며, 교체 후보도 같은 변위만큼 옮겨 ALI 관습으로 맞춤',
          offset_policy='기존점 편차에서 85명 기준점·원천별 중앙값(ALI 표기 관습 오프셋)을 뺀 뒤 판정; 엑셀 미사용',
          trigger=dict(both_sources_off_mm=1.,support_mm=1.,stable_disagreement_mm=3.,roi_u_mm=15.,roi_v_extra_mm=0.),
          specs={k:{x:v for x,v in s.items()} for k,s in SPECS.items()},
          lateral_center='N·ANS·PNS: 두개 정중면 L=0 / B·Pog·Me·Gn·Go: 하악 SSM 대응 시드 L(하악 편위 반영)',
          note='공학적 윤곽 정의. N은 봉합선, Go는 접선 이등분 정의의 대리 조건이며 해부학적 확정이 아님',
          support_method='가장 큰 연결 구조의 열별 최외곽 u에서 최댓값; 창 끝 1 mm 제외; 양쪽 4 mm 안에서 0.5 mm 이상 낮아지거나 구조가 끊겨야 함',
          symmetry_used=False,Excel_used_for_selection=False)

VARIANTS=[]
for angle in RULE['axis_angles_deg']:
    for window in RULE['windows']:
        for family,delta in [('AMASSS',None),('native_CT',-100.),('native_CT',0.),('native_CT',100.)]:
            name=f'{family}_{angle:+g}_{window}_{delta}'
            if angle==0 and window=='base':name='contour_base' if family=='AMASSS' else f'native_otsu_{delta:+g}'
            VARIANTS.append(dict(name=name,family=family,angle_deg=angle,window=window,otsu_delta=delta))


def basis(d,angle_deg):
    t=np.radians(angle_deg);d=np.asarray(d,float)
    eu=np.array([np.cos(t)*d[0]-np.sin(t)*d[1],np.sin(t)*d[0]+np.cos(t)*d[1]]);eu/=np.linalg.norm(eu)
    return eu,np.array([-eu[1],eu[0]])


def center_L(seed,key):
    """두개·상악 정중점(N·ANS·PNS)은 두개 정중면 L=0. 하악점은 하악 비대칭(턱 편위)을 따르도록 하악 표면 대응 시드의 L."""
    return float(seed[0]) if SPECS[key]['model']=='MAND' else 0.


def grid(seed,key,variant):
    """정중 좌표계 국소 좌표의 표본 격자 (L, u, v, 3)."""
    s=RULE['step_mm'];spec=SPECS[key];eu,ev=basis(spec['d'],variant['angle_deg'])
    half=spec['window'][variant['window']=='expanded']
    Lc=center_L(seed,key)
    L=np.arange(Lc-RULE['slab_mm'],Lc+RULE['slab_mm']+s/2,s)
    u=np.arange(-RULE['profile_half_mm'],RULE['profile_half_mm']+s/2,s);v=np.arange(-half,half+s/2,s)
    G=np.zeros((len(L),len(u),len(v),3))
    G[...,0]=L[:,None,None]
    G[...,1]=seed[1]+u[None,:,None]*eu[0]+v[None,None,:]*ev[0]
    G[...,2]=seed[2]+u[None,:,None]*eu[1]+v[None,None,:]*ev[1]
    return G,L,u,v,eu,ev


def detect(sampler,seed,key,variant):
    """sampler(G, kind) -> 격자 값. kind='mask'|'ct'."""
    G,L,u,v,eu,ev=grid(seed,key,variant)
    out=dict(variant=variant['name'],family=variant['family'],angle_deg=variant['angle_deg'],window=variant['window'],
             winner=None,eligible=[],threshold=None,profile={},basis=[eu.tolist(),ev.tolist()])
    ct=sampler(G,'ct')
    if variant['family']=='AMASSS':m=sampler(G,'mask')>0
    else:
        vals=ct[np.isfinite(ct)]
        if not len(vals) or np.ptp(vals)==0:return dict(out,status='no_image_contrast')
        lo,hi=np.percentile(vals,[1,99]);out['threshold']=float(threshold_multiotsu(np.clip(vals,lo,hi),classes=3)[-1])+variant['otsu_delta']
        m=ct>out['threshold']
    m&=np.isfinite(ct)
    if SPECS[key].get('method')=='support':
        front,cands,status=support_candidates(m.any(0),u,v)
        out['profile']=dict(v=v.tolist(),u=front.tolist(),trace_status='support')
        return finish(out,cands,status,m,L,u,seed,eu,ev)
    front,trace=trace_front(m.any(0),u,v)
    out['profile']=dict(v=v.tolist(),u=front.tolist(),trace_status=trace)
    if trace!='tracked':return dict(out,status=trace)
    cands,status=contour_candidates(front,v,np.zeros(3),'A' if SPECS[key]['mode']=='min' else 'max',0.)
    return finish(out,cands,status,m,L,u,seed,eu,ev)


def support_candidates(binary,u,v):
    lab,n=label(binary)
    if not n:return np.full(len(v),np.nan),[],'no_connected_structure'
    sizes=np.bincount(lab.ravel());sizes[0]=0;m=lab==sizes.argmax()
    has=m.any(0);front=np.where(has,u[::-1][np.argmax(m[::-1],axis=0)],np.nan)
    if not has.any():return front,[],'no_connected_structure'
    best=np.nanmax(front);ties=np.flatnonzero(abs(front-best)<=.25);i=int(ties[len(ties)//2])
    if min(v[i]-v[0],v[-1]-v[i])<1.:return front,[],'extremum_at_search_boundary'
    for side in (v<v[i]-.01,v>v[i]+.01):
        near=side&(abs(v-v[i])<=4.)
        f=front[near]
        if not (np.isnan(f).any() or best-np.nanmin(f)>=.5):return front,[],'no_supported_interior_extremum'
    return front,[dict(index=i,front=float(front[i]),sup=float(v[i]),prominence=float(best-np.nanmax(front[abs(v-v[i])>=3.5]) if (abs(v-v[i])>=3.5).any() and np.isfinite(front[abs(v-v[i])>=3.5]).any() else 4.))],'candidate'


def finish(out,cands,status,m,L,u,seed,eu,ev):
    eligible=[]
    for c in cands:
        iu=int(np.argmin(abs(u-c['front'])));j=c['index'];lat=L[m[:,iu,j]]
        if not len(lat):continue
        p=np.array([np.median(lat),*(np.asarray(seed[1:])+c['front']*eu+c['sup']*ev)])
        eligible.append(dict(point=p,score=c['prominence'],prominence_mm=c['prominence']))
    eligible.sort(key=lambda c:(-c['score'],np.linalg.norm(c['point']-seed)))
    return dict(out,status=status,eligible=eligible,winner=eligible[0]['point'] if eligible else None)


def deviation(run,seed,original):
    """기존점의 윤곽 법선 방향 편차(u_original - u_contour). 윤곽 밖이면 None."""
    p=run.get('profile',{});v=np.asarray(p.get('v',[]),float);u=np.asarray(p.get('u',[]),float)
    if not len(u):return None
    ok=np.isfinite(u);eu,ev=(np.asarray(x) for x in run['basis'])
    d=np.asarray(original,float)[1:]-np.asarray(seed,float)[1:];u0,v0=d@eu,d@ev
    if ok.sum()<2 or not v[ok].min()<=v0<=v[ok].max():return None
    return float(u0-np.interp(v0,v[ok],u[ok]))


def raw_deviation(original,seed,results):
    base={r['variant']:r for r in results}
    return {'mask':deviation(base['contour_base'],seed,original),'native':deviation(base['native_otsu_+0'],seed,original)}


def decide(original,seed,results,stab,key,offset=None):
    t=RULE['trigger'];base={r['variant']:r for r in results};offset=offset or {}
    raw=raw_deviation(original,seed,results)
    dev={k:(None if v is None else v-offset.get(k,0.)) for k,v in raw.items()}
    eu,ev=(np.asarray(x) for x in base['contour_base']['basis'])
    d=np.asarray(original,float)[1:]-np.asarray(seed,float)[1:]
    outside=bool(abs(d@eu)>t['roi_u_mm'] or abs(d@ev)>SPECS[key]['window'][1] or abs(original[0]-center_L(seed,key))>3)
    a,b=dev['mask'],dev['native']
    both_off=a is not None and b is not None and abs(a)>t['both_sources_off_mm'] and abs(b)>t['both_sources_off_mm'] and np.sign(a)==np.sign(b)
    supported=any(x is not None and abs(x)<=t['support_mm'] for x in (a,b)) and not both_off
    rep=stab['representative'];disp=np.asarray(offset.get('displacement',np.zeros(3)),float)
    conflict=rep is not None and np.linalg.norm(np.asarray(rep)-disp-original)>t['stable_disagreement_mm']
    reasons=[r for r,f in (('original_outside_independent_roi',outside),('both_sources_disagree_with_original_boundary',both_off),
                           ('stable_definition_candidate_disagrees_with_original',conflict)) if f]
    triggered=bool(reasons)
    route='existing_point_supported' if supported else 'existing_point_retained_not_confirmed'
    if triggered:
        route=('single_reconstruction_candidate' if stab['status']=='stable_single_candidate' else
               'candidate_region' if stab['adequate_support'] else 'automatic_withhold')
    return dict(route=route,triggered=triggered,reasons=reasons,original_boundary_deviation=dev,raw_deviation=raw,
                independent_support=bool(supported and not triggered))
