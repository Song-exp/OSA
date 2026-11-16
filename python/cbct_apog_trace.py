"""연속 경계 추적 및 창/축 요인 분리. 현재 A/Pog·Me·엑셀 입력 없음."""
import numpy as np
from scipy.ndimage import label,map_coordinates,gaussian_filter1d,median_filter
from skimage.filters import threshold_otsu
from cbct_apog_ensemble import decide

RULE=dict(version='continuous-front-cortex-window-axis-v2',reference_patients=[5,11,28],
          exclude_self_reference=True,symmetry_used=False,Excel_used_for_selection=False,
          trace='largest connected sagittal component, dynamic programming over anterior run ends',
          continuity_cost=.25,front_cost=.25,maximum_jump_mm=3.,maximum_gap_mm=1.,
          minimum_profile_coverage=.8,endpoint_guard_mm=1.,minimum_prominence_mm=.5,
          prominence_support_mm=4.,smooth_mm=.75,
          axis_angles_deg=[-5.,0.,5.],window_modes=['base','expanded'],
          A_base='ANS -15..-2 intersect independent seed +/-8 mm',
          A_expanded='ANS -18..-1 intersect independent seed +/-12 mm',
          Pog_base='independent seed +/-10 mm',Pog_expanded='independent seed +/-14 mm',
          sampling_step_mm=.5,slab_mm=1.,
          note='engineering sensitivity experiment, not calibrated HU or anatomical confirmation')

VARIANTS=[]
for angle in RULE['axis_angles_deg']:
    for window in RULE['window_modes']:
        for family,delta in [('AMASSS',None),('native_CT',-100.),('native_CT',0.),('native_CT',100.)]:
            name=f'trace_{family}_{angle:+g}_{window}_{delta}'
            if angle==0 and window=='base':
                name='contour_base' if family=='AMASSS' else f'native_otsu_{delta:+g}'
            VARIANTS.append(dict(name=name,family=family,angle_deg=angle,window=window,otsu_delta=delta,step=.5,slab=1.))


def trace_front(binary,ant,sup):
    """각 높이의 전방 경계 후보를 연속 경로로 선택한다."""
    # Keep only the largest connected structure, preventing detached teeth/metal islands.
    lab,n=label(binary)
    if not n:return np.full(len(sup),np.nan),'no_connected_structure'
    sizes=np.bincount(lab.ravel());sizes[0]=0;m=lab==sizes.argmax()
    nodes=[];rows=[]
    for j in range(len(sup)):
        ends=np.flatnonzero(m[:,j]&~np.r_[m[1:,j],False])
        ends=ends[(ends>0)&(ends<len(ant)-1)]
        if len(ends):nodes.append(ant[ends]);rows.append(j)
    if len(rows)<7:return np.full(len(sup),np.nan),'insufficient_contour_coverage'
    costs=[RULE['front_cost']*(nodes[0].max()-nodes[0])];parents=[]
    for t in range(1,len(nodes)):
        gap=sup[rows[t]]-sup[rows[t-1]]
        if gap>RULE['maximum_gap_mm']+.5:
            return np.full(len(sup),np.nan),'disconnected_contour'
        delta=nodes[t][:,None]-nodes[t-1][None,:]
        penalty=RULE['continuity_cost']*delta**2/max(gap,.5)
        penalty[abs(delta)>RULE['maximum_jump_mm']*max(gap/.5,1)]=1e6
        transitions=costs[-1][None,:]+penalty
        parent=transitions.argmin(1);parents.append(parent)
        costs.append(transitions[np.arange(len(parent)),parent]+RULE['front_cost']*(nodes[t].max()-nodes[t]))
    i=int(costs[-1].argmin());front=np.full(len(sup),np.nan)
    for t in range(len(nodes)-1,-1,-1):
        front[rows[t]]=nodes[t][i]
        if t:i=int(parents[t-1][i])
    if np.any(abs(np.diff(front[rows]))>RULE['maximum_jump_mm']*np.maximum(np.diff(sup[rows])/.5,1)):
        return front,'disconnected_contour'
    # Isolated protrusions are not continuous cortical evidence.
    finite=np.isfinite(front)
    filled=np.interp(np.arange(len(front)),np.flatnonzero(finite),front[finite])
    filtered=median_filter(filled,size=5,mode='nearest')
    front[finite&(abs(filled-filtered)>2.)]=np.nan
    if np.isfinite(front).mean()<RULE['minimum_profile_coverage']:
        return front,'insufficient_contour_coverage'
    return front,'tracked'


def contour_candidates(front,sup,seed,key,angle_deg):
    ok=np.isfinite(front)
    if len(front)<7 or ok.mean()<RULE['minimum_profile_coverage']:
        return [],'insufficient_contour_coverage'
    filled=np.interp(sup,sup[ok],front[ok]);step=float(np.median(np.diff(sup)))
    curve=gaussian_filter1d(filled,RULE['smooth_mm']/step,mode='nearest')
    a=curve-seed[1];s=sup-seed[2];theta=np.deg2rad(angle_deg)
    qa=np.cos(theta)*a+np.sin(theta)*s
    qs=-np.sin(theta)*a+np.cos(theta)*s
    if np.any(np.diff(qs)<=0):return [],'axis_folds_contour'
    extreme=(qa[ok].min() if key=='A' else qa[ok].max())
    indices=np.flatnonzero(ok&(abs(qa-extreme)<=.25))
    indices=[i for i in indices if min(qs[i]-qs[0],qs[-1]-qs[i])>=RULE['endpoint_guard_mm']]
    if not indices:return [],'extremum_at_search_boundary'
    candidates=[]
    for i in indices:
        left=qa[(qs<qs[i])&(qs>=qs[i]-RULE['prominence_support_mm'])]
        right=qa[(qs>qs[i])&(qs<=qs[i]+RULE['prominence_support_mm'])]
        if not len(left) or not len(right):continue
        prominence=min(left.max(),right.max())-qa[i] if key=='A' else qa[i]-max(left.min(),right.min())
        if prominence>=RULE['minimum_prominence_mm']:
            candidates.append(dict(index=int(i),front=float(front[i]),sup=float(sup[i]),prominence=float(prominence)))
    return candidates,'candidate' if candidates else 'no_supported_interior_extremum'


def detect_variant(mask,ct,base_axes,seed,ans_sup,key,variant):
    step=variant['step'];L=np.arange(-variant['slab'],variant['slab']+step/2,step)
    ant=np.arange(seed[1]-17,seed[1]+17+step/2,step)
    expanded=variant['window']=='expanded'
    if key=='A':
        half=12 if expanded else 8
        lo,hi=max(ans_sup-(18 if expanded else 15),seed[2]-half),min(ans_sup-(1 if expanded else 2),seed[2]+half)
    else:
        half=14 if expanded else 10;lo,hi=seed[2]-half,seed[2]+half
    sup=np.arange(lo,hi+step/2,step)
    empty=dict(variant=variant['name'],family=variant['family'],winner=None,eligible=[],threshold=None,profile={},
               angle_deg=variant['angle_deg'],window=variant['window'])
    if len(sup)<7:return dict(empty,status='empty_search_window')
    G=np.stack(np.meshgrid(L,ant,sup,indexing='ij'),0)
    coords=np.stack([(G[i]-base_axes[i][0])/(base_axes[i][1]-base_axes[i][0]) for i in range(3)],0)
    image=map_coordinates(ct,coords,order=1,cval=np.nan,prefilter=False)
    if variant['family']=='AMASSS':m=map_coordinates(mask.astype(np.uint8),coords,order=0,cval=0,prefilter=False)>0
    else:
        vals=image[np.isfinite(image)]
        if not len(vals) or np.ptp(vals)==0:return dict(empty,status='no_image_contrast')
        low,high=np.percentile(vals,[1,99]);empty['threshold']=float(threshold_otsu(np.clip(vals,low,high)))+variant['otsu_delta']
        m=image>empty['threshold']
    m&=np.isfinite(image)
    front,trace_status=trace_front(m.any(0),ant,sup)
    candidates,status=contour_candidates(front,sup,seed,key,variant['angle_deg']) if trace_status=='tracked' else ([],trace_status)
    eligible=[]
    for c in candidates:
        j=c['index'];aindex=int(np.argmin(abs(ant-c['front'])));lateral=L[m[:,aindex,j]]
        if not len(lateral):continue
        eligible.append(dict(point=np.array([np.median(lateral),c['front'],c['sup']]),score=c['prominence'],prominence_mm=c['prominence']))
    eligible.sort(key=lambda c:(-c['score'],np.linalg.norm(c['point']-seed)))
    return dict(empty,winner=eligible[0]['point'] if eligible else None,eligible=eligible,status=status,
                profile=dict(sup=sup,front=front,coverage=float(np.isfinite(front).mean()),trace_status=trace_status))
