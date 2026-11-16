"""A/Pog 독립 윤곽 후보. 현재 A/Pog 좌표와 Me는 후보 추출 입력이 아니다."""
import numpy as np
from scipy.ndimage import gaussian_filter1d,map_coordinates
from skimage.filters import threshold_otsu

VARIANTS=[dict(name='contour_base',family='AMASSS',step=.5,offset=[0.,0.,0.],slab=1.,smooth=.5)]
for axis,delta in ((1,1.5),(2,1.5),(0,.75)):
    for sign in (-1,1):
        offset=[0.,0.,0.];offset[axis]=sign*delta
        VARIANTS.append(dict(name=f'contour_shift_{axis}_{sign}',family='AMASSS',step=.5,offset=offset,slab=1.,smooth=.5))
for step in (.25,.75):
    VARIANTS.append(dict(name=f'contour_step_{step}',family='AMASSS',step=step,offset=[0.,0.,0.],slab=1.,smooth=.5))
for slab in (.5,1.5):
    VARIANTS.append(dict(name=f'contour_slab_{slab}',family='AMASSS',step=.5,offset=[0.,0.,0.],slab=slab,smooth=.5))
for sigma in (0.,1.):
    VARIANTS.append(dict(name=f'contour_smooth_{sigma}',family='AMASSS',step=.5,offset=[0.,0.,0.],slab=1.,smooth=sigma))
for delta in (-100.,0.,100.):
    VARIANTS.append(dict(name=f'native_otsu_{delta:+g}',family='native_CT',step=.5,offset=[0.,0.,0.],slab=1.,smooth=.5,otsu_delta=delta))

RULE=dict(version='independent-A-Pog-contour-v1',reference_patients=[5,11,28],exclude_self_reference=True,
          seed='median of independent reference-to-patient surface transfers; not a final coordinate',
          A_window='ANS superior -15 to -2 mm intersect transferred seed +/-8 mm',
          Pog_window='transferred chin seed +/-10 mm; no Me or current Pog window',
          minimum_profile_coverage=.8,endpoint_guard_mm=1.,minimum_prominence_mm=.5,
          original_boundary_deviation_mm=1.,definition_disagreement_mm=3.,
          symmetry_used=False,Excel_used_for_selection=False,reference_annotations='unreviewed ALI, ROI guidance only')


def extrema_candidates(front,sup,key,smooth_mm=.5,step=.5):
    ok=np.isfinite(front)
    if len(front)<7 or ok.mean()<RULE['minimum_profile_coverage']:
        return [],'insufficient_contour_coverage'
    filled=np.interp(np.arange(len(front)),np.flatnonzero(ok),front[ok])
    curve=gaussian_filter1d(filled,smooth_mm/step,mode='nearest') if smooth_mm>0 else filled
    extreme=np.min(curve[ok]) if key=='A' else np.max(curve[ok])
    indices=np.flatnonzero(ok & (abs(curve-extreme)<=.25))
    interior=[i for i in indices if min(sup[i]-sup[0],sup[-1]-sup[i])>=RULE['endpoint_guard_mm']]
    if not interior:return [],'extremum_at_search_boundary'
    output=[]
    for i in interior:
        n=max(1,int(round(2./step)))
        left=curve[max(0,i-n):i];right=curve[i+1:min(len(curve),i+n+1)]
        if not len(left) or not len(right):continue
        prominence=(min(np.max(left),np.max(right))-curve[i] if key=='A' else
                    curve[i]-max(np.min(left),np.min(right)))
        if prominence<RULE['minimum_prominence_mm']:continue
        output.append(dict(index=int(i),front=float(front[i]),sup=float(sup[i]),prominence=float(prominence)))
    return output,'candidate' if output else 'no_supported_interior_extremum'


def detect_variant(mask,ct,base_axes,seed,ans_sup,key,variant):
    step=variant['step'];offset=np.asarray(variant['offset'])
    L=np.arange(offset[0]-variant['slab'],offset[0]+variant['slab']+step/2,step)
    ant=np.arange(seed[1]-15+offset[1],seed[1]+15+offset[1]+step/2,step)
    if key=='A':lo,hi=max(ans_sup-15,seed[2]-8),min(ans_sup-2,seed[2]+8)
    else:lo,hi=seed[2]-10,seed[2]+10
    sup=np.arange(lo+offset[2],hi+offset[2]+step/2,step)
    if len(sup)<7:
        return dict(variant=variant['name'],family=variant['family'],winner=None,eligible=[],status='empty_search_window',threshold=None,profile={})
    G=np.stack(np.meshgrid(L,ant,sup,indexing='ij'),0)
    coords=np.stack([(G[i]-base_axes[i][0])/(base_axes[i][1]-base_axes[i][0]) for i in range(3)],0)
    m=map_coordinates(mask.astype(np.uint8),coords,order=0,cval=0,prefilter=False)>0
    image=map_coordinates(ct,coords,order=1,cval=np.nan,prefilter=False)
    threshold=None
    if variant['family']=='native_CT':
        values=image[np.isfinite(image)]
        if len(values) and np.ptp(values)>0:
            low,high=np.percentile(values,[1,99]);threshold=float(threshold_otsu(np.clip(values,low,high)))+variant['otsu_delta']
            m=np.isfinite(image)&(image>threshold)
        else:m=np.zeros(image.shape,bool)
    m&=np.isfinite(image)
    profile=m.any(0);has=profile.any(0)
    indices=len(ant)-1-np.argmax(profile[::-1],axis=0)
    front=np.where(has,ant[indices],np.nan)
    candidates,status=extrema_candidates(front,sup,key,variant['smooth'],step)
    eligible=[]
    for c in candidates:
        j=c['index'];aindex=indices[j]
        lateral=L[m[:,aindex,j]]
        if not len(lateral):continue
        point=np.array([np.median(lateral),c['front'],c['sup']])
        eligible.append(dict(point=point,score=c['prominence'],prominence_mm=c['prominence']))
    eligible.sort(key=lambda c:(-c['score'],np.linalg.norm(c['point']-seed)))
    return dict(variant=variant['name'],family=variant['family'],winner=eligible[0]['point'] if eligible else None,
                eligible=eligible,status=status,threshold=threshold,
                profile=dict(sup=sup,front=front,coverage=float(has.mean())))


def decide(original_local,seed,results,stability):
    """원래 점은 후보 추출 후 교체 필요 여부 판단에만 사용한다."""
    deviations={}
    for family,name in (('mask','contour_base'),('native','native_otsu_+0')):
        run=next(r for r in results if r['variant']==name)
        profile=run.get('profile',{});sup=np.asarray(profile.get('sup',[]),float);front=np.asarray(profile.get('front',[]),float)
        ok=np.isfinite(front)
        if ok.sum()>=2 and sup[ok].min()<=original_local[2]<=sup[ok].max():
            deviations[family]=float(original_local[1]-np.interp(original_local[2],sup[ok],front[ok]))
        else:deviations[family]=None
    outside=bool(abs(original_local[1]-seed[1])>15 or abs(original_local[2]-seed[2])>12 or abs(original_local[0])>3)
    a,b=deviations['mask'],deviations['native']
    both_off=(a is not None and b is not None and abs(a)>1. and abs(b)>1. and np.sign(a)==np.sign(b))
    mismatch=(a is not None and abs(a)>1. and b is not None and abs(b)<=.75)
    representative=stability['representative']
    conflict=(representative is not None and np.linalg.norm(np.asarray(representative)-original_local)>3.)
    reasons=[]
    if outside:reasons.append('original_outside_independent_roi')
    if both_off:reasons.append('both_sources_disagree_with_original_front_boundary')
    if conflict:reasons.append('stable_definition_candidate_disagrees_with_original')
    triggered=bool(reasons)
    route=('mask_mismatch_original_retained' if mismatch and not triggered else 'existing_point_retained_not_confirmed')
    if triggered:
        route=('single_reconstruction_candidate' if stability['status']=='stable_single_candidate' else
               'candidate_region' if stability['adequate_support'] else 'automatic_withhold')
    return dict(route=route,triggered=triggered,reasons=reasons,original_boundary_deviation=deviations)
