"""Po 후보의 조건 변화 안정성. 반복 추론 정확도·통계적 신뢰구간을 뜻하지 않는다."""
import numpy as np
from scipy.ndimage import binary_closing,binary_erosion,binary_fill_holes,label,map_coordinates
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu

from cbct_po_tube85 import connected_tracks

VARIANTS=[dict(name='mask_base',family='AMASSS',step=.5,offset=[0,0,0],closing=1.,contrast=100.)]
for axis in range(3):
    for delta in (-1.5,1.5):
        offset=[0.,0.,0.];offset[axis]=delta
        VARIANTS.append(dict(name=f'mask_shift_{axis}_{delta:+g}',family='AMASSS',step=.5,offset=offset,closing=1.,contrast=100.))
for step in (.75,1.):
    VARIANTS.append(dict(name=f'mask_step_{step}',family='AMASSS',step=step,offset=[0,0,0],closing=1.,contrast=100.))
for radius in (.5,1.5):
    VARIANTS.append(dict(name=f'mask_close_{radius}',family='AMASSS',step=.5,offset=[0,0,0],closing=radius,contrast=100.))
for contrast in (75.,150.):
    VARIANTS.append(dict(name=f'mask_contrast_{contrast}',family='AMASSS',step=.5,offset=[0,0,0],closing=1.,contrast=contrast))
for delta in (-100.,0.,100.):
    VARIANTS.append(dict(name=f'native_otsu_{delta:+g}',family='native_CT',step=.5,offset=[0,0,0],closing=1.,contrast=100.,otsu_delta=delta))

ACCEPT=dict(mask_success_fraction=.75,native_success_fraction=2/3,point_p90_radius_mm=1.5,
            family_medoid_distance_mm=2.,candidate_range_min_mask_fraction=.5,
            metric_width={'ANB':1.,'MP-FH':1.,'Nperp-A':2.,'Nperp-Pog':2.},
            interpretation='fixed engineering criteria; no calibrated probability or clinical accuracy guarantee')


def crop_variant(mask,ct,base_axes,seed,variant):
    step=variant['step'];center=np.asarray(seed)+variant['offset']
    axes=[np.arange(center[i]-h,center[i]+h+step/2,step) for i,h in enumerate((12.,15.,15.))]
    G=np.stack(np.meshgrid(*axes,indexing='ij'),0)
    idx=np.stack([(G[i]-base_axes[i][0])/(base_axes[i][1]-base_axes[i][0]) for i in range(3)],0)
    m=map_coordinates(mask.astype(np.uint8),idx,order=0,cval=0,prefilter=False)>0
    image=map_coordinates(ct,idx,order=1,cval=np.nan,prefilter=False)
    threshold=None
    if variant['family']=='native_CT':
        values=image[np.isfinite(image)]
        if len(values) and np.ptp(values)>0:
            lo,hi=np.percentile(values,[1,99]);values=np.clip(values,lo,hi)
            threshold=float(threshold_otsu(values))+variant['otsu_delta']
            m=np.isfinite(image)&(image>threshold)
        else:m=np.zeros(image.shape,bool)
    return m,image,axes,threshold


def nodes_for(mask,ct,axes,step,closing):
    L,A,S=axes;r=int(np.ceil(closing/step));yy,zz=np.mgrid[-r:r+1,-r:r+1]
    disk=(yy*step)**2+(zz*step)**2<=closing**2+1e-8
    nodes=[]
    for i,bone in enumerate(mask):
        closed=binary_closing(bone,structure=disk,border_value=0)
        void=binary_fill_holes(closed)&~closed
        labs,n=label(void)
        for j in range(1,n+1):
            region=labs==j;aa,ss=np.where(region);area=len(aa)*step**2
            if not 6<=area<=180:continue
            if not (3<=np.ptp(A[aa])+step<=20 and 3<=np.ptp(S[ss])+step<=20):continue
            roof=int(ss.max());ai=int(np.rint(np.median(aa[S[ss]>=S[roof]-.5])))
            da=max(1,int(round(1./step)));dz=max(1,int(round(1.5/step)))
            cap=bone[max(0,ai-da):ai+da+1,roof+1:min(roof+dz+1,len(S))]
            if not cap.size or cap.mean()<.3:continue
            interior=binary_erosion(region)
            lumen=ct[i][interior] if interior.any() else ct[i][region]
            above=ct[i,max(0,ai-da):ai+da+1,roof+1:min(roof+dz+1,len(S))]
            roof_values=above[cap]
            if not len(lumen) or not len(roof_values):continue
            contrast=float(np.nanmedian(roof_values)-np.nanmedian(lumen))
            if not np.isfinite(contrast):continue
            nodes.append(dict(index=i,region=region,area_mm2=float(area),
                              centroid=np.array([np.mean(A[aa]),np.mean(S[ss])]),
                              local=np.array([L[i],A[ai],S[roof]+step/2]),contrast=contrast))
    return nodes


def detect_variant(mask,ct,base_axes,seed,variant):
    m,image,axes,threshold=crop_variant(mask,ct,base_axes,seed,variant)
    nodes=nodes_for(m,image,axes,variant['step'],variant['closing'])
    groups=connected_tracks(nodes);eligible=[]
    for ids in groups:
        group=[nodes[i] for i in ids];indices=sorted({n['index'] for n in group})
        span=(indices[-1]-indices[0]+1)*variant['step']
        volume=sum(n['area_mm2'] for n in group)*variant['step']
        if span<4 or volume<40 or max(n['area_mm2'] for n in group)<12:continue
        plausible=[n for n in group if n['contrast']>=variant['contrast'] and np.linalg.norm(n['local']-seed)<=12]
        if not plausible:continue
        plausible.sort(key=lambda n:(-abs(n['local'][0]),np.linalg.norm((n['local']-seed)[1:])))
        point=plausible[0]['local']
        if min(abs(point[0]-axes[0][0]),abs(point[0]-axes[0][-1]))<1:continue
        median=np.median([n['local'] for n in group],axis=0);distance=np.linalg.norm((median-seed)[1:])
        eligible.append(dict(point=point,score=float(volume/(1+(distance/8.)**2)),span_mm=float(span)))
    eligible.sort(key=lambda c:-c['score'])
    unique=not(len(eligible)>1 and eligible[0]['score']/eligible[1]['score']<1.25)
    winner=eligible[0]['point'] if eligible and unique else None
    return dict(variant=variant['name'],family=variant['family'],winner=winner,
                eligible=eligible,threshold=threshold,status='winner' if winner is not None else 'ambiguous' if eligible else 'no_candidate')


def medoid(points):
    points=np.asarray(points,float)
    return points[int(np.argmin(cdist(points,points).sum(1)))]


def stability(results,variants=VARIANTS):
    families={};all_points=[]
    for family in ('AMASSS','native_CT'):
        runs=[r for r in results if r['family']==family]
        points=np.array([r['winner'] for r in runs if r['winner'] is not None],float).reshape(-1,3)
        center=medoid(points) if len(points) else None
        radius=float(np.percentile(np.linalg.norm(points-center,axis=1),90)) if len(points) else None
        fraction=len(points)/len([v for v in variants if v['family']==family])
        families[family]=dict(runs=len(runs),successes=len(points),success_fraction=fraction,
                              medoid=center,p90_radius_mm=radius)
        all_points.extend(points)
    a,b=families['AMASSS'],families['native_CT']
    distance=float(np.linalg.norm(a['medoid']-b['medoid'])) if a['medoid'] is not None and b['medoid'] is not None else None
    stable=(a['success_fraction']>=ACCEPT['mask_success_fraction'] and b['success_fraction']>=ACCEPT['native_success_fraction']
            and a['p90_radius_mm']<=ACCEPT['point_p90_radius_mm'] and b['p90_radius_mm']<=ACCEPT['point_p90_radius_mm']
            and distance is not None and distance<=ACCEPT['family_medoid_distance_mm'])
    adequate=(a['success_fraction']>=ACCEPT['candidate_range_min_mask_fraction'] or b['success_fraction']>=ACCEPT['native_success_fraction'])
    # 범위는 한 우승 후보에만 한정하지 않고 경쟁하는 적격 후보까지 포함한다.
    candidate_points=[c['point'] for r in results for c in r['eligible']]
    unique=np.unique(np.round(np.asarray(candidate_points,float).reshape(-1,3),6),axis=0) if candidate_points else np.empty((0,3))
    if stable and len(unique) and np.linalg.norm(unique-medoid(all_points),axis=1).max()>3.:
        stable=False
    return dict(status='stable_single_candidate' if stable else 'candidate_region' if adequate else 'insufficient_support',
                families=families,family_distance_mm=distance,
                representative=medoid(all_points) if stable else None,candidates=unique,
                candidate_count=len(unique),adequate_support=bool(adequate),
                coordinate_span_mm=np.ptp(unique,axis=0) if len(unique) else None)


def metric_decision(values,adequate,metric,legacy=False):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if legacy:return dict(status='existing_value_retained_unverified',value=float(a[0]),low=float(a[0]),high=float(a[0]))
    if not adequate or not len(a):return dict(status='automatic_withhold',value=None,
                                           low=float(a.min()) if len(a) else None,high=float(a.max()) if len(a) else None)
    low,high=float(a.min()),float(a.max())
    if high-low<=ACCEPT['metric_width'][metric]:
        return dict(status='candidate_stable_metric',value=float(np.median(a)),low=low,high=high)
    return dict(status='candidate_interval_only',value=None,low=low,high=high)
