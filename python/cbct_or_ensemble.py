"""Or 국소 후보: 안와저 전체의 저점 대신 앞쪽 하연으로 이어지는 상연 전환부를 탐색.

MAX 분할과 원본 밝기 경계는 실제 안와 하연의 정답 라벨이 아니다.
동일 환자의 동일 측만 분석하고 모든 결과를 자동 복원 후보로 기록한다.
"""
import numpy as np
from scipy.ndimage import binary_closing,minimum_filter1d

from cbct_po_ensemble import VARIANTS as PO_VARIANTS,crop_variant

VARIANTS=[{**v,'name':'or_'+v['name']} for v in PO_VARIANTS]
RULE=dict(version='or-anterior-rim-transition-v1',forward_mm=2.,min_forward_drop_mm=3.,
          roof_neighborhood_mm=3.,seed_window_mm=[8.,8.,8.],
          near_lowest_roof_mm=.75,contrast='bone below minus image above roof',
          interpretation='operational rim proxy, not expert-validated Orbitale')


def rim_candidates(mask,ct,axes,seed,step,closing,contrast):
    L,A,S=axes
    r=max(1,int(np.ceil(closing/step)))
    x,y,z=np.mgrid[-r:r+1,-r:r+1,-r:r+1]
    ball=(x*step)**2+(y*step)**2+(z*step)**2<=closing**2+1e-8
    mask=binary_closing(mask,structure=ball,border_value=0)
    has=mask.any(2)
    top=np.where(has,len(S)-1-np.argmax(mask[:,:,::-1],axis=2),0)
    heights=np.where(has,S[top]+step/2,np.nan)
    forward=max(1,int(round(RULE['forward_mm']/step)))
    # 각 좌우 위치에서 뒤쪽 바닥이 앞쪽 외면으로 내려가는 전환부를 찾는다.
    profile=[]
    for i in range(2,len(L)-2):
        if abs(L[i]-seed[0])>8:continue
        nodes=[]
        for j in range(2,len(A)-forward-2):
            if not has[i,j] or abs(A[j]-seed[1])>8:continue
            k=int(top[i,j]);s=heights[i,j]
            if k<3 or k>=len(S)-3 or abs(s-seed[2])>8:continue
            later=heights[i,j+forward]
            drop=s-later if np.isfinite(later) else 20.
            if drop<RULE['min_forward_drop_mm']:continue
            da=max(1,int(round(1./step)));dz=max(1,int(round(1.5/step)))
            below=ct[max(i-1,0):i+2,max(j-1,0):j+2,max(k-dz+1,0):k+1]
            above=ct[max(i-1,0):i+2,max(j-1,0):j+2,k+1:min(k+dz+1,len(S))]
            lower_mask=mask[max(i-1,0):i+2,max(j-1,0):j+2,max(k-dz+1,0):k+1]
            dense=below[lower_mask]
            if not len(dense) or not above.size:continue
            native_contrast=float(np.nanmedian(dense)-np.nanmedian(above))
            if not np.isfinite(native_contrast) or native_contrast<contrast:continue
            nodes.append(dict(point=np.array([L[i],A[j],s]),contrast=native_contrast,drop_mm=float(drop),i=i,j=j))
        if nodes:
            # 내부 바닥 대신 이어진 바닥이 끝나는 앞쪽 가장자리를 선택한다.
            nodes.sort(key=lambda n:(-n['point'][1],np.linalg.norm(n['point']-seed)))
            profile.append(nodes[0])
    if not profile:return []
    roof=np.full(len(L),np.inf)
    for n in profile:roof[n['i']]=n['point'][2]
    size=2*max(1,int(round(RULE['roof_neighborhood_mm']/step)))+1
    local_min=minimum_filter1d(roof,size=size,mode='constant',cval=np.inf)
    candidates=[]
    for n in profile:
        i=n['i']
        nearby=[x for x in profile if abs(L[x['i']]-L[i])<=1.5 and np.linalg.norm((x['point']-n['point'])[[1,2]])<=3.]
        if len(nearby)<max(3,int(round(2./step))):continue
        if n['point'][2]>local_min[i]+.25:continue
        if np.linalg.norm(n['point']-seed)>10.:continue
        candidates.append(n)
    if not candidates:return []
    low=min(n['point'][2] for n in candidates)
    # 극단적인 끝점 하나 대신 같은 낮은 가장자리의 범위를 후보로 남긴다.
    return [n for n in candidates if n['point'][2]<=low+RULE['near_lowest_roof_mm']]


def detect_variant(mask,ct,base_axes,seed,variant):
    m,image,axes,threshold=crop_variant(mask,ct,base_axes,seed,variant)
    nodes=rim_candidates(m,image,axes,seed,variant['step'],variant['closing'],variant['contrast'])
    # 격자의 이웃 점은 같은 가장자리이므로 우승 후보는 seed까지 거리로 고른다.
    nodes.sort(key=lambda n:np.linalg.norm(n['point']-seed))
    eligible=[dict(point=n['point'],score=float(1/(1+np.linalg.norm(n['point']-seed))),
                   contrast=n['contrast'],forward_drop_mm=n['drop_mm']) for n in nodes]
    return dict(variant=variant['name'],family=variant['family'],winner=eligible[0]['point'] if eligible else None,
                eligible=eligible,threshold=threshold,status='winner' if eligible else 'no_candidate')
