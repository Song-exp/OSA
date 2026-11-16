"""v2 후보를 치아 인접·전방 피질·하연 조건으로 선별하는 실험용 게이트."""
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates,gaussian_filter1d

RULE=dict(version='APog-structure-gate-v3',tooth_clearance_mm=.75,bone_boundary_distance_mm=.75,
          minimum_CT_contrast=75.,normal_offsets_mm=[.75,1.,1.25],minimum_front_normal_component=.707,
          Pog_inferior_clearance_mm=3.,inferior_patch_half_mm=2.,minimum_valid_samples=.8,
          tooth_mask='UNIVERSALLAB_MODEL labels 1..52 (permanent/deciduous); exclude bone/canal 53..55',
          inherited_crop='legacy tooth segmentation crop used original Pog/Me; domain checked, not independent truth',
          trigger_policy='freeze v2 triggered branches; filter existing candidates only',
          symmetry_used=False,Excel_used_for_selection=False,production_replacement=False)


def tooth_binary(labels):
    return (labels>=1)&(labels<=52)


def at(array,axes,point,order=1,cval=np.nan):
    q=np.atleast_2d(point)
    coords=np.stack([(q[:,i]-axes[i][0])/(axes[i][1]-axes[i][0]) for i in range(3)])
    return map_coordinates(array,coords,order=order,cval=cval,prefilter=False,output=np.float32)


def prepare(mask,ct,teeth,domain,axes):
    spacing=[float(x[1]-x[0]) for x in axes]
    return dict(mask=mask,ct=ct,teeth=teeth,domain=domain,axes=axes,
                tooth_distance=distance_transform_edt(~teeth,sampling=spacing) if teeth.any() else None,
                boundary_distance=np.where(mask,distance_transform_edt(mask,sampling=spacing),distance_transform_edt(~mask,sampling=spacing)))


def anterior_normal(profile,point):
    s=np.asarray(profile.get('sup',[]),float);f=np.asarray(profile.get('front',[]),float)
    ok=np.isfinite(f)
    if ok.sum()<3 or point[2]<s[ok].min() or point[2]>s[ok].max():return None
    filled=np.interp(s,s[ok],f[ok]);step=float(np.median(np.diff(s)))
    smooth=gaussian_filter1d(filled,.75/step,mode='nearest')
    slope=float(np.interp(point[2],s,np.gradient(smooth,s)))
    n=np.array([0.,1.,-slope]);return n/np.linalg.norm(n)


def inferior_clearance(mask,axes,point):
    L,A,S=axes
    li=abs(L-point[0])<=1.;ai=abs(A-point[1])<=RULE['inferior_patch_half_mm']
    if not li.any() or not ai.any():return None,'inferior_patch_outside_roi'
    patch=mask[li][:,ai,:].any((0,1));rows=np.flatnonzero(patch)
    if not len(rows):return None,'inferior_structure_missing'
    if rows[0]==0:return None,'inferior_boundary_truncated'
    return float(point[2]-S[rows[0]]),None


def check(point,profile,key,data):
    p=np.asarray(point,float);axes=data['axes'];reasons=[];info={}
    def value(name):return float(at(data[name],axes,p)[0])
    if value('domain')<.99 or not np.isfinite(value('domain')):
        reasons.append('tooth_crop_unknown_at_candidate')
    distance=data['tooth_distance']
    gap=float(at(distance,axes,p)[0]) if distance is not None else None
    info['tooth_distance_mm']=gap
    if gap is not None and gap<=RULE['tooth_clearance_mm']:reasons.append('tooth_or_root_proximity')
    boundary=value('boundary_distance');info['bone_boundary_distance_mm']=boundary
    if not np.isfinite(boundary) or boundary>RULE['bone_boundary_distance_mm']:reasons.append('not_near_AMASSS_bone_boundary')
    n=anterior_normal(profile,p);info['normal']=n
    if n is None:reasons.append('normal_unavailable')
    else:
        info['front_normal_component']=float(n[1])
        if n[1]<RULE['minimum_front_normal_component']:reasons.append('surface_faces_inferior_or_superior')
        offsets=np.asarray(RULE['normal_offsets_mm'])[:,None]
        inner=at(data['ct'],axes,p-offsets*n);outer=at(data['ct'],axes,p+offsets*n)
        valid=np.isfinite(inner)&np.isfinite(outer)
        contrast=float(np.median(inner[valid]-outer[valid])) if valid.any() else None
        info['CT_contrast']=contrast
        if valid.mean()<RULE['minimum_valid_samples']:reasons.append('CT_normal_samples_missing')
        elif contrast<RULE['minimum_CT_contrast']:reasons.append('CT_cortical_contrast_insufficient')
        inside=at(data['mask'],axes,p-offsets*n,order=0,cval=0).mean()
        outside=at(data['mask'],axes,p+offsets*n,order=0,cval=0).mean()
        info.update(inward_mask_fraction=float(inside),outward_mask_fraction=float(outside))
        if inside<.5 or outside>.5:reasons.append('not_anterior_outer_cortex')
    if key=='Pog':
        clearance,issue=inferior_clearance(data['mask'],axes,p)
        info['inferior_clearance_mm']=clearance
        if issue:reasons.append(issue)
        elif clearance<RULE['Pog_inferior_clearance_mm']:reasons.append('too_close_to_inferior_border')
    return dict(accepted=not reasons,reasons=reasons,**info)


def filter_variant(run,key,data):
    result=dict(run);accepted=[];checks=[]
    for candidate in run['eligible']:
        evidence=check(candidate['point'],run['profile'],key,data)
        checks.append(dict(point=candidate['point'],**evidence))
        if evidence['accepted']:accepted.append(candidate)
    # Preserve the original score order; never choose by Excel or original target point.
    result.update(eligible=accepted,winner=accepted[0]['point'] if accepted else None,
                  status='candidate' if accepted else 'structural_gate_rejected' if checks else run['status'],
                  structural_checks=checks)
    return result
