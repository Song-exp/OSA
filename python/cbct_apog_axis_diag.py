"""1-2 진단 엔진: 같은 FH에서 점 이동 효과와 기준축(FH 앞뒤 기울기) 효과를 분리한다.

선택·보정용이 아니다. 엑셀은 후보·분기가 모두 고정된 뒤 오차 분해에만 쓴다.
"""
import numpy as np
from osa.ceph import head_frame,_msp_angle
from osa.refine import front_contour,STEP,MAX_SHIFT,WINDOWS

RULE=dict(version='APog-axis-diagnosis-v1',
          pitch_grid_deg=[round(x,2) for x in np.arange(-6,6.001,.5)],
          pitch_sign='delta>0: 앞쪽 축을 위쪽으로 회전, N 아래 점의 Nperp 감소',
          small_error={'Nperp-A':1.,'Nperp-Pog':1.,'ANB':1.,'MP-FH':1.},
          pitch_explained=dict(max_abs_delta_deg=3.,max_residual_mm=1.),
          frame={'ANB':'jung','Nperp-A':'yao4','Nperp-Pog':'yao4','MP-FH':'yao4'},
          Pog_legacy='osa.refine Pog window Me+3..Me+20, frontmost MAND contour, fallback to ALI',
          excel_zero_policy='엑셀 0은 결측 의심으로 분리 집계',
          Excel_used_for_selection=False,symmetry_used=False,production_replacement=False)


def pitched(R,deg):
    """머리 좌표계(행=좌,앞,위)를 좌우 축 기준으로 deg만큼 회전."""
    t=np.radians(deg);left,ant,sup=R
    return np.vstack([left,np.cos(t)*ant+np.sin(t)*sup,np.cos(t)*sup-np.sin(t)*ant])


def frame(P,fh,deg=0.):
    o,R=head_frame(P,fh);return o,pitched(R,deg)


def local(P,o,R):
    return {k:R@(np.asarray(v,float)-o) for k,v in P.items()}


def nperp(P,key,o,R):
    return float(R[1]@(np.asarray(P[key],float)-o))


def anb(P,o,R):
    h=local({k:P[k] for k in ('A','B')},o,R)
    return float(_msp_angle(h['A'])-_msp_angle(h['B']))


def mpfh(P,o,R):
    h=local({k:P[k] for k in ('Me','RGo','LGo')},o,R);d=(h['RGo']+h['LGo'])/2-h['Me']
    return float(np.degrees(np.arctan2(d[2],-d[1])))


def legacy_pog(P,mask,dirs,org,o,R):
    """기존 refine의 Pog 부분과 같은 규칙. 축이 바뀌면 창과 앞쪽 방향도 같이 바뀐다."""
    ref,lo,hi,_=WINDOWS['Pog']
    me=R@(np.asarray(P[ref],float)-o);pog=R@(np.asarray(P['Pog'],float)-o)
    sup=np.arange(me[2]+lo,me[2]+hi+1e-9,STEP)
    front=front_contour(mask,dirs,org,o,R,sup,(pog[1]-25,pog[1]+25))
    coverage=float(np.isfinite(front).mean())
    if coverage<.5:return dict(value=float(pog[1]),status='coverage_fallback_ALI',coverage=coverage,edge=None,height=None)
    best=np.nanmax(front);ties=np.flatnonzero(abs(front-best)<=STEP/2);i=ties[len(ties)//2]
    new=np.array([0.,front[i],sup[i]])
    if np.linalg.norm(new-pog)>MAX_SHIFT:
        return dict(value=float(pog[1]),status='shift_fallback_ALI',coverage=coverage,edge=None,height=None)
    edge='lower' if ties.min()==0 else 'upper' if ties.max()==len(sup)-1 else None
    return dict(value=float(new[1]),status='refined',coverage=coverage,edge=edge,height=float(sup[i]-me[2]))


def slope(deg,values):
    d=np.asarray(deg,float);v=np.asarray(values,float);ok=np.isfinite(v)
    return float(np.polyfit(d[ok],v[ok],1)[0]) if ok.sum()>=2 else np.nan


def common_pitch(errors,slopes):
    """e_k ≈ s_k·δ 를 최소제곱으로 푼다. errors = 엑셀 − 자동."""
    e=np.asarray(errors,float);s=np.asarray(slopes,float);ok=np.isfinite(e)&np.isfinite(s)
    if not ok.any() or not (s[ok]**2).sum():return np.nan,np.full(len(e),np.nan)
    delta=float((s[ok]*e[ok]).sum()/(s[ok]**2).sum())
    return delta,e-s*delta


def reach(values,target):
    """후보 범위에서 엑셀까지의 거리. 범위 안이면 0."""
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if not len(v) or target is None or not np.isfinite(target):return np.nan
    return float(max(v.min()-target,0.,target-v.max()))


def classify(error,small,reach_gap,pitch_ok):
    if not np.isfinite(error):return 'no_reference'
    if abs(error)<=small:return 'small_error'
    if np.isfinite(reach_gap) and reach_gap==0:return 'point_candidate_reaches_reference'
    if pitch_ok:return 'common_pitch_consistent'
    return 'unexplained_by_point_or_pitch'
