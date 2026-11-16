"""Illustrate report definitions with existing, local CBCT data.

Example cases are fixed by cohort order, not Excel agreement. All displayed
points are automatic candidates, never expert ground truth. No inputs changed.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from cbct_patient_pipeline import CB, OUT, landmarks, digest, save
from cbct_fh_repair import repair
from cbct_landmark_review import panel
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd
from osa.ceph import head_frame, measures

DEST = OUT / 'report/glossary_images'
manifest = {}


def register(keys, filename, patient, caption):
    for key in keys:
        manifest[key] = dict(src='glossary_images/'+filename, patient=patient,
                             caption=caption+' 자동 계측 예시이며 전문가 확인 전입니다.')


def finish(fig, filename):
    fig.savefig(DEST / filename, dpi=125, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def point_example(key, point, raw, directions, origin, patient, caption, half=25):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    for ax, plane in zip(axes, ('sagittal', 'coronal')):
        panel(ax, raw, directions, origin, point, plane, {key: (point, 'cyan', '+')}, half=half)
        ax.set_title(f'{key}: {plane}', fontsize=11)
        ax.tick_params(labelsize=7)
    fig.suptitle(f'{patient}: automatic {key} example; cyan + = point', fontsize=11)
    filename=key+'.png'
    finish(fig, filename)
    register([key], filename, patient, caption)


def sample(raw, directions, origin, q, shape):
    return map_coordinates(raw, np.linalg.solve(directions.T, (q-origin).T)[::-1],
                           order=1, cval=-1000, output=np.float32).reshape(shape)


def sagittal(raw, directions, origin, points):
    o, R = head_frame(points)
    ant, sup = np.arange(-115, 55, .5), np.arange(-150, 55, .5)
    A, Z = np.meshgrid(ant, sup, indexing='ij')
    q=o+np.stack([A*0, A, Z], -1).reshape(-1, 3)@R
    ct=sample(raw, directions, origin, q, A.shape)
    local={k:R@(p-o) for k,p in points.items()}
    fig, ax=plt.subplots(figsize=(5.6,6.3),layout='constrained')
    ax.imshow(ct.T,origin='lower',cmap='gray',vmin=-700,vmax=1800,extent=[ant[0],ant[-1],sup[0],sup[-1]])
    ax.set(xlabel='Anterior + (mm)',ylabel='Superior + (mm)')
    ax.tick_params(labelsize=8)
    return fig,ax,local


def mark(ax, h, key, color='cyan'):
    p=h[key]
    ax.plot(p[1],p[2],'+',color=color,ms=11,mew=2)
    ax.annotate(key,(p[1],p[2]),xytext=(5,5),textcoords='offset points',color=color,fontsize=10,
                bbox=dict(facecolor='black',alpha=.6,edgecolor='none',pad=1))


def skull_examples(pid, points, raw, directions, origin):
    patient=f'P_{pid}'
    for key in ('N','A','B','S','Ba','Pog','Me','ANS','PNS','RPo','LPo','ROr','LOr','RGo','LGo'):
        point_example(key,points[key],raw,directions,origin,patient,f'{patient} 실제 단면의 {key} 위치. 하늘색 +가 해당 자동 좌표입니다.')
    for alias,key in {'Na':'N','Se':'S','Po':'RPo','Or':'ROr','Go':'RGo'}.items():
        manifest[alias]=dict(manifest[key])
        if alias in ('Po','Or','Go'):
            manifest[alias]['caption']='오른쪽 점의 예시입니다. '+manifest[alias]['caption']
        if alias=='Se':
            manifest[alias]['caption']='Se=Sella라는 현재 가정을 보여 주는 예시이며 원 정의는 미확인입니다. '+manifest[alias]['caption']
    for kind in ('FH','ANB','Nperp','MP'):
        fig,ax,h=sagittal(raw,directions,origin,points)
        fh=float(np.mean([h[k][2] for k in ('RPo','LPo','ROr')]))
        if kind in ('FH','Nperp','MP'):
            ax.axhline(fh,color='yellow',ls='--',lw=1.5,label='FH (projected)')
        if kind=='FH':
            for key in ('RPo','LPo','ROr'):mark(ax,h,key)
            caption=f'{patient} FH를 만드는 세 점의 정중면 투영 예시입니다. 좌우 점은 실제 같은 단면 위에 있지 않습니다.'
        elif kind=='ANB':
            for key,color in [('N','cyan'),('A','yellow'),('B','lime')]:mark(ax,h,key,color)
            for key,color in [('A','yellow'),('B','lime')]:ax.plot([0,h[key][1]],[0,h[key][2]],color=color,lw=1.8)
            ax.text(.03,.03,f'ANB = {measures(points)["ANB"]:.2f} deg',transform=ax.transAxes,color='white',bbox=dict(facecolor='black',alpha=.7))
            caption=f'{patient}의 N–A·N–B 선으로 ANB를 설명하는 정중면 투영 예시입니다.'
        elif kind=='Nperp':
            ax.axvline(0,color='cyan',ls='--',label='N perpendicular')
            for key,color in [('A','lime'),('Pog','orange')]:
                mark(ax,h,key,color)
                ax.plot([0,h[key][1]],[h[key][2],h[key][2]],color=color,lw=2,label='Nperp-'+key)
            mark(ax,h,'N')
            caption=f'{patient}의 N 수직선과 A·Pog의 앞뒤 거리 예시입니다.'
        else:
            h['Go mean']=(h['RGo']+h['LGo'])/2
            mark(ax,h,'Go mean','lime');mark(ax,h,'Me','cyan')
            ax.plot([h['Go mean'][1],h['Me'][1]],[h['Go mean'][2],h['Me'][2]],color='lime',lw=2,label='MP: Go mean - Me')
            caption=f'{patient}의 좌우 Go 평균–Me 선과 FH 기준선의 정중면 투영입니다.'
        if kind!='ANB':ax.legend(loc='upper left',fontsize=8)
        ax.set_title(f'{patient}: {kind} on real CBCT',fontsize=12)
        finish(fig,kind+'.png')
        keys={'FH':['FH'],'ANB':['ANB'],'Nperp':['Nperp','Nperp-A','Nperp-Pog'],'MP':['MP','MP-FH','MP–FH']}[kind]
        register(keys,kind+'.png',patient,caption)
    o,R=head_frame(points)
    ls,zs=np.arange(-90,90,.5),np.arange(-150,55,.5)
    L,Z=np.meshgrid(ls,zs,indexing='ij')
    q=o+np.stack([L,L*0,Z],-1).reshape(-1,3)@R
    fig,ax=plt.subplots(figsize=(5.5,6.2),layout='constrained')
    ax.imshow(sample(raw,directions,origin,q,L.shape).T,origin='lower',cmap='gray',vmin=-700,vmax=1800,extent=[ls[0],ls[-1],zs[0],zs[-1]])
    ax.axvline(0,color='cyan',lw=2,label='MSP intersection');ax.plot(0,0,'+',color='yellow',ms=12)
    ax.set(title=f'{patient}: MSP in N coronal plane',xlabel='Left + (mm)',ylabel='Superior + (mm)');ax.legend(fontsize=8)
    finish(fig,'MSP.png');register(['MSP'],'MSP.png',patient,'N을 지나는 관상면에서 정중시상면이 만나는 선(하늘색)입니다. 평면 전체의 3차원 모양은 아닙니다.')


def nasal_and_dental(pid, points, raw, directions, origin):
    import SimpleITK as sitk
    from run_ali import ascii_path
    from cbct_nasal_label_audit import sample as sample_label
    from cbct_maxilla_section import section
    from cbct_evidence_figs import teeth_points
    patient=f'P_{pid}'
    image=sitk.ReadImage(str(ascii_path(CB/f'seg_totalseg/head_glands_cavities/{patient}.nii.gz')))
    labels=sitk.GetArrayFromImage(image)
    o,R,_=midline_frame(points);ans=(R@(points['ANS']-o))[1]
    ls,zs=np.arange(-40,40,.3),np.arange(-10,65,.3);L,Z=np.meshgrid(ls,zs,indexing='ij')
    q=o+np.stack([L,np.full_like(L,ans*.5),Z],-1).reshape(-1,3)@R
    ct=sample(raw,directions,origin,q,L.shape);mask=sample_label(labels,image,q).reshape(L.shape)
    fig,ax=plt.subplots(figsize=(6,5.5),layout='constrained')
    ax.imshow(ct.T,origin='lower',cmap='gray',vmin=-1000,vmax=1200,extent=[ls[0],ls[-1],zs[0],zs[-1]])
    for area,color in [(mask,'lime'),(mask&(ct>-1200)&(ct<-400),'yellow')]:ax.contour(ls,zs,area.T.astype(float),[.5],colors=color,linewidths=1)
    ax.set(title=f'{patient}: nasal label (green), air (yellow)',xlabel='Left + (mm)',ylabel='Superior + (mm)')
    finish(fig,'NAV.png');register(['NAV','HU'],'NAV.png',patient,'실제 비강 라벨(초록)과 라벨 안 공기(노랑)의 관상면 예시입니다. NAV는 이 한 장의 넓이가 아니라 경계 안 3차원 공기 부피입니다.')
    nav=pd.read_csv(OUT/'nav_review/v1/nav.csv').set_index('Patient').loc[patient]
    rh=o+np.array([0,nav.Rh_ant,nav.Rh_sup])@R
    point_example('Rh',rh,raw,directions,origin,patient,'비골 하단을 찾는 영상 규칙의 자동 후보점입니다.')
    molars=pd.read_csv(OUT/'tdi_review/v1/molars.csv')
    value=pd.read_csv(OUT/'phi_ncwr_review/v1/phi_ncwr.csv').set_index('Patient').loc[patient]
    ct,mx,ls,zs,q_teeth,upper,lc,ts=section(pid,molars)
    for kind in ('NCWR','PHI'):
        fig,ax=plt.subplots(figsize=(7,5.7),layout='constrained')
        ax.imshow(ct.T,origin='lower',cmap='gray',vmin=-1000,vmax=1500,extent=[ls[0],ls[-1],zs[0],zs[-1]])
        ax.contour(ls,zs,ts.T.astype(float),[.5],colors='lime',linewidths=.9)
        if kind=='NCWR':
            ax.plot([value.ts_l,value.ts_r],[value.Z_ts,value.Z_ts],color='cyan',lw=2,label=f'NCW = {value.NCW_ts:.2f} mm')
            ax.set_title(f'{patient}: NCWR = NCW / MxBW x 100\n{value.NCW_ts:.2f} / {value.MxBW_jugal:.2f} x 100 = {value.NCWR_ts:.2f}')
            caption='실제 대구치 단면의 비강 폭 측정선(하늘색)입니다. 분모 MxBW는 그림에 임의의 기준선을 그리지 않고 현재 자동값만 병기했습니다. 원 측정 높이·기준점은 미확인입니다.'
        else:
            ax.contour(ls,zs,mx.T.astype(float),[.5],colors='orange',linewidths=.8)
            ax.plot([lc,lc],[value.Z_occ,value.Z_palate],color='cyan',lw=2,label=f'PH = {value.PH:.2f} mm')
            ax.plot(upper.L,upper.Z,'+',color='yellow',ms=12)
            ax.plot([upper.L.UR6,upper.L.UL6],[upper.Z.mean(),upper.Z.mean()],color='yellow',ls='--',label=f'Width = {value.MxOW_rc:.2f} mm')
            ax.set_title(f'{patient}: PHI = PH / width x 100 = {value.PHI_rc:.2f}')
            caption='구개 높이(하늘색)와 상악 제1대구치 저항중심 폭(노랑)의 실제 단면 예시입니다. 표시한 자동 높이·점은 전문가 확인 전입니다.'
        ax.set(xlabel='Left + (mm)',ylabel='Superior + (mm)');ax.legend(fontsize=8)
        finish(fig,kind+'.png')
        keys=['NCWR','NCW','MxBW'] if kind=='NCWR' else ['PHI','PH','MxOW']
        register(keys,kind+'.png',patient,caption)
    h0,frame=head_frame(points)
    selected=molars[(molars.Patient==patient)&molars.status.eq('ok')].set_index('tooth')
    fig,axes=plt.subplots(1,2,figsize=(9,5),layout='constrained');widths=[]
    for ax,teeth,name in [(axes[0],['UR6','UL6'],'Upper'),(axes[1],['LR6','LL6'],'Lower')]:
        u=selected.loc[teeth];width=float(u.L.iloc[1]-u.L.iloc[0]);widths.append(width)
        ls,zs=np.arange(-48,48,.3),np.arange(u.Z.mean()-25,u.Z.mean()+35,.3);L,Z=np.meshgrid(ls,zs,indexing='ij')
        Q=h0+np.stack([L,np.full_like(L,u.A.mean()),Z],-1).reshape(-1,3)@frame
        ax.imshow(sample(raw,directions,origin,Q,L.shape).T,origin='lower',cmap='gray',vmin=-700,vmax=1800,extent=[ls[0],ls[-1],zs[0],zs[-1]])
        ax.plot(u.L,u.Z,'+',color='cyan',ms=12,mew=2);ax.plot(u.L,[u.Z.mean()]*2,color='cyan',lw=2)
        ax.set(title=f'{patient}: {name} width {width:.2f} mm',xlabel='Left + (mm)',ylabel='Superior + (mm)')
    fig.suptitle(f'TDI = upper - lower = {widths[0]-widths[1]:.2f} mm')
    finish(fig,'TDI.png');register(['TDI','MxAW','MnAW'],'TDI.png',patient,'각 턱의 실제 대구치 단면에서 좌우 저항중심의 수평 좌표 차이를 잽니다. TDI는 위턱 폭에서 아래턱 폭을 뺀 값입니다.')
    for tooth in ('UR6','UL6','LR6','LL6'):
        row=selected.loc[tooth];pt=h0+np.array([row.L,row.A,row.Z])@frame
        point_example(tooth,pt,raw,directions,origin,patient,'해당 제1대구치의 자동 저항중심 추정점을 표시한 위치 예시입니다.',half=17)
    for key in ('UR6MB','UL6MB'):
        point_example(key,teeth_points(pid)[key],raw,directions,origin,patient,'ALI 치아 모델이 출력한 근심협측 교두 후보점입니다.',half=17)


def airway_examples(pid):
    import cbct_pmca as pm
    from cbct_pav import volume,above_plane,BELOW
    patient=f'P_{pid}';points=repair(landmarks(pid)[0])[0]
    raw,directions,origin=read_nrrd(CB/f'nrrd/{patient}.nrrd')
    e=pd.read_csv(OUT/'eb_review/v2/eb_candidates.csv').set_index('Patient').loc[patient]
    assert e.method=='vallecula'
    eb=e[['Eb_x','Eb_y','Eb_z']].to_numpy(float)
    point_example('Eb',eb,raw,directions,origin,patient,'후두개곡 직접 검출 규칙(vallecula)의 위치 예시입니다. 설골 기반 근사 환자도 별도로 존재합니다.',half=25)
    o,R,_=midline_frame(points);s=R@(points['S']-o);ebm=np.array([0.,e.Eb_ant,e.Eb_sup])
    ant,zs=np.arange(-70,40,.4),np.arange(-95,35,.4);A,Z=np.meshgrid(ant,zs,indexing='ij')
    Q=o+np.stack([A*0,A,Z],-1).reshape(-1,3)@R
    ct=sample(raw,directions,origin,Q,A.shape)
    fig,ax=plt.subplots(figsize=(6,6),layout='constrained')
    ax.imshow(ct.T,origin='lower',cmap='gray',vmin=-1000,vmax=1200,extent=[ant[0],ant[-1],zs[0],zs[-1]])
    ax.plot([0,ebm[1]],[0,ebm[2]],color='cyan',lw=2)
    for name,q in [('PNS',np.zeros(3)),('Eb',ebm)]:
        ax.plot(q[1],q[2],'+',color='cyan',ms=11);ax.annotate(name,(q[1],q[2]),xytext=(6,4),textcoords='offset points',color='cyan')
    ax.set(title=f'{patient}: PAL {e.PAL_mm:.2f} mm (PNS to Eb)',xlabel='Anterior + (mm)',ylabel='Superior + (mm)')
    finish(fig,'PAL.png');register(['PAL'],'PAL.png',patient,'후비극에서 Eb까지 직선 거리의 정중면 투영 예시입니다. 저장된 PAL은 3차원 거리여서 그림의 투영 길이와 약간 다를 수 있습니다.')
    ls=np.arange(-pm.HALF_L,pm.HALF_L+1e-9,pm.STEP);aa=np.arange(pm.ANT[0],pm.ANT[1]+1e-9,pm.STEP);zz=np.arange(e.Eb_sup-BELOW,pm.TOP+1e-9,pm.STEP)
    grid=np.stack(np.meshgrid(ls,aa,zz,indexing='ij'),-1).reshape(-1,3)
    ct3=sample(raw,directions,origin,o+grid@R,(len(ls),len(aa),len(zz)))
    fh=R@head_frame(points)[1][2];fh=fh if fh[2]>0 else -fh
    mask,pav=volume(ct3,ls,aa,zz,s,above_plane(ls,aa,zz,ebm,fh))
    fig,ax=plt.subplots(figsize=(6,6),layout='constrained')
    ax.imshow(ct.T,origin='lower',cmap='gray',vmin=-1000,vmax=1200,extent=[ant[0],ant[-1],zs[0],zs[-1]])
    ax.contour(aa,zz,mask[np.abs(ls)<.8].any(0).T.astype(float),[.5],colors='lime')
    ax.plot([0,s[1]],[0,s[2]],color='yellow',label='S - PNS boundary')
    xs=np.array([-65,25]);ax.plot(xs,ebm[2]-(xs-ebm[1])*fh[1]/fh[2],color='cyan',label='Eb plane parallel to FH')
    ax.set(title=f'{patient}: PAV mask, volume {pav/1000:.2f} mL',xlabel='Anterior + (mm)',ylabel='Superior + (mm)');ax.legend(fontsize=7)
    finish(fig,'PAV.png');register(['PAV'],'PAV.png',patient,'정중 부근 인두 기도 마스크(초록), 위 경계(노랑), Eb를 지나는 아래 경계(하늘색)입니다. 부피는 전체 3차원 마스크로 계산합니다.')
    # Match PMCA's own grid and its centerline-perpendicular section direction.
    zz=np.arange(e.Eb_sup,pm.TOP+1e-9,pm.STEP)
    grid=np.stack(np.meshgrid(ls,aa,zz,indexing='ij'),-1).reshape(-1,3)
    ct3=sample(raw,directions,origin,o+grid@R,(len(ls),len(aa),len(zz)))
    mask=pm.airway(ct3,ls,aa,zz,s[1],s[2]);_,areas,centers=pm.sections(mask,ls,aa,zz)
    area,z=pm.pick(areas,zz,e.Eb_sup+pm.BOTTOM_CENTER,0.)
    saved=pd.read_csv(OUT/'pmca_review/v1/pmca_candidates.csv').set_index('Patient').loc[patient]
    assert np.isclose(area,saved.PMCA_center,atol=.02), (area,saved.PMCA_center)
    k=int(np.argmin(abs(zz-z)));pts=np.c_[centers,zz];t=np.gradient(pts,axis=0)[k];t=t/np.linalg.norm(t)
    u=np.cross(t,[1.,0.,0.]);u/=np.linalg.norm(u);v=np.cross(t,u)
    space=np.arange(-pm.HALF_L,pm.HALF_L+1e-9,pm.STEP);U,V=np.meshgrid(space,space,indexing='ij')
    local=pts[k]+U[...,None]*u+V[...,None]*v
    pixels=sample(raw,directions,origin,o+local.reshape(-1,3)@R,U.shape)
    index=np.stack([(local[...,0]-ls[0])/pm.STEP,(local[...,1]-aa[0])/pm.STEP,(local[...,2]-zz[0])/pm.STEP])
    section_mask=map_coordinates(mask.astype(np.float32),index,order=1,cval=0)>.5
    fig,ax=plt.subplots(figsize=(6,5.4),layout='constrained')
    ax.imshow(pixels.T,origin='lower',cmap='gray',vmin=-1000,vmax=1200,extent=[space[0],space[-1],space[0],space[-1]])
    ax.contour(space,space,section_mask.T.astype(float),[.5],colors='lime',linewidths=1.4)
    ax.set(title=f'{patient}: PMCA {area:.2f} mm2\nActual centerline-perpendicular minimum section',xlabel='Section axis U (mm)',ylabel='Section axis V (mm)')
    finish(fig,'PMCA.png');register(['PMCA'],'PMCA.png',patient,'현재 규칙으로 찾은 실제 최소 단면입니다. 초록 윤곽 안의 넓이를 잽니다. 단면은 기도 중심선에 수직이며 원 측정 정의는 확인 전입니다.')


def other_examples(pid,points,raw,directions,origin):
    from scipy.ndimage import label
    from scipy.stats import rankdata
    from cbct_extra import kappa
    patient=f'P_{pid}'
    fig,axes=plt.subplots(1,2,figsize=(8.5,5),layout='constrained')
    indices=np.rint(np.linalg.solve(directions.T,points['N']-origin)).astype(int)
    spacing=np.diag(directions);low=origin-spacing/2;high=origin+(np.array(raw.shape[::-1])-.5)*spacing
    for ax,image,extent,title in [(axes[0],raw[:,:,indices[0]],[low[1],high[1],low[2],high[2]],'Sagittal full FOV'),(axes[1],raw[:,indices[1],:],[low[0],high[0],low[2],high[2]],'Coronal full FOV')]:
        ax.imshow(image,origin='lower',cmap='gray',vmin=-700,vmax=1800,extent=extent)
        ax.set_title(title,fontsize=11);ax.set_xlabel('LPS horizontal (mm)');ax.set_ylabel('LPS z (mm)');ax.tick_params(labelsize=7)
    fig.suptitle(f'{patient}: original CBCT coverage, no patient identifiers')
    finish(fig,'FOV.png');register(['CBCT','FOV'],'FOV.png',patient,'실제 CBCT 촬영 범위 전체의 단면입니다. 영상 끝 바깥의 구조는 현재 촬영본에서 확인할 수 없습니다.')
    c=pd.read_csv(OUT/'condyle_review/v1/condyle.csv').set_index('Patient').loc[patient]
    o,R=head_frame(points);L0,A0,Z0=c.Sup_R_L,c.Sup_R_ant,c.Sup_R_sup
    ant,sup=np.arange(A0-25,A0+25,.3),np.arange(c.Mid_R_sup-18,Z0+12,.3)
    A,Z=np.meshgrid(ant,sup,indexing='ij');q=o+np.stack([A*0+L0,A,Z],-1).reshape(-1,3)@R
    fig,axes=plt.subplots(1,2,figsize=(9,5.5),layout='constrained');ax=axes[0]
    ax.imshow(sample(raw,directions,origin,q,A.shape).T,origin='lower',cmap='gray',vmin=-700,vmax=1800,extent=[ant[0],ant[-1],sup[0],sup[-1]])
    ax.plot([A0,c.Mid_R_ant],[Z0,c.Mid_R_sup],color='cyan',lw=2)
    for key,a,z,color in [('Sup',A0,Z0,'yellow'),('Mid',c.Mid_R_ant,c.Mid_R_sup,'cyan')]:
        ax.plot(a,z,'+',color=color,ms=12,mew=2);ax.annotate(key,(a,z),xytext=(5,5),textcoords='offset points',color=color)
    ax.axhline(c.Mid_R_sup,color='yellow',ls='--',label='FH-parallel reference')
    ax.set(title=f'{patient}: right condyle, sagittal',xlabel='Anterior + (mm)',ylabel='Superior + (mm)');ax.legend(fontsize=7)
    ls,ant=np.arange(L0-15,L0+15,.3),np.arange(A0-20,A0+20,.3);L,A=np.meshgrid(ls,ant,indexing='ij')
    q=o+np.stack([L,A,L*0+(Z0-3)],-1).reshape(-1,3)@R
    axes[1].imshow(sample(raw,directions,origin,q,L.shape).T,origin='lower',cmap='gray',vmin=-700,vmax=1800,extent=[ls[0],ls[-1],ant[0],ant[-1]])
    seg,sd,so=read_nrrd(CB/f'seg/{patient}_seg_MAND.nrrd')
    masks=[]
    for dz in (-.3,0,.3):
        Q=o+np.stack([L,A,L*0+(Z0-3+dz)],-1).reshape(-1,3)@R
        masks.append(map_coordinates(seg,np.linalg.solve(sd.T,(Q-so).T)[::-1],order=0,cval=0).reshape(L.shape)>0)
    mask=np.any(masks,axis=0);components,_=label(mask);ix=np.argwhere(mask)
    nearest=ix[np.argmin((ls[ix[:,0]]-L0)**2+(ant[ix[:,1]]-A0)**2)]
    mask=components==components[tuple(nearest)];ij=np.argwhere(mask)
    XY=np.c_[ls[ij[:,0]],ant[ij[:,1]]];center=XY.mean(0);axis=np.linalg.eigh(np.cov(XY.T))[1][:,-1]
    axes[1].contour(ls,ant,mask.T.astype(float),[.5],colors='lime')
    line=center+np.array([-10,10])[:,None]*axis;axes[1].plot(line[:,0],line[:,1],color='cyan',lw=2)
    axes[1].axhline(center[1],color='yellow',ls='--');axes[1].set(title=f'{patient}: axial, 3 mm below Sup',xlabel='Left + (mm)',ylabel='Anterior + (mm)')
    finish(fig,'condyle.png');register(['Sup','Mid','TMJ'],'condyle.png',patient,'오른쪽 과두의 실제 시상면과 최상점 아래 축상면입니다. Sup–Mid 및 장축의 자동 후보를 설명하는 예시이며 원래 각도 정의·통증 여부의 검증 자료가 아닙니다.')
    d=pd.read_csv(OUT/'extra_review/v1/extra.csv').dropna(subset=['Septal_xl','SDp_mm','Septal_auto'])
    truth=d.Septal_xl.to_numpy(int);pred=d.Septal_auto.to_numpy(int);values=d.SDp_mm.to_numpy(float)
    matrix=np.array([[((truth==t)&(pred==p)).sum() for p in (0,1)] for t in (0,1)])
    n1=int((truth==1).sum());n0=int((truth==0).sum());auc=(rankdata(values)[truth==1].sum()-n1*(n1+1)/2)/(n1*n0)
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.6),layout='constrained')
    axes[0].imshow(matrix,cmap='Blues')
    for i in (0,1):
        for j in (0,1):axes[0].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=18,color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    axes[0].set(xticks=[0,1],yticks=[0,1],xlabel='Automatic 0 / 1',ylabel='Excel 0 / 1',title=f'All {len(d)} patients: kappa = {kappa(truth,pred):.2f}')
    axes[1].boxplot([values[truth==0],values[truth==1]],tick_labels=['Excel 0','Excel 1'])
    rng=np.random.default_rng(20261110)
    for pos,key in [(1,0),(2,1)]:
        v=values[truth==key];axes[1].scatter(pos+rng.uniform(-.1,.1,len(v)),v,s=17,alpha=.65)
    axes[1].axhline(2,ls='--',color='red',label='Current threshold: 2 mm');axes[1].legend(fontsize=7)
    axes[1].set(ylabel='Septal deviation (mm)',title=f'Continuous measurement: AUC = {auc:.2f}')
    finish(fig,'binary_statistics.png')
    register(['κ','AUC'],'binary_statistics.png',f'{len(d)}명','실제 비중격 편위 자료의 0/1 일치표(왼쪽)와 연속값 분포(오른쪽)입니다. 전체 환자 예시여서 본문의 평가군 통계와 대상 수가 다를 수 있습니다.')


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    cohort=pd.read_csv(OUT/'cohort.csv');pid=int(cohort.loc[cohort.Include,'Patient'].iloc[0][2:])
    points=repair(landmarks(pid)[0])[0];raw,directions,origin=read_nrrd(CB/f'nrrd/P_{pid}.nrrd')
    protected={str(p):digest(p) for p in OUT.glob('P_*/result.json')}
    print(f'Landmark and dental examples: P_{pid}',flush=True)
    skull_examples(pid,points,raw,directions,origin)
    nasal_and_dental(pid,points,raw,directions,origin)
    other_examples(pid,points,raw,directions,origin)
    eb=pd.read_csv(OUT/'eb_review/v2/eb_candidates.csv');airway=int(eb.loc[eb.method=='vallecula','Patient'].iloc[0][2:])
    print(f'Airway examples: P_{airway}',flush=True)
    airway_examples(airway)
    for key in ('MAE','CCC','r','ρ'):
        manifest[key]=dict(src='scatter/ANB.svg',patient='54명',caption='실제 54명의 ANB 엑셀값–복원값 산점도입니다. 통계 용어의 비교 데이터 예시이며 해부학적 점의 예시는 아닙니다.')
    manifest['QC']=dict(src='../fh_repair/v2/figs/P_37_landmarks.png',patient='P_37',caption='자동 점·보정 후보를 비교하는 실제 검토 화면입니다. QC 통과가 해부학적 정확도 확정을 뜻하지 않습니다.')
    save(DEST/'manifest.json',manifest)
    assert protected=={str(p):digest(p) for p in OUT.glob('P_*/result.json')}
    sources=[CB/f'nrrd/P_{p}.nrrd' for p in (pid,airway)]
    sources += [p for p in (CB/'landmarks').glob('*.mrk.json') if p.name.startswith((f'P_{pid}_',f'P_{airway}_'))]
    sources += [OUT/'phi_ncwr_review/v1/phi_ncwr.csv',OUT/'tdi_review/v1/molars.csv',OUT/'eb_review/v2/eb_candidates.csv',OUT/'pmca_review/v1/pmca_candidates.csv']
    sources += [CB/f'seg/P_{pid}_seg_MAND.nrrd',OUT/'condyle_review/v1/condyle.csv',OUT/'extra_review/v1/extra.csv',CB/f'seg_totalseg/head_glands_cavities/P_{pid}.nii.gz']
    save(DEST/'provenance.json',dict(selection='First included cohort patient; first direct vallecula detection. Not chosen by Excel agreement.',
         automatic_examples_not_expert_ground_truth=True,inputs_sha256={str(p):digest(p) for p in sources},script_sha256=digest(Path(__file__)),
         example_images=len(list(DEST.glob('*.png'))),mapped_terms=len(manifest),original_results_unchanged=True))
    print(f'Generated {len(list(DEST.glob("*.png")))} images for {len(manifest)} terms; original results unchanged')


if __name__=='__main__':main()
