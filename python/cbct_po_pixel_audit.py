"""Exploratory native-voxel and aligned local-patch comparison, one patient at a time.

No inference, landmark replacement, exclusion, or diagnostic bone-density claim.
Reference patients passed distance QC but have no expert-approved Po coordinates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label, map_coordinates
from scipy.spatial.distance import jensenshannon

from cbct_patient_pipeline import CB, OUT, ROOT, digest, landmarks, patient_id, save
from check_landmarks import read_nrrd

AUDIT=OUT/'po_pixel_audit/v1'
CASES={16:'RPo',20:'LPo',37:'RPo'}
RADIUS=12.0
STEP=0.5
AXIS=np.arange(-RADIUS,RADIUS+STEP/2,STEP)
BINS=np.r_[-np.inf,np.arange(-1500.,4001.,100.),np.inf]
METRICS=['median_value','p90_value','airlike_fraction','denselike_fraction',
         'air_components_ge_2mm3','largest_air_component_fraction','gradient_p90_value_per_mm',
         'high_frequency_mad_value','high_ge_3000_fraction','clipped_high_fraction',
         'fov_margin_mm']
LABELS=['Median gray value','P90 gray value','Fraction < -500','Fraction > 300',
        'Air components >=2 mm3','Largest air component fraction','Gradient P90 / mm',
        'High-frequency residual MAD','Fraction >=3000','Fraction >= model clip high',
        'Distance to image boundary (mm)']
MATCH=['A_ManufacturerModelName','A_KVP','A_XRayTubeCurrent','A_ExposureTime',
       'A_voxel_mm','A_FOV_x_mm','A_FOV_z_mm']


def frame(points, side):
    mid=np.array([points[k] for k in ['N','S','Ba','ANS','PNS']])
    left=np.linalg.svd(mid-mid.mean(0))[2][-1]
    if left[0]<0:left=-left
    anterior=points['N']-points['S']
    anterior=anterior-left*np.dot(anterior,left)
    anterior=anterior/np.linalg.norm(anterior)
    superior=np.cross(anterior,left)
    if superior[2]<0:raise ValueError('Unexpected cranial frame orientation')
    outward=left if side=='LPo' else -left
    # Columns map outward/anterior/superior coordinates into patient LPS.
    return np.column_stack([outward,anterior,superior])


def aligned_patch(volume,directions,origin,center,basis):
    k,j,i=np.meshgrid(AXIS,AXIS,AXIS,indexing='ij') # array (sup,ant,out)
    local=np.stack([i.ravel(),j.ravel(),k.ravel()])
    physical=center[:,None]+basis@local
    index=np.linalg.solve(directions.T,physical-origin[:,None])
    sampled=map_coordinates(volume,index[::-1],order=1,output=np.float32,
                            mode='constant',cval=np.nan,prefilter=False)
    return sampled.reshape(i.shape)


def model_clip_bounds(volume):
    # Current ALI _prepare_one_patient uses CorrectHisto(..., .01, .99).
    # Historical cache provenance is not known; these are current-code proxies.
    counts,edges=np.histogram(volume,bins=1000)
    cdf=np.cumsum(counts).astype(float);cdf-=cdf.min();cdf/=cdf.max()
    low=edges[np.flatnonzero(cdf>.01)[0]]
    high=edges[np.flatnonzero(cdf>.99)[0]]
    return max(float(low),-1500.),min(float(high),4000.)


def native_stats(volume,directions,origin,center,clip):
    if not np.allclose(directions,np.diag(np.diag(directions))) or np.any(np.diag(directions)<=0):
        raise ValueError('Native audit requires positive axis-aligned LPS data')
    spacing=np.diag(directions)
    c=np.linalg.solve(directions.T,center-origin)
    n=np.array(volume.shape[::-1])
    margin=float(np.min(np.r_[c*spacing,(n-1-c)*spacing]))
    lo=np.maximum(0,np.floor(c-(RADIUS+3)/spacing).astype(int))
    hi=np.minimum(n,np.ceil(c+(RADIUS+3)/spacing).astype(int)+1)
    crop=volume[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]].astype(np.float32)
    z,y,x=np.ogrid[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]
    mask=((x-c[0])*spacing[0])**2+((y-c[1])*spacing[1])**2+((z-c[2])*spacing[2])**2<=RADIUS**2
    values=crop[mask]
    if not len(values):raise ValueError('Empty native ROI')
    grad=np.sqrt(sum(g*g for g in np.gradient(crop,*spacing[::-1])))
    residual=crop-gaussian_filter(crop,sigma=0.6/spacing[::-1])
    air=(crop < -500)&mask
    components,_=label(air) # 6-neighbor connectivity, fixed for every patient
    sizes=np.bincount(components.ravel())[1:]
    big=sizes*np.prod(spacing)>=2.0
    hist=np.histogram(values,bins=BINS)[0].astype(float);hist/=hist.sum()
    metrics={
        'native_voxels':int(len(values)),'native_roi_mm3':float(len(values)*np.prod(spacing)),
        'median_value':float(np.median(values)),'p10_value':float(np.percentile(values,10)),
        'p90_value':float(np.percentile(values,90)),
        'airlike_fraction':float(np.mean(values < -500)),
        'denselike_fraction':float(np.mean(values > 300)),
        'air_components_ge_2mm3':int(big.sum()),
        'largest_air_component_fraction':float(sizes.max()/air.sum()) if air.any() else 0.,
        'gradient_p90_value_per_mm':float(np.percentile(grad[mask],90)),
        'high_frequency_mad_value':float(np.median(abs(residual[mask]-np.median(residual[mask])))),
        'high_ge_3000_fraction':float(np.mean(values>=3000)),
        'clipped_high_fraction':float(np.mean(values>=clip[1])),
        'fov_margin_mm':margin,'roi_complete':bool(margin>=RADIUS),
    }
    return metrics,hist


def metadata():
    return pd.read_csv(ROOT/'Result_py/cbct_eda/cbct_eda_patient.csv').set_index('Patient')


def included():
    cohort=pd.read_csv(OUT/'cohort.csv',dtype=str,keep_default_na=False)
    if cohort.Patient.duplicated().any():raise ValueError('Duplicate cohort patient')
    if not cohort.Include.str.lower().isin(['true','false','1','0']).all():raise ValueError('Invalid Include')
    return [patient_id(p) for p in cohort.loc[cohort.Include.str.lower().isin(['true','1']),'Patient']]


def process(pid):
    if pid not in included():raise ValueError('Patient excluded from cohort')
    result_path=OUT/f'P_{pid}/result.json'
    if not result_path.exists():raise ValueError('Patient measurement result missing')
    baseline=json.loads(result_path.read_text(encoding='utf-8'))
    points,files=landmarks(pid)
    scan=CB/f'nrrd/P_{pid}.nrrd'
    scan_hash=digest(scan)
    if scan_hash!=baseline['fingerprint']['nrrd_sha256']:raise ValueError('Image changed since measurement')
    for f in files:
        if digest(f)!=baseline['fingerprint']['landmarks'][f.name]:raise ValueError('Landmarks changed since measurement')
    evidence_path=OUT/f'P_{pid}/review/{CASES.get(pid,"RPo")}_evidence.json'
    evidence=None
    if pid in CASES:
        evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
        if evidence['image_sha256']!=scan_hash or evidence['patient']!=f'P_{pid}':raise ValueError('Candidate provenance mismatch')
    fingerprint={'scan':scan_hash,'landmarks':baseline['fingerprint']['landmarks'],
                 'candidate':digest(evidence_path) if evidence else None,
                 'code':digest(Path(__file__)),'cohort':digest(OUT/'cohort.csv'),
                 'baseline':digest(result_path),'eda':digest(ROOT/'Result_py/cbct_eda/cbct_eda_patient.csv')}
    folder=AUDIT/'patients'/f'P_{pid}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'metrics.json';patch_path=folder/'patches.npz'
    if path.exists() and patch_path.exists():
        cache=json.loads(path.read_text(encoding='utf-8'))
        if cache.get('fingerprint')==fingerprint and cache.get('patch_sha256')==digest(patch_path):
            print(f'P_{pid}: reused',flush=True);return cache
    volume,directions,origin=read_nrrd(scan)
    clip=model_clip_bounds(volume)
    rows=[];patches={}
    for side in ['RPo','LPo']:
        variants=[('automatic',points[side])]
        if evidence and CASES[pid]==side:
            variants.append(('candidate',np.array(evidence['candidate']['position_lps_mm'])))
        basis=frame(points,side)
        for variant,center in variants:
            metrics,hist=native_stats(volume,directions,origin,center,clip)
            patch=aligned_patch(volume,directions,origin,center,basis)
            metrics.update({'Patient':f'P_{pid}','Side':side,'Variant':variant,
                            'center_lps_mm':center.tolist(),'basis_lps_columns':basis.tolist(),
                            'histogram_probability':hist.tolist(),
                            'patch_finite_fraction':float(np.isfinite(patch).mean())})
            rows.append(metrics);patches[f'{side}_{variant}']=patch
    del volume
    tmp=folder/'patches.tmp.npz';np.savez_compressed(tmp,**patches);tmp.replace(patch_path)
    cache={'patient':f'P_{pid}','fingerprint':fingerprint,'rows':rows,
           'current_code_histogram_clip_proxy':list(clip),
           'patch_sha256':digest(patch_path),'coordinate_system':'LPS',
           'automatic_qc_status':baseline['variants']['automatic']['release_status']}
    save(path,cache);print(f'P_{pid}: saved {len(rows)} ROIs',flush=True)
    return cache


def same_protocol(a,b):
    for key in MATCH:
        if pd.isna(a[key]) or pd.isna(b[key]):return False
        if key=='A_ManufacturerModelName':
            if a[key]!=b[key]:return False
        elif not np.isclose(float(a[key]),float(b[key]),rtol=0,atol=1e-5):return False
    return True


def one_row(cache,side,variant):
    return next(row for row in cache['rows'] if row['Side']==side and row['Variant']==variant)


def report(caches):
    meta=metadata();reference=[pid for pid,c in caches.items() if c['automatic_qc_status']=='qc_pass_pending_visual_review' and pid not in CASES]
    comparisons=[];case_sections=[]
    for pid,side in CASES.items():
        if pid not in caches:continue
        refs=[p for p in reference if same_protocol(meta.loc[f'P_{pid}'],meta.loc[f'P_{p}'])]
        if len(refs)<10:raise ValueError('Too few acquisition-matched reference patients')
        rows=[one_row(caches[p],side,'automatic') for p in refs]
        automatic=one_row(caches[pid],side,'automatic');candidate=one_row(caches[pid],side,'candidate')
        hist=np.array([r['histogram_probability'] for r in rows]);mid_hist=np.median(hist,axis=0);mid_hist/=mid_hist.sum()
        fig,axs=plt.subplots(1,2,figsize=(18,6))
        centers=np.r_[-1600.,np.arange(-1450.,4000.,100.),4100.]
        med=np.median(hist,axis=0);low,high=np.percentile(hist,[5,95],axis=0)
        axs[0].fill_between(centers,low,high,color='#cbd5e1',alpha=.7,label='Reference patient curves: 5-95%')
        axs[0].plot(centers,med,color='#334155',label=f'Reference median (N={len(refs)})')
        for row,color,name in [(automatic,'#dc2626','ALI location'),(candidate,'#15803d','Image candidate')]:
            axs[0].plot(centers,row['histogram_probability'],color=color,lw=2,label=name)
        axs[0].set(xlabel='Native rescaled DICOM gray value (not calibrated density)',ylabel='Probability per gray-value bin',xlim=(-1600,4100))
        axs[0].legend(fontsize=9);axs[0].grid(alpha=.2)
        x=np.arange(len(METRICS))
        for row,color,name,offset in [(automatic,'#dc2626','ALI location',-.12),(candidate,'#15803d','Image candidate',.12)]:
            ranks=[100*np.mean(np.array([r[m] for r in rows])<row[m])+50*np.mean(np.array([r[m] for r in rows])==row[m]) for m in METRICS]
            axs[1].scatter(ranks,x+offset,color=color,s=45,label=name)
        axs[1].axvspan(5,95,color='#e2e8f0',alpha=.6);axs[1].axvline(50,color='gray',ls='--')
        axs[1].set_yticks(x,LABELS);axs[1].invert_yaxis();axs[1].set(xlim=(-2,102),xlabel='Descriptive reference-patient percentile (not a p-value)')
        axs[1].legend(fontsize=9)
        fig.suptitle(f'P_{pid} {side}: native 12 mm sphere, protocol-matched QC-pass provisional references')
        fig.tight_layout();fig.savefig(AUDIT/f'P_{pid}_{side}_distributions.png',dpi=150);plt.close(fig)
        ref_patches=[]
        for p in refs:
            with np.load(AUDIT/f'patients/P_{p}/patches.npz') as z:
                ref_patches.append(z[f'{side}_automatic'].copy())
        stack=np.array(ref_patches)
        with np.load(AUDIT/f'patients/P_{pid}/patches.npz') as z:
            ca=z[f'{side}_candidate'].copy();au=z[f'{side}_automatic'].copy()
        mid=len(AXIS)//2
        reference_image=np.nanmedian(stack,axis=0)
        air_probability=np.nanmean(np.where(np.isfinite(stack),(stack < -500).astype(float),np.nan),axis=0)
        fig,axs=plt.subplots(3,4,figsize=(18,14))
        def plane(a,v):return [a[mid],a[:,mid,:],a[:,:,mid]][v]
        for row,(patch,name) in enumerate([(au,'ALI location'),(ca,'Image candidate'),(reference_image,f'Reference median N={len(refs)}')]):
            for col,view in enumerate(['axial','coronal','sagittal']):
                axs[row,col].imshow(plane(patch,col),origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=[-RADIUS,RADIUS,-RADIUS,RADIUS])
                axs[row,col].plot(0,0,'+',color='red' if row==0 else 'lime',ms=12)
                axs[row,col].set_title(name+' | '+view);axs[row,col].set_xlabel(['outward','outward','anterior'][col]+' mm');axs[row,col].set_ylabel(['anterior','superior','superior'][col]+' mm')
        diff=ca-reference_image
        im=axs[0,3].imshow(plane(diff,2),origin='lower',cmap='coolwarm',vmin=-1500,vmax=1500,extent=[-12,12,-12,12]);axs[0,3].set_title('Candidate minus reference median\nSagittal; approximate correspondence');fig.colorbar(im,ax=axs[0,3],shrink=.7)
        im=axs[1,3].imshow(plane(air_probability,2),origin='lower',cmap='viridis',vmin=0,vmax=1,extent=[-12,12,-12,12]);axs[1,3].set_title('Reference airlike probability\nSagittal, < -500');fig.colorbar(im,ax=axs[1,3],shrink=.7)
        im=axs[2,3].imshow(plane((ca < -500).astype(float)-air_probability,2),origin='lower',cmap='coolwarm',vmin=-1,vmax=1,extent=[-12,12,-12,12]);axs[2,3].set_title('Candidate air mask minus reference probability');fig.colorbar(im,ax=axs[2,3],shrink=.7)
        fig.suptitle(f'P_{pid} {side} | 24 mm cube, 0.5 mm trilinear grid; Po-free N-S/midline orientation\nEach row centered on its own point. Reference median is a composite, not an individual anatomical image.')
        fig.tight_layout();fig.savefig(AUDIT/f'P_{pid}_{side}_patches.png',dpi=150);plt.close(fig)
        # Fixed IDs chosen before looking at the case histograms; real reference
        # examples reveal how much detail the cohort-median image blurs away.
        examples=[p for p in [5,11,28] if p in refs]
        fig,axs=plt.subplots(1,len(examples)+1,figsize=(6*(len(examples)+1),6),squeeze=False)
        for ax,patch,name in zip(axs.flat,[ca]+[ref_patches[refs.index(p)] for p in examples],[f'P_{pid} candidate']+[f'P_{p} automatic (provisional)' for p in examples]):
            ax.imshow(plane(patch,2),origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=[-12,12,-12,12]);ax.plot(0,0,'+',color='lime',ms=12);ax.set_title(name);ax.set_xlabel('anterior mm');ax.set_ylabel('superior mm')
        fig.suptitle(f'{side}: actual sagittal patches, identical physical scale and window')
        fig.tight_layout();fig.savefig(AUDIT/f'P_{pid}_{side}_examples.png',dpi=150);plt.close(fig)
        table=[]
        for metric in METRICS:
            values=np.array([r[metric] for r in rows]);q=np.percentile(values,[5,50,95])
            record={'Patient':f'P_{pid}','Side':side,'Metric':metric,'Reference_N':len(refs),
                    'Reference_p05':q[0],'Reference_median':q[1],'Reference_p95':q[2],
                    'Automatic':automatic[metric],'Candidate':candidate[metric],
                    'Candidate_percentile':100*(np.mean(values<candidate[metric])+.5*np.mean(values==candidate[metric]))}
            comparisons.append(record)
            table.append(f"<tr><td>{escape(metric)}</td><td>{automatic[metric]:.3f}</td><td>{candidate[metric]:.3f}</td><td>{q[1]:.3f}</td><td>{q[0]:.3f}–{q[2]:.3f}</td><td>{record['Candidate_percentile']:.1f}</td></tr>")
        js_auto=float(jensenshannon(automatic['histogram_probability'],mid_hist,base=2))
        js_candidate=float(jensenshannon(candidate['histogram_probability'],mid_hist,base=2))
        save(AUDIT/f'P_{pid}_{side}_comparison.json',{'patient':f'P_{pid}','side':side,'reference_ids':[f'P_{p}' for p in refs],
             'protocol_match_fields':MATCH,'reference_status':'distance-QC pass, NOT expert-validated normal or correct Po',
             'native_histogram_JS_distance_to_reference_median':{'automatic':js_auto,'candidate':js_candidate},
             'reference_source_fingerprints':{f'P_{p}':caches[p]['fingerprint'] for p in refs},
             'case_source_fingerprint':caches[pid]['fingerprint'],'cause_status':'exploratory association; no search trajectories or adjudicated ground truth'})
        images=''.join(f'<h3>{title}</h3><a href="P_{pid}_{side}_{suffix}.png"><img src="P_{pid}_{side}_{suffix}.png"></a>' for suffix,title in [('examples','실제 비교 환자'),('distributions','원본 복셀 분포와 환자별 순위'),('patches','정렬 단면과 분포 차이 지도')])
        case_sections.append(f'''<section id="P_{pid}"><h2>P_{pid} · {side}</h2><p>같은 쪽, 촬영 조건이 일치하는 거리 QC 통과 환자 {len(refs)}명과 비교했습니다. 참고군은 전문가가 확정한 정상군이 아닙니다. 히스토그램 Jensen–Shannon 거리: 자동 위치 {js_auto:.3f}, 영상 후보 {js_candidate:.3f}. 값이 작을수록 밝기 분포가 비슷하며, 해부학적 정확도를 보증하지 않습니다.</p>
<p><a href="../../P_{pid}/review/{side}_review.html">원본 영상·수정 후보·엑셀 비교</a> · <a href="P_{pid}_{side}_comparison.json">참고 환자 명단·근거</a></p>
<table><tr><th>지표</th><th>자동 위치</th><th>영상 후보</th><th>참고군 중앙값</th><th>참고군 5–95백분위</th><th>후보 백분위</th></tr>{''.join(table)}</table>{images}</section>''')
        del stack,ref_patches
    pd.DataFrame(comparisons).to_csv(AUDIT/'case_metrics_comparison.csv',index=False,encoding='utf-8-sig')
    all_rows=[{k:v for k,v in row.items() if k not in ['histogram_probability','basis_lps_columns','center_lps_mm']} for c in caches.values() for row in c['rows']]
    pd.DataFrame(all_rows).to_csv(AUDIT/'all_roi_metrics.csv',index=False,encoding='utf-8-sig')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Po 오류 부위 복셀 분포 비교</title>
<style>body{font:17px system-ui,sans-serif;max-width:1700px;margin:32px auto;padding:0 25px;color:#17222c;line-height:1.6}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}.note{background:#fff0d5;padding:20px}section{margin-top:60px}a{color:#0759a8}</style>
<h1>Po 오류 부위: 원본 복셀 분포와 구조 비교</h1><p><a href="#P_16">P_16</a> · <a href="#P_20">P_20</a> · <a href="#P_37">P_37</a> · <a href="case_metrics_comparison.csv">비교 CSV</a></p>
<p class="note">탐색 분석입니다. 세 환자의 후보 좌표와 참고군 Po 모두 전문가 확정 전입니다. 픽셀 분포 차이는 검출 실패 원인에 대한 단서를 주지만 원인을 확정하지 않습니다. 환자별 숫자를 비교하며 복셀 수를 독립 표본 수로 취급하지 않았습니다.</p>
<h2>비교 방법</h2><p>문의 32명 제외 명단에서 기존 계측이 있는 54명을 환자 한 명씩 처리했습니다. 거리 QC 통과 48명을 잠정 참고군으로 두고 같은 쪽 Po 및 장비·관전압·관전류·노출시간·복셀 크기·FOV가 일치하는 환자를 비교했습니다. 엑셀 오차로 참고군을 고르지 않았습니다.</p>
<p>히스토그램과 지표는 반경 12 mm 구 안의 원본 NRRD 복셀값으로 계산합니다. 그림은 24 mm 정육면체를 0.5 mm 간격으로 보간하고, N·S·Ba·ANS·PNS의 정중면과 S→N 방향으로 자세를 맞췄습니다. 양쪽은 바깥쪽 방향을 통일했습니다. 위치·자세 정렬만 했으며 서로 다른 환자의 개별 뼈 경계가 픽셀마다 정확히 대응하지 않습니다.</p>
<p>−500/300/3000은 고정한 탐색 임계값으로, 공기 유사·고밝기 영역 비율을 기술합니다. 골밀도나 금속 진단 기준이 아닙니다. 원본은 rescale 적용 DICOM 값이며 CBCT의 절대 HU 정확도는 검증되지 않았습니다. 촬영 조건을 맞춘 이유는 <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5863017/">FOV에 따른 gray value 변동을 확인한 원저 연구</a>를 참고했습니다.</p>
<p>현재 ALI 코드는 전역 히스토그램 1–99% 근처에서 밝기를 잘라내고 1/0.3 mm로 재표본화한 뒤, 각 64³ 패치를 −1~1로 정규화합니다. 본 자료의 clipped-high 비율은 현재 코드로 계산한 근사 지표이며 과거 추론 캐시와 실행 조건은 확정되지 않았습니다. 원본 밝기 차이가 그대로 모델 입력 차이가 되는 것은 아닙니다. 고주파 잔차·경사도는 구조와 잡음이 섞인 지표로 움직임/잡음을 확진하지 않습니다.</p>
<p>그림의 5–95백분위는 환자별 분포이며 통계적 신뢰구간이나 정상 판정 범위가 아닙니다. 3명으로 유의성 검정·오류 예측 모델을 만들지 않았습니다. 원인 확정에는 전문가 좌표, 모델 탐색 경로·시드·종료 상태, 분리된 검증 사례가 필요합니다.</p>'''+''.join(case_sections)+'</html>'
    (AUDIT/'index.html').write_text(html,encoding='utf-8')
    save(AUDIT/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'patients':sorted(caches),'reference_ids':reference,
         'case_sides':CASES,'native_radius_mm':RADIUS,'patch_step_mm':STEP,'missing_selected':[p for p in included() if p not in caches],
         'cohort_sha256':digest(OUT/'cohort.csv'),'script_sha256':digest(Path(__file__)),
         'reference_selection':'distance QC pass; no Excel error thresholds','scope':'exploratory; not validated cause or final corrected measurement'})
    print(f'Report: {AUDIT / "index.html"}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--patient',type=patient_id)
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if args.patient:
        process(args.patient);return
    if not args.run:
        print('Use --patient N to cache one selected patient, or --run for sequential selected cohort + report.');return
    AUDIT.mkdir(parents=True,exist_ok=True)
    caches={};failures=[]
    for pid in included():
        try:caches[pid]=process(pid)
        except Exception as error:
            failures.append({'Patient':f'P_{pid}','Error':str(error)})
            print(f'P_{pid}: {error}',flush=True)
    save(AUDIT/'failures.json',failures)
    report(caches)


if __name__=='__main__':main()
