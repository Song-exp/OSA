"""환자별 재개 가능한 CBCT 계측. 기존 평가 결과와 분리, QC 실패 값은 배포하지 않음."""
from __future__ import annotations
import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pandas as pd
from batch_cbct_pipeline import LMS, STRUCTS
from eval_measure import QC_PAIRS
from osa.ceph import measures
from osa.cbct import _read
from pilot_measure import excel

ROOT=Path(__file__).resolve().parents[1]
CB=ROOT/"Result_py/CBCT"
OUT=CB/"patient_pipeline/v2"
PAIRS={"ANB":"ANB","Nperp_A":"Nperp_A","Nperp_Pog":"Nperp_Pog","MPFH_GoMe":"MPFH"}


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def patient_id(value):
    m=re.fullmatch(r'(?:P_)?([1-9]\d*)',str(value).strip())
    if not m:raise ValueError(f'잘못된 환자번호: {value}')
    return int(m.group(1))


def source_signature(pid):
    folders=list((ROOT/'Data/CBCT').glob(f'P_{pid}_*'))
    if len(folders)!=1:raise ValueError('원본 환자 폴더 없음/중복')
    files=sorted(folders[0].glob('*.dcm'))
    if not files:raise ValueError('상위 DICOM 없음')
    listing=[(f.name,f.stat().st_size,f.stat().st_mtime_ns) for f in files]
    h=_read(files[0],True)
    return {'folder':folders[0].name,'files':len(files),'series_uid':str(h.SeriesInstanceUID),
            'file_stat_sha256':hashlib.sha256(json.dumps(listing).encode()).hexdigest()}


def landmarks(pid):
    files=sorted((CB/'landmarks').glob(f'P_{pid}_*.mrk.json'))
    if not files:raise ValueError('랜드마크 파일 없음')
    points={}
    for f in files:
        for markup in json.loads(f.read_text(encoding='utf-8'))['markups']:
            system=markup.get('coordinateSystem')
            if system not in ('LPS','RAS'):raise ValueError(f'좌표계 미확인: {f.name}')
            for control in markup['controlPoints']:
                if control.get('positionStatus','defined')!='defined':raise ValueError('미확정 랜드마크 위치')
                point=np.asarray(control['position'],dtype=float)
                if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('유효하지 않은 좌표')
                if system=='RAS':point=point*np.array([-1,-1,1])
                label=control['label']
                if label in points and not np.allclose(points[label],point,atol=1e-6,rtol=0):
                    raise ValueError(f'중복 랜드마크 좌표 충돌: {label}')
                points[label]=point
    missing=set(LMS)-points.keys()
    if missing:raise ValueError('랜드마크 누락: '+', '.join(sorted(missing)))
    return points,files


def valid_nrrd(path):
    """헤더·gzip 끝의 크기 점검. CRC/분할 정확도 검증을 대신하지 않는다."""
    with Path(path).open('rb') as f:
        head=f.read(8192).replace(b'\r\n',b'\n')
        if not head.startswith(b'NRRD') or b'\n\n' not in head:raise ValueError(f'NRRD 헤더 오류: {path}')
        meta={k.strip():v.strip() for line in head.split(b'\n\n')[0].decode('ascii').splitlines() if ':' in line and not line.startswith('#') for k,v in [line.split(':',1)]}
        sizes=[int(v) for v in meta['sizes'].split()]
        if len(sizes)!=3 or min(sizes)<=0:raise ValueError('NRRD 크기 오류')
        widths={'short':2,'short int':2,'int16':2,'unsigned char':1,'uchar':1,'uint8':1,'unsigned short':2,'uint16':2,'float':4}
        if meta['type'] not in widths:raise ValueError('NRRD type 미지원')
        if meta['encoding']!='gzip':raise ValueError('NRRD gzip만 지원')
        f.seek(-4,2);actual=int.from_bytes(f.read(4),'little')
        expected=int(np.prod(sizes))*widths[meta['type']]
        if actual!=expected%(2**32):raise ValueError(f'NRRD 잘림/크기 오류: {path}')
    return sizes


def freeze_qc(ids,path):
    if path.exists():raise ValueError('기존 QC 기준은 덮어쓰지 않습니다. 새 버전 파일을 지정하세요.')
    if len(ids)<10:raise ValueError('QC 기준 작성에는 최소 10명 필요')
    rows=[];sources={}
    for pid in ids:
        p,files=landmarks(pid)
        rows.append([np.linalg.norm(p[a]-p[b]) for a,b in QC_PAIRS])
        sources[f'P_{pid}']={f.name:digest(f) for f in files}
    a=np.asarray(rows);med=np.median(a,axis=0);mad=np.median(abs(a-med),axis=0)
    save(path,{'version':'distance-mad-v1','cutoff':4.0,'patients':[f'P_{p}' for p in ids],
               'pairs':[list(x) for x in QC_PAIRS],'median':med.tolist(),'scale':(1.4826*mad).tolist(),
               'sources':sources,'note':'결과변수/엑셀 오차를 사용하지 않는 거리 QC. 임상 정확도 보증이나 자동 환자 제외 기준 아님'})


def qc(points,profile):
    if profile.get('version')!='distance-mad-v1':raise ValueError('QC 기준 버전 오류')
    pairs=profile.get('pairs',[]);medians=profile.get('median',[]);scales=profile.get('scale',[])
    if len(pairs)!=len(QC_PAIRS) or len(medians)!=len(pairs) or len(scales)!=len(pairs):raise ValueError('QC 기준 항목 수 오류')
    if [tuple(p) for p in pairs]!=QC_PAIRS:raise ValueError('QC 거리 정의 불일치')
    if not np.isfinite(medians).all() or not np.isfinite(scales).all() or min(scales)<0:raise ValueError('QC 기준 수치 오류')
    if not np.isfinite(profile.get('cutoff',float('nan'))) or profile['cutoff']<=0:raise ValueError('QC cutoff 오류')
    flags=[]
    for (a,b),med,scale in zip(profile['pairs'],profile['median'],profile['scale']):
        distance=float(np.linalg.norm(points[a]-points[b]))
        z=abs(distance-med)/scale if scale>1e-8 else (0 if abs(distance-med)<1e-8 else float('inf'))
        if z>profile['cutoff']:flags.append({'pair':f'{a}-{b}','distance_mm':distance,'z':float(z) if np.isfinite(z) else None})
    # 같은 점·일직선으로 퇴화한 FH와 MSP를 먼저 차단한다.
    u=points['LPo']-points['RPo'];v=points['ROr']-points['RPo']
    if np.linalg.norm(np.cross(u,v))<1e-6:flags.append({'pair':'FH','reason':'평면 퇴화'})
    return flags


def corrections(points,path,pid):
    if path is None:return dict(points),[]
    d=json.loads(path.read_text(encoding='utf-8'))
    if d.get('patient')!=f'P_{pid}' or d.get('coordinate_system')!='LPS':raise ValueError('수정 좌표 환자/좌표계 불일치')
    if not str(d.get('reviewer','')).strip() or not str(d.get('reason','')).strip():raise ValueError('검수자·수정 근거 필수')
    out=dict(points)
    for label,value in d['points'].items():
        a=np.asarray(value,float)
        if label not in points or a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('수정 좌표 오류')
        out[label]=a
    return out,sorted(d['points'])


def execute(pid,reference,allow_inference=False,correction=None):
    case=OUT/f'P_{pid}';case.mkdir(parents=True,exist_ok=True)
    state={'patient':f'P_{pid}','method':'jung-msp-gome-v2','status':'running','started':datetime.now().isoformat(),'steps':{}}
    save(case/'status.json',state)
    nrrd=CB/f'nrrd/P_{pid}.nrrd'
    stages=[('nrrd',[nrrd],['export_nrrd.py',str(pid)]),
            ('ali',[],['run_ali.py',str(pid),'--lm',*LMS]),
            ('segmentation',[CB/f'seg/P_{pid}_seg_{s}.nrrd' for s in STRUCTS],['run_amasss.py',str(pid),'--struct',','.join(STRUCTS)])]
    try:
        source=source_signature(pid)
        prior=case/'result.json'
        if prior.exists() and json.loads(prior.read_text(encoding='utf-8')).get('fingerprint',{}).get('source') not in (None,source):
            raise ValueError('원본 DICOM 변경 감지: 이전 파생물 재사용 금지. 새 영상·결과 버전으로 분리 필요')
        for name,files,args in stages:
            try:
                if name=='ali':landmarks(pid)
                else:
                    for f in files:valid_nrrd(f)
                state['steps'][name]='reused_existing_artifact'
            except (ValueError,OSError,KeyError,json.JSONDecodeError):
                if not allow_inference:raise ValueError(f'{name} 결과 없음/불완전. 추론 허용 실행 또는 원인 확인 필요')
                # 단계 실패 시 다음 환자로 넘어가되 같은 환자의 후속 단계는 실행하지 않음.
                result=subprocess.run([sys.executable,*args],cwd=ROOT/'python',capture_output=True,text=True,encoding='utf-8',errors='replace')
                (case/f'{name}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
                if result.returncode:raise RuntimeError(f'{name} 실패: {case/name}.log')
                if name=='ali':landmarks(pid)
                else:
                    for f in files:valid_nrrd(f)
                state['steps'][name]='computed'
            save(case/'status.json',state)
        original,files=landmarks(pid)
        corrected,changed=corrections(original,correction,pid)
        profile=json.loads(reference.read_text(encoding='utf-8'))
        fingerprint={'source':source,'nrrd_sha256':digest(nrrd),
                     'segmentation':{s:{'size':(CB/f'seg/P_{pid}_seg_{s}.nrrd').stat().st_size,
                                        'mtime_ns':(CB/f'seg/P_{pid}_seg_{s}.nrrd').stat().st_mtime_ns} for s in STRUCTS},
                     'landmarks':{f.name:digest(f) for f in files},'qc_reference':digest(reference),
                     'ceph_code':digest(ROOT/'python/osa/ceph.py'),'pipeline_code':digest(Path(__file__)),
                     'excel':digest(ROOT/'Patient_Info_Data_87명_261019.xlsx'),
                     'corrections':digest(correction) if correction else None}
        if prior.exists() and json.loads(prior.read_text(encoding='utf-8')).get('fingerprint')==fingerprint:
            state.update(status='cached',result=str(prior));save(case/'status.json',state);return json.loads(prior.read_text(encoding='utf-8'))
        variants={};xl=excel().loc[pid]
        for label,p in [('automatic',original)]+([('reviewed',corrected)] if changed else []):
            flags=qc(p,profile)
            try:
                values={k:float(v) for k,v in measures(p,'jung').items()}
                if not all(np.isfinite(list(values.values()))):raise ValueError('유한하지 않은 계측값')
            except (ValueError,ArithmeticError,np.linalg.LinAlgError):
                values={};flags.append({'pair':'measurement','reason':'기하 계산 실패'})
            variants[label]={'measurements_diagnostic':values,'qc_flags':flags,
                             'release_status':'needs_review' if flags else 'qc_pass_pending_visual_review',
                             'excel_comparison':{key:{'auto':values.get(key),'excel':float(xl[col]) if pd.notna(xl[col]) else None,
                                 'difference':values[key]-float(xl[col]) if key in values and pd.notna(xl[col]) else None} for key,col in PAIRS.items()}}
        result={'patient':f'P_{pid}','fingerprint':fingerprint,'variants':variants,'corrected_points':changed,
                'inference_provenance':'existing artifacts without historical stage receipts remain legacy/unverified; current model use is recorded only for newly computed stages',
                'implemented_variables':list(PAIRS),'not_implemented':['TDI','PHI','NCWR','NAV','PAV','PMCA','PAL'],
                'historical_evaluation':'Result_py/eval is not modified; new reviewed results are not an independent accuracy validation'}
        if prior.exists():
            previous=json.loads(prior.read_text(encoding='utf-8'))
            key=hashlib.sha256(json.dumps(previous.get('fingerprint',{}),sort_keys=True).encode()).hexdigest()[:16]
            save(case/f'history/{key}.json',previous)
        save(prior,result);state.update(status='completed',result=str(prior));save(case/'status.json',state)
        return result
    except Exception as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}');save(case/'status.json',state);raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--ids',nargs='+')
    group.add_argument('--manifest',type=Path,help='CSV: Patient,Include,Reason. Include가 true/1인 환자만 순서대로 처리')
    ap.add_argument('--qc-reference',type=Path,default=OUT/'qc_reference.json')
    ap.add_argument('--freeze-qc',action='store_true',help='명시한 ids로 QC 기준 생성 후 종료. 기존 기준 덮어쓰기 금지')
    ap.add_argument('--corrections-dir',type=Path)
    ap.add_argument('--run',action='store_true',help='없으면 실행 계획만 출력')
    ap.add_argument('--infer-missing',action='store_true')
    a=ap.parse_args()
    if a.ids:ids=[patient_id(p) for p in a.ids]
    else:
        manifest=pd.read_csv(a.manifest,dtype={'Patient':str,'Include':str}).fillna('')
        if not {'Patient','Include','Reason'}<=set(manifest):ap.error('manifest 필수 열: Patient,Include,Reason')
        if manifest.Patient.duplicated().any():ap.error('manifest 환자 중복')
        if not manifest.Include.str.lower().isin(['true','false','1','0']).all():ap.error('Include는 true/false/1/0만 허용')
        ids=[patient_id(p) for p in manifest.loc[manifest.Include.str.lower().isin(['true','1']),'Patient']]
    if len(ids)!=len(set(ids)):ap.error('환자번호 중복')
    if a.freeze_qc:freeze_qc(ids,a.qc_reference);print(f'QC 기준 고정: {a.qc_reference}');return
    if not a.run:print(json.dumps({'patients':[f'P_{p}' for p in ids],'sequential':True,'run':False},ensure_ascii=False));return
    if not a.qc_reference.is_file():ap.error('먼저 명시한 개발 표본으로 --freeze-qc 또는 --qc-reference를 지정하세요.')
    rows=[]
    for pid in ids:
        correction=a.corrections_dir/f'P_{pid}.json' if a.corrections_dir else None
        if correction and not correction.exists():correction=None
        try:
            r=execute(pid,a.qc_reference,a.infer_missing,correction)
            rows.append({'Patient':f'P_{pid}','Status':'completed','QC':r['variants'].get('reviewed',r['variants']['automatic'])['release_status']})
        except Exception as e:rows.append({'Patient':f'P_{pid}','Status':'failed','Error':str(e)})
        print(rows[-1],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'last_run.csv',index=False,encoding='utf-8-sig')
    compared=[]
    for pid in ids:
        file=OUT/f'P_{pid}/result.json'
        if not file.is_file() or next(row for row in rows if row['Patient']==f'P_{pid}')['Status']!='completed':continue
        result=json.loads(file.read_text(encoding='utf-8'))
        for variant,data in result['variants'].items():
            for variable,values in data['excel_comparison'].items():
                compared.append({'Patient':f'P_{pid}','Variant':variant,'Variable':variable,
                                 'QC':data['release_status'],**values})
    pd.DataFrame(compared).to_csv(OUT/'selected_measures_diagnostic.csv',index=False,encoding='utf-8-sig')
    if a.manifest:
        save(OUT/'last_selection.json',{'manifest_sha256':digest(a.manifest),'patients':[f'P_{p}' for p in ids]})
    if any(r['Status']=='failed' for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
