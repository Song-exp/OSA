"""복구가 발동한 모든 Po를 자기 위치의 실제 직교 단면에서 대조한다."""
from html import escape
import json
import re

import numpy as np
import pandas as pd

from cbct_po_tube85 import DEST,PREV
from cbct_patient_pipeline import CB,landmarks,digest
from cbct_fh_repair import repair
from cbct_seg_landmarks85 import load_mask,save
from check_landmarks import read_nrrd


def native_panel(ax,volume,mask,dirs,origin,point,plane,color,name):
    assert np.allclose(dirs,np.diag(np.diag(dirs))) and (np.diag(dirs)>0).all()
    spacing=np.diag(dirs);idx=np.rint(np.linalg.solve(dirs.T,point-origin)).astype(int)
    axis,h,v={'sagittal':(0,1,2),'coronal':(1,0,2),'axial':(2,0,1)}[plane]
    if not 0<=idx[axis]<volume.shape[::-1][axis]:
        ax.text(.5,.5,'영상 범위 밖',transform=ax.transAxes,ha='center');return
    take=lambda a:(a[:,:,idx[0]] if axis==0 else a[:,idx[1],:] if axis==1 else a[idx[2]])
    image=take(volume);seg=take(mask)>0
    lo=origin-spacing/2;hi=origin+(np.array(volume.shape[::-1])-.5)*spacing
    extent=[lo[h],hi[h],lo[v],hi[v]]
    ax.imshow(image,origin='lower',cmap='gray',vmin=-700,vmax=2000,extent=extent)
    if seg.any():
        # 넓은 원본 격자의 점을 사용해 마스크와 CT의 물리 위치를 맞춘다.
        X=origin[h]+np.arange(seg.shape[1])*spacing[h]
        Y=origin[v]+np.arange(seg.shape[0])*spacing[v]
        ax.contour(X,Y,seg.astype(float),[.5],colors=['#00c8ff'],linewidths=.65)
    off=point[axis]-(origin[axis]+idx[axis]*spacing[axis])
    ax.plot(point[h],point[v],'+',color=color,ms=14,mew=2.2)
    ax.set_xlim(point[h]-20,point[h]+20);ax.set_ylim(point[v]-20,point[v]+20)
    ax.set_title(f'{name} / {plane}\n단면 축 거리 {off:+.2f} mm',fontsize=10)
    ax.set_xlabel(('LPS x','LPS y','LPS z')[h]+' mm');ax.set_ylabel(('LPS x','LPS y','LPS z')[v]+' mm')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    routes=pd.read_csv(DEST/'point_routes.csv');focus=routes[routes.triggered]
    assert len(focus)==6 and focus.Patient.nunique()==5
    cards=[];coordinate_rows=[]
    for p,g in focus.groupby('Patient',sort=False):
        c=json.loads((DEST/'cases'/f'{p}.json').read_text(encoding='utf-8'))
        P,files=landmarks(int(p[2:]));legacy,_=repair(P)
        volume,dirs,org=read_nrrd(CB/f'nrrd/{p}.nrrd');mask,mdirs,morg=load_mask(p,'CB')
        assert np.allclose(dirs,mdirs) and np.allclose(org,morg)
        for _,r in g.iterrows():
            side=r.point;info=c['points'][side]
            variants=[('원래 ALI',P[side],'#ff4040')]
            if np.linalg.norm(legacy[side]-P[side])>.01:
                variants.append(('이전 대칭 추정: 비교 전용',legacy[side],'#ce93d8'))
            if info['previous_candidate'] is not None:
                variants.append(('이전 작은 구멍 후보',np.array(info['previous_candidate']),'#ff9800'))
            if info['candidate'] is not None:
                variants.append(('관 모양 조건 통과 후보',np.array(info['candidate']),'#00e676'))
            fig,axes=plt.subplots(len(variants),3,figsize=(16,4.6*len(variants)),squeeze=False)
            for row,(name,point,color) in enumerate(variants):
                coordinate_rows.append(dict(Patient=p,point=side,variant=name,x=point[0],y=point[1],z=point[2]))
                for col,plane in enumerate(('sagittal','coronal','axial')):
                    native_panel(axes[row,col],volume,mask,dirs,org,point,plane,color,name)
            fig.suptitle(f'{p} {side}: 각 점의 자기 위치에서 자른 원본 / 하늘색=CB 분할 / 모든 칸 40×40 mm',fontsize=13)
            fig.text(.5,.012,'점 하나당 실제 3방향 단면. 다른 깊이의 점은 겹쳐 그리지 않았습니다. 대칭 추정은 이전 결과 비교용이며 새 복구에는 사용하지 않습니다.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.025,1,.96));path=DEST/'figs'/f'{p}_{side}_native_comparison.png';fig.savefig(path,dpi=105);plt.close(fig)
            tracks=pd.DataFrame([{k:v for k,v in t.items() if k!='candidate'} for t in info['tracks']])
            if 'reasons' in tracks:tracks['reasons']=tracks.reasons.map(lambda x:';'.join(x))
            options=''.join(f'<label><input type="radio" name="{p}|{side}|{field}" value="{value}">{label}</label> '
                            for field,choices in [('canal_structure',[('yes','외이도 구조 확인'),('no','다른 공간 의심'),('uncertain','구조 판정 보류')]),
                                                  ('external_rim',[('yes','외측 입구 상연 확인'),('no','외측 입구 위치 불일치'),('uncertain','상연 위치 판정 보류')])]
                            for value,label in choices)
            coords=pd.DataFrame([x for x in coordinate_rows if x['Patient']==p and x['point']==side])
            cards.append(f'<article id="{p}-{side}"><h2>{p} {side}</h2><p>{escape(info["route"])}</p>'
                         f'{coords.to_html(index=False,border=0,float_format=lambda x:f"{x:.3f}")}'
                         f'<a href="figs/{path.name}"><img src="figs/{path.name}"></a>'
                         f'<details><summary>인접 단면의 관 모양 검사</summary>{tracks.to_html(index=False,border=0)}</details>'
                         f'<div class="choices">{options}</div><textarea name="{p}|{side}|note" placeholder="확인한 구조·위치와 판단 근거"></textarea></article>')
        del volume,mask
    pd.DataFrame(coordinate_rows).to_csv(DEST/'review_coordinates.csv',index=False,encoding='utf-8-sig')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Po 5명·6개 점 원본 검토</title>
<style>body{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;margin:0}main{max-width:1500px;margin:auto;padding:24px}article{background:white;margin:24px 0;padding:16px;overflow:auto}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:5px}label{display:inline-block;margin:6px 12px 6px 0}textarea{width:100%;min-height:60px}button{font:inherit;padding:8px 16px}</style><main>
<h1>Po 복구 분기 5명·6개 점: 실제 단면 비교</h1><p>각 점의 자기 위치에서 자른 원본이며, 다른 깊이의 점을 한 단면 위에 투영하지 않았습니다. 하늘색은 같은 위치의 CB 분할 경계입니다. 기존 대칭 추정점은 비교를 위해서만 표시했고 새 복구에는 사용하지 않았습니다.</p>
<p>후보의 구멍이 실제 외이도인지, 점이 외측 입구의 상연에 있는지를 분리해서 확인해야 합니다. 아래 선택은 검토 기록일 뿐 측정값을 변경하지 않습니다. 전문가 확인 전이며 자동 정답 표시가 없습니다.</p>
<button type="button" onclick="saveReview()">판단 기록 JSON 저장</button>'''+''.join(cards)+'''
<script>function saveReview(){const decisions={};document.querySelectorAll('input[type=radio]:checked,textarea').forEach(e=>{if(e.value)decisions[e.name]=e.value});const data={workflow:'po_tube_v1',status:'review_record_only_not_applied',saved_at:new Date().toISOString(),decisions};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='po_tube_review_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}</script></main></html>'''
    (DEST/'point_review.html').write_text(html,encoding='utf-8')
    path=DEST/'report.html';s=path.read_text(encoding='utf-8')
    link='<p><b><a href="point_review.html">5명·6개 Po의 원래 점·이전 추정·새 후보를 실제 단면에서 대조하고 판단 기록 남기기</a></b></p>'
    if 'href="point_review.html"' not in s:s=s.replace('<h2>1. 확인된 약점과 새 조건</h2>',link+'<h2>1. 확인된 약점과 새 조건</h2>')
    path.write_text(s,encoding='utf-8')
    # 기존 버전 입력 해시·결과표가 유지됐는지 확인한다.
    source=json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    assert digest(PREV/'protocol.json')==source['source_protocol_sha256']
    assert len(routes)==170 and routes.Patient.nunique()==85
    comp=pd.read_csv(DEST/'measurement_comparison.csv');assert comp.groupby('metric').size().eq(85).all()
    assert comp.loc[~comp.recovery_complete,'proposed_value'].isna().all()
    for page in ('report.html','point_review.html'):
        refs=re.findall(r'(?:href|src)="([^"]+)"',(DEST/page).read_text(encoding='utf-8'))
        assert all((DEST/x).exists() for x in refs if not x.startswith('#'))
    save(DEST/'verification.json',dict(patients=85,Po_points=170,focus_patients=5,focus_points=6,
         all85_retained_in_evaluation=True,unresolved_proposed_values_blank=True,
         prior_protocol_unchanged=True,report_links_valid=True,
         candidate_selection_uses_Excel=False,symmetry_used_for_new_recovery=False,
         review_choices_applied_to_measurements=False))
    print('Saved point_review.html and verified 85 patients / 170 Po points / 6 native comparison figures')


if __name__=='__main__':main()
