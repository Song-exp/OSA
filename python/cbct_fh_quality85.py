"""85명 공통 FH 품질·복구 규칙의 탐색 평가. 원본/기존 결과는 변경하지 않는다.

추출 단계는 엑셀을 읽지 않는다. 고정된 Po-free 좌표 분포와 원본 픽셀만 사용한다.
Po 후보는 외이도처럼 보이는 폐쇄 공기 단면의 외측 상연이라는 운영 정의이며,
전문가가 확정한 Po 정답이 아니다. 검수 실패 사례는 측정값을 숨기지 않고 별도 기록한다.
"""
from collections import Counter
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label, map_coordinates

from cbct_patient_pipeline import CB, OUT, landmarks, digest
from cbct_eb_detect import midline_frame
from cbct_fh_repair import DEST as REF_DIR, local, repair
from check_landmarks import read_nrrd
from osa.ceph import measures

DEST = OUT/'develop85/fh_quality_v1'
KEYS = ('RPo', 'LPo', 'ROr', 'LOr')
METRICS = {'ANB': ('ANB', 'jung'), 'Nperp-A': ('Nperp_A', 'yao4'),
           'Nperp-Pog': ('Nperp_Pog', 'yao4'), 'MP-FH': ('MPFH_GoMe', 'yao4')}
RULE = dict(version='fh-image-quality-v1', geometry_z=5., bone_threshold=300.,
            bone_max_distance_mm=2.5, roi_radius_mm=4., voxel_step_mm=.5,
            canal_air_threshold=-400., canal_sensitivity_thresholds=[-500., -300.],
            canal_area_mm2=[3., 120.], canal_width_height_mm=[2., 16.],
            search_lateral_half_mm=8., search_ant_half_mm=14., search_sup_half_mm=18.,
            candidate_min_adjacent_slices=3, candidate_position_sensitivity_mm=2.,
            roof_bone_fraction_min=.15, bilateral_ant_sup_difference_mm=6.,
            candidate_max_shift_mm=40., minimum_fov_margin_mm=2.,
            reference='historical fh_repair/v1/reference.json; not refitted',
            input_selection='all 85 patients in compare_all.csv; Excel never used in extraction',
            deployment='separate exploratory branch, no replacement of prior results')


def json_safe(x):
    if isinstance(x, dict): return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return json_safe(x.tolist())
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x


def save(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sample(v, dirs, org, positions):
    shape = positions.shape[:-1]
    ijk = np.linalg.solve(dirs.T, (positions.reshape(-1, 3)-org).T)
    return map_coordinates(v, ijk[::-1], order=1, output=np.float32,
                           mode='constant', cval=np.nan, prefilter=False).reshape(shape)


def margin(v, dirs, org, point):
    ijk = np.linalg.solve(dirs.T, point-org)
    spacing = np.linalg.norm(dirs, axis=1)
    return float(np.min(np.r_[ijk*spacing, (np.array(v.shape[::-1])-1-ijk)*spacing]))


def support(v, dirs, org, point):
    r = np.arange(-4., 4.01, .5)
    offsets = np.stack(np.meshgrid(r, r, r, indexing='ij'), -1)
    values = sample(v, dirs, org, point+offsets)
    distances = np.linalg.norm(offsets, axis=-1)
    dense = (values > RULE['bone_threshold']) & (distances <= 4.)
    near = float(distances[dense].min()) if dense.any() else float('inf')
    return dict(nearest_dense_mm=near, patch_finite_fraction=float(np.isfinite(values).mean()),
                gray_at_point=float(sample(v, dirs, org, point[None])[0]),
                fov_margin_mm=margin(v, dirs, org, point))


def geometry(points, key, ref):
    q, _, _ = local(points, key)
    z = np.abs((q-np.array(ref[key]['median']))/np.maximum(ref[key]['mad'], 1e-6))
    return float(z.max())


def assess(v, dirs, org, points, key, ref):
    s = support(v, dirs, org, points[key])
    z = geometry(points, key, ref)
    reasons = []
    if s['fov_margin_mm'] < RULE['minimum_fov_margin_mm']: reasons.append('outside_or_near_fov')
    if z > RULE['geometry_z']: reasons.append('geometry_outlier')
    if s['nearest_dense_mm'] > RULE['bone_max_distance_mm']: reasons.append('no_nearby_dense_boundary')
    return dict(point=points[key], z=z, **s, reasons=reasons, passed=not reasons)


def holes(image, ant, sup, threshold):
    """시상 단면에서 영상 경계와 연결되지 않은 공기 구멍과 바로 위 뼈를 확인."""
    air = np.isfinite(image) & (image < threshold)
    labs, n = label(air)
    out = []
    for k in range(1, n+1):
        aa, zz = np.where(labs == k)
        if len(aa)*.25 < 3 or len(aa)*.25 > 120: continue
        if aa.min() == 0 or aa.max() == len(ant)-1 or zz.min() == 0 or zz.max() == len(sup)-1: continue
        width = (aa.max()-aa.min()+1)*.5; height = (zz.max()-zz.min()+1)*.5
        if not (2 <= width <= 16 and 2 <= height <= 16): continue
        roof = zz.max()
        top_a = aa[zz >= roof-1]
        a = float(np.median(ant[top_a])); s = float(sup[roof]+.25)
        ai = int(np.argmin(abs(ant-a)))
        above = image[max(0,ai-2):ai+3, roof+1:min(len(sup),roof+5)]
        bone = float(np.mean(above > RULE['bone_threshold'])) if above.size else 0.
        if bone < RULE['roof_bone_fraction_min']: continue
        out.append(dict(ant=a, sup=s, area_mm2=float(len(aa)*.25), roof_bone_fraction=bone))
    return out


def canal_candidate(v, dirs, org, points, key, ref, opposite_ok):
    """공통 영상 탐색 규칙. 반사 위치는 검색 중심에만 쓰며 후보는 픽셀에서 구한다."""
    o, R, _ = midline_frame(points)
    sign = 1 if key == 'LPo' else -1
    other = 'RPo' if key == 'LPo' else 'LPo'
    if opposite_ok:
        center = R@(points[other]-o); center[0] *= -1
        search_source = 'opposite_reflection_search_only'
    else:
        center = np.array(ref[key]['median'], float); center[0] *= sign
        search_source = 'fixed_population_search_only'
    lateral = np.arange(abs(center[0])-8, abs(center[0])+8.01, 1.)
    ant = np.arange(center[1]-14, center[1]+14.01, .5)
    sup = np.arange(center[2]-18, center[2]+18.01, .5)
    G = np.stack(np.meshgrid(sign*lateral, ant, sup, indexing='ij'), -1)
    scan = sample(v, dirs, org, o+G@R)
    candidates = []
    for i, lateral_distance in enumerate(lateral):
        for h in holes(scan[i], ant, sup, -400):
            distance = np.hypot(h['ant']-center[1], h['sup']-center[2])
            if distance > 14: continue
            q = np.array([sign*lateral_distance, h['ant'], h['sup']])
            # 공기 문턱을 바꾸어도 같은 공기 구멍 상연이 나타나는지 확인.
            changes = []
            for threshold in (-500., -300.):
                alt = holes(scan[i], ant, sup, threshold)
                changes.append(min([np.hypot(x['ant']-h['ant'], x['sup']-h['sup']) for x in alt], default=float('inf')))
            candidates.append(dict(slice=i, local=q, position=o+q@R,
                                   threshold_sensitivity_mm=max(changes), **h))
    stable = []
    for c in candidates:
        neighbors = {x['slice'] for x in candidates if abs(x['slice']-c['slice']) <= 1
                     and np.linalg.norm((x['local']-c['local'])[1:]) <= 2.}
        if len(neighbors) < 3 or c['threshold_sensitivity_mm'] > 2.: continue
        qpoints = dict(points); qpoints[key] = c['position']
        if geometry(qpoints, key, ref) > RULE['geometry_z']: continue
        if margin(v, dirs, org, c['position']) < 2.: continue
        if np.linalg.norm(c['position']-points[key]) > 40.: continue
        stable.append(c)
    if not stable:
        return dict(found=False, source=search_source, raw_candidates=len(candidates),
                    reason='no_candidate_passed_image_geometry_stability')
    # 가장 외측의 안정된 폐쇄 공기 단면 상연. 결과변수·환자 번호는 참조하지 않는다.
    stable.sort(key=lambda c: (-abs(c['local'][0]),
                             np.linalg.norm((c['local']-center)[1:])))
    chosen = stable[0]
    return dict(found=True, source=search_source, position=chosen['position'],
                local=chosen['local'], threshold_sensitivity_mm=chosen['threshold_sensitivity_mm'],
                raw_candidates=len(candidates), stable_candidates=len(stable),
                area_mm2=chosen['area_mm2'], roof_bone_fraction=chosen['roof_bone_fraction'])


def run_case(patient, ref):
    points, files = landmarks(int(patient[2:]))
    image_path = CB/f'nrrd/{patient}.nrrd'
    image_hash = digest(image_path)
    lm_hash = {f.name:digest(f) for f in files}
    cache_path = DEST/'cases'/f'{patient}.json'
    fingerprint = dict(image=image_hash, landmarks=lm_hash, reference=digest(REF_DIR/'reference.json'),
                       code=digest(Path(__file__)), rule=RULE)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        if cached['fingerprint'] == fingerprint:
            print(patient, cached['status'], '(cached)', flush=True); return cached
    v, dirs, org = read_nrrd(image_path)
    if not np.allclose(dirs, np.diag(np.diag(dirs))) or np.any(np.diag(dirs) <= 0):
        raise ValueError('Expected positive axis-aligned LPS volume')
    original = {k:assess(v, dirs, org, points, k, ref) for k in KEYS}
    recovered = dict(points); attempts = {}; routes = []
    for key in ('RPo', 'LPo'):
        if original[key]['passed']: continue
        other = 'LPo' if key == 'RPo' else 'RPo'
        attempt = canal_candidate(v, dirs, org, points, key, ref, original[other]['passed'])
        attempts[key] = attempt
        if attempt['found']:
            recovered[key] = np.array(attempt['position'])
            routes.append(key)
    final = {k:assess(v, dirs, org, recovered, k, ref) for k in KEYS}
    o, R, _ = midline_frame(points)
    asymmetry = {}
    for a, b in (('RPo','LPo'), ('ROr','LOr')):
        delta = abs(R@(recovered[a]-recovered[b]))
        asymmetry[a[1:]] = dict(ant=float(delta[1]), sup=float(delta[2]))
        if np.max(delta[1:]) > RULE['bilateral_ant_sup_difference_mm']:
            for key in (a,b):
                final[key]['reasons'].append('bilateral_ant_sup_asymmetry'); final[key]['passed'] = False
    # 안전장치: 실패 복구는 확인용 좌표로 남기고 배포값으로 내보내지 않는다.
    missing = any(x['fov_margin_mm'] < 0 for x in original.values())
    release = all(x['passed'] for x in final.values())
    status = ('not_measurable' if missing and not release else 'needs_review') if not release else ('image_recovered' if routes else 'automatic_unchanged')
    legacy, repair_log = repair(points, ref)
    vals = {}
    for name, p in (('raw', points), ('legacy_repair', legacy), ('image_rule', recovered)):
        vals[name] = {m:float(measures(p, fh)[key]) for m, (key, fh) in METRICS.items()}
    result = dict(patient=patient, fingerprint=fingerprint, status=status, release=release,
                  original=original, final=final, attempts=attempts, recovered_points=recovered,
                  changed_points=routes, asymmetry=asymmetry, legacy_repair_log=repair_log,
                  diagnostic_measurements=vals,
                  released_measurements=vals['image_rule'] if release else None)
    save(cache_path, result)
    print(patient, status, routes, flush=True)
    return json_safe(result)


def summarize(cases):
    # 정확도 평가는 추출이 끝난 뒤에만 엑셀 비교표를 읽는다.
    excel = pd.read_csv(OUT/'restore_ext/v1/compare_all.csv').set_index(['Patient','metric']).excel
    rows, patients, comparisons = [], [], []
    for c in cases:
        p = c['patient']; patients.append(dict(Patient=p, status=c['status'], release=c['release'],
                        changed_points=';'.join(c['changed_points']),
                        original_failed=';'.join(k for k,x in c['original'].items() if not x['passed']),
                        final_failed=';'.join(k for k,x in c['final'].items() if not x['passed'])))
        for key in KEYS:
            for stage in ('original','final'):
                r = c[stage][key]
                rows.append(dict(Patient=p, point=key, stage=stage, z=r['z'],
                                 nearest_dense_mm=r['nearest_dense_mm'], fov_margin_mm=r['fov_margin_mm'],
                                 passed=r['passed'], reasons=';'.join(r['reasons'])))
        for metric in METRICS:
            y = excel.get((p,metric), np.nan)
            old = c['diagnostic_measurements']['legacy_repair'][metric]
            new = c['diagnostic_measurements']['image_rule'][metric]
            comparisons.append(dict(Patient=p, metric=metric, status=c['status'], release=c['release'],
                                    legacy=old, candidate=new, excel=y, legacy_error=abs(old-y),
                                    candidate_error=abs(new-y), changed=abs(old-new)>1e-6,
                                    released=new if c['release'] else np.nan))
    pdf = pd.DataFrame(patients); pointdf = pd.DataFrame(rows); comp = pd.DataFrame(comparisons)
    for name, frame in [('patient_routes.csv', pdf), ('point_quality.csv', pointdf), ('measurement_comparison.csv', comp)]:
        frame.to_csv(DEST/name,index=False,encoding='utf-8-sig')
    summary = []
    for m,g in comp.groupby('metric'):
        good = g[g.release]; changed = g[g.changed]
        prior_acceptable = changed[changed.legacy_error <= (2. if m in ('ANB','MP-FH') else 2.)]
        summary.append(dict(metric=m, total_n=len(g), released_n=len(good), rejected_n=int((~g.release).sum()),
                            coverage=float(g.release.mean()), all_legacy_MAE=float(g.legacy_error.mean()),
                            all_candidate_diagnostic_MAE=float(g.candidate_error.mean()),
                            same_released_legacy_MAE=float(good.legacy_error.mean()),
                            same_released_candidate_MAE=float(good.candidate_error.mean()),
                            changed_n=len(changed), improved_n=int((changed.candidate_error < changed.legacy_error).sum()),
                            worsened_n=int((changed.candidate_error > changed.legacy_error).sum()),
                            previously_within_2_worsened_over_2=int((prior_acceptable.candidate_error > 2).sum()),
                            withheld_legacy_MAE=float(g[~g.release].legacy_error.mean())))
    sdf = pd.DataFrame(summary); sdf.to_csv(DEST/'evaluation.csv',index=False,encoding='utf-8-sig')
    save(DEST/'summary.json', dict(status_counts=Counter(pdf.status), patients=len(pdf),
          notes=['85명 전체 재개발 자료, 독립 검증 아님', '픽셀 문턱은 CBCT 장비 보정된 HU가 아님',
                 '예외 분기 결정에는 엑셀 미사용', 'dense boundary 근접만으로 Po/Or 정답을 판정하지 않음',
                 '복구 후보는 외이도 폐쇄 공기 단면 상연이라는 운영 정의; 전문가 정답 없음']))
    def table(t):return t.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    cards=[]
    for c in cases:
        if c['status'] == 'automatic_unchanged':continue
        p=c['patient']; f=DEST/'figs'/f'{p}.png'
        cards.append(f'<article id="{p}"><h3>{p} / {c["status"]}</h3>'
                     f'{table(comp[comp.Patient==p][["metric","legacy","candidate","excel","legacy_error","candidate_error"]])}'
                     +(f'<a href="figs/{p}.png"><img src="figs/{p}.png"></a>' if f.exists() else '')
                     +f'<p><a href="cases/{p}.json">품질·후보·입력 근거</a></p></article>')
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>85명 FH 영상 품질·복구 규칙</title><style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;margin:0;background:#f8f8f5}}main{{max-width:1450px;margin:auto;padding:24px}}table{{border-collapse:collapse;background:#fff;font-size:13px}}td,th{{border:1px solid #ddd;padding:5px}}.scroll{{overflow:auto}}img{{max-width:100%}}article{{padding:16px;background:white;margin:20px 0}}h2{{margin-top:36px}}</style><main>
<h1>85명 공통 FH 영상 품질·복구 규칙</h1><p>기본 측정 → 입력 품질 점검 → 영상 기반 Po 재탐색 → 재점검 → 측정 제공/검수 필요 분기. 모든 환자에게 같은 규칙을 적용했습니다. 기존 자동점·수정점·보고서는 변경하지 않았습니다.</p>
<p><b>이 결과는 공통 규칙의 탐색 평가입니다.</b> 품질 통과는 해부학적 정답 확인이 아닙니다. 내보낸 값도 연구용 후보이며 임상 확정값이 아닙니다. 기존 54명 고정 분포를 재사용했고 85명 모두 개발 자료입니다.</p>
<h2>1. 처리 결과</h2>{table(pdf.groupby('status').size().rename('환자 수').reset_index())}
<p>automatic_unchanged: 원래 점으로 품질 통과 / image_recovered: 영상 탐색 좌표로 복구 후 통과 / needs_review: 실패 원인을 남기고 신뢰값 제공 보류 / not_measurable: 필요한 기준점이 촬영 범위 밖이며 복구 실패.</p>
<h2>2. 오차와 측정 제공률을 함께 평가</h2><p>all_*는 보류 환자까지 포함한 85명 진단용 값입니다. same_released_*는 같은 제공 환자끼리의 비교입니다. 보류 환자의 오차도 withheld_legacy_MAE로 공개했습니다. 실패 환자를 빼고 평균이 좋아졌다고 주장하지 않습니다. 오차 2 이내→2 초과 악화는 진단용 운영 지표이며 임상 허용오차가 아닙니다.</p><div class="scroll">{table(sdf)}</div>
<p>Nperp-Pog의 비교에는 Pog 윤곽 보정을 넣지 않고 FH 변경 효과만 분리했습니다. ANB는 3점, 나머지는 4점 FH를 고정했습니다. 엑셀의 0도 그대로 포함했습니다.</p>
<h2>3. 전체 85명 처리 경로</h2><div class="scroll">{table(pdf)}</div>
<h2>4. 예외 사례</h2>{''.join(cards)}
<h2>5. 공통 규칙과 한계</h2><pre>{escape(json.dumps(RULE,ensure_ascii=False,indent=2))}</pre>
<p>분포 z&gt;5, 근처 뼈 신호 없음, FOV 부족을 먼저 표시합니다. Po 재탐색은 정상 반대편의 반사 위치 또는 고정 분포 중앙값을 검색 중심으로만 쓰고, 실제 영상의 폐쇄 공기 단면·상연 뼈·이웃 단면 지속성·문턱 민감도·좌우 관계를 확인합니다. Or은 현재 자동 재검출 없이 보수적으로 보류합니다. 정상 해부학적 비대칭도 보류될 수 있습니다. 단면 연결 조건이 외이도를 놓치거나 다른 공기 공간을 고를 가능성이 있어 전문가 검수가 필요합니다.</p>
<p><a href="patient_routes.csv">전체 처리 경로</a> · <a href="point_quality.csv">340개 점의 전후 품질</a> · <a href="measurement_comparison.csv">85명×4지표 비교</a> · <a href="evaluation.csv">평가표</a> · <a href="protocol.json">고정 규칙·실행 근거</a></p></main></html>'''
    (DEST/'report.html').write_text(html,encoding='utf-8')
    print(pdf.groupby('status').size().to_string(),flush=True)
    print(sdf.to_string(index=False),flush=True)
    return pdf


def figures(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cbct_landmark_review import panel
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    for c in cases:
        if c['status']=='automatic_unchanged':continue
        p=c['patient']; out=DEST/'figs'/f'{p}.png'
        P=landmarks(int(p[2:]))[0]
        failed=[k for k in KEYS if not c['original'][k]['passed']]
        keys=failed or [k for k in KEYS if not c['final'][k]['passed']]
        fig,axes=plt.subplots(len(keys),3,figsize=(15,4.8*len(keys)),squeeze=False)
        v,dirs,org=read_nrrd(CB/f'nrrd/{p}.nrrd')
        for row,key in enumerate(keys):
            candidate=c['attempts'].get(key,{})
            center=P[key]
            if candidate.get('found'):center=np.array(candidate['position'])
            marks={'ALI':(P[key],'#ff4040','+')}
            if candidate.get('found'):marks['image candidate']=(center,'#00e676','x')
            for col,plane in enumerate(('sagittal','coronal','axial')):
                panel(axes[row,col],v,dirs,org,center,plane,marks,half=25)
                axes[row,col].set_title(f'{key}: {plane} / '+('후보 위치' if candidate.get('found') else '원래 위치'),fontsize=10)
        fig.suptitle(f'{p}: {c["status"]} / 실제 단면과 기준점 (엑셀로 좌표 선택하지 않음)',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out,dpi=100);plt.close(fig)
        del v


def main():
    for folder in (DEST,DEST/'cases',DEST/'figs'):folder.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'restore_ext/v1/compare_all.csv'
    patients=sorted(pd.read_csv(manifest,usecols=['Patient']).Patient.unique(),key=lambda p:int(p[2:]))
    assert len(patients)==85
    save(DEST/'protocol.json',dict(rule=RULE,patients=patients,cohort_manifest_sha256=digest(manifest),
                                 reference_sha256=digest(REF_DIR/'reference.json'),code_sha256=digest(Path(__file__))))
    ref=json.loads((REF_DIR/'reference.json').read_text(encoding='utf-8'))['points']
    cases=[run_case(p,ref) for p in patients]
    summarize(cases)
    figures(cases)
    summarize(cases)
    print('saved',DEST/'report.html',flush=True)


if __name__=='__main__':main()
