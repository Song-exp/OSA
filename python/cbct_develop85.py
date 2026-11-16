"""85명 계측법 재개발: 기존 결과를 보존한 오차 감사와 중첩 후보 선택.

실제 기하 계산 후보만 비교한다. 엑셀 환산이나 오차 기반 환자 제외는 하지 않는다.
이미 검토한 코호트의 탐색적 재평가이며 외부 검증이 아니다.
"""
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import cbct_accuracy_experiments as X
import cbct_excel_alignment as alignment
from cbct_patient_pipeline import CB, OUT, landmarks
from cbct_fh_repair import repair
from osa.ceph import measures, head_frame

DEST = OUT/'develop85/v1'
REPEATS, FOLDS, INNER_REPEATS = 20, 5, 5
SEED = 851115
ORDER = ['ANB', 'Nperp-A', 'Nperp-Pog', 'MP-FH', 'TDI', 'PHI', 'NCWR', 'PAL', 'PAV', 'NAV', 'PMCA']


def dump(name, data):
    (DEST/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(name, frame):
    frame.to_csv(DEST/name, index=False, encoding='utf-8-sig')


def policy(errors, base, seed):
    """훈련 환자 안에서만 후보 선택. 원래 계산보다 5% 감소·내부 fold 80% 우세."""
    if len(errors) < 10 or errors.shape[1] == 1:
        return base
    fold_mae = []
    for rep in range(INNER_REPEATS):
        for _, valid in KFold(5, shuffle=True, random_state=seed+rep).split(errors):
            fold_mae.append(errors[valid].mean(0))
    mae = np.asarray(fold_mae)
    eligible = [i for i in range(errors.shape[1]) if i != base
                and mae[:, i].mean() <= .95*mae[:, base].mean()
                and np.mean(mae[:, i] < mae[:, base]) >= .8]
    return min(eligible, key=lambda i: (mae[:, i].mean(), i)) if eligible else base


def evaluate(t, columns, current, metric):
    # 결측을 다른 정의 값으로 채우지 않는다. 후보 모두 계산된 동일 환자끼리 비교.
    valid = t.dropna(subset=columns+['excel']).sort_values('Patient').reset_index(drop=True)
    e = np.abs(valid[columns].to_numpy(float)-valid.excel.to_numpy(float)[:, None])
    base = columns.index(current)
    selected_errors, baseline_errors, assignments = [], [], []
    for rep in range(REPEATS):
        for fold, (tr, te) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED+rep).split(valid)):
            selected = policy(e[tr], base, SEED+rep*100+fold)
            for j in te:
                assignments.append(dict(metric=metric, Patient=valid.Patient.iloc[j], repeat=rep, fold=fold,
                                        selected=columns[selected], baseline=current,
                                        excel=float(valid.excel.iloc[j]),
                                        baseline_value=float(valid[columns[base]].iloc[j]),
                                        selected_value=float(valid[columns[selected]].iloc[j]),
                                        baseline_error=float(e[j, base]), selected_error=float(e[j, selected])))
            selected_errors.extend(e[te, selected]); baseline_errors.extend(e[te, base])
    out = pd.DataFrame(assignments)
    patient = out.groupby('Patient')[['baseline_error', 'selected_error']].mean()
    delta = (patient.baseline_error-patient.selected_error).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    choice = policy(e, base, SEED)
    result = dict(metric=metric, n=len(valid), current=current, proposed=columns[choice],
                  baseline_MAE=float(np.mean(baseline_errors)), nested_MAE=float(np.mean(selected_errors)),
                  gain_pct=float(100*(1-np.mean(selected_errors)/np.mean(baseline_errors))),
                  paired_MAE_reduction_low=float(low), paired_MAE_reduction_high=float(high),
                  changed_folds=int(sum(g.selected.iloc[0] != current for _, g in out.groupby(['repeat', 'fold']))),
                  total_folds=REPEATS*FOLDS)
    result['status'] = '개선 후보' if result['gain_pct'] >= 5 and low > 0 and columns[choice] != current else '현 방식 유지'
    full = [dict(metric=metric, candidate=c, n=len(valid), MAE=float(e[:, i].mean()),
                 is_current=c == current, is_proposal=i == choice) for i, c in enumerate(columns)]
    return result, out, full


def collect():
    F = alignment.features()
    first = json.loads((X.EXP/'stage1_choices.json').read_text(encoding='utf-8'))
    second = json.loads((X.EXP/'stage2_choices.json').read_text(encoding='utf-8'))
    current = {**{m: v['chosen'] for m, v in first.items()},
               **{m: v['chosen'] for m, v in second.items()}, 'PAL': 'PAL_mm'}
    diagnostics = []
    # 신규 후보: 반대편 Orbitale을 쓰는 3점 FH. 기존 계측 함수의 정의를 그대로 사용.
    for p in F['ANB'].Patient:
        P = landmarks(int(p[2:]))[0]
        Q, logs = repair(P)
        values = measures(Q, 'lor')
        _, R3 = head_frame(Q, 'jung'); _, R4 = head_frame(Q, 'yao4')
        angle = np.degrees(np.arccos(np.clip(abs(R3[2]@R4[2]), 0, 1)))
        diagnostics.append(dict(Patient=p, FH_3_vs_4_deg=float(angle),
                                repairs=';'.join(k+':'+str(action) for k, _, action in logs)))
        for m, key in X.SKEL.items():
            F[m].loc[F[m].Patient == p, 'lor'] = values[key]
    csv('geometry_diagnostics.csv', pd.DataFrame(diagnostics))
    metadata = pd.read_csv(X.EXT/'compare_all.csv', encoding='utf-8-sig')
    units = metadata.drop_duplicates('metric').set_index('metric').unit.to_dict()
    old_dev = set(X.sets()[0])
    ranks, baseline, availability = [], [], []
    for m in ORDER:
        t = F[m]
        # TTA는 기존 좌우 대응 문제로 기하 후보 비교에서 제외. 별도 인공지능 재학습 없음.
        columns = [c for c in t if c not in ('Patient', 'excel', 'tta')]
        value = t[current[m]]
        ok = value.notna() & t.excel.notna()
        paired = t[ok].copy()
        error = value[ok]-paired.excel
        for p, x, y, err in zip(paired.Patient, value[ok], paired.excel, error):
            ranks.append(dict(Patient=p, metric=m, unit=units[m], current=current[m],
                              automatic=x, excel=y, difference=err, abs_error=abs(err),
                              excel_zero=bool(y == 0), old_set='dev54' if p in old_dev else 'former_holdout31',
                              rank_within_metric=int(error.abs().rank(method='min', ascending=False).loc[paired.Patient.eq(p)].iloc[0])))
        baseline.append(dict(metric=m, unit=units[m], n=int(ok.sum()), MAE=float(error.abs().mean()),
                             median_AE=float(error.abs().median()), p90_AE=float(error.abs().quantile(.9)),
                             max_AE=float(error.abs().max()), bias=float(error.mean()),
                             missing_auto=int(value.isna().sum()), missing_excel=int(t.excel.isna().sum()),
                             paired_zeros=int((paired.excel == 0).sum())))
        availability.extend(dict(metric=m, candidate=c, automatic_available=int(t[c].notna().sum()),
                                 paired_available=int((t[c].notna() & t.excel.notna()).sum())) for c in columns)
        F[m] = t[['Patient']+columns+['excel']]
    ranked = pd.DataFrame(ranks).merge(pd.DataFrame(diagnostics), on='Patient', how='left')
    ranked = ranked.sort_values(['metric', 'rank_within_metric'])
    csv('error_ranking.csv', ranked)
    csv('metric_baseline.csv', pd.DataFrame(baseline))
    csv('candidate_availability.csv', pd.DataFrame(availability))
    csv('candidate_values.csv', pd.concat([t.assign(metric=m).melt(
        id_vars=['Patient', 'metric', 'excel'], var_name='candidate', value_name='automatic')
        for m, t in F.items()], ignore_index=True))
    return F, current, ranked, pd.DataFrame(baseline)


def evidence(ranks):
    """큰 오차 사례의 실제 점 단면. 엑셀에 맞추기 위해 점을 이동하지 않는다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cbct_landmark_review import panel
    from check_landmarks import read_nrrd
    from scipy.ndimage import map_coordinates
    from osa.refine import refine
    import SimpleITK as sitk
    from run_ali import ascii_path

    plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
    figdir = DEST/'figs'; figdir.mkdir(exist_ok=True)
    patients = list(dict.fromkeys(ranks[(ranks.metric.isin(['Nperp-A', 'Nperp-Pog'])) & (ranks.rank_within_metric <= 3)].Patient))
    observations = []
    for p in patients:
        f = figdir/f'{p}.png'
        P = landmarks(int(p[2:]))[0]; Q, logs = repair(P)
        v, dirs, org = read_nrrd(CB/f'nrrd/{p}.nrrd')
        im = sitk.ReadImage(str(ascii_path(CB/f'seg/{p}_seg_MAND.nrrd')))
        mand = sitk.GetArrayFromImage(im)
        im2 = sitk.ReadImage(str(ascii_path(CB/f'seg/{p}_seg_MAX.nrrd')))
        maxm = sitk.GetArrayFromImage(im2)
        for im_ in (im, im2):
            assert np.allclose(im_.GetOrigin(), org)
            assert np.allclose(np.array(im_.GetDirection()).reshape(3, 3)@np.diag(im_.GetSpacing()), dirs.T)
            assert tuple(im_.GetSize()[::-1]) == v.shape
        refined, info = refine(Q, mand, maxm, dirs, org, 'yao4')
        fig, axes = plt.subplots(3, 3, figsize=(16, 16))
        for ax, key in zip(axes.ravel()[:8], ('RPo', 'LPo', 'ROr', 'LOr', 'N', 'A', 'B', 'Pog')):
            marks = {'ALI': (P[key], '#ff4040', '+')}
            if np.linalg.norm(Q[key]-P[key]) > .01:
                marks['FH repair'] = (Q[key], '#00c8ff', 'x')
            if key in ('A', 'Pog'):
                marks['contour candidate'] = (refined[key], '#00e676', 'x')
            panel(ax, v, dirs, org, np.asarray(P[key]), 'sagittal', marks, half=20)
            ax.set_title(f'{key}: 원래 점을 지나는 실제 시상 단면', fontsize=10)
            if key in ('A', 'Pog'):
                mask = maxm if key == 'A' else mand
                idx = np.rint(np.linalg.solve(dirs.T, P[key]-org)).astype(int)
                radius = np.ceil(12/np.linalg.norm(dirs, axis=1)).astype(int)
                lo = np.maximum(idx-radius, 0); hi = np.minimum(idx+radius+1, v.shape[::-1])
                zz, yy, xx = np.where(mask[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] > 0)
                nearest = None
                if len(xx):
                    coords = np.c_[xx+lo[0], yy+lo[1], zz+lo[2]]@dirs+org
                    nearest = float(np.linalg.norm(coords-P[key], axis=1).min())
                observations.append(dict(Patient=p, point=key, nearest_seg_voxel_center_mm=nearest,
                                         refinement_shift_mm=float(np.linalg.norm(refined[key]-Q[key])),
                                         refinement_status=info[key]['status'], diagnosis='영상 검토 필요; 엑셀 오류로 단정하지 않음'))
        ax = axes[2, 2]
        o, R = head_frame(Q, 'jung')
        A = np.arange(-115, 31, 1.); S = np.arange(-145, 36, 1.)
        L = np.arange(-70, 71, 2.)
        local = np.stack(np.meshgrid(L, A, S, indexing='ij'), -1)
        sample = map_coordinates(v, np.linalg.solve(dirs.T, (o+local.reshape(-1, 3)@R-org).T)[::-1],
                                 order=1, cval=-1000).reshape(local.shape[:-1]).max(0)
        ax.imshow(sample.T, origin='lower', cmap='gray', vmin=0, vmax=2500,
                  extent=[A[0]-.5, A[-1]+.5, S[0]-.5, S[-1]+.5])
        _, R4 = head_frame(Q, 'yao4')
        slope = -(R4[2]@R[1])/(R4[2]@R[2])
        ax.plot(A, np.zeros(len(A)), color='#00c8ff', label='3-point FH direction')
        ax.plot(A, slope*A, color='#00e676', label='4-point FH direction')
        # N을 지나는 방향 비교선이며 실제 FH 평면의 위치가 아님.
        for key in ('N', 'A', 'Pog'):
            q = R@(Q[key]-o)
            ax.plot(q[1], q[2], 'r+'); ax.annotate(key, (q[1], q[2]), color='yellow')
        ax.legend(fontsize=8); ax.set_title('전체 위치 안내 / N에서 FH 방향만 비교', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{p}: Nperp 큰 오차 원인 점검 — 빨강 원래 점 / 하늘색 FH 보정 / 초록 윤곽 후보', fontsize=13)
        fig.text(.5, .015, '각 확대 그림은 해당 원래 점을 지나는 실제 단면. 후보가 단면에서 떨어진 거리는 범례 off-plane에 표시. 전체 투영은 위치 안내용.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, .95)); fig.savefig(f, dpi=105); plt.close(fig)
        del v, mand, maxm, im, im2
        print('evidence', p, flush=True)
    csv('image_observations.csv', pd.DataFrame(observations))
    return patients


def report(summary, base, ranked, patients):
    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x: f'{x:.3f}', escape=True)
    cards = []
    for p in patients:
        t = ranked[(ranked.Patient == p) & ranked.metric.isin(['Nperp-A', 'Nperp-Pog'])]
        cards.append(f'<article><h3>{p}</h3>{table(t[["metric", "automatic", "excel", "difference", "FH_3_vs_4_deg", "repairs"]])}'
                     f'<a href="figs/{p}.png"><img src="figs/{p}.png" alt="{p} 원본 점검"></a></article>')
    brief = summary[['metric', 'n', 'current', 'proposed', 'baseline_MAE', 'nested_MAE', 'gain_pct',
                     'paired_MAE_reduction_low', 'paired_MAE_reduction_high', 'status']]
    links = ''.join(f'<li><a href="{n}">{label}</a></li>' for n, label in [
        ('error_ranking.csv', '85명 지표별 전체 오차 순위'), ('metric_baseline.csv', '지표별 현재 오차·결측·제로값'),
        ('candidate_comparison.csv', '동일 환자 집합의 기하 후보 비교'), ('candidate_availability.csv', '후보별 측정 가능 인원'),
        ('nested_summary.csv', '중첩 교차검증 요약'), ('nested_predictions.csv', '환자별 외부 fold 예측 및 선택 기록'),
        ('image_observations.csv', '영상 사례의 뼈 마스크 거리·보정 이동량'), ('protocol.json', '평가 규칙 및 입력 해시')])
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>85명 계측법 재개발 1차 점검</title><style>body{{font:15px/1.65 'Malgun Gothic',sans-serif;background:#f8f8f5;color:#222;margin:0}}main{{max-width:1450px;margin:auto;padding:24px}}table{{border-collapse:collapse;background:white;font-size:13px}}td,th{{border:1px solid #ddd;padding:6px}}img{{max-width:100%}}article,.box{{background:white;border:1px solid #ddd;padding:16px;margin:20px 0}}.scroll{{overflow:auto}}h2{{margin-top:40px}}</style><main>
<h1>85명 계측법 재개발: 오차 감사와 기하 후보 평가</h1>
<p>기존 개발 54명 + 기존 검증 31명을 모두 개발 자료로 사용합니다. 기존 31명은 이 분석의 독립 검증군이 아닙니다. 기존 보고서·측정값은 변경하지 않았습니다.</p>
<div class="box">원본 계측 개선과 엑셀 환산을 구분했습니다. 이번 비교에는 직선 환산을 넣지 않았고, 엑셀값 0이나 큰 오차를 이유로 환자를 제외하지 않았습니다. 0은 정상 측정값일 수 있습니다. 결측은 채우지 않았습니다.</div>
<h2>1. 현재 방식의 전체 오차</h2><p>지표마다 단위가 다르므로 MAE 크기로 서로 다른 지표의 좋고 나쁨을 순위 매기지 않습니다. n은 자동값과 엑셀값이 모두 있는 환자 수입니다.</p><div class="scroll">{table(base)}</div>
<h2>2. 계산법 후보를 평가에서 분리해 선택</h2>
<p>바깥 5겹 × 20회, 안쪽 5겹 × 5회. 후보 선택은 안쪽 자료만 사용하고 바깥 환자에서는 오차만 계산했습니다. 안쪽 MAE 5% 이상 감소 및 fold 80% 이상 우세일 때만 변경합니다. 현재 방식은 이전 실험 선택 파일에 기록된 기하 설정입니다.</p>
<p>후보 모두 계산된 동일 환자에서 비교했습니다. TTA는 기존 좌우 대응 문제 때문에 제외했고, 새 후보는 왼쪽 Orbitale을 이용하는 FH(lor)입니다. proposed는 85명에서 다시 선택한 배포 후보이며 nested_MAE는 해당 선택 절차의 바깥 평가값입니다. 같은 숫자로 취급하지 않습니다.</p>
<div class="scroll">{table(brief)}</div>
<p>gain_pct는 오차 감소율, reduction_low/high는 환자 단위 짝지은 MAE 감소의 부트스트랩 95% 구간입니다. 이는 새 코호트 독립 검증이나 임상 정확도 보장이 아닙니다. 기존 54명으로 만든 고정 FH 보정 기준을 재사용했고, 모든 후보는 이미 검토했던 데이터에서 나온 탐색 후보라는 한계가 있습니다.</p>
<h2>3. 지표별 큰 오차 상위 5건</h2><p>원인은 아직 확정하지 않았습니다. 자동점 오류·기준면 오류·수기 정의 차이·입력 오류 가능성을 대조해야 합니다.</p>
<div class="scroll">{table(ranked[ranked.rank_within_metric <= 5][['Patient', 'metric', 'unit', 'automatic', 'excel', 'difference', 'rank_within_metric', 'excel_zero', 'old_set']])}</div>
<h2>4. 원본에서 확인한 우선 점검 사항</h2>
<p><b>P_35:</b> 오른쪽 Po가 뼈에서 벗어나 있고 양쪽 Po가 모두 이상치로 표시되어 기존 규칙이 보정을 멈춥니다. 3점·4점 FH 법선 차이는 약 8.42°입니다. 반대쪽 점도 이상치이므로 단순 반사로 해결하면 안 됩니다. 두 Po를 원본의 외이도에서 다시 확인하는 것이 우선입니다. 엑셀 Nperp 값은 둘 다 0이지만, 이것만으로 입력 오류라고 판단하지 않았습니다.</p>
<p><b>P_67·P_2:</b> Pog와 하악 분할의 가까운 복셀 중심 사이 거리가 각각 약 0.50·0.17 mm인데 Nperp-Pog 차이는 약 15.07·11.83 mm입니다. 점을 뼈로 조금 옮기는 것만으로 차이가 설명되지는 않습니다. N 위치, 기준면, 투영·부호 및 수기 측정 기준의 대조가 필요합니다.</p>
<p>위 거리는 분할 마스크 복셀 중심까지의 거리이며 전문가가 판정한 랜드마크 오차가 아닙니다. 그림에서 녹색 윤곽 후보가 빨강과 겹치면 안전장치가 보정을 거부했을 수도 있으므로 image_observations의 상태를 함께 봅니다.</p>
<h2>5. Nperp-A / Nperp-Pog 각각 상위 3명의 원본 대조</h2>{''.join(cards)}
<h2>6. 다음 개선 순서</h2><p>기존 기하 후보를 재선택하는 것만으로는 채택 기준을 통과한 추가 개선이 없었습니다. 양쪽 Po 이상 사례의 영상 기반 재검출을 먼저 개발하고, Nperp의 수기 기준선과 N 위치를 대조해야 합니다. 엑셀에 맞추기 위해 환자마다 점을 이동하지 않습니다. TDI·PHI·NCWR의 수기 측정 정의 확인이 필요한 부분은 숫자만으로 결정하지 않습니다.</p>
<h2>자료</h2><ul>{links}</ul></main></html>'''
    (DEST/'report.html').write_text(html, encoding='utf-8')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source_paths = [X.EXP/'stage2_values.csv', X.EXP/'stage4_values.csv', X.EXP/'stage1_choices.json', X.EXP/'stage2_choices.json',
                    X.EXT/'compare_all.csv', X.EXT/'new_alternatives.csv', X.FH_REF/'reference.json',
                    OUT/'fh_repair/v2/comparison.csv', OUT/'eb_review/v2/eb_candidates.csv']
    source_paths += [OUT/path for path, _, _ in X.ALT.values()]
    from cbct_patient_pipeline import digest
    dump('protocol.json', dict(created=datetime.now().isoformat(), cohort_n=85,
         old_holdout_reused_for_development=True, independent_validation=False,
         outer_folds=FOLDS, outer_repeats=REPEATS, inner_folds=5, inner_repeats=INNER_REPEATS,
         seed=SEED, adoption_min_gain=.05, adoption_min_fold_win=.8, error_based_exclusion=False,
         zero_exclusion=False, calibration=False, fixed_historical_FH_reference=True,
         inputs={str(p.relative_to(OUT)): digest(p) for p in source_paths}))
    F, current, ranked, baseline = collect()
    results, predictions, candidates = [], [], []
    for m in ORDER:
        t = F[m]; cols = [c for c in t if c not in ('Patient', 'excel')]
        result, pred, full = evaluate(t, cols, current[m], m)
        results.append(result); predictions.append(pred); candidates.extend(full)
        print(m, 'n', result['n'], 'MAE', round(result['baseline_MAE'], 3), '->', round(result['nested_MAE'], 3), result['status'], flush=True)
    summary = pd.DataFrame(results)
    csv('nested_summary.csv', summary); csv('nested_predictions.csv', pd.concat(predictions, ignore_index=True))
    csv('candidate_comparison.csv', pd.DataFrame(candidates))
    patients = evidence(ranked)
    report(summary, baseline, ranked, patients)
    print('saved', DEST/'report.html', flush=True)


def verify():
    """입력 불변 및 모든 바깥 fold의 선택이 훈련 환자만으로 재현되는지 확인."""
    from cbct_patient_pipeline import digest
    protocol = json.loads((DEST/'protocol.json').read_text(encoding='utf-8'))
    for path, signature in protocol['inputs'].items():
        assert digest(OUT/path) == signature, path
    long = pd.read_csv(DEST/'candidate_values.csv')
    pred = pd.read_csv(DEST/'nested_predictions.csv')
    summary = pd.read_csv(DEST/'nested_summary.csv')
    folds_checked = 0
    for _, row in summary.iterrows():
        m = row.metric
        g = long[long.metric == m]
        cols = list(g.candidate.drop_duplicates())
        t = g.pivot(index='Patient', columns='candidate', values='automatic')[cols]
        t['excel'] = g.drop_duplicates('Patient').set_index('Patient').excel
        t = t.dropna().sort_index()
        assert len(t) == row.n
        err = np.abs(t[cols].to_numpy()-t.excel.to_numpy()[:, None])
        for rep in range(REPEATS):
            for fold, (tr, te) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED+rep).split(t)):
                saved = pred[(pred.metric == m) & (pred['repeat'] == rep) & (pred.fold == fold)]
                assert set(t.index[tr]).isdisjoint(saved.Patient)
                assert set(t.index[te]) == set(saved.Patient)
                picked = policy(err[tr], cols.index(row.current), SEED+rep*100+fold)
                assert set(saved.selected) == {cols[picked]}
                expected = pd.Series(err[te, picked], index=t.index[te]).sort_index()
                actual = saved.set_index('Patient').selected_error.sort_index()
                assert np.allclose(expected, actual)
                folds_checked += 1
        assert pred[pred.metric == m].groupby('Patient').size().eq(REPEATS).all()
    rank = pd.read_csv(DEST/'error_ranking.csv')
    assert rank.Patient.nunique() == 85
    assert ((rank.Patient == 'P_35') & (rank.metric == 'Nperp-A') & (rank.excel == 0)).any()
    assert len(pred) == int(summary.n.sum()*REPEATS)
    dump('verification.json', dict(inputs_unchanged=True, outer_folds_replayed=folds_checked,
                                  test_patients_excluded_from_selection=True,
                                  prediction_rows=len(pred), patients=85, zeros_retained=True))
    print('Verified unchanged inputs,', folds_checked, 'training-only fold selections and', len(pred), 'predictions')


if __name__ == '__main__':
    import sys
    verify() if '--verify' in sys.argv else main()
