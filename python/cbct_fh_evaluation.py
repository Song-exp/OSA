"""Frozen FH repair evaluation and 11-metric error-review queue.

Creates fh_repair/v2 only. No inference, tuning, source landmark edits, or
replacement of the original patient result.json files. Existing QC is frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cbct_fh_repair import DEST as REF_DIR, Z_CUT, repair
from cbct_patient_pipeline import CB, OUT, digest, landmarks, qc, save
from cbct_report_scatter import load_pairs, stats
from osa.ceph import measures

DEST = OUT / "fh_repair/v2"
METRICS = {"ANB": "ANB", "Nperp-A": "Nperp_A", "Nperp-Pog": "Nperp_Pog", "MP-FH": "MPFH_GoMe"}
EVIDENCE = {"ANB": "s3", "Nperp-A": "s3", "Nperp-Pog": "s2", "MP-FH": "s3",
            "TDI": "s9", "PHI": "s10", "NCWR": "s10", "PAL": "s5", "PAV": "s11", "NAV": "s12b", "PMCA": "s7"}
ROOT = Path(__file__).resolve().parents[1]


def snapshots():
    paths = list(OUT.glob("P_*/result.json")) + list((CB / "landmarks").glob("P_*.mrk.json"))
    paths += [OUT / "qc_reference.json", REF_DIR / "reference.json"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def calculate():
    baseline, specs, n = load_pairs()
    assert n == 54, "Recheck cohort before evaluating a different sample"
    ref_path = REF_DIR / "reference.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    profile = json.loads((OUT / "qc_reference.json").read_text(encoding="utf-8"))
    assert ref["z_cut"] == Z_CUT
    assert all(np.isfinite(v["mad"]).all() and min(v["mad"]) > 0 for v in ref["points"].values())
    records, changes, cases = [], [], {}
    for patient in sorted(baseline.Patient.unique(), key=lambda x: int(x[2:])):
        points, files = landmarks(int(patient[2:]))
        prior_path = OUT / patient / "result.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        # This version is deliberately automatic-only; refuse to silently
        # discard manual landmark corrections from a future selected variant.
        assert "reviewed" not in prior["variants"], f"Manual correction requires explicit provenance: {patient}"
        selected = prior["variants"]["automatic"]
        for path in files:
            assert prior["fingerprint"]["landmarks"][path.name] == digest(path), f"Stale landmarks: {patient}"
        before = measures(points, "jung")
        fixed, log = repair(points, ref["points"])
        after = measures(fixed, "jung")
        pre_flags, post_flags = qc(points, profile), qc(fixed, profile)
        pre_qc = "needs_review" if pre_flags else "qc_pass_pending_visual_review"
        post_qc = "needs_review" if post_flags else "qc_pass_pending_visual_review"
        assert pre_qc == selected["release_status"], f"Stale baseline QC: {patient}"
        changed = [k for k in points if not np.array_equal(points[k], fixed[k])]
        assert all(k in ("RPo", "LPo", "ROr", "LOr") for k in changed)
        case = dict(patient=patient, version="fh-repair-evaluation-v2", coordinate_system="LPS",
                    baseline_result_sha256=digest(prior_path), landmark_sha256={p.name: digest(p) for p in files},
                    fh_reference_sha256=digest(ref_path), qc_reference_sha256=digest(OUT / "qc_reference.json"),
                    repaired_points={k: fixed[k].tolist() for k in changed}, repair_log=log,
                    measurements_before=before, measurements_after=after,
                    qc_before=dict(status=pre_qc, flags=pre_flags), qc_after=dict(status=post_qc, flags=post_flags),
                    status="exploratory_pending_expert_review; post-development reevaluation; original result unchanged")
        save(DEST / "patients" / patient / "evaluation.json", case)
        cases[patient] = (points, fixed, case)
        for k, z, action in log:
            changes.append(dict(Patient=patient, point=k, z=z, action=action,
                                before_lps=json.dumps(points[k].tolist()), after_lps=json.dumps(fixed[k].tolist()),
                                qc_before=pre_qc, qc_after=post_qc))
        for metric, key in METRICS.items():
            base_row = baseline[(baseline.Patient == patient) & (baseline.metric == metric)].iloc[0]
            assert np.isclose(before[key], base_row.auto, atol=1e-8, rtol=0), f"Baseline mismatch: {patient}/{metric}"
            records.append(dict(Patient=patient, metric=metric, Set=base_row.Set, excel=base_row.excel,
                                before=before[key], after=after[key], difference_before=before[key]-base_row.excel,
                                difference_after=after[key]-base_row.excel, delta=after[key]-before[key],
                                qc_before=pre_qc, qc_after=post_qc, repaired_points=",".join(changed),
                                source=f"fh_repair/v2/patients/{patient}/evaluation.json"))
    d = pd.DataFrame(records)
    assert len(d) == 216 and not d.duplicated(["Patient", "metric"]).any()
    d["abs_improvement"] = d.difference_before.abs()-d.difference_after.abs()
    assert np.isfinite(d[["excel", "before", "after"]]).all().all()
    d.to_csv(DEST / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(DEST / "repair_log.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for metric, g in d.groupby("metric", sort=False):
        # Paired comparisons on fixed populations; post-QC population is never
        # used to selectively remove hard pre-repair cases.
        for group, subset in [("all", g), ("eval", g[g.Set == "eval"]),
                              ("baseline_qc_pass", g[g.qc_before == "qc_pass_pending_visual_review"])]:
            for version in ("before", "after"):
                v = subset.rename(columns={version: "auto"})
                s = stats(v)
                summaries.append(dict(metric=metric, group=group, version=version, **s,
                                      median_abs=float(np.median(np.abs(v.auto-v.excel)))))
    summary = pd.DataFrame(summaries)
    summary.to_csv(DEST / "summary.csv", index=False, encoding="utf-8-sig")
    latest = baseline.copy()
    for _, row in d.iterrows():
        ix = (latest.Patient == row.Patient) & (latest.metric == row.metric)
        latest.loc[ix, ["auto", "qc", "source", "rule"]] = [row.after, row.qc_after, row.source, "frozen_fh_repair_v2"]
    latest["difference"] = latest.auto-latest.excel
    latest["abs_error"] = latest.difference.abs()
    latest["relative_error_pct"] = np.where(latest.excel > 0, latest.difference/latest.excel*100, np.nan)
    latest.to_csv(DEST / "latest_comparison_pairs.csv", index=False, encoding="utf-8-sig")
    queue = []
    for metric, g in latest[latest.paired].groupby("metric", sort=False):
        kinds = [("absolute_top5", g.abs_error)]
        if metric in ("NAV", "PAV", "PMCA"):
            kinds.append(("relative_top5", g.relative_error_pct.abs()))
        for kind, score in kinds:
            for rank, idx in enumerate(score.nlargest(5).index, 1):
                row = g.loc[idx].to_dict()
                row.update(selection=kind, rank=rank, evidence_section=EVIDENCE[metric])
                queue.append(row)
    queue = pd.DataFrame(queue)
    return d, summary, latest, queue, cases, specs


def make_landmark_evidence(patients, cases):
    from cbct_landmark_review import panel
    from check_landmarks import read_nrrd
    result = {}
    for patient in patients:
        print(f"Image evidence: {patient}", flush=True)
        p, q, case = cases[patient]
        volume, directions, origin = read_nrrd(CB / f"nrrd/{patient}.nrrd")
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        panels = [("N/A/B", ("N", "A", "B"), "sagittal", 55),
                  ("Pog/Me", ("Pog", "Me"), "sagittal", 30),
                  ("RPo", ("RPo",), "sagittal", 28), ("LPo", ("LPo",), "sagittal", 28),
                  ("Or", ("ROr", "LOr"), "coronal", 55),
                  ("Go/Me", ("RGo", "LGo", "Me"), "sagittal", 65),
                  ("RPo", ("RPo",), "coronal", 28), ("LPo", ("LPo",), "coronal", 28)]
        for ax, (label, keys, plane, half) in zip(axes.flat, panels):
            center = np.mean([q[k] for k in keys], axis=0)
            marks = {k: (p[k], color, "+") for k, color in zip(keys, ("red", "lime", "orange"))}
            for k in keys:
                if k in case["repaired_points"]:
                    marks[k+" mirrored"] = (q[k], "cyan", "x")
            panel(ax, volume, directions, origin, center, plane, marks, half=half)
            ax.set_title(label+" | "+ax.get_title())
        fig.suptitle(f"{patient}: original landmarks (+); FH mirrored points (cyan x).\nNative LPS slices, projected markers; off-plane distances in legend. Exploratory, not expert reviewed.")
        fig.tight_layout(rect=(0, 0, 1, .94))
        target = DEST / "figs" / f"{patient}_landmarks.png"
        fig.savefig(target, dpi=95)
        plt.close(fig)
        result[patient] = target
    return result


def make_comparison_figures(d):
    from cbct_report_scatter import plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
    errors, error_axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    for col, metric in enumerate(METRICS):
        g = d[d.metric == metric]
        unit = "mm" if metric.startswith("Nperp") else "°"
        low, high = g[["excel", "before", "after"]].to_numpy().min(), g[["excel", "before", "after"]].to_numpy().max()
        pad = max((high-low)*.08, .1)
        for row, version in enumerate(("before", "after")):
            v = g.assign(auto=g[version], difference=g[f"difference_{version}"], qc=g[f"qc_{version}"])
            plot(axes[row, col], v, f"{metric}: {version}", unit)
            axes[row, col].set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad))
            axes[row, col].lines[0].set_data((low-pad, high+pad), (low-pad, high+pad))
        ax = error_axes[col]
        box = ax.boxplot([g.difference_before, g.difference_after], tick_labels=["Before", "After"], patch_artist=True)
        for patch, color in zip(box["boxes"], ("#94a3b8", "#3b82b6")):
            patch.set(facecolor=color, alpha=.4)
        ax.axhline(0, color="#b3261e", ls="--", lw=1)
        ax.set(title=f"{metric} (n = 54)", ylabel=f"Auto - Excel ({unit})")
        ax.grid(axis="y", alpha=.18)
    fig.savefig(DEST / "figs/fh_before_after_scatter.png", dpi=130, facecolor="white")
    errors.savefig(DEST / "figs/fh_before_after_errors.png", dpi=130, facecolor="white")
    plt.close(fig)
    plt.close(errors)


def evidence_links(metric, patient, figures):
    links = [(OUT / "evidence_review/v1/index.html", EVIDENCE[metric], "기존 영상 검토 절")]
    audit = OUT / "nav_review/v2_label_audit/index.html"
    if metric in ("NAV", "NCWR") and audit.exists():
        links.insert(0, (audit, patient, "54명 비강 라벨 점검: 해당 환자"))
    if patient in figures and metric in METRICS:
        links.insert(0, (figures[patient], "", "환자별 새 랜드마크 단면"))
    individual = OUT / f"evidence_review/v1/figs/pmca_{patient}.png"
    if metric == "PMCA" and individual.exists():
        links.insert(0, (individual, "", "환자별 기존 기도 단면"))
    montage = {"TDI": "tdi_sections.png", "PHI": "phi_ncwr_sections.png", "NCWR": "phi_ncwr_sections.png",
               "PAV": "pav_sections.png", "NAV": "nav_sections.png", "PMCA": ""}.get(metric, "")
    if metric == "PAL":
        links.append((OUT / "eb_review/v2/eb_all_patients.png", "", "전원 Eb 몽타주에서 환자 번호 확인"))
    if montage:
        links.append((OUT / "evidence_review/v1/figs" / montage, "", "기존 전체 몽타주에서 환자 번호 확인"))
    if metric == "NAV":
        # nav_sections shows the earlier image-rule mask, not NAV_ts. Keep
        # only the matching TotalSegmentator review sources for this metric.
        links = [x for x in links if x[0].name != "nav_sections.png"]
        for filename in ("nav_ts_coronal.png", "nav_ts_coronal_low.png"):
            links.append((OUT / "evidence_review/v1/figs" / filename, "", "TotalSegmentator 검토 몽타주(일부 환자)"))
    return links


def build_page(d, summary, queue, figures, specs):
    import os
    def link(path, fragment="", label=None):
        assert path.exists(), path
        url = os.path.relpath(path, DEST).replace("\\", "/")+("#"+fragment if fragment else "")
        return f'<a href="{escape(url, quote=True)}">{escape(label or path.name)}</a>'
    units = {name: unit for name, unit, _ in specs}
    parts = ['''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FH 보정 전후와 큰 오차 환자 검토</title><style>body{max-width:1250px;margin:30px auto;padding:0 18px;font:15px/1.65 system-ui,"Malgun Gothic",sans-serif;color:#17222c;background:#fbfbfa}h2{margin-top:40px;border-top:1px solid #ccd5df;padding-top:20px}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d5dce3;padding:7px;text-align:left}th{background:#edf2f7}.wrap{overflow:auto}.note{background:#edf2f7;padding:14px;border-left:4px solid #2457a6}a{color:#2457a6}img{width:100%;height:auto}code{overflow-wrap:anywhere}td.num{text-align:right;font-variant-numeric:tabular-nums}</style></head><body><h1>FH 보정 전후 비교와 큰 오차 환자 검토</h1>''']
    parts.append('<p class="note">54명에게 고정 FH 규칙(5 MAD, 반대쪽 점의 정중면 대칭)을 동일하게 적용했습니다. 엑셀로 임계값을 조정하지 않았습니다. 기존 result.json·랜드마크·QC 기준은 보존했습니다. 이 결과는 개발 후 재평가이며 전문가 확인 전입니다. 원 측정 좌표가 없어 큰 오차의 원인을 확정할 수 없습니다.</p>')
    parts.append('<p>'+link(OUT / "report/CBCT_보고서_2026-11-10.html", label="기존 보고서")+' · '+link(DEST / "comparison.csv", label="54명 보정 전후 CSV")+' · '+link(DEST / "outliers.csv", label="큰 오차 목록 CSV")+'</p>')
    parts.append('<h2 id="comparison">1. 54명 전체: 같은 환자에서 보정 전후 비교</h2><p>오차 = 자동 − 엑셀. MAE는 평균 절대오차, MedAE는 중앙 절대오차, CCC는 일치도입니다. 환자 제외 없이 동일한 54명을 비교합니다. QC 변화와 관계없이 전체 집계는 54명을 유지합니다.</p><figure><img src="figs/fh_before_after_scatter.png" alt="4지표 보정 전후 54명 산점도"><figcaption>위: 보정 전, 아래: 보정 후. 각 지표의 두 행은 동일한 축 범위이며 54명을 모두 표시합니다.</figcaption></figure><figure><img src="figs/fh_before_after_errors.png" alt="4지표 보정 전후 오차 박스플롯"><figcaption>같은 54명의 보정 전후 오차 분포. 빨간 점선은 오차 0이며 이상치를 포함합니다.</figcaption></figure><div class="wrap"><table><tr><th>지표</th><th>단위</th><th>MAE 전 → 후</th><th>MedAE 전 → 후</th><th>r 전 → 후</th><th>CCC 전 → 후</th><th>평균 차이 전 → 후</th></tr>')
    overall = summary[summary.group == "all"].set_index(["metric", "version"])
    def num(v):
        return f"{v:,.3f}"
    for metric in METRICS:
        before, after = overall.loc[(metric, "before")], overall.loc[(metric, "after")]
        cells = "".join(f'<td class="num">{num(before[k])} → {num(after[k])}</td>' for k in ("mae", "median_abs", "r", "ccc", "bias"))
        parts.append(f'<tr><td>{metric}</td><td>{units[metric]}</td>{cells}</tr>')
    parts.append('</table></div><p>평가군47명과「보정 전 QC 통과」48명을 고정한 비교도 '+link(DEST / "summary.csv", label="전체 요약 CSV")+'에 포함했습니다. 48명은 현재 거리 QC 통과 인원이며, 과거 보고서의 41명 집계는 필터 출처가 확인되지 않아 재사용하지 않았습니다.</p>')
    parts.append('<h2 id="changed">2. 좌표가 바뀐 환자와 영향</h2><div class="wrap"><table><tr><th>환자</th><th>지표</th><th>엑셀</th><th>보정 전</th><th>보정 후</th><th>절대오차 감소</th><th>QC 전 → 후</th><th>근거</th></tr>')
    for _, r in d[d.repaired_points != ""].iterrows():
        view = link(figures[r.Patient], label="원래 점·보정점 단면") if r.Patient in figures else "미생성"
        cells = "".join(f'<td class="num">{num(r[k])}</td>' for k in ("excel", "before", "after", "abs_improvement"))
        parts.append(f'<tr><td>{r.Patient} ({r.repaired_points})</td><td>{r.metric}</td>{cells}<td>{r.qc_before} → {r.qc_after}</td><td>{view}</td></tr>')
    parts.append('</table></div><p>절대오차 감소가 음수이면 악화입니다. QC 통과는 해부학적 정확도나 전문가 승인과 같지 않습니다.</p>')
    parts.append('<h2 id="outliers">3. 보정 후 지표별 큰 오차 환자</h2><p>각 지표의 절대오차 상위5명을 표시했습니다. NAV·PAV·PMCA는 상대오차 상위5명도 추가했습니다. 순위는 검토 우선순위이며, 환자를 제외하거나 규칙을 맞추는 기준이 아닙니다. 4지표는 새 FH 값, 나머지7개는 기존 선택 규칙의 값을 사용합니다. 단위가 다른 지표끼리 오차 크기를 비교하지 않습니다. 비율은 엑셀값이 양수인 경우에만 계산했습니다.</p>')
    queue["evidence_links"] = ""
    queue["review_status"] = "원인 미확정; 영상 근거 확인 및 전문가 검토 필요"
    for metric in queue.metric.drop_duplicates():
        parts.append(f'<h3 id="metric-{metric}">{metric} ({units[metric]})</h3><div class="wrap"><table><tr><th>선정</th><th>순위</th><th>환자</th><th>엑셀</th><th>복원</th><th>오차</th><th>절대오차</th><th>상대오차%</th><th>구분·QC</th><th>영상 근거</th></tr>')
        for idx, r in queue[queue.metric == metric].iterrows():
            sources = evidence_links(metric, r.Patient, figures)
            links = "<br>".join(link(path, fragment, label) for path, fragment, label in sources)
            queue.loc[idx, "evidence_links"] = " | ".join(str(p.relative_to(ROOT))+("#"+f if f else "") for p, f, _ in sources)
            cells = "".join(f'<td class="num">{num(r[k]) if pd.notna(r[k]) else "—"}</td>' for k in ("excel", "auto", "difference", "abs_error", "relative_error_pct"))
            label = "절대오차" if r.selection.startswith("absolute") else "상대오차"
            qc_text = r.qc if pd.notna(r.qc) and r.qc else "별도 QC 상태 없음"
            parts.append(f'<tr><td>{label}</td><td>{r["rank"]}</td><td>{r.Patient}</td>{cells}<td>{r.Set}<br>{qc_text}</td><td>{links}</td></tr>')
        parts.append('</table></div>')
    nasal_note = 'NAV·NCWR은 후속 전수 점검의 환자별 관상면에 추가 연결했습니다.' if (OUT / "nav_review/v2_label_audit/index.html").exists() else 'NAV 라벨 영상은 일부 환자만 있습니다.'
    parts.append('<p class="note">새 랜드마크 단면은 4지표의 큰 오차 환자와 FH 보정 환자를 대상으로 만들었습니다. '+nasal_note+' 다른 지표의 링크는 기존 환자별 그림 또는 전체 몽타주입니다. 링크가 있다는 이유로 전문가 검토가 완료된 것은 아닙니다. 랜드마크 점은 단면에 투영되므로 범례의 off-plane 거리를 함께 확인해야 합니다.</p>')
    parts.append('<h2 id="images">4. 새 환자별 랜드마크 단면</h2>')
    for patient, path in figures.items():
        url = path.relative_to(DEST).as_posix()
        parts.append(f'<details><summary>{patient} 랜드마크 단면</summary><a href="{url}"><img loading="lazy" src="{url}" alt="{patient} 원래 랜드마크와 FH 보정점"></a></details>')
    parts.append('</body></html>')
    (DEST / "index.html").write_text("\n".join(parts), encoding="utf-8")
    queue.to_csv(DEST / "outliers.csv", index=False, encoding="utf-8-sig")
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-links", action="store_true", help="Refresh follow-up evidence links without recalculating measurements or images")
    args = parser.parse_args()
    if args.refresh_links:
        manifest = json.loads((DEST / "provenance.json").read_text(encoding="utf-8"))
        for name, expected in manifest["frozen_originals_sha256"].items():
            assert digest(ROOT / name) == expected, f"Protected source changed: {name}"
        for name, expected in manifest["output_sha256"].items():
            assert digest(DEST / name) == expected, f"Evaluation data changed: {name}"
        d = pd.read_csv(DEST / "comparison.csv").fillna({"repaired_points": ""})
        summary, queue = pd.read_csv(DEST / "summary.csv"), pd.read_csv(DEST / "outliers.csv")
        figures = {p: DEST / "figs" / f"{p}_landmarks.png" for p in manifest["figure_patients"]}
        _, specs, _ = load_pairs()
        build_page(d, summary, queue, figures, specs)
        manifest["output_sha256"]["outliers.csv"] = digest(DEST / "outliers.csv")
        manifest["links_refreshed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["link_refresh_script_sha256"] = digest(Path(__file__))
        audit = OUT / "nav_review/v2_label_audit/audit.csv"
        if audit.exists():
            manifest["followup_nasal_audit_sha256"] = digest(audit)
        save(DEST / "provenance.json", manifest)
        print("Updated 11-metric review links; numerical results and original sources unchanged")
        return
    original = snapshots()
    (DEST / "figs").mkdir(parents=True, exist_ok=True)
    d, summary, latest, queue, cases, specs = calculate()
    make_comparison_figures(d)
    patients = set(queue.loc[queue.metric.isin(METRICS), "Patient"])
    patients |= {p for p, (_, _, c) in cases.items() if c["repaired_points"]}
    figures = make_landmark_evidence(sorted(patients, key=lambda p: int(p[2:])), cases)
    # Save once before constructing HTML links to the queue CSV.
    queue.to_csv(DEST / "outliers.csv", index=False, encoding="utf-8-sig")
    queue = build_page(d, summary, queue, figures, specs)
    assert original == snapshots(), "Protected original artifacts changed"
    changed = d.loc[d.repaired_points != "", "Patient"].unique().tolist()
    assert set(d.loc[d.repaired_points == "", "delta"].round(10)) == {0.0}
    assert len(queue[queue.selection == "absolute_top5"]) == 55
    counts = {kind: int((d.drop_duplicates("Patient")[column] == "qc_pass_pending_visual_review").sum())
              for kind, column in [("before", "qc_before"), ("after", "qc_after")]}
    input_files = [OUT / source for source in latest.source.unique() if (OUT / source).is_file()]
    save(DEST / "provenance.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
         version="fh-repair-evaluation-v2", n_patients=54, changed_patients=changed,
         qc_pass_counts=counts, top_k=5, figure_patients=list(figures),
         frozen_originals_sha256=original,
         input_sources_sha256={str(p.relative_to(ROOT)): digest(p) for p in input_files},
         output_sha256={name: digest(DEST / name) for name in ("comparison.csv", "summary.csv", "repair_log.csv", "latest_comparison_pairs.csv", "outliers.csv")},
         script_sha256=digest(Path(__file__)), repair_script_sha256=digest(ROOT / "python/cbct_fh_repair.py"),
         ceph_script_sha256=digest(ROOT / "python/osa/ceph.py"),
         excel_sha256=digest(ROOT / "Patient_Info_Data_87명_261019.xlsx"),
         validation="216 finite matched measurements; unchanged baseline recomputes; original hashes preserved; unmodified patients have zero delta",
         caveat="Exploratory post-development reevaluation, not independent validation or expert approval"))
    print(summary[summary.group == "all"].round(4).to_string(index=False))
    print(f"Changed: {changed}; QC pass: {counts}; outlier rows: {len(queue)}; evidence patients: {len(figures)}")
    print(f"Review: {DEST / 'index.html'}")


if __name__ == "__main__":
    main()
