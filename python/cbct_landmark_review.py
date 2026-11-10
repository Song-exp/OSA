"""One-patient image evidence for Po review; candidates never replace ALI output.

Contralateral reflection is a search aid only. It uses midline landmarks, not
Excel values, the faulty FH plane, or an outcome-selected population template.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cbct_patient_pipeline import CB, OUT, PAIRS, digest, landmarks, patient_id, save, qc, source_signature
from check_landmarks import read_nrrd
from osa.ceph import measures
from pilot_measure import excel


def reflection(points, label):
    keys = ["N", "S", "Ba", "ANS", "PNS"]
    mid = np.array([points[k] for k in keys])
    center = mid.mean(axis=0)
    normal = np.linalg.svd(mid-center)[2][-1]
    opposite = points[{"RPo": "LPo", "LPo": "RPo"}[label]]
    return opposite - 2*np.dot(opposite-center, normal)*normal, {
        "landmarks": keys, "center_lps_mm": center.tolist(),
        "normal": normal.tolist(),
        "max_plane_residual_mm": float(np.max(abs((mid-center)@normal))),
    }


def panel(ax, volume, directions, origin, center, plane, marks, half=28):
    ijk = np.rint(np.linalg.solve(directions.T, center-origin)).astype(int)
    # This exporter produces axis-aligned positive-LPS volumes. Refuse others
    # instead of silently drawing an incorrect physical-coordinate overlay.
    if not np.allclose(directions, np.diag(np.diag(directions))) or np.any(np.diag(directions)<=0):
        raise ValueError("Review renderer requires positive axis-aligned LPS directions")
    axis, horizontal, vertical = {"axial": (2,0,1), "coronal": (1,0,2), "sagittal": (0,1,2)}[plane]
    if not 0<=ijk[axis]<volume.shape[::-1][axis]:
        ax.set_title("Outside image"); return
    image = {"axial": lambda: volume[ijk[2]],
             "coronal": lambda: volume[:,ijk[1],:],
             "sagittal": lambda: volume[:,:,ijk[0]]}[plane]()
    spacing = np.diag(directions)
    lo = origin-spacing/2
    hi = origin+(np.array(volume.shape[::-1])-0.5)*spacing
    ax.imshow(image, cmap="gray", vmin=-700, vmax=2000, origin="lower",
              extent=[lo[horizontal],hi[horizontal],lo[vertical],hi[vertical]])
    for name, (p, color, marker) in marks.items():
        # All markers are projected onto this slice; off-plane distance explicit.
        delta = p[axis]-(origin[axis]+ijk[axis]*spacing[axis])
        ax.plot(p[horizontal],p[vertical],marker,color=color,ms=9,mew=1.7,
                label=f"{name} (off-plane {delta:+.1f} mm)")
    ax.set_xlim(center[horizontal]-half,center[horizontal]+half)
    ax.set_ylim(center[vertical]-half,center[vertical]+half)
    ax.set_aspect("equal")
    names=["LPS x", "LPS y", "LPS z"]
    ax.set_xlabel(names[horizontal]+" (mm)");ax.set_ylabel(names[vertical]+" (mm)")
    ax.set_title(f"{plane}; {names[axis]}={origin[axis]+ijk[axis]*spacing[axis]:.2f}")
    if marks:ax.legend(fontsize=6,loc="upper right")


def review(pid, focus="RPo", candidate=None, reason=None):
    manifest = OUT/"cohort.csv"
    cohort = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    match = cohort[cohort.Patient==f"P_{pid}"]
    if len(match)!=1 or match.iloc[0].Include.lower() not in ("true","1"):
        raise ValueError("Patient is not included in the current cohort.csv")
    points, files = landmarks(pid)
    image = CB/f"nrrd/P_{pid}.nrrd"
    with image.open("rb") as stream:
        header=stream.read(8192).split(b"\n\n",1)[0].decode("ascii")
    if "space: left-posterior-superior" not in header or "endian: little" not in header:
        raise ValueError("Review reader requires little-endian LPS NRRD")
    volume, directions, origin = read_nrrd(image)
    mirror, plane = reflection(points, focus)
    out = OUT/f"P_{pid}/review"; out.mkdir(parents=True,exist_ok=True)
    stem=focus if candidate is not None else focus+"_search"
    marks = {"ALI": (points[focus],"red","+"), "reflected search": (mirror,"cyan","x")}
    centers = [("automatic",points[focus]),("search only",mirror)]
    if candidate is not None:
        candidate = np.asarray(candidate,dtype=float)
        if not reason or not np.isfinite(candidate).all():
            raise ValueError("A candidate needs finite coordinates and an image-based reason")
        voxel = np.linalg.solve(directions.T,candidate-origin)
        if np.any(voxel<0) or np.any(voxel>np.array(volume.shape[::-1])-1):
            raise ValueError("Candidate is outside image")
        marks["image candidate"]=(candidate,"lime","+")
        centers.append(("candidate: NOT approved",candidate))
    fig, axes = plt.subplots(len(centers),3,figsize=(18,5.4*len(centers)),squeeze=False)
    for row,(name,center) in enumerate(centers):
        for col,view in enumerate(["axial","coronal","sagittal"]):
            panel(axes[row,col],volume,directions,origin,center,view,marks)
        axes[row,0].set_ylabel(name+"\n"+axes[row,0].get_ylabel())
    fig.suptitle(f"P_{pid} {focus}: physical LPS coordinates; markers projected; Excel hidden")
    fig.tight_layout();fig.savefig(out/f"{stem}_mpr.png",dpi=150);plt.close(fig)
    # Broad sagittal sweep locates the outer canal without treating reflection
    # as a correction. The exact physical x of every displayed slice is shown.
    fig, axes = plt.subplots(3,5,figsize=(20,12))
    for ax,offset in zip(axes.flat,np.arange(-14,16,2)):
        center=mirror.copy();center[0]+=offset
        panel(ax,volume,directions,origin,center,"sagittal",{},half=24)
    fig.suptitle(f"P_{pid} {focus}: sagittal search slices (no Excel-derived coordinates)")
    fig.tight_layout();fig.savefig(out/f"{focus}_sagittal_search.png",dpi=150);plt.close(fig)
    evidence={"patient":f"P_{pid}","focus":focus,"coordinate_system":"LPS",
              "created_utc":datetime.now(timezone.utc).isoformat(),
              "cohort_manifest_sha256":digest(manifest),"source_identity":source_signature(pid),
              "image_sha256":digest(image),"landmarks_sha256":{p.name:digest(p) for p in files},
              "automatic":points[focus].tolist(),"reflected_search_only":mirror.tolist(),
              "reflection_plane":plane,
              "definition":"External and uppermost point of the auditory canal",
              "definition_source":"cbct_인자추출.pdf, page 4, Table 1",
              "definition_source_sha256":digest(CB.parents[1]/"cbct_인자추출.pdf"),
              "review_code_sha256":digest(Path(__file__)),
              "status":"awaiting_image_candidate" if candidate is None else "candidate_requires_expert_review",
              "warning":"Reflection is not a measured Po. Candidate is not approved or applied to the pipeline."}
    if candidate is not None:
        evidence["candidate"]={"position_lps_mm":candidate.tolist(),"reason":reason,
                               "selected_by":"assistant visual inspection; not a human specialist annotation",
                               "displacement_mm":float(np.linalg.norm(candidate-points[focus]))}
        changed=dict(points);changed[focus]=candidate
        before,after=measures(points),measures(changed)
        profile_path=OUT/"qc_reference.json"
        profile=json.loads(profile_path.read_text(encoding="utf-8"))
        evidence["qc_reference_sha256"]=digest(profile_path)
        evidence["automatic_qc_flags"]=qc(points,profile)
        evidence["candidate_qc_flags"]=qc(changed,profile)
        # A deterministic local perturbation grid, not a confidence interval or
        # a validated landmark repeatability estimate. Never select by Excel.
        sensitivity=[]
        for offset in product([-1.,0.,1.],repeat=3):
            shifted=dict(changed);shifted[focus]=candidate+np.array(offset)
            voxel=np.linalg.solve(directions.T,shifted[focus]-origin)
            if np.any(voxel<0) or np.any(voxel>np.array(volume.shape[::-1])-1):
                raise ValueError("Sensitivity grid exceeds image extent")
            values=measures(shifted)
            sensitivity.append({"dx_mm":offset[0],"dy_mm":offset[1],"dz_mm":offset[2],
                                **{k:values[k] for k in PAIRS}})
        grid=pd.DataFrame(sensitivity)
        grid.to_csv(out/f"{focus}_sensitivity_grid.csv",index=False,encoding="utf-8-sig")
        evidence["sensitivity"]={"description":"27-point Cartesian grid: each axis -1/0/+1 mm; not a confidence interval",
                                  "variable_ranges":{k:[float(grid[k].min()),float(grid[k].max())] for k in PAIRS}}
        xl=excel().loc[pid]
        rows=[{"Patient":f"P_{pid}","Variable":k,"Unit":"degree" if k in ["ANB","MPFH_GoMe"] else "mm",
               "Automatic":before[k],"ImageCandidate":after[k],"Excel":float(xl[x]),
               "Automatic_minus_Excel":before[k]-float(xl[x]),
               "Candidate_minus_Excel":after[k]-float(xl[x]),
               "Candidate_grid_min":float(grid[k].min()),"Candidate_grid_max":float(grid[k].max()),
               "Status":"candidate_requires_expert_review"} for k,x in PAIRS.items()]
        pd.DataFrame(rows).to_csv(out/f"{focus}_candidate_comparison.csv",index=False,encoding="utf-8-sig")
        table="".join(f"<tr><td>{escape(r['Variable'])}</td><td>{r['Automatic']:.2f}</td><td>{r['ImageCandidate']:.2f}</td><td>{r['Excel']:.2f}</td><td>{r['Automatic_minus_Excel']:+.2f}</td><td>{r['Candidate_minus_Excel']:+.2f}</td><td>{r['Candidate_grid_min']:.2f}–{r['Candidate_grid_max']:.2f}</td><td>{r['Unit']}</td></tr>" for r in rows)
        report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>P_{pid} {focus} 영상 검토</title>
<style>body{{font:17px system-ui,sans-serif;max-width:1700px;margin:36px auto;padding:0 24px;color:#17222c}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ccc;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;height:auto}}.note{{background:#fff4d8;padding:18px;line-height:1.7}}a{{color:#0759a8}}code{{font-size:15px}}</style>
<h1>P_{pid} · {focus} 원본 영상 검토</h1><p class="note">수정 후보이며 전문가 검수 전입니다. 원본 자동 좌표와 본 분석 계측값은 변경하지 않았습니다. 엑셀은 비교값이며 정답으로 확정된 수작업 좌표가 없습니다.</p>
<p>Po 정의: 외이도 외측·최상방 경계. 근거: cbct_인자추출.pdf 4쪽 Table 1.</p>
<p>자동 LPS 좌표(mm): <code>{points[focus].round(3).tolist()}</code> → 영상 후보: <code>{candidate.round(3).tolist()}</code> (이동 {evidence['candidate']['displacement_mm']:.2f} mm)</p>
<p>선택 근거: {escape(reason)} 반대편 Po의 반사 위치는 탐색 범위에만 사용했습니다. 후보 위치는 AI의 영상 검토 결과이며 전문가가 승인한 좌표가 아닙니다.</p>
<table><thead><tr><th>지표</th><th>자동</th><th>영상 후보</th><th>엑셀</th><th>자동−엑셀</th><th>후보−엑셀</th><th>후보 ±1 mm 격자 결과</th><th>단위</th></tr></thead><tbody>{table}</tbody></table>
<p>±1 mm 격자는 각 축 −1/0/+1 mm를 조합한 27개 위치입니다. 해부학적 허용 오차·신뢰구간·정확도 기준을 뜻하지 않습니다. 후보 거리 QC 표시는 {len(evidence['candidate_qc_flags'])}개이며, 0개여도 좌표가 맞다는 보증은 아닙니다.</p>
<p><a href="{focus}_candidate_comparison.csv">비교 CSV</a> · <a href="{focus}_evidence.json">좌표·입력 근거 JSON</a> · <a href="{focus}_sensitivity_grid.csv">27개 위치 결과</a></p>
<h2>자동 위치 · 탐색 위치 · 영상 후보</h2><p>빨강: 자동 / 하늘색: 반대편 반사 탐색 위치 / 초록: 영상 후보. 다른 단면의 점도 투영 표시하며, 범례에 단면에서 떨어진 거리를 표시했습니다. 이미지를 누르면 원래 크기로 볼 수 있습니다.</p>
<a href="{focus}_mpr.png"><img src="{focus}_mpr.png" alt="원본 직교 단면과 랜드마크"></a>
<h2>주변 시상면 탐색</h2><a href="{focus}_sagittal_search.png"><img src="{focus}_sagittal_search.png" alt="시상면 연속 단면"></a>
<p>전문가가 실제 외이도 외측 상연을 확인한 뒤 별도 corrections JSON으로 등록해야 본 환자 파이프라인의 reviewed 변형을 계산합니다. 후보 비교는 독립 검증 성능에 합산하지 않습니다.</p></html>'''
        (out/f"{focus}_review.html").write_text(report,encoding="utf-8")
    save(out/f"{stem}_evidence.json",evidence)
    print(json.dumps({"patient":f"P_{pid}","output":str(out),"reflection_search":mirror.tolist(),"status":evidence['status']}))
    return evidence


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient",required=True,type=patient_id)
    ap.add_argument("--focus",choices=["RPo","LPo"],default="RPo")
    ap.add_argument("--candidate",nargs=3,type=float,metavar=("X","Y","Z"))
    ap.add_argument("--reason")
    args=ap.parse_args()
    review(args.patient,args.focus,args.candidate,args.reason)
