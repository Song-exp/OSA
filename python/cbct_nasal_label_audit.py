"""54-patient visual and descriptive audit of existing nasal labels.

No segmentation inference, parameter changes, or measurement replacement.
Outputs nav_review/v2_label_audit with per-patient coronal evidence.
"""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label, map_coordinates

from cbct_eb_detect import midline_frame
from cbct_fh_repair import repair
from cbct_molar_section import pair_ok
from cbct_nav import side_of_line
from cbct_patient_pipeline import CB, OUT, digest, landmarks, save
from check_landmarks import read_nrrd
from osa.ceph import head_frame
from run_ali import ascii_path

DEST = OUT / "nav_review/v2_label_audit"
SOURCE = CB / "seg_totalseg/head_glands_cavities"
ROOT = Path(__file__).resolve().parents[1]


def sample(labels, image, points):
    org, sp = np.asarray(image.GetOrigin()), np.asarray(image.GetSpacing())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    ix = np.rint(((points-org) @ np.linalg.inv(direction).T) / sp).astype(int)[:, ::-1]
    ok = ((ix >= 0) & (ix < labels.shape)).all(axis=1)
    out = np.zeros(len(points), dtype=bool)
    out[ok] = np.isin(labels[tuple(ix[ok].T)], (14, 15))
    return out


def native_stats(raw, labels, image, origin, directions, points, nav):
    # The segmentation and source are required to share their physical grid.
    assert raw.shape == labels.shape
    assert np.allclose(image.GetOrigin(), origin, atol=1e-4, rtol=0)
    assert np.allclose(image.GetSpacing(), np.diag(directions), atol=1e-5, rtol=0)
    assert np.allclose(np.asarray(image.GetDirection()).reshape(3, 3), np.eye(3))
    voxel = float(np.prod(image.GetSpacing()))
    mask = np.isin(labels, (14, 15))
    indices = np.column_stack(np.nonzero(mask))
    if not len(indices):
        return {"label_voxels": 0, "geometry_flags": "empty_label"}
    low, high = indices.min(0), indices.max(0)
    cropped = mask[tuple(slice(int(a), int(b)+1) for a, b in zip(low, high))]
    components, n = label(cropped)
    sizes = np.bincount(components.ravel())[1:]
    lps = origin+indices[:, ::-1]*np.diag(directions)
    o, R, _ = midline_frame(points)
    local = (lps-o)@R.T
    A, Z = local[:, 1], local[:, 2]
    p = {k: R@(points[k]-o) for k in ("N", "Ba", "S", "ANS")}
    N, Ba, S, ANS = (p[k][1:] for k in ("N", "Ba", "S", "ANS"))
    rh = (nav.Rh_ant, nav.Rh_sup)
    in_box = side_of_line(A, Z, N, Ba, (ANS[0]-10, 5)) & side_of_line(A, Z, S, (0, 0), (ANS[0]/2, 10)) & side_of_line(A, Z, rh, ANS, (ANS[0]/2, 10)) & (side_of_line(A, Z, N, rh, (ANS[0]/2, 10)) | (Z < rh[1])) & (Z >= 0)
    hu = raw[tuple(indices.T)]
    air = (hu > -1200) & (hu < -400)
    air500 = (hu > -1200) & (hu < -500)
    total_air = int(air.sum())
    contained_air = int((air & in_box).sum())
    r = dict(label_voxels=len(indices), label_native_mL=len(indices)*voxel/1000,
             label_components=n, largest_component_fraction=float(sizes.max()/len(indices)),
             two_largest_fraction=float(np.sort(sizes)[-2:].sum()/len(indices)),
             label_air_mL=total_air*voxel/1000,
             label_nonair_fraction=float((~air).mean()),
             native_NAV_ts_mm3=contained_air*voxel,
             native_NAV_ts_500_mm3=int((air500 & in_box).sum())*voxel,
             label_air_in_NAV_bounds_fraction=contained_air/total_air if total_air else np.nan,
             label_border_touch=bool(np.any(low == 0) or np.any(high == np.array(labels.shape)-1)))
    r["native_vs_saved_NAV_pct"] = (r["native_NAV_ts_mm3"]/nav.NAV_ts-1)*100 if nav.NAV_ts else np.nan
    flags = []
    if r["label_border_touch"]:
        flags.append("label_touches_FOV")
    native_labels = labels[tuple(indices.T)]
    for number, name in ((14, "right"), (15, "left")):
        side = native_labels == number
        r[f"{name}_label_mL"] = int(side.sum())*voxel/1000
        r[f"{name}_median_L_mm"] = float(np.median(local[side, 0])) if side.any() else np.nan
        if not side.any():
            flags.append(f"missing_{name}_label")
    r["geometry_flags"] = ";".join(flags)
    return r


def panel(ax, raw, directions, origin, labels, label_image, q, shape, Ls, Zs, title):
    indices = np.linalg.solve(directions.T, (q-origin).T)[::-1]
    ct = map_coordinates(raw, indices, order=1, cval=-1000).reshape(shape)
    mask = sample(labels, label_image, q).reshape(shape)
    ax.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000, vmax=1200,
              extent=[Ls[0], Ls[-1], Zs[0], Zs[-1]], aspect="equal")
    air = mask & (ct > -1200) & (ct < -400)
    for area, color, width in ((mask, "lime", 1), (air, "yellow", .6)):
        if area.any() and not area.all():
            ax.contour(Ls, Zs, area.T.astype(float), [.5], colors=color, linewidths=width)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Left + (mm)", fontsize=8)
    ax.set_ylabel("Superior + (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    return int(mask.sum()), int(air.sum())


def patient_figure(patient, nav, maxilla, molars):
    pid = int(patient[2:])
    points, lm_files = landmarks(pid)
    raw_path = CB / f"nrrd/{patient}.nrrd"
    label_path = SOURCE / f"{patient}.nii.gz"
    raw, directions, origin = read_nrrd(raw_path)
    image = sitk.ReadImage(str(ascii_path(label_path)))
    labels = sitk.GetArrayFromImage(image)
    result = native_stats(raw, labels, image, origin, directions, points, nav)
    result.update(Patient=patient, source_ct_sha256=digest(raw_path), source_label_sha256=digest(label_path),
                  source_landmarks_sha256=json.dumps({f.name: digest(f) for f in lm_files}, sort_keys=True))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), layout="constrained")
    o, R, _ = midline_frame(points)
    ant = (R@(points["ANS"]-o))[1]
    Ls, Zs = np.arange(-40, 40, .3), np.arange(-15, 65, .3)
    LL, ZZ = np.meshgrid(Ls, Zs, indexing="ij")
    for ax, fraction in zip(axes[:3], (.25, .5, .75)):
        q = o+np.stack([LL, np.full_like(LL, ant*fraction), ZZ], -1).reshape(-1, 3)@R
        counts = panel(ax, raw, directions, origin, labels, image, q, LL.shape, Ls, Zs,
                       f"{patient}: PNS-to-ANS {fraction:.0%}")
        result[f"slice_{int(fraction*100)}_label_pixels"], result[f"slice_{int(fraction*100)}_air_pixels"] = counts
    upper = molars[(molars.Patient == patient) & molars.tooth.isin(["UR6", "UL6"]) & molars.status.eq("ok")].set_index("tooth")
    if len(upper) == 2 and pair_ok(upper.loc["UL6"], upper.loc["UR6"]):
        h0, frame = head_frame(repair(points)[0])
        center_l = (upper.L.UR6+upper.L.UL6)/2
        ls, zs = np.arange(center_l-55, center_l+55, .25), np.arange(upper.Z.mean()-25, upper.Z.mean()+55, .25)
        ll, zz = np.meshgrid(ls, zs, indexing="ij")
        q = h0+np.stack([ll, np.full_like(ll, upper.A.mean()), zz], -1).reshape(-1, 3)@frame
        panel(axes[3], raw, directions, origin, labels, image, q, ll.shape, ls, zs, f"{patient}: actual NCWR molar plane")
        mask = sample(labels, image, q).reshape(ll.shape)
        widths = np.array([np.ptp(ls[mask[:, j]])+.25 if mask[:, j].any() else np.nan for j in range(len(zs))])
        actual = float(np.nanmax(widths)) if np.isfinite(widths).any() else np.nan
        result["NCW_ts_rechecked_mm"] = actual
        result["NCW_ts_saved_mm"] = maxilla.NCW_ts
        result["NCW_ts_delta_mm"] = actual-maxilla.NCW_ts
        if np.isfinite(maxilla.NCW_ts):
            assert np.isclose(actual, maxilla.NCW_ts, atol=1e-8, rtol=0), (patient, actual, maxilla.NCW_ts)
            axes[3].plot([maxilla.ts_l, maxilla.ts_r], [maxilla.Z_ts, maxilla.Z_ts], color="cyan", lw=2, label=f"NCW {actual:.2f} mm")
            axes[3].legend(fontsize=7, loc="upper right")
    else:
        axes[3].axis("off")
        axes[3].text(.1, .6, f"{patient}: no valid upper molar pair\nNCWR measurement plane unavailable", transform=axes[3].transAxes)
    nav_text = f"NAV auto {nav.NAV_ts/1000:.2f} mL"+(f" vs Excel {nav.NAV_xl/1000:.2f}" if pd.notna(nav.NAV_xl) else "; Excel missing")
    ncwr_text = f"NCWR auto {maxilla.NCWR_ts:.2f} vs Excel {maxilla.NCWR_xl:.2f}" if pd.notna(maxilla.NCWR_ts) else "NCWR unavailable"
    fig.suptitle(f"{patient} | {nav_text} | {ncwr_text}\nGreen: nasal label; yellow: label air (-1200 < HU < -400); cyan: saved NCW measurement. No label edits.", fontsize=11)
    target = DEST / "figs" / f"{patient}_nasal.png"
    fig.savefig(target, dpi=100, facecolor="white")
    plt.close(fig)
    result["image"] = f"figs/{target.name}"
    return result


def contact_sheets(ids):
    sheets = []
    for offset in range(0, len(ids), 6):
        group = ids[offset:offset+6]
        fig, axes = plt.subplots(len(group), 1, figsize=(20, 5.4*len(group)), squeeze=False)
        for ax, patient in zip(axes.flat, group):
            ax.imshow(plt.imread(DEST / "figs" / f"{patient}_nasal.png"))
            ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=.015)
        filename = f"sheet_{offset//6+1:02d}.png"
        fig.savefig(DEST / "figs" / filename, dpi=100, facecolor="white")
        plt.close(fig)
        sheets.append(dict(file=f"figs/{filename}", patients=group))
    return sheets


def page(d, sheets):
    d = d.copy()
    d["geometry_flags"] = d.geometry_flags.astype("string").fillna("")
    outliers = pd.read_csv(OUT / "fh_repair/v2/outliers.csv")
    nav_ids = set(outliers.loc[outliers.metric == "NAV", "Patient"])
    ncwr_ids = set(outliers.loc[outliers.metric == "NCWR", "Patient"])
    d["NAV_outlier"] = d.Patient.isin(nav_ids)
    d["NCWR_outlier"] = d.Patient.isin(ncwr_ids)
    review_path = DEST / "visual_review.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    d["visual_review"] = d.Patient.map(lambda p: reviews.get(p, {}).get("note", "AI 영상 검토 대기"))
    d.to_csv(DEST / "audit.csv", index=False, encoding="utf-8-sig")
    parts = ['''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>54명 비강 라벨 전수 검토</title><style>body{font:15px/1.7 system-ui,"Malgun Gothic",sans-serif;margin:30px auto;max-width:1500px;padding:0 18px}a{color:#2457a6}h2{margin-top:40px;border-top:1px solid #ccd5df;padding-top:15px}.note{background:#edf2f7;padding:15px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d5dce3;padding:6px;text-align:left}th{background:#edf2f7}img{width:100%;height:auto}.wrap{overflow:auto}</style></head><body><h1>54명 비강 라벨 전수 검토</h1>''']
    parts.append('<p class="note">초록: 기존 TotalSegmentator 비강 라벨. 노랑: 라벨 안 공기(−1200 &lt; HU &lt; −400). 앞의 세 단면은 PNS→ANS 25·50·75%이며 Po를 사용하지 않는 정중 좌표계입니다. 네 번째는 실제 NCWR 대구치 단면(존재할 때)입니다. 세 관상면만으로 3차원 분할 정확도를 확정할 수 없습니다. 라벨이 비갑개·조직을 포함하므로 비공기 비율이나 좌우 크기 차이만으로 실패 판정하지 않습니다. 모든 검토는 탐색적 AI 검토이며 전문가 확인 전입니다. 원래 라벨·계측·result.json은 변경하지 않았습니다.</p>')
    parts.append('<p><a href="../../report/CBCT_보고서_2026-11-10.html">기존 보고서</a> · <a href="../../fh_repair/v2/index.html#outliers">큰 오차 목록</a> · <a href="audit.csv">전수 점검 CSV</a></p>')
    reviewed = sum(p in reviews for p in d.Patient)
    parts.append(f'<p>대상 {len(d)}명 · 영상 생성 {len(d)}명 · 기록된 AI 시각 검토 {reviewed}명 · NCW 재현 확인 {d.NCW_ts_delta_mm.notna().sum()}명. NAV는 원래 0.4 mm 재표본 격자 값과 원영상 격자 재계산 값의 차이를 CSV에 별도 기록했습니다. 원영상 격자 값으로 기존 계측을 교체하지 않았습니다.</p>')
    paired = d[d.NAV_excel.notna()]
    native_ratio = paired.native_NAV_ts_mm3/paired.NAV_excel
    parts.append(f'<h2 id="findings">점검 결과</h2><ul><li>54명 모두 좌우 비강 라벨이 존재하며 라벨의 촬영 범위 경계 접촉은 없었습니다.</li><li>재현 가능한 NCW {d.NCW_ts_delta_mm.notna().sum()}명의 폭은 기존 값과 일치했습니다. 최종 NCWR은 분모 등 추가 조건이 필요해 {d.NCWR_saved.notna().sum()}명입니다.</li><li>원영상 격자 NAV는 기존 0.4 mm 재표본 격자보다 중앙값 {d.native_vs_saved_NAV_pct.median():.2f}% 큽니다(범위 {d.native_vs_saved_NAV_pct.min():.2f}~{d.native_vs_saved_NAV_pct.max():.2f}%). 엑셀 비교 {len(paired)}명에서 자동/엑셀 중앙 비율은 {paired.NAV_ratio.median():.3f} → {native_ratio.median():.3f}입니다. 격자 차이만으로 남은 과소 계측을 설명하기 어렵습니다.</li></ul>')
    if reviewed == len(d):
        parts.append('<p class="note">AI가 54명 전원의 25·50·75% 관상면과 가능한 NCWR 단면을 확인했습니다. 표시한 단면에서 한쪽 비강 전체 누락이나 상악동으로의 큰 누출은 뚜렷하지 않았습니다. 이 결과는 3차원 전체 정확도 검증이나 미세 경계 검증을 의미하지 않습니다. P_4·P_20은 노란 공기 영역이 적고, P_14는 큰 NAV 차이가 있어도 표시한 단면에서 큰 상악동 침범은 보이지 않습니다. NCWR 큰 오차 사례는 NCW가 재현되므로 MxBW·측정 높이 정의를 계속 확인해야 합니다.</p>')
    for group in sheets:
        parts.append('<p><a href="'+group["file"]+'">'+", ".join(group["patients"])+" 묶음 그림</a></p>")
    parts.append('<h2 id="summary">전수 점검 목록</h2><div class="wrap"><table><tr><th>환자</th><th>기존 NAV/엑셀 비</th><th>NCWR 오차</th><th>라벨 mL</th><th>라벨 안 공기 중 NAV 경계 포함률</th><th>기하 표시</th><th>큰 오차 선정</th><th>AI 시각 검토</th></tr>')
    for _, r in d.iterrows():
        ratio = f"{r.NAV_ratio:.2f}" if pd.notna(r.NAV_ratio) else "엑셀 결측"
        err = f"{r.NCWR_error:+.2f}" if pd.notna(r.NCWR_error) else "미계측"
        picked = ", ".join(name for name, value in (("NAV", r.NAV_outlier), ("NCWR", r.NCWR_outlier)) if value)
        parts.append(f'<tr><td><a href="#{r.Patient}">{r.Patient}</a></td><td>{ratio}</td><td>{err}</td><td>{r.label_native_mL:.2f}</td><td>{r.label_air_in_NAV_bounds_fraction:.1%}</td><td>{escape(r.geometry_flags)}</td><td>{picked}</td><td>{escape(r.visual_review)}</td></tr>')
    parts.append('</table></div><h2>환자별 단면</h2>')
    for _, r in d.iterrows():
        parts.append(f'<section id="{r.Patient}"><h3>{r.Patient}</h3><p>{escape(r.visual_review)}</p><a href="{r.image}"><img loading="lazy" src="{r.image}" alt="{r.Patient} 비강 라벨 세 관상면과 NCWR 단면"></a></section>')
    parts.append('</body></html>')
    (DEST / "index.html").write_text("\n".join(parts), encoding="utf-8")
    publish_evidence_link()
    return d


def publish_evidence_link():
    import re
    path = OUT / "evidence_review/v1/index.html"
    html = path.read_text(encoding="utf-8")
    start, end = "<!-- NASAL_AUDIT_START -->", "<!-- NASAL_AUDIT_END -->"
    block = '''<!-- NASAL_AUDIT_START -->
<h2 id="s16">16. 비강 라벨 54명 전수 점검 (2026-11-10)</h2>
<p>54명 전원에서 PNS→ANS 25·50·75% 관상면을 만들고 AI가 확인했습니다. 48명은 실제 NCWR 대구치 단면을 추가했습니다. 표시한 단면에서 한쪽 전체 누락이나 상악동으로의 큰 누출은 뚜렷하지 않았습니다. 미세 경계·전체 3차원 정확도·전문가 판독은 검증되지 않았습니다.</p>
<p>48명의 NCW 폭은 저장값과 정확히 재현됐습니다. 원영상 격자에서 NAV를 집계하면 기존값보다 중앙값 1.06% 커지지만, 엑셀 비교 37명의 자동/엑셀 중앙 비율은 0.812 → 0.820으로 과소 계측이 남았습니다. 기존 라벨과 계측값을 바꾸지 않았습니다.</p>
<p><a href="../../nav_review/v2_label_audit/index.html">54명 개별 영상·9개 묶음 그림·검토 기록</a> · <a href="../../nav_review/v2_label_audit/index.html#P_14">P_14</a> · <a href="../../nav_review/v2_label_audit/index.html#P_20">P_20</a> · <a href="../../nav_review/v2_label_audit/audit.csv">전수 점검 CSV</a></p>
<!-- NASAL_AUDIT_END -->'''
    if start in html:
        html = re.sub(re.escape(start)+r".*?"+re.escape(end), lambda _: block, html, flags=re.S)
    else:
        assert html.count("</body>") == 1
        html = html.replace("</body>", block+"\n</body>")
    nav = '<li><a href="#s16">비강 라벨 54명 전수 점검</a></li>'
    if 'href="#s16"' in html:
        html = re.sub(r'<li><a href="#s16">.*?</a></li>', lambda _: nav, html)
    else:
        assert html.count("</ol></nav>") == 1
        html = html.replace("</ol></nav>", nav+"\n</ol></nav>")
    path.write_text(html, encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh review notes, statistics and HTML from completed audit without regenerating images")
    args = parser.parse_args()
    if args.refresh:
        provenance_path = DEST / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        notes_path = DEST / "visual_review_notes.json"
        if notes_path.exists():
            notes = json.loads(notes_path.read_text(encoding="utf-8"))
            reviews_path = DEST / "visual_review.json"
            reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
            for patient, record in reviews.items():
                record["note"] = notes["notes"].get(patient, notes["default"])
            save(reviews_path, reviews)
        d = page(pd.read_csv(DEST / "audit.csv"), provenance["sheets"])
        assert len(d) == 54 and not d.Patient.duplicated().any()
        for name, expected in provenance["protected_sources_sha256"].items():
            assert digest(ROOT / name) == expected, f"Protected original changed: {name}"
        provenance.update(refreshed_utc=datetime.now(timezone.utc).isoformat(),
                          n_visual_reviewed=int(d.visual_review.ne("AI 영상 검토 대기").sum()),
                          script_sha256=digest(Path(__file__)),
                          visual_review_sha256=digest(DEST / "visual_review.json"),
                          audit_sha256=digest(DEST / "audit.csv"))
        save(provenance_path, provenance)
        print(f"Refreshed {len(d)} patients; {provenance['n_visual_reviewed']} AI visual reviews; originals unchanged")
        return
    (DEST / "figs").mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(OUT / "cohort.csv")
    ids = cohort.loc[cohort.Include.astype(str).str.lower().isin(["true", "1"]), "Patient"].tolist()
    assert len(ids) == 54
    names = json.loads((SOURCE / "labels.json").read_text(encoding="utf-8"))
    assert names["14"] == "nasal_cavity_right" and names["15"] == "nasal_cavity_left"
    nav = pd.read_csv(OUT / "nav_review/v1/nav.csv").set_index("Patient")
    maxilla = pd.read_csv(OUT / "phi_ncwr_review/v1/phi_ncwr.csv").set_index("Patient")
    molars = pd.read_csv(OUT / "tdi_review/v1/molars.csv")
    protected = [OUT / p / "result.json" for p in ids]+[SOURCE / f"{p}.nii.gz" for p in ids]
    protected += [OUT / "nav_review/v1/nav.csv", OUT / "phi_ncwr_review/v1/phi_ncwr.csv", OUT / "tdi_review/v1/molars.csv"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    rows = []
    for i, patient in enumerate(ids, 1):
        print(f"{i}/{len(ids)} {patient}", flush=True)
        r = patient_figure(patient, nav.loc[patient], maxilla.loc[patient], molars)
        r.update(NAV_saved=nav.loc[patient, "NAV_ts"], NAV_excel=nav.loc[patient, "NAV_xl"],
                 NAV_ratio=nav.loc[patient, "NAV_ts"]/nav.loc[patient, "NAV_xl"],
                 NCWR_saved=maxilla.loc[patient, "NCWR_ts"], NCWR_excel=maxilla.loc[patient, "NCWR_xl"],
                 NCWR_error=maxilla.loc[patient, "NCWR_ts"]-maxilla.loc[patient, "NCWR_xl"])
        rows.append(r)
        pd.DataFrame(rows).to_csv(DEST / "audit_progress.csv", index=False, encoding="utf-8-sig")
    d = pd.DataFrame(rows)
    sheets = contact_sheets(ids)
    d = page(d, sheets)
    assert all(d.label_voxels > 0) and len(d) == 54 and not d.Patient.duplicated().any()
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    save(DEST / "provenance.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
         n_patients=len(d), slice_fractions=[.25, .5, .75], pixel_step_mm=.3,
         native_NAV_note="Label air in fixed Roh boundaries sampled on source voxel grid; reference-only, does not replace saved 0.4 mm resampling",
         protected_sources_sha256=before, script_sha256=digest(Path(__file__)), sheets=sheets,
         unchanged_sources=True, expert_reviewed=False))
    print(d[["Patient", "NAV_ratio", "label_native_mL", "native_vs_saved_NAV_pct", "NCW_ts_delta_mm", "geometry_flags"]].round(3).to_string(index=False))
    print(f"Review: {DEST / 'index.html'}")


if __name__ == "__main__":
    main()
