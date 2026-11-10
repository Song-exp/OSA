"""Add reproducible Excel-versus-restoration plots to the existing CBCT report.

Reads existing measurements only; never runs inference or changes result.json.
Run from any directory: .venv/Scripts/python.exe python/cbct_report_scatter.py
"""
from html import escape
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Result_py/CBCT/patient_pipeline/v2"
REPORT = BASE / "report/CBCT_보고서_2026-11-10.html"
PILOT = {5, 7, 9, 24, 30, 32, 41, 44, 50, 54, 60, 69}
START, END = "<!-- CBCT_SCATTER_START -->", "<!-- CBCT_SCATTER_END -->"


def read(relative):
    return pd.read_csv(BASE / relative, encoding="utf-8-sig")


def load_pairs(fh_repair=False):
    cohort = read("cohort.csv")
    ids = set(cohort.loc[cohort.Include.astype(str).str.lower().isin(["true", "1"]), "Patient"])
    rows = []
    specs = []
    # Read the selected variant from the original result, without recalculation.
    for key, name, unit in [("ANB", "ANB", "°"), ("Nperp_A", "Nperp-A", "mm"),
                            ("Nperp_Pog", "Nperp-Pog", "mm"), ("MPFH_GoMe", "MP-FH", "°")]:
        specs.append((name, unit, "P_n/result.json: reviewed 우선, 없으면 automatic; FH 신규 보정 미반영"))
        for patient in sorted(ids, key=lambda p: int(p[2:])):
            result = json.loads((BASE / patient / "result.json").read_text(encoding="utf-8"))
            variant = "reviewed" if "reviewed" in result["variants"] else "automatic"
            data = result["variants"][variant]
            pair = data["excel_comparison"][key]
            rows.append(dict(metric=name, Patient=patient, excel=pair["excel"], auto=pair["auto"],
                             qc=data["release_status"], source=f"{patient}/result.json", rule=variant))
    for path, name, auto, xl, unit in [
        ("tdi_review/v1/tdi.csv", "TDI", "TDI_rc", "TDI_xl", "mm"),
        ("phi_ncwr_review/v1/phi_ncwr.csv", "PHI", "PHI_rc", "PHI_xl", "%"),
        ("eb_review/v2/eb_candidates.csv", "PAL", "PAL_mm", None, "mm"),
        ("pav_review/v1/pav.csv", "PAV", "PAV_fh", "PAV_xl", "mm³"),
        ("nav_review/v1/nav.csv", "NAV", "NAV_ts", "NAV_xl", "mm³"),
        ("phi_ncwr_review/v1/phi_ncwr.csv", "NCWR", "NCWR_ts", "NCWR_xl", "%"),
        ("pmca_review/v1/pmca_candidates.csv", "PMCA", "PMCA_center", None, "mm²"),
    ]:
        d = read(path)
        if xl is None:
            sheet = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None).iloc[3:]
            sheet = sheet[sheet[0].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x))]
            lookup = dict(zip(sheet[0].astype(int), pd.to_numeric(sheet[32 if name == "PAL" else 31], errors="coerce")))
            xl = name + "_xl"
            d[xl] = d.Patient.map(lambda p: lookup.get(int(p[2:]), np.nan))
        specs.append((name, unit, f"{path}：{auto}"))
        for _, row in d[d.Patient.isin(ids)].iterrows():
            rows.append(dict(metric=name, Patient=row.Patient, excel=row[xl], auto=row[auto],
                             qc="", source=path, rule=auto))
    d = pd.DataFrame(rows)
    assert not d.duplicated(["metric", "Patient"]).any(), "Duplicate patient/metric"
    d["Set"] = d.Patient.map(lambda p: "pilot" if int(p[2:]) in PILOT else "eval")
    d[["excel", "auto"]] = d[["excel", "auto"]].apply(pd.to_numeric, errors="coerce")
    d["paired"] = np.isfinite(d.excel) & np.isfinite(d.auto)
    d["difference"] = d.auto - d.excel
    d["qc_before"] = d.qc
    if fh_repair:
        provenance = json.loads((BASE / "fh_repair/v2/provenance.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((BASE / "fh_repair/v2/comparison.csv").read_bytes()).hexdigest() == provenance["output_sha256"]["comparison.csv"], "FH evaluation CSV changed since validation"
        evaluated = read("fh_repair/v2/comparison.csv")
        assert len(evaluated) == 216 and not evaluated.duplicated(["Patient", "metric"]).any()
        for _, row in evaluated.iterrows():
            ix = (d.Patient == row.Patient) & (d.metric == row.metric)
            assert ix.sum() == 1
            original = d.loc[ix].iloc[0]
            assert np.isclose(original.auto, row.before, atol=1e-8, rtol=0)
            assert np.isclose(original.excel, row.excel, atol=1e-8, rtol=0)
            d.loc[ix, ["auto", "qc", "rule", "source"]] = [row.after, row.qc_after, "frozen_fh_repair_v2", row.source]
        specs = [(name, unit, "fh_repair/v2/comparison.csv: 고정 FH 보정 후") if name in set(evaluated.metric) else (name, unit, source) for name, unit, source in specs]
        d["difference"] = d.auto-d.excel
    return d, specs, len(ids)


def stats(g):
    x, y = g.excel.to_numpy(), g.auto.to_numpy()
    n = len(g)
    if not n:
        return dict(n=0, r=np.nan, ccc=np.nan, mae=np.nan, bias=np.nan)
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x-x.mean())*(y-y.mean()))
    denominator = vx+vy+(x.mean()-y.mean())**2
    return dict(n=n, r=cov/np.sqrt(vx*vy) if vx > 0 and vy > 0 else np.nan,
                ccc=2*cov/denominator if denominator > 0 else np.nan,
                mae=np.mean(np.abs(y-x)), bias=np.mean(y-x))


def fmt(value):
    return f"{value:,.2f}" if np.isfinite(value) else "—"


def boxplots(g, name, unit, dest):
    """Distributions of the same paired patients shown in the scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), layout="constrained")
    rng = np.random.default_rng(20261110)
    for ax, values, labels, colors, ylabel, title in [
        (axes[0], [g.excel.to_numpy(), g.auto.to_numpy()], ["Excel", "Restored"],
         ["#2563a6", "#d98419"], f"Value ({unit})", f"{name}: value distributions (n = {len(g)})"),
        (axes[1], [g.difference.to_numpy()], ["Restored - Excel"],
         ["#8b5ab5"], f"Difference ({'percentage points' if unit == '%' else unit})", f"{name}: signed error distribution"),
    ]:
        boxes = ax.boxplot(values, tick_labels=labels, widths=.45, patch_artist=True,
                           showfliers=False, medianprops=dict(color="#111827", linewidth=1.8),
                           whiskerprops=dict(color="#64748b"), capprops=dict(color="#64748b"))
        for patch, color in zip(boxes["boxes"], colors):
            patch.set(facecolor=color, alpha=.2, edgecolor=color)
        for position, (series, color) in enumerate(zip(values, colors), 1):
            # Jitter only the horizontal display position; the observed values
            # and errors remain unchanged. All patients, including outliers, show.
            ax.scatter(position+rng.uniform(-.11, .11, len(series)), series,
                       s=17, color=color, alpha=.65, edgecolors="none", zorder=3)
        ax.set(title=title, ylabel=ylabel)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", alpha=.18)
    axes[1].axhline(0, color="#b3261e", ls="--", lw=1.1, label="Zero error")
    axes[1].legend(fontsize=8)
    buffer = StringIO()
    fig.savefig(buffer, format="svg", facecolor="white", metadata={"Date": None})
    if name == "ANB":
        fig.savefig(dest / "ANB_boxplots_preview.png", dpi=130, facecolor="white")
    plt.close(fig)
    svg = ET.fromstring(buffer.getvalue())
    for element in svg.iter():
        href = "{http://www.w3.org/1999/xlink}href"
        if href in element.attrib:
            element.set("href", element.attrib.pop(href))
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    text = ET.tostring(svg, encoding="unicode")
    for element_id in re.findall(r' id="([^"]+)"', text):
        prefix = f"box-{name}-{element_id}"
        text = text.replace(f'id="{element_id}"', f'id="{prefix}"')
        text = text.replace(f'#{element_id}"', f'#{prefix}"')
        text = text.replace(f'#{element_id})', f'#{prefix})')
    (dest / f"{name}_boxplots.svg").write_text(text, encoding="utf-8")
    return text


def plot(ax, g, name, unit, label_all=False):
    lo = min(g.excel.min(), g.auto.min())
    hi = max(g.excel.max(), g.auto.max())
    pad = max((hi-lo)*.08, .1)
    bounds = (lo-pad, hi+pad)
    ax.plot(bounds, bounds, "--", color="#64748b", lw=1.2, label="Identity (y = x)")
    for group, color, marker, label in [("eval", "#2563a6", "o", "Evaluation"), ("pilot", "#ca720c", "^", "Pilot")]:
        for index, row in g[g.Set == group].iterrows():
            fail = row.qc == "needs_review"
            point = ax.scatter(row.excel, row.auto, s=36, marker=marker, facecolor="none" if fail else color,
                               edgecolor="#c13737" if fail else color, linewidth=1, alpha=.85, zorder=3)
            point.set_gid(f"point-{name}-{index}")
        ax.scatter([], [], s=30, marker=marker, color=color, label=label)
    if (g.qc == "needs_review").any():
        ax.scatter([], [], s=30, facecolor="none", edgecolor="#c13737", label="QC: needs review")
    ax.set(xlim=bounds, ylim=bounds, xlabel=f"Excel ({unit})", ylabel=f"Restored ({unit})", title=name)
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(alpha=.18)
    legend = ax.legend(fontsize=7, loc="upper left", framealpha=.85)
    if not label_all:
        for _, row in g.loc[g.difference.abs().nlargest(3).index].iterrows():
            ax.annotate(row.Patient, (row.excel, row.auto), xytext=(4, -10), textcoords="offset points", fontsize=7, color="#6b3541")
        return
    # Keep the actual measurement positions fixed; move only the labels.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = [legend.get_window_extent(renderer).extents]
    locations = ax.transData.transform(g[["excel", "auto"]].to_numpy())
    scale = ax.figure.dpi / 72
    bounds_px = ax.get_window_extent(renderer)
    for (index, row), (px, py) in zip(g.iterrows(), locations):
        width, height = len(row.Patient)*4.3*scale, 9*scale
        candidates = []
        for radius in (8, 16, 24, 36, 50, 68):
            for angle in np.arange(0, 360, 45):
                dx, dy = radius*np.cos(np.radians(angle)), radius*np.sin(np.radians(angle))
                cx, cy = px+dx*scale, py+dy*scale
                box = (cx-width/2-2, cy-height/2-2, cx+width/2+2, cy+height/2+2)
                overlap = sum(max(0, min(box[2], b[2])-max(box[0], b[0])) * max(0, min(box[3], b[3])-max(box[1], b[1])) for b in occupied)
                point_hits = sum((box[0] <= x <= box[2] and box[1] <= y <= box[3]) for x, y in locations)
                outside = not (bounds_px.x0 <= box[0] and box[2] <= bounds_px.x1 and bounds_px.y0 <= box[1] and box[3] <= bounds_px.y1)
                candidates.append((overlap*100 + point_hits*300 + outside*1e7 + radius, dx, dy, box))
        _, dx, dy, box = min(candidates, key=lambda c: c[0])
        occupied.append(box)
        text = ax.annotate(row.Patient, (row.excel, row.auto), xytext=(dx, dy),
                           textcoords="offset points", fontsize=7, ha="center", va="center",
                           color="#334155", arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=.5),
                           bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=.3))
        text.set_gid(f"patient-label-{name}-{index}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Show the preserved original four metrics instead of the completed FH evaluation")
    args = parser.parse_args()
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob("P_*/result.json")}
    use_repair = not args.baseline and (BASE / "fh_repair/v2/provenance.json").exists()
    data, specs, n_cohort = load_pairs(fh_repair=use_repair)
    dest = REPORT.parent / "scatter"
    dest.mkdir(exist_ok=True)
    data.to_csv(dest / "comparison_pairs.csv", index=False, encoding="utf-8-sig")
    cards, summaries = [], []
    overview, axes = plt.subplots(3, 4, figsize=(19, 15), layout="constrained")
    for i, (name, unit, source) in enumerate(specs):
        g = data[(data.metric == name) & data.paired]
        assert len(g) >= 2, name
        fig, ax = plt.subplots(figsize=(8.5, 8), layout="constrained")
        plot(ax, g, name, unit, label_all=True)
        buffer = StringIO()
        fig.savefig(buffer, format="svg", facecolor="white", metadata={"Date": None})
        plt.close(fig)
        svg = ET.fromstring(buffer.getvalue())
        ns = "{http://www.w3.org/2000/svg}"
        points = {f"point-{name}-{idx}": row for idx, row in g.iterrows()}
        for element in svg.iter():
            # Inline SVG is parsed as HTML, where arbitrary XML prefixes such
            # as ns4:href do not resolve xlink references. Use SVG2 href so the
            # actual point markers and font glyphs render in the report.
            xlink_href = "{http://www.w3.org/1999/xlink}href"
            if xlink_href in element.attrib:
                element.set("href", element.attrib.pop(xlink_href))
            row = points.get(element.get("id"))
            if row is not None:
                title = ET.Element(ns+"title")
                title.text = f"{row.Patient} | {row.Set} | Excel {row.excel:.3f} | Restored {row.auto:.3f} | Δ {row.difference:+.3f} {unit}" + (f" | {row.qc}" if row.qc else "")
                element.insert(0, title)
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        svg_text = ET.tostring(svg, encoding="unicode")
        for element_id in re.findall(r' id="([^"]+)"', svg_text):
            svg_text = svg_text.replace(f'id="{element_id}"', f'id="{name}-{element_id}"')
            svg_text = svg_text.replace(f'#{element_id}"', f'#{name}-{element_id}"')
            svg_text = svg_text.replace(f'#{element_id})', f'#{name}-{element_id})')
        (dest / f"{name}.svg").write_text(svg_text, encoding="utf-8")
        box_svg = boxplots(g, name, unit, dest)
        plot(axes.flat[i], g, name, unit)
        group_rows = []
        groups = [("전체", g), ("평가군", g[g.Set == "eval"])]
        if i < 4:
            if use_repair:
                groups.append(("보정 전 QC 통과군 고정", g[g.qc_before == "qc_pass_pending_visual_review"]))
            groups.append(("보정 후 QC 통과" if use_repair else "QC 통과 전체", g[g.qc == "qc_pass_pending_visual_review"]))
        for label, group in groups:
            values = stats(group)
            summaries.append(dict(metric=name, unit=unit, group=label, **values))
            group_rows.append("<tr><td>"+label+"</td>"+"".join(f'<td class="n">{int(value) if key == "n" else fmt(value)}</td>' for key, value in values.items())+"</tr>")
        details = "".join(f"<tr><td>{r.Patient}</td><td>{r.Set}</td><td class='n'>{fmt(r.excel)}</td><td class='n'>{fmt(r.auto)}</td><td class='n'>{fmt(r.difference)}</td></tr>" for _, r in g.iterrows())
        missing = data[(data.metric == name) & ~data.paired]
        missing_details = "".join(f"<li>{r.Patient}: "+("엑셀값 없음" if not np.isfinite(r.excel) else "")+(" / " if not np.isfinite(r.excel) and not np.isfinite(r.auto) else "")+("복원값 없음" if not np.isfinite(r.auto) else "")+"</li>" for _, r in missing.iterrows())
        missing_html = f"<details><summary>점을 찍을 수 없는 환자 {len(missing)}명</summary><ul>{missing_details}</ul></details>" if len(missing) else "<p>대상 54명 모두 점으로 표시했습니다.</p>"
        audit_link = '<p><a href="../nav_review/v2_label_audit/index.html">54명 비강 라벨 전수 점검·환자별 영상·NCW 재현 확인</a></p>' if name in ("NAV", "NCWR") and (BASE / "nav_review/v2_label_audit/index.html").exists() else ""
        cards.append(f'''<article class="scatter-card"><h3>{name} <small>({unit})</small></h3>
<p class="sub">환자 1명 = 점 1개 · 표시된 점 {len(g)}개 / 대상 {n_cohort}명 · 결측 또는 미계측 {n_cohort-len(g)}명</p>
<label><input type="checkbox" checked onchange="this.closest('article').classList.toggle('hide-patient-labels', !this.checked)"> 모든 환자 번호 표시</label>
{svg_text}
<figure class="distribution-boxplots">{box_svg}<figcaption>왼쪽: 엑셀 실제값과 복원값의 분포. 오른쪽: 오차(복원 − 엑셀)의 분포이며 빨간 점선은 오차 0입니다. 산점도와 같은 {len(g)}명의 비교쌍을 사용했습니다. 상자는 25–75백분위, 가운데 선은 중앙값, 수염은 사분위 범위의 1.5배 이내입니다. 모든 환자를 점으로 겹쳐 표시했으며 수염 밖 값도 포함합니다. <a href="scatter/{name}_boxplots.svg" download>박스플롯 SVG 저장</a></figcaption></figure>
<div class="tw"><table><tr><th>대상</th><th>n</th><th>Pearson r</th><th>CCC</th><th>MAE</th><th>평균 차이</th></tr>{''.join(group_rows)}</table></div>
<p class="scatter-source">사용 값: <code>{escape(source)}</code> · <a href="scatter/{name}.svg" download>SVG 저장</a></p>
<p><a href="../fh_repair/v2/index.html#metric-{name}">이 지표의 큰 오차 환자와 영상 근거</a></p>
{audit_link}
<details><summary>환자별 비교값</summary><div class="tw"><table><tr><th>환자</th><th>구분</th><th>엑셀</th><th>복원</th><th>복원 − 엑셀</th></tr>{details}</table></div></details>{missing_html}</article>''')
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .9, "Excel vs restored: 11 metrics\n\nDashed: identity (y = x)\nBlue circles: evaluation\nOrange triangles: pilot\nRed outlines: QC needs review\n\nAll finite pairs; no outlier removal.\nExploratory measurements.", va="top", fontsize=12)
    overview.savefig(dest / "overview.png", dpi=150, facecolor="white")
    plt.close(overview)
    pd.DataFrame(summaries).to_csv(dest / "agreement_summary.csv", index=False, encoding="utf-8-sig")
    n_qc = data.loc[(data.metric == "ANB") & (data.qc == "qc_pass_pending_visual_review"), "Patient"].nunique()
    version_note = (f'4지표 산점도·박스플롯은 <code>fh_repair/v2</code>의 고정 FH 보정 후 값입니다. 원래 result.json은 보존했습니다. 동일한 54명 비교에서 보정 전 QC 통과 48명, 보정 후 {n_qc}명입니다. 「보정 전 QC 통과군 고정」통계는 같은 48명으로 계산합니다. 과거 본문 집계 41명과는 대상이 다릅니다. 전문가 검토 완료를 뜻하지 않습니다. <a href="../fh_repair/v2/index.html#comparison">보정 전후 비교</a> · <a href="../fh_repair/v2/index.html#outliers">11지표 큰 오차 환자·영상 근거</a>'
                    if use_repair else f'기존 result.json의 선택 변형을 사용하며 신규 FH 보정은 반영하지 않았습니다. 현재 QC 통과는 {n_qc}명으로 본문 과거 집계 41명과 대상이 다릅니다. 아래 QC 통과 통계는 현재 파일 기준이며, 전문가 검토 완료를 뜻하지 않습니다.')
    fh_summary_html = ""
    if use_repair:
        evaluation = read("fh_repair/v2/summary.csv")
        all_patients = evaluation[evaluation.group == "all"].set_index(["metric", "version"])
        summary_rows = []
        for metric in ("ANB", "Nperp-A", "Nperp-Pog", "MP-FH"):
            old, new = all_patients.loc[(metric, "before")], all_patients.loc[(metric, "after")]
            cells = "".join(f'<td class="n">{old[k]:.3f} → {new[k]:.3f}</td>' for k in ("mae", "median_abs", "r", "ccc"))
            summary_rows.append(f'<tr><td>{metric}</td>{cells}</tr>')
        fh_summary_html = '<h3 id="fh-comparison">FH 보정 전후: 동일한 54명 전체</h3><div class="tw"><table><tr><th>지표</th><th>MAE 전 → 후</th><th>중앙 절대오차 전 → 후</th><th>r 전 → 후</th><th>CCC 전 → 후</th></tr>'+"".join(summary_rows)+'</table></div><p>P_16 RPo·P_20 LPo·P_37 RPo만 바뀌었으며 다른 51명의 값은 그대로입니다. P_20은 MP-FH가 개선됐지만 ANB·Nperp 오차는 커졌습니다. 이 결과는 전문가 확인 전 개발 후 재평가입니다. <a href="../fh_repair/v2/index.html">보정 전후 그림·환자별 영향·11지표 큰 오차 환자 검토</a></p>'
    section = f'''{START}
<style>.scatter-grid{{display:grid;grid-template-columns:minmax(0,1fr);gap:18px}}.hide-patient-labels [id*="-patient-label-"]{{display:none}}.scatter-card{{min-width:0;padding:14px;background:var(--card);border:1px solid var(--line);border-radius:8px}}.scatter-card h3{{margin:0}}.scatter-card svg{{display:block;width:100%;height:auto;background:white;border-radius:4px}}.scatter-card table{{font-size:.85rem}}.scatter-card th,.scatter-card td{{padding:5px}}.scatter-source{{font-size:.78rem;color:var(--muted);overflow-wrap:anywhere}}.scatter-card .sub{{font-size:.83rem;margin:.3rem 0 .6rem}}</style>
<h2 id="scatter">1-1. 엑셀 값 vs 복원값 산점도</h2>
<p>환자 한 명을 점 하나로 표시했습니다. X축은 엑셀, Y축은 복원값입니다. 두 축은 같은 범위·축척이며, 점선(y=x)에 가까울수록 두 값이 일치합니다. 점선 위는 과대 계측, 아래는 과소 계측입니다. 모든 점에 환자 번호를 표시했고, 번호와 점을 잇는 가는 선은 위치 안내선입니다. 점이 겹치더라도 측정값 위치는 바꾸지 않았습니다. 번호 표시를 끄거나 점에 마우스를 올려 비교값을 확인할 수 있습니다. 비교값이 없는 환자와 결측 항목은 각 그래프 아래에 명시했습니다.</p>
<div class="note"><strong>해석:</strong> Pearson r은 상관, CCC는 상관과 치우침을 함께 반영한 일치도입니다(1에 가까울수록 일치). MAE는 평균 절대 차이, 평균 차이는 복원 − 엑셀입니다. PHI·NCWR의 차이 단위는 퍼센트포인트입니다. 임상 허용오차 기준은 설정하지 않았습니다.<br>
<strong>대상:</strong> Include {n_cohort}명 중 엑셀·복원값이 모두 유한한 환자만 비교하며 이상치를 제거하지 않았습니다. 파란 원은 평가군, 주황 삼각형은 파일럿, 빨간 테두리는 QC 검토 필요 환자입니다. 평가군 표시는 기존 분할을 뜻하며 독립 검증을 보장하지 않습니다. PAL은 개발 중 엑셀을 참고했고, 대구치 규칙은 평가군 확인 후 수정된 이력이 있습니다.<br>
<strong>4지표 버전:</strong> {version_note}</div>
{fh_summary_html}
<p><a href="scatter/overview.png">11지표 전체 그림</a> · <a href="scatter/comparison_pairs.csv" download>환자별 비교 CSV</a> · <a href="scatter/agreement_summary.csv" download>일치도 요약 CSV</a></p>
<div class="scatter-grid">{''.join(cards)}</div>
{END}
'''
    html = REPORT.read_text(encoding="utf-8")
    if START in html:
        html = re.sub(re.escape(START)+r".*?"+re.escape(END)+r"\s*", "", html, flags=re.S)
    by_metric = {spec[0]: card for spec, card in zip(specs, cards)}

    def place(key, content, anchor=None, old_figure=None):
        nonlocal html
        start = f"<!-- CBCT_METRIC_{key}_START -->"
        end = f"<!-- CBCT_METRIC_{key}_END -->"
        block = start+"\n"+content+"\n"+end
        if start in html:
            html = re.sub(re.escape(start)+r".*?"+re.escape(end), lambda _: block, html, flags=re.S)
        elif old_figure:
            pattern = r'<figure>\s*<img\b[^>]*src="[^"]*/'+re.escape(old_figure)+r'"[^>]*>.*?</figure>'
            html, count = re.subn(pattern, lambda _: block, html, flags=re.S)
            assert count == 1, f"Missing original figure: {old_figure}"
        else:
            assert anchor and html.count(anchor) == 1, key
            html = html.replace(anchor, block+"\n\n"+anchor)

    # Put each chart in its measurement section, replacing the original figures.
    # The general reading guide belongs with the first group of measurements.
    guide = section.split('<div class="scatter-grid">', 1)[0]
    guide = guide.replace('<h2 id="scatter">1-1. 엑셀 값 vs 복원값 산점도</h2>',
                          '<h3 id="scatter">엑셀 값 vs 복원값 산점도 읽는 법</h3>')
    guide = guide.replace(START, "")
    place("skeletal", guide+'<div class="scatter-grid">'+"".join(by_metric[k] for k in ("ANB", "Nperp-A", "Nperp-Pog", "MP-FH"))+"</div>", anchor='<h2 id="s6">')
    place("PAL", by_metric["PAL"], anchor='<h3>PMCA (최소 단면적)</h3>')
    place("PMCA", by_metric["PMCA"], anchor='<h3>PAV (인두 기도 부피)</h3>')
    place("PAV", by_metric["PAV"], old_figure="pav_sections.png")
    place("NAV", by_metric["NAV"], old_figure="nav_ts_coronal.png")
    place("TDI", by_metric["TDI"], old_figure="tdi_sections.png")
    place("PHI_NCWR", '<div class="scatter-grid">'+by_metric["PHI"]+by_metric["NCWR"]+'</div>', old_figure="phi_ncwr_sections.png")
    nav = '<li><a href="#scatter">엑셀 vs 복원값 산점도 (11지표)</a></li>'
    if nav not in html:
        html = html.replace('<li><a href="#s1">현황 요약</a></li>', '<li><a href="#s1">현황 요약</a></li>\n'+nav)
    if use_repair and (BASE / "nav_review/v2_label_audit/audit.csv").exists():
        from cbct_update_daily_report import update_html
        html = update_html(html)
    REPORT.write_text(html, encoding="utf-8")
    class Validator(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
            self.cards = 0

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.append(attrs["id"])
            self.cards += attrs.get("class") == "scatter-card"

    validator = Validator()
    validator.feed(REPORT.read_text(encoding="utf-8"))
    assert validator.cards == 11
    assert len(validator.ids) == len(set(validator.ids)), "Duplicate HTML/SVG IDs"
    assert START not in html and END not in html
    for filename in ("pav_sections.png", "nav_ts_coronal.png", "tdi_sections.png", "phi_ncwr_sections.png"):
        assert filename not in html, f"Old report figure still present: {filename}"
    expected_sections = {"ANB": "s5", "Nperp-A": "s5", "Nperp-Pog": "s5", "MP-FH": "s5",
                         "PAL": "s6", "PMCA": "s6", "PAV": "s6", "NAV": "s7",
                         "TDI": "s8", "PHI": "s8", "NCWR": "s8"}
    for metric, section_id in expected_sections.items():
        position = html.index(f'<h3>{metric} <small>')
        previous_sections = re.findall(r'<h2 id="(s\d+)">', html[:position])
        assert previous_sections[-1] == section_id, (metric, section_id)
    for name, _, _ in specs:
        svg_file = ET.parse(dest / f"{name}.svg")
        assert len(svg_file.findall(".//{http://www.w3.org/2000/svg}title")) == int(((data.metric == name) & data.paired).sum())
        assert sum("-patient-label-" in e.get("id", "") for e in svg_file.iter()) == int(((data.metric == name) & data.paired).sum())
        svg_ids = {e.get("id") for e in svg_file.iter() if e.get("id")}
        uses = svg_file.findall(".//{http://www.w3.org/2000/svg}use")
        assert uses and all(e.get("href", "").startswith("#") and e.get("href")[1:] in svg_ids for e in uses), "Broken inline SVG marker/glyph reference"
        for group in svg_file.iter():
            if f"{name}-point-{name}-" in group.get("id", ""):
                assert group.findall(".//{http://www.w3.org/2000/svg}use") or group.findall("./{http://www.w3.org/2000/svg}path"), "Patient marker missing"
        box_file = ET.parse(dest / f"{name}_boxplots.svg")
        box_ids = {e.get("id") for e in box_file.iter() if e.get("id")}
        assert all(e.get("href", "").startswith("#") and e.get("href")[1:] in box_ids for e in box_file.findall(".//{http://www.w3.org/2000/svg}use"))
    assert html.count('class="distribution-boxplots"') == 11
    assert not re.search(r'\bns\d+:href=', html), "Unsupported href prefix in inline HTML SVG"
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob("P_*/result.json")}
    assert before == after, "Original result files changed"
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    print(f"Updated: {REPORT}\nValidated plots and hover values: {len(specs)}; original result files unchanged: {len(before)}")


if __name__ == "__main__":
    main()
