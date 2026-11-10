"""PSG 차이 분포 확대와 허용차 후보 탐색. 기준을 확정하거나 원본을 변경하지 않는다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Result_py/patient_matching"
OUT = BASE / "figures"
COUNTS = ["OA", "CA", "MA", "H", "RERA", "RESP"]
METRICS = ["TST", "AHI", "RDI", "ODI", "LSaO2", *COUNTS]
NAMES = {"OA": "폐쇄성 무호흡", "CA": "중추성 무호흡", "MA": "혼합성 무호흡",
         "H": "저호흡", "RERA": "호흡노력 관련 각성", "RESP": "호흡각성"}
LABELS = ["엑셀 − 리포트", "원본 − 리포트", "엑셀 − 원본"]
COLORS = ["#2879A6", "#DA8539", "#55866D"]
UNITS = {"TST": "분", "AHI": "회/시간", "RDI": "회/시간", "ODI": "회/시간", "LSaO2": "%p",
         **{key: "건" for key in COUNTS}}
MIN_SPAN = {"TST": .25, "AHI": .2, "RDI": .3, "ODI": 1., "LSaO2": 1.25,
            **{key: 1.25 for key in COUNTS}}


def stats(v):
    v = np.asarray(v, dtype=float)
    q025, q1, med, q3, q975 = np.quantile(v, [.025, .25, .5, .75, .975])
    inside = v[(v >= q1 - 1.5 * (q3-q1)) & (v <= q3 + 1.5 * (q3-q1))]
    return {"n": len(v), "median": float(med), "q1": float(q1), "q3": float(q3),
            "lo95": float(q025), "hi95": float(q975), "abs95": float(np.quantile(abs(v), .95)),
            "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "whislo": float(inside.min()), "whishi": float(inside.max())}


def load_data():
    master = pd.read_csv(BASE / "patient_matching_with_raw.csv")
    source = BASE / "psg_values_with_raw.csv"
    d = pd.read_csv(source)
    ids = master.loc[master["엑셀_원신호검사"].eq("동일 검사"), "환자"]
    d = d[d["환자"].isin(ids) & d["변수"].isin(METRICS)].copy()
    assert len(d) == len(ids) * len(METRICS)
    assert d["엑셀대응리포트"].eq(d["원신호대응리포트"]).all()
    for name, a, b in zip(LABELS, ["엑셀원값", "원본판독·파형산출값", "엑셀원값"],
                          ["동일검사리포트원값", "동일검사리포트원값", "원본판독·파형산출값"]):
        d[name] = d[a] - d[b]
    assert d[LABELS].notna().all().all()
    d["최대절대차이"] = d[LABELS].abs().max(axis=1)
    d["초기기준초과"] = d["최대절대차이"].gt(d["허용차이"] + 1e-8)
    flags = d.groupby("환자")["초기기준초과"].sum().astype(int)
    mismatch = set(flags[flags.gt(0)].index)
    d["초기불일치군"] = d["환자"].isin(mismatch)
    payload = {"total": len(master), "eligible": len(ids), "initialMismatch": len(mismatch),
               "deferred": len(master)-len(ids), "metrics": [], "patientIds": list(ids)}
    rows = []
    for metric in METRICS:
        sub = d[d["변수"].eq(metric)]
        tol = float(sub["허용차이"].iloc[0])
        all_values = sub[LABELS].to_numpy().ravel()
        span = max(MIN_SPAN[metric], tol * 2, float(np.quantile(abs(all_values), .90)) * 1.25)
        full = max(span, float(abs(all_values).max()) * 1.12)
        data = {"key": metric, "name": NAMES.get(metric, metric), "unit": UNITS[metric], "initial": tol,
                "zoom": span, "full": full, "maxAbs95": float(sub["최대절대차이"].quantile(.95)), "rows": []}
        for _, r in sub.iterrows():
            data["rows"].append({"id": r["환자"], "excel": float(r["엑셀원값"]),
                "raw": float(r["원본판독·파형산출값"]), "report": float(r["동일검사리포트원값"]),
                "diff": [float(r[x]) for x in LABELS], "initialBad": bool(r["초기불일치군"]),
                "kind": r["값유형"], "maxAbs": float(r["최대절대차이"])})
        for scope, selected in [("동일검사 전체", sub), ("초기 불일치군", sub[sub["초기불일치군"]])]:
            for label in LABELS:
                rows.append({"지표": metric, "표본": scope, "비교": label,
                             "초기허용차": tol, **stats(selected[label]), "지표최대차이_절대P95_전체73명": data["maxAbs95"]})
        payload["metrics"].append(data)
    d.to_csv(OUT / "psg_error_review_data.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "psg_error_distribution_stats.csv", index=False, encoding="utf-8-sig")
    groups = master[["환자", "엑셀_원신호검사"]].copy()
    groups["초기기준초과지표수"] = groups["환자"].map(flags).astype("Int64")
    groups.to_csv(OUT / "psg_error_review_groups.csv", index=False, encoding="utf-8-sig")
    return payload, source


def draw_axis(ax, metric, selected, span, zoom=False):
    tau = metric["initial"]
    ax.axhspan(-tau, tau, facecolor="#E5F1E8", alpha=.9, zorder=0)
    if tau:
        for y in [-tau, tau]:
            ax.axhline(y, color="#579767", ls="--", lw=1.2)
    ax.axhline(0, color="#354657", lw=1, ls=":")
    rng = np.random.default_rng(20261105)
    summaries = []
    for j, color in enumerate(COLORS):
        vals = np.array([r["diff"][j] for r in selected])
        s = stats(vals); summaries.append(s)
        box = ax.bxp([{"med": s["median"], "q1": s["q1"], "q3": s["q3"],
                       "whislo": s["whislo"], "whishi": s["whishi"]}], positions=[j+1], widths=.38,
                     showfliers=False, patch_artist=True,
                     medianprops={"color": "#203547", "lw": 2.2})
        box["boxes"][0].set(facecolor=color, alpha=.16, edgecolor=color)
        for bound in [s["lo95"], s["hi95"]]:
            ax.hlines(bound, j+.74, j+1.26, color="#896DA4", ls="-.", lw=1.4)
        jitter = rng.uniform(-.12, .12, len(vals))
        for k, r in enumerate(selected):
            y = vals[k]
            if abs(y) <= span:
                ax.scatter(j+1+jitter[k], y, s=28 if zoom else 17,
                           c="#C75B58" if abs(y)>tau+1e-8 else color, alpha=.72,
                           edgecolors="white", linewidths=.45, zorder=4)
        if zoom:
            hi, lo = int((vals>span).sum()), int((vals < -span).sum())
            ax.text(j+1, 1.025, f"화면 밖 ↑{hi} / ↓{lo}명", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=10, color="#A05248" if hi+lo else "#71808B")
        else:
            for r in sorted(selected, key=lambda r: abs(r["diff"][j]), reverse=True)[:1]:
                if abs(r["diff"][j]) > metric["zoom"]:
                    ax.annotate(r["id"], (j+1, r["diff"][j]), xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_ylim(-span, span)
    ax.set_xlim(.55, 3.45)
    ax.set_xticks([1, 2, 3], LABELS if zoom else ["엑셀-보고", "원본-보고", "엑셀-원본"])
    ax.set_ylabel(f"차이 ({metric['unit']})")
    ax.grid(axis="y", color="#CDD5DD", alpha=.45)
    return summaries


def export_figures(payload):
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname=font).get_name(), "DejaVu Sans"]
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    with PdfPages(OUT / "psg_error_review.pdf") as pdf:
        for metric in payload["metrics"]:
            selected = [r for r in metric["rows"] if r["initialBad"]]
            fig = plt.figure(figsize=(17, 10))
            gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1], left=.07, right=.97, top=.79, bottom=.28, wspace=.24)
            main, overview = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
            title = metric["key"] + (f" · {metric['name']}" if metric["key"] in COUNTS else "")
            fig.suptitle(f"{title} · 차이 분포와 허용차 후보 검토", x=.07, y=.96, ha="left", fontsize=24, weight="bold")
            fig.text(.07, .895, f"초기 불일치군 {len(selected)}명 표시 / 기준 검토의 참조 표본은 동일 검사 전체 {payload['eligible']}명", fontsize=13)
            fig.text(.07, .85, f"중앙 구간 확대 · 초기 허용차 ±{metric['initial']:g} {metric['unit']}  |  최대 절대차 P95 = {metric['maxAbs95']:.3g} {metric['unit']} (전체 73명)", fontsize=12, color="#3D6950")
            summaries = draw_axis(main, metric, selected, metric["zoom"], True)
            draw_axis(overview, metric, selected, metric["full"])
            overview.set_title("전체 범위", fontsize=13, pad=18)
            overview.tick_params(axis="x", labelrotation=28, labelsize=9)
            for j, s in enumerate(summaries):
                main.text((j+.5)/3, -.16, f"중앙값 {s['median']:.3g}\n중앙 95% [{s['lo95']:.3g}, {s['hi95']:.3g}]",
                          transform=main.transAxes, ha="center", va="top", fontsize=11, linespacing=1.7)
            fig.text(.07, .12, "녹색 영역·점선: 현재 허용차 후보  |  빨간 점: 후보 범위 초과  |  보라색 짧은 선: 표시군의 2.5~97.5백분위", fontsize=11)
            fig.text(.07, .075, "박스: 중앙값·사분위 범위, 수염: 1.5×IQR. 중앙 95%와 P95는 분포 요약이며, 임상 허용오차를 확정하는 기준이 아닙니다.", fontsize=10, color="#687582")
            note = ("사건 수는 Embla TXT 전체 집계 또는 Nox Manual 우선 후보 규칙입니다. 최종 리포트와 집계 범위·판독층 차이가 포함될 수 있습니다."
                    if metric["key"] in COUNTS else "ODI는 자체 알고리즘, AHI/RDI는 기존 판독 기반 재계산, LSaO₂는 수면 중 최저입니다. HTML에서 허용차·축 범위·표시군을 바꿀 수 있습니다.")
            fig.text(.07, .04, note, fontsize=10, color="#687582")
            fig.savefig(OUT / f"psg_error_review_{metric['key']}.png", dpi=160, facecolor="white")
            pdf.savefig(fig)
            plt.close(fig)


HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>PSG 오차 범위 검토</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#233747;font:15px 'Malgun Gothic',sans-serif}main{max-width:1500px;margin:auto;padding:32px 28px}h1{font-size:32px;margin:0 0 14px}h2{font-size:27px;margin:0}p{line-height:1.8}.muted{color:#647786}.top,.card{background:white;border:1px solid #dce4e9;border-radius:12px;padding:26px;margin-bottom:24px}.top{border-top:5px solid #355d78}.toolbar,.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}button,select,input{font:inherit;border:1px solid #aebfc9;border-radius:5px;padding:8px;background:white}button{cursor:pointer}button:hover{background:#edf4f8}input[type=number]{width:100px}input[type=range]{width:220px;padding:0}.pill{display:inline-block;background:#edf2f6;padding:7px 11px;border-radius:6px}.summary{background:#eaf3ed;border:1px solid #bdd8c6;border-radius:6px;padding:12px;margin:14px 0}.controls{background:#f7f9fa;padding:14px;margin-top:18px}.plotwrap{overflow-x:auto}svg{display:block;width:100%;min-width:850px;height:auto}svg text{font-family:'Malgun Gothic',sans-serif}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin:12px 0}.key{display:inline-block;width:12px;height:12px;margin-right:5px;vertical-align:middle}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce4e9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f4f7f9}.count{color:#a64640;font-weight:bold}.note{font-size:13px;color:#647786}.jump a{margin-right:18px;color:#2e6589}details{margin:16px 0}.groupnote{font-size:13px;color:#647786}
</style></head><body><main>
<section class="top"><h1>PSG 차이 분포 · 오차 판정 범위 검토</h1>
<p>요약지표 5개와 사건 수 6개, <strong>총 11개 PSG 지표</strong>의 차이 분포입니다. <strong>허용차 후보 ±T</strong>를 움직여 기준을 넘는 환자가 얼마나 달라지는지 확인합니다.<br>같은 검사 <strong>__ELIGIBLE__명</strong>이 참조 표본입니다. 다른 검사·검사 미확인 __DEFERRED__명은 비교에서 제외했습니다.</p>
<div class="toolbar"><label>분포에 표시할 환자 <select id="scope"><option value="bad">초기 기준 불일치군 __MISMATCH__명</option><option value="all">동일 검사 전체 __ELIGIBLE__명</option></select></label><button id="reset">초기 후보로 복원</button><button id="download">현재 후보·환자 판정 CSV 저장</button></div>
<div id="overall" class="summary"></div><p class="note">불일치군은 초기 기준으로 고정되어 있습니다. 후보를 바꿔도 표시군이 자동으로 바뀌지는 않습니다. 기준을 정할 때는 전체 73명 보기로도 확인하세요.</p>
<div class="jump"><a href="#TST">TST</a><a href="#AHI">AHI</a><a href="#RDI">RDI</a><a href="#ODI">ODI</a><a href="#LSaO2">LSaO₂</a><a href="#OA">OA 폐쇄성 무호흡</a><a href="#CA">CA 중추성 무호흡</a><a href="#MA">MA 혼합성 무호흡</a><a href="#H">H 저호흡</a><a href="#RERA">RERA</a><a href="#RESP">RESP 호흡각성</a></div>
<p class="note">초기 불일치군은 11개 중 하나라도 기존 기준을 초과한 환자입니다. 사건 수 6종은 초기 후보가 0건(정확 일치)입니다. 중앙 95% 범위 = 표시군 차이의 2.5~97.5백분위. 절대차 P95 후보 = 전체 __ELIGIBLE__명에서 환자별 세 쌍 차이 중 최대 절대차의 95백분위입니다. 둘 다 기술통계이며 최종 허용오차를 뜻하지 않습니다. 원본 값의 산출 방법·판독층·집계 범위 차이도 포함됩니다.</p></section>
<div id="cards"></div><p class="note">원본 자료와 기존 대응표는 변경하지 않습니다. 이 화면의 후보 기준은 CSV로 저장할 때만 별도 파일로 남습니다.</p>
<script>
const DATA=__DATA__;
const PAIRS=['엑셀 − 리포트','원본 − 리포트','엑셀 − 원본'];
const COLORS=['#2879A6','#DA8539','#55866D'];
const state={scope:'bad',tau:{},zoom:{}};
const fmt=(v)=>Number(v.toFixed(3)).toString();
const quantile=(v,p)=>{const a=[...v].sort((a,b)=>a-b),k=(a.length-1)*p,i=Math.floor(k);return a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(k-i)};
function stats(a){let q1=quantile(a,.25),q3=quantile(a,.75),b=a.filter(v=>v>=q1-1.5*(q3-q1)&&v<=q3+1.5*(q3-q1));return {q1,q3,med:quantile(a,.5),lo:quantile(a,.025),hi:quantile(a,.975),loW:Math.min(...b),hiW:Math.max(...b),mean:a.reduce((x,y)=>x+y,0)/a.length};}
function exceeds(m,r){return r.maxAbs>state.tau[m.key]+1e-8}
function totalBad(){return DATA.patientIds.filter(id=>DATA.metrics.some(m=>exceeds(m,m.rows.find(r=>r.id===id)))).length}
const NS='http://www.w3.org/2000/svg';
function node(tag,attrs={},txt){const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(txt!==undefined)n.textContent=txt;return n}
function add(svg,tag,attrs,txt){let n=node(tag,attrs,txt);svg.appendChild(n);return n}
function setup(){
 for(const m of DATA.metrics){state.tau[m.key]=m.initial;state.zoom[m.key]=m.zoom;const card=document.createElement('section');card.className='card';card.id=m.key;
 card.innerHTML=`<h2>${m.key}${m.name!==m.key?' · '+m.name:''} <span class="muted" style="font-size:17px">차이 단위: ${m.unit}</span></h2>
 <div class="controls"><label>허용차 후보 ± <input id="tau-${m.key}" type="number" min="0" step="0.01" value="${m.initial}"> ${m.unit}</label><input id="range-${m.key}" type="range" min="0" max="${m.full}" step="0.001" value="${m.initial}" aria-label="${m.key} 후보 허용차"><button id="p95-${m.key}">절대차 P95 후보 넣기 (${fmt(m.maxAbs95)})</button></div>
 <div class="toolbar" style="margin-top:14px"><label>세로축 확대 ± <input id="zoom-${m.key}" type="number" min="0.001" step="0.1" value="${fmt(m.zoom)}"></label><button id="central-${m.key}">중앙 구간 확대</button><button id="full-${m.key}">전체 범위 보기</button></div>
 <div id="count-${m.key}" class="summary"></div><div class="legend"><span><i class="key" style="background:#e5f1e8;border:1px dashed #579767"></i>후보 허용 범위</span><span><i class="key" style="background:#c75b58"></i>범위 초과 점</span><span><i class="key" style="background:#896da4"></i>중앙 95% 경계</span><span>박스: 중앙값·IQR / 수염: 1.5×IQR</span></div>
 <div class="plotwrap"><svg id="svg-${m.key}" viewBox="0 0 1180 570" role="img" aria-label="${m.key} 차이 분포"></svg></div><div id="table-${m.key}"></div>
 <p class="groupnote">각 점에 마우스를 올리면 환자 번호와 세 원값이 표시됩니다. 화면 밖 점은 화살표·인원수로 남기며, 전체 범위 보기에서 실제 위치를 확인할 수 있습니다.</p>`;
 document.getElementById('cards').appendChild(card);
 const updateTau=v=>{v=Number(v);if(!Number.isFinite(v)||v<0)return;state.tau[m.key]=v;document.getElementById('tau-'+m.key).value=v;const slider=document.getElementById('range-'+m.key);slider.max=Math.max(m.full,v);slider.value=v;renderAll()};
 document.getElementById('tau-'+m.key).addEventListener('input',e=>updateTau(e.target.value));
 document.getElementById('range-'+m.key).addEventListener('input',e=>updateTau(e.target.value));
 document.getElementById('p95-'+m.key).addEventListener('click',()=>updateTau(m.maxAbs95));
 const zoom=v=>{v=Number(v);if(!Number.isFinite(v)||v<=0)return;state.zoom[m.key]=v;document.getElementById('zoom-'+m.key).value=fmt(v);render(m)};
 document.getElementById('zoom-'+m.key).addEventListener('input',e=>zoom(e.target.value));
 document.getElementById('central-'+m.key).addEventListener('click',()=>zoom(m.zoom));
 document.getElementById('full-'+m.key).addEventListener('click',()=>zoom(m.full));
 }
 document.getElementById('scope').addEventListener('change',e=>{state.scope=e.target.value;renderAll()});
 document.getElementById('reset').addEventListener('click',()=>{for(const m of DATA.metrics){state.tau[m.key]=m.initial;document.getElementById('tau-'+m.key).value=m.initial;document.getElementById('range-'+m.key).value=m.initial;}renderAll()});
 document.getElementById('download').addEventListener('click',download);
 renderAll();
}
function renderAll(){document.getElementById('overall').textContent=`현재 후보 기준: 전체 동일 검사 ${DATA.eligible}명 중 ${totalBad()}명이 하나 이상의 지표에서 범위 초과 · 초기 기준 ${DATA.initialMismatch}명. 후보는 아직 확정되지 않았습니다.`;DATA.metrics.forEach(render)}
function render(m){
 const rows=m.rows.filter(r=>state.scope==='all'||r.initialBad),tau=state.tau[m.key],span=state.zoom[m.key];
 const svg=document.getElementById('svg-'+m.key);svg.replaceChildren();
 const L=86,R=1145,T=55,B=465,H=B-T,xs=[260,615,970],Y=v=>T+(span-v)/(span*2)*H;
 add(svg,'defs',{}).appendChild(node('clipPath',{id:'clip-'+m.key}));svg.querySelector('clipPath').appendChild(node('rect',{x:L,y:T,width:R-L,height:H}));
 add(svg,'rect',{x:L,y:Y(Math.min(tau,span)),width:R-L,height:Math.max(0,Y(-Math.min(tau,span))-Y(Math.min(tau,span))),fill:'#e5f1e8'});
 for(let i=0;i<=8;i++){let v=-span+2*span*i/8,y=Y(v);add(svg,'line',{x1:L,x2:R,y1:y,y2:y,stroke:'#dde4ea'});add(svg,'text',{x:L-14,y:y+5,'text-anchor':'end','font-size':14,fill:'#4b6375'},fmt(v));}
 add(svg,'text',{x:L,y:20,'font-size':14,fill:'#526c7e'},`표시군 ${rows.length}명 · 중앙 구간 ±${fmt(span)} ${m.unit}`);
 for(const v of [-tau,tau])if(Math.abs(v)<=span)add(svg,'line',{x1:L,x2:R,y1:Y(v),y2:Y(v),stroke:'#579767','stroke-width':2,'stroke-dasharray':'7 5'});
 add(svg,'line',{x1:L,x2:R,y1:Y(0),y2:Y(0),stroke:'#344e61','stroke-dasharray':'3 4'});
 add(svg,'line',{x1:L,x2:L,y1:T,y2:B,stroke:'#5c7283'});add(svg,'line',{x1:L,x2:R,y1:B,y2:B,stroke:'#5c7283'});
 const summaries=[];
 for(let j=0;j<3;j++){
  const x=xs[j],a=rows.map(r=>r.diff[j]),s=stats(a);summaries.push(s);
  const g=add(svg,'g',{'clip-path':`url(#clip-${m.key})`});
  add(g,'line',{x1:x,x2:x,y1:Y(s.loW),y2:Y(s.hiW),stroke:'#6e8594','stroke-width':2});
  for(const v of [s.loW,s.hiW])add(g,'line',{x1:x-30,x2:x+30,y1:Y(v),y2:Y(v),stroke:'#6e8594','stroke-width':2});
  add(g,'rect',{x:x-52,y:Y(s.q3),width:104,height:Math.max(1,Y(s.q1)-Y(s.q3)),fill:COLORS[j],'fill-opacity':.18,stroke:COLORS[j],'stroke-width':2});
  add(g,'line',{x1:x-52,x2:x+52,y1:Y(s.med),y2:Y(s.med),stroke:'#203b50','stroke-width':3});
  for(const v of [s.lo,s.hi])add(g,'line',{x1:x-70,x2:x+70,y1:Y(v),y2:Y(v),stroke:'#896da4','stroke-width':2,'stroke-dasharray':'6 4'});
  rows.forEach((r,i)=>{const v=r.diff[j],jitter=(((i*37+j*17)%101)/100-.5)*74;
   if(Math.abs(v)<=span){let p=add(g,'circle',{cx:x+jitter,cy:Y(v),r:4.8,fill:Math.abs(v)>tau+1e-8?'#c75b58':COLORS[j],'fill-opacity':.78,stroke:'white','stroke-width':.6});p.appendChild(node('title',{},`${r.id} | ${PAIRS[j]} ${fmt(v)} ${m.unit}\n엑셀 ${fmt(r.excel)} / 원본 ${fmt(r.raw)} / 리포트 ${fmt(r.report)}\n${r.kind}`));}
  });
  for(const [which,v,up] of [['상단',span,true],['하단',-span,false]]){const outside=rows.filter(r=>up?r.diff[j]>span:r.diff[j]<-span);if(outside.length){let y=up?T+7:B-7;let p=add(svg,'path',{d:`M ${x-7} ${y+(up?6:-6)} L ${x+7} ${y+(up?6:-6)} L ${x} ${y+(up?-6:6)} Z`,fill:'#ae514b'});p.appendChild(node('title',{},outside.map(r=>`${r.id}: ${fmt(r.diff[j])}`).join('\n')));add(svg,'text',{x:x+14,y:up?T+15:B-10,'font-size':13,fill:'#a14d47'},`${up?'↑':'↓'} 화면 밖 ${outside.length}명`);}}
  add(svg,'text',{x,y:505,'text-anchor':'middle','font-size':18,fill:'#28495f'},PAIRS[j]);
  add(svg,'text',{x,y:536,'text-anchor':'middle','font-size':14,fill:'#677d8d'},`중앙값 ${fmt(s.med)} · 중앙 95% [${fmt(s.lo)}, ${fmt(s.hi)}]`);
 }
 let bad=m.rows.filter(r=>exceeds(m,r)).length;
 document.getElementById('count-'+m.key).innerHTML=`후보 허용차 <strong>±${fmt(tau)} ${m.unit}</strong> → 전체 ${DATA.eligible}명 중 <span class="count">${bad}명 (${(100*bad/DATA.eligible).toFixed(1)}%)</span>이 세 자료 중 하나 이상의 차이에서 범위 초과. <span class="muted">참조 절대차 P95: ${fmt(m.maxAbs95)} ${m.unit}</span>`;
 document.getElementById('table-'+m.key).innerHTML='<table><thead><tr><th>표시군의 비교</th><th>N</th><th>중앙값</th><th>중앙 95% 범위</th><th>후보 밖 / 표시군</th><th>전체 최소~최대</th></tr></thead><tbody>'+summaries.map((s,j)=>`<tr><td>${PAIRS[j]}</td><td>${rows.length}</td><td>${fmt(s.med)}</td><td>${fmt(s.lo)} ~ ${fmt(s.hi)}</td><td>${rows.filter(r=>Math.abs(r.diff[j])>tau+1e-8).length} / ${rows.length}</td><td>${fmt(Math.min(...rows.map(r=>r.diff[j])))} ~ ${fmt(Math.max(...rows.map(r=>r.diff[j])))}</td></tr>`).join('')+'</tbody></table>';
}
function download(){const lines=[['환자',...DATA.metrics.map(m=>m.key+'_허용차후보'),...DATA.metrics.map(m=>m.key+'_초과'),'하나이상초과'].join(',')];for(const id of DATA.patientIds){const flags=DATA.metrics.map(m=>exceeds(m,m.rows.find(r=>r.id===id)));lines.push([id,...DATA.metrics.map(m=>state.tau[m.key]),...flags,flags.some(Boolean)].join(','));}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\n')],{type:'text/csv;charset=utf-8;'}));a.download='PSG_허용차후보_검토.csv';a.click();URL.revokeObjectURL(a.href);}
setup();
</script></main></body></html>'''


def main():
    OUT.mkdir(exist_ok=True)
    payload, source = load_data()
    export_figures(payload)
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    for key, value in {"__ELIGIBLE__": payload["eligible"], "__DEFERRED__": payload["deferred"],
                       "__MISMATCH__": payload["initialMismatch"]}.items():
        html = html.replace(key, str(value))
    (OUT / "psg_error_review.html").write_text(html, encoding="utf-8")
    (OUT / "psg_error_review_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"지표수": len(METRICS), "지표목록": METRICS,
               "전체": payload["total"], "같은검사": payload["eligible"], "초기불일치군": payload["initialMismatch"],
               "보류": payload["deferred"], "95퍼센트기준참조표본": "같은검사 전체 73명",
               "원본SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "초기후보": {m["key"]: m["initial"] for m in payload["metrics"]},
               "최대절대차P95": {m["key"]: m["maxAbs95"] for m in payload["metrics"]}}
    (OUT / "psg_error_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
