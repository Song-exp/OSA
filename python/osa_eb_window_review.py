"""Eb 탐색 아래 한계 판정 페이지(아티팩트용). 현재 규칙(설골 기준점 −3 mm까지)과 새 규칙(그 환자 설골 몸통 아래 끝까지)의 Eb를 비교한다.
새 규칙 근거: 후두개곡 바닥(설골–후두개 인대)은 설골 몸통에 붙으므로 Eb는 몸통 아래 끝보다 낮을 수 없다.
대상: 두 규칙의 Eb가 다른 환자 + 채택 줄이 1~2줄뿐인 환자 (rule_params/eb_rows.csv, hyoid_blobs.csv 기준).
영상: 정중 ±6 mm 최소 투영(공기)과 정중 단면 한 장. 엑셀 PAL과 두 PAL을 함께 적는다.
틀·기준 모식도: osa_eb_review_template.html (환자 정보 없음)
출력: Result_py/analysis/all83/eb_window_review/{artifact.html, P_n.png} (git 제외, 환자 영상 포함)
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import cbct_eb_detect as E
from cbct_patient_pipeline import CB, OUT as PIPE, landmarks
from check_landmarks import read_nrrd
import osa_experiments as X

SRC = X.R_/"analysis/all83/rule_params"; OUT = X.R_/"analysis/all83/eb_window_review"
TPL = Path(__file__).with_name("osa_eb_review_template.html")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False


def body_bottom(src=SRC):
    """환자별 설골 몸통 아래 끝(기준점 = 덩어리 뒤쪽 1/3 꼭대기 대비 mm)."""
    b = pd.read_csv(src/"hyoid_blobs.csv", encoding="utf-8-sig"); b = b[b.chosen].set_index("Patient")
    p = pd.read_csv(src/"per_patient.csv", encoding="utf-8-sig").set_index("Patient")
    return np.maximum(b.back_sup, b.front_sup)-b.si_len-p.hy_sup.reindex(b.index)


def pick(rows, lo, min_rows=3):
    ok = rows[(rows.gap >= 1) & (rows.gap <= 10) & (rows.dc >= -18) & (rows.dc <= 2) & (rows.dz >= lo) & (rows.dz <= 22)].sort_values("dz")
    return ok.iloc[0] if len(ok) >= min_rows else None


def images(pid, ant_lo=None):
    P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); o, R, _ = E.midline_frame(P)
    ant, sup, Ls = np.arange(E.ANT[0] if ant_lo is None else ant_lo, E.ANT[1], E.STEP), np.arange(*E.SUP, E.STEP), np.arange(-6, 6+1e-9, E.STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-3000).reshape(len(Ls), len(ant), len(sup)); del v
    vol[vol < -2500] = np.nan
    return np.nanmin(vol, 0), vol[len(Ls)//2], ant, sup


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("P_*.png"): f.unlink()
    r = pd.read_csv(SRC/"eb_rows.csv", encoding="utf-8-sig"); p = pd.read_csv(SRC/"per_patient.csv", encoding="utf-8-sig").set_index("Patient")
    lim = body_bottom()
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'PAL'").set_index("Patient").excel
    cases = []
    for pt, g in r.groupby("Patient"):
        if pt not in lim.index: continue
        a, b = pick(g, -3), pick(g, lim[pt])
        one2 = b is None and 0 < len(g[(g.gap >= 1) & (g.gap <= 10) & (g.dc >= -18) & (g.dc <= 2) & (g.dz >= lim[pt]) & (g.dz <= 22)]) < 3
        if (a is None) != (b is None) or (a is not None and abs(a.dz-b.dz) > .01) or one2: cases.append((pt, a, b, one2))
    cases.sort(key=lambda c: int(c[0][2:])); cards = []
    for pt, a, b, one2 in cases:
        pid = int(pt[2:]); hy = p.loc[pt]; lo = lim[pt]; mip, mid, ant, sup = images(pid)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained"); ext = [ant[0], ant[-1], sup[0], sup[-1]]
        for x_, im, t in ((ax[0], mip, "정중 ±6 mm 최소 투영 (공기 보기)"), (ax[1], mid, "정중 단면 한 장")):
            x_.imshow(im.T, origin="lower", cmap="gray", vmin=-1000, vmax=600, extent=ext); x_.set_title(t, fontsize=9)
            x_.axhline(hy.hy_sup-3, color="#ffd400", lw=1.2, ls="--"); x_.axhline(hy.hy_sup+lo, color="#ff6b6b", lw=1.2, ls=":")
            x_.plot(hy.hy_ant, hy.hy_sup, "y+", ms=12, mew=2)
            if a is not None: x_.plot(hy.hy_ant+a.dc, hy.hy_sup+a.dz, "o", mfc="none", mec="#2fd3ff", ms=12, mew=2.5)
            if b is not None: x_.plot(hy.hy_ant+b.dc, hy.hy_sup+b.dz, "x", color="#ff6b6b", ms=12, mew=2.5)
            x_.set_xlim(hy.hy_ant-30, hy.hy_ant+12); x_.set_ylim(hy.hy_sup-18, hy.hy_sup+28); x_.tick_params(labelsize=7)
        f = OUT/f"{pt}.png"; fig.savefig(f, dpi=80); plt.close(fig)
        pal = lambda c: f"{np.hypot(hy.hy_ant+c.dc, hy.hy_sup+c.dz):.1f}" if c is not None else "직접 검출 없음"
        tag = "새 규칙에서도 1~2줄뿐" if one2 else "새 규칙에서 바뀜"
        cards.append(f'<div class="c"><h3>{pt} <span>{tag} · 몸통 아래 끝 {lo:.1f} mm</span></h3><a href="{f.name}"><img src="{f.name}" alt="{pt} Eb 비교"></a>'
                     f'<p>엑셀 PAL <b>{xl.get(pt, np.nan):.1f}</b> · 현재 ○ {pal(a)} · 새 규칙 × {pal(b)}</p>'
                     f'<p class="j" data-p="{pt}">판정: <label><input type="radio" name="{pt}" value="현재">현재(○)</label> <label><input type="radio" name="{pt}" value="새규칙">새 규칙(×)</label>'
                     f' <label><input type="radio" name="{pt}" value="둘다틀림">둘 다 틀림</label> <label><input type="radio" name="{pt}" value="애매">애매</label></p></div>')
        print(pt, end=" ", flush=True)
    tpl = TPL.read_text(encoding="utf-8"); head, ref = tpl.split("<!--REF-->")
    head = head.replace("<title>Eb 탐색 한계 판정</title>", "<title>Eb 탐색 한계 판정</title>")
    body = f"""<main><h1>Eb(후두개곡 최하점) 탐색 아래 한계 판정 — {len(cases)}명</h1>
<p class="note"><b>새 규칙:</b> Eb를 찾는 아래 한계를 그 환자의 <b>설골 몸통 아래 끝</b>으로 둡니다(후두개곡 바닥인 설골–후두개 인대가 설골 몸통에 붙기 때문). 지금 규칙은 설골 기준점 3 mm 아래까지만 봅니다.
엑셀 PAL과 비교하면 새 규칙은 직접 검출 44명 → 53명, 평균 절대오차 2.90 → 2.56 mm(같은 44명 기준)였습니다.</p>
<p class="note">노랑 + = 설골 기준점(덩어리 뒤쪽 1/3 꼭대기), 노랑 점선 = 지금 아래 한계(기준점 3 mm 아래), 빨강 점선 = 새 아래 한계(설골 몸통 아래 끝).
○ 하늘 = 지금 규칙의 Eb, × 빨강 = 새 규칙의 Eb. 지금 규칙에서 직접 검출이 없으면 실제 계산은 설골 기준 추정값을 씁니다.</p>
{ref}
<div class="g">{''.join(cards)}</div>
<h2>판정 결과</h2><p class="note">고른 결과가 아래에 모입니다. "결과 복사"를 눌러 대화창에 붙여 주세요.</p>
<button type="button" id="cp">결과 복사</button><span id="cnt"></span><textarea id="out" readonly></textarea>
<script>(function(){{
const K="eb_window_judge_v2";let saved={{}};try{{saved=JSON.parse(localStorage.getItem(K)||"{{}}")}}catch(e){{}}
document.querySelectorAll(".j").forEach(p=>{{const v=saved[p.dataset.p];if(v){{const i=p.querySelector('input[value="'+v+'"]');if(i)i.checked=true}}}});
function u(){{const o={{}};const r=[...document.querySelectorAll(".j")].map(p=>{{const c=p.querySelector("input:checked");if(c)o[p.dataset.p]=c.value;return p.dataset.p+": "+(c?c.value:"-")}});
document.getElementById("out").value=r.join("\\n");document.getElementById("cnt").textContent=Object.keys(o).length+" / "+r.length+"명 판정";try{{localStorage.setItem(K,JSON.stringify(o))}}catch(e){{}}}}
document.addEventListener("change",u);u();
document.getElementById("cp").addEventListener("click",async()=>{{const t=document.getElementById("out");try{{await navigator.clipboard.writeText(t.value);document.getElementById("cnt").textContent="복사됨"}}catch(e){{t.focus();t.select()}}}});
}})();</script></main>"""
    (OUT/"artifact.html").write_text(head+body, encoding="utf-8")
    print("\nsaved", OUT/"artifact.html", len(cases), "명")
    return [c[0] for c in cases]


def compare(cur="W6_gap1-10_rows3", new="W6_gap0.8-10_rows2", out_name="artifact_v4.html", src="variants_combo.csv", prefix="v4_", ant_lo=None, win=None,
            title="Eb 검출 조건 완화 판정", lead=None, key="eb_relax_judge_v1"):
    """민감도 결과(eb_sensitivity/variants_combo.csv)의 두 설정을 비교하는 판정 페이지. 직접 검출이 없으면 설골 기준 추정점(□)을 표시."""
    v = pd.read_csv(X.R_/"analysis/all83/eb_sensitivity"/src, encoding="utf-8-sig")
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'PAL'").set_index("Patient").excel
    lim = body_bottom(); T = {}
    for k in (cur, new):
        g = v[v.variant == k].set_index("Patient"); d = g.Eb_ant.notna(); off = ((g.Eb_ant-g.hy_ant)[d].median(), (g.Eb_sup-g.hy_sup)[d].median())
        T[k] = g.assign(direct=d, ea=g.Eb_ant.where(d, g.hy_ant+off[0]), es=g.Eb_sup.where(d, g.hy_sup+off[1]))
    a, b = T[cur], T[new]    # 둘 다 추정인 환자(보정량 변화로만 움직임)는 빼고, 직접 검출 여부나 직접 검출 위치가 달라진 환자만
    ch = a.index[(a.direct != b.direct) | (a.direct & b.direct & (((a.ea-b.ea).abs() > .01) | ((a.es-b.es).abs() > .01)))]
    ch = sorted(ch, key=lambda s: int(s[2:])); cards = []
    for pt in ch:
        pid = int(pt[2:]); ra, rb = a.loc[pt], b.loc[pt]; mip, mid, ant, sup = images(pid, ant_lo); ext = [ant[0], ant[-1], sup[0], sup[-1]]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
        for x_, im, t in ((ax[0], mip, "정중 ±6 mm 최소 투영 (검출에 쓰는 영상)"), (ax[1], mid, "정중 단면 한 장 (참고)")):
            x_.imshow(im.T, origin="lower", cmap="gray", vmin=-1000, vmax=600, extent=ext); x_.set_title(t, fontsize=9)
            x_.axhline(ra.hy_sup+lim.get(pt, -3), color="#ff6b6b", lw=1.2, ls=":"); x_.plot(ra.hy_ant, ra.hy_sup, "y+", ms=12, mew=2)
            for w_, c_ in (win or ()): x_.axvline(w_, color=c_, lw=1.4, ls="--")
            for r_, col, mk in ((ra, "#2fd3ff", "o"), (rb, "#ff6b6b", "x")):
                x_.plot(r_.ea, r_.es, mk if r_.direct else "s", mfc="none" if mk == "o" or not r_.direct else col, mec=col, color=col, ms=12, mew=2.5)
            x_.set_xlim(max(ant[0], ra.hy_ant-(45 if ant_lo else 30)), ra.hy_ant+12); x_.set_ylim(ra.hy_sup-18, ra.hy_sup+28); x_.tick_params(labelsize=7)
        f = OUT/f"{prefix}{pt}.png"; fig.savefig(f, dpi=80); plt.close(fig)
        lab = lambda r_: f"{np.hypot(r_.ea, r_.es):.1f}" + ("" if r_.direct else " (추정)")
        cards.append(f'<div class="c"><h3>{pt} <span>n줄 {int(ra.n_rows)}</span></h3><a href="{f.name}"><img src="{f.name}" alt="{pt} Eb 비교"></a>'
                     f'<p>엑셀 PAL <b>{xl.get(pt, np.nan):.1f}</b> · 지금 ○ {lab(ra)} · 후보 × {lab(rb)}</p>'
                     f'<p class="j" data-p="{pt}">판정: <label><input type="radio" name="{pt}" value="지금">지금(○)</label> <label><input type="radio" name="{pt}" value="후보">후보(×)</label>'
                     f' <label><input type="radio" name="{pt}" value="둘다틀림">둘 다 틀림</label> <label><input type="radio" name="{pt}" value="애매">애매</label></p></div>')
        print(pt, end=" ", flush=True)
    head, ref = TPL.read_text(encoding="utf-8").split("<!--REF-->")
    lead = lead or ("<b>후보:</b> 후두개 두께 하한 1 → 0.8 mm(영상 2칸), 최소 연속 줄 수 3 → 2줄. 정중 두께 ±6 mm, 아래 한계(설골 몸통 아래 끝)는 그대로입니다. "
                    "엑셀 PAL과 비교하면(81명, 추정 포함) 직접 검출 53 → 61명, 평균 절대오차 2.62 → 2.50 mm, 상관 0.816 → 0.834였습니다.")
    body = f"""<main><h1>{title} — {len(ch)}명</h1>
<p class="note">{lead}</p>
<p class="note">○ 하늘 = 지금 조건의 Eb, × 빨강 = 후보 조건의 Eb. 네모(□·■)는 직접 검출에 실패해 설골 기준으로 추정한 점입니다. 노랑 + = 설골 기준점, 빨강 점선 = 아래 한계(설골 몸통 아래 끝).</p>
{ref}
<div class="g">{''.join(cards)}</div>
<h2>판정 결과</h2><p class="note">"결과 복사"를 눌러 대화창에 붙여 주세요.</p>
<button type="button" id="cp">결과 복사</button><span id="cnt"></span><textarea id="out" readonly></textarea>
<script>(function(){{
const K="{key}";let saved={{}};try{{saved=JSON.parse(localStorage.getItem(K)||"{{}}")}}catch(e){{}}
document.querySelectorAll(".j").forEach(p=>{{const v=saved[p.dataset.p];if(v){{const i=p.querySelector('input[value="'+v+'"]');if(i)i.checked=true}}}});
function u(){{const o={{}};const r=[...document.querySelectorAll(".j")].map(p=>{{const c=p.querySelector("input:checked");if(c)o[p.dataset.p]=c.value;return p.dataset.p+": "+(c?c.value:"-")}});
document.getElementById("out").value=r.join("\\n");document.getElementById("cnt").textContent=Object.keys(o).length+" / "+r.length+"명 판정";try{{localStorage.setItem(K,JSON.stringify(o))}}catch(e){{}}}}
document.addEventListener("change",u);u();
document.getElementById("cp").addEventListener("click",async()=>{{const t=document.getElementById("out");try{{await navigator.clipboard.writeText(t.value);document.getElementById("cnt").textContent="복사됨"}}catch(e){{t.focus();t.select()}}}});
}})();</script></main>"""
    (OUT/out_name).write_text(head+body, encoding="utf-8"); print("saved", OUT/out_name, ch)
    return ch


if __name__ == "__main__":
    import sys
    if "--window" in sys.argv:
        print(compare("clo-18_chi2_back25_win-35", "clo-18_chi2_back25_win-45", "artifact_v5.html", "variants_ap.csv", "v5_", ant_lo=-55.0,
                      win=((-35.0, "#2fd3ff"), (-45.0, "#ff6b6b")), title="Eb 탐색 창 뒤쪽 끝 판정", key="eb_window_back_judge_v1",
                      lead="<b>후보:</b> Eb를 찾는 영상 창의 뒤쪽 끝을 PNS 35 mm 뒤 → 45 mm 뒤로 넓힙니다(그 밖의 조건은 확정값: 아래 한계 = 설골 몸통 아래 끝, ±6 mm, 후두개 두께 하한 0.8 mm, 2줄). "
                           "엑셀 PAL과 비교하면(81명, 추정 포함) 직접 검출 61 → 71명, 평균 절대오차 2.50 → 2.35 mm, 상관 0.834 → 0.854였습니다. "
                           "세로 점선 = 창 뒤쪽 끝(하늘: 지금 35 mm, 빨강: 후보 45 mm). 그림은 두 창을 모두 보이도록 더 넓게 잘랐습니다."))
    else:
        print(compare() if "--relax" in sys.argv else main())
