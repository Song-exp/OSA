"""83명 Step 1~4 (분할 없음, 계수 유의성 판단용).
Step 1: Roh 2025 Table 3 동시 모델 Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + PMCA + PAL. 엑셀 판 / 복원값 판(복원값 결측은 엑셀로 채움), 83명.
Step 2~4 (같은 템플릿): 1. 변수 설명 2. 상관 히트맵(Step 1 인자 + 신규 인자끼리) 3. 대체 관계(도메인 짝) 4. X 전체 요약표(대체·추가) + 행을 누르면 회귀식 단위 표.
기준 모델은 복원값 판. 비교는 X가 있는 같은 환자에서.
출력: Result_py/analysis/all83/전체_83명_단계별_회귀.html (+ step1_roh.csv, step1_4.csv, figs/heat_*.png)
"""
import json, warnings; warnings.filterwarnings("ignore")
from html import escape
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

import osa_all83 as A
from osa_var_detail import detail
from cbct_final_report import CSS
from report_terms import check
from report_editor import apply

OUT = A.OUT; FIG = OUT/"figs"; B0 = A.B0
ROH = ["ANB", "MP-FH", "TDI", "PMCA", "PAL"]
XCOL = dict(zip(ROH, ["x_ANB", "x_MPFH", "x_TDI", "x_PMCA", "x_PAL"])); MCOL = dict(zip(ROH, ["m_ANB", "m_MPFH", "m_TDI", "a_PMCA", "a_PAL"]))
BCOL = {k: f"b_{k}" for k in ROH}
YS = ["logAHI", "logAHI_SUP", "AHI", "AHI_SUP"]
YL = {"logAHI": "log AHI", "logAHI_SUP": "log 앙와위 AHI", "AHI": "AHI 원값 (Roh와 같은 척도)", "AHI_SUP": "앙와위 AHI 원값"}
TERM = {"AWshape_PC1": "AWshape_PC1 (기도 굵기)", "H_col": "H_col (기도 높이)", "upper_width_M1": "upper_width_molar", "const": "절편", "Age": "나이", "Male": "성별(남=1)", "BMI": "BMI", **{v: f"{k} (엑셀)" for k, v in XCOL.items()}, **{v: f"{k} (복원값)" for k, v in BCOL.items()}}
DISP = {"upper_width_M1": "upper_width_molar", "lower_width_M1": "lower_width_molar"}      # 표시용 이름(보고서 금지 표현 'M1 ' 회피)

SAG = ("ANB", "앞뒤(시상) 턱 관계: 상악 대비 하악이 앞인지 뒤인지"); VER = ("MP-FH", "수직 얼굴 패턴: 하악평면 기울기·아래 얼굴 높이")
TRA = ("TDI", "횡방향 턱 폭: 상하악 치조·치열궁의 가로 폭"); CSA = ("PMCA", "기도 단면 크기: 인두가 얼마나 넓은가"); LEN = ("PAL", "기도 길이: 인두가 위아래로 얼마나 긴가")
STEP2 = {"a_PAV": (CSA[0], "기도 크기(부피 = 단면 × 길이)의 단면 성분"), "a_PAV_h": CSA, "AW_min": CSA, "HD_mm": (CSA[0], "기도 단면 크기(수력학적 직경)"),
         "nasopharynx_min_area": CSA, "nasopharynx_mean_area": CSA, "oropharynx_min_area": CSA, "oropharynx_mean_area": CSA,
         "hypopharynx_min_area": CSA, "hypopharynx_mean_area": CSA, "oro_naso_min_ratio": CSA, "H_col": LEN, "AW_length": LEN,
         "AW_min_pos": None, "m_NAV": None, "soft_palate_length": None, "soft_palate_thickness": None, "soft_palate_angle": None, "soft_palate_PNS_gap": None,
         "AWshape_PC1": None, "AWshape_PC2": None, "AWshape_PC3": None}
STEP3 = {**{k: SAG for k in ("Pog_Ba_ant", "SNB", "SNA", "Pog_PNS_ant", "PNS_B_ant", "Overjet_mm", "Co_Gn", "Go_Me", "body_len", "body_len_res", "McN_diff", "F1_PC1", "m_NperpPog", "m_NperpA")},
         **{k: VER for k in ("PP_MP", "SN_MP", "LAFH_ratio", "VERT_Z", "gonial", "Jarabak", "ANS_Me", "N_Me", "S_Go", "Co_Go", "ramus_body", "Yaxis", "FacialAxis_proxy", "F1_PC4", "overbite_mm")},
         **{k: TRA for k in ("upper_width_M1", "lower_width_M1", "InterGonial", "Upper_Taper", "upper_depth", "m_PHI", "m_NCWR")},
         **{k: None for k in ("SIZE_PC1", "SIZE_MAND", "SIZE_MAX", "SIZE_CB", "S_N", "S_Ba", "NSBa", "ANS_PNS", "PNS_Ba_depth", "Co_A", "palate_cb", "mand_cb", "logCS_all", "AlloShape",
                              "Atyp_D2", "Atyp_Procrustes", "PLS_t1", "MaxPost_Clivus_ant")}}

# Step 3 변수 출처 구분 (색): ① 고전 두부계측 ② Roh 지표(Step 1 모델 밖) ③ 자체 3D 기준점 지표 ④ 형상 요약값
KIND3 = {**{k: "c1" for k in ("SNA", "SNB", "S_N", "S_Ba", "NSBa", "ANS_PNS", "N_Me", "ANS_Me", "S_Go", "Co_Gn", "Co_A", "Co_Go", "Go_Me", "gonial", "PP_MP", "SN_MP",
                               "Jarabak", "McN_diff", "Yaxis", "FacialAxis_proxy", "LAFH_ratio", "Overjet_mm", "overbite_mm", "upper_width_M1", "lower_width_M1",
                               "InterGonial", "Upper_Taper", "upper_depth", "ramus_body", "palate_cb", "mand_cb")},
         **{k: "c2" for k in ("m_NperpPog", "m_NperpA", "m_PHI", "m_NCWR")},
         **{k: "c3" for k in ("Pog_Ba_ant", "Pog_PNS_ant", "PNS_B_ant", "PNS_Ba_depth", "body_len", "body_len_res", "MaxPost_Clivus_ant")},
         **{k: "c4" for k in ("F1_PC1", "F1_PC4", "SIZE_PC1", "SIZE_MAND", "SIZE_MAX", "SIZE_CB", "logCS_all", "AlloShape", "Atyp_D2", "Atyp_Procrustes", "PLS_t1", "VERT_Z")}}
KIND_LAB = {"c1": ("고전 두부계측 (선행연구 기준 계측값)", "#2563eb", "#e6efff"), "c2": ("Roh 지표 중 Step 1 모델 밖", "#7c3aed", "#f1e9ff"),
            "c3": ("자체 3D 기준점 지표 (골격 위치 벡터 기반)", "#059669", "#e3f7ee"), "c4": ("형상 요약값 (AMASSS 대응점 PC·크기·PLS 등)", "#d97706", "#fff1dc")}


def kc(sid, v):
    return f' class="{KIND3[v]}"' if sid == "3" and v in KIND3 else ""


def legend3():
    return '<p class="legend">' + " ".join(f'<span class="{k}">{escape(l)}</span>' for k, (l, _, _) in KIND_LAB.items()) + '</p>'


def step4(d):
    skip = set(STEP2) | set(STEP3) | set(B0) | set(XCOL.values()) | set(MCOL.values()) | set(BCOL.values())
    names = ["tongue_mL", "tongue_c_ant", "tongue_c_sup", "tongue_low_sup", "tongue_len", "tongue_height", "tongue_back_ant", "tongue_mandible_ratio", "tongue_MMV",
             "tongue_MMVhull", "SoftLoad_MMV", "MMV_mL", "MMV_hull_mL", "masseter_mL", "lat_pterygoid_mL", "med_pterygoid_mL", "chew_mL", "sinus_max_mL", "sinus_max_air_frac",
             "sinus_frontal_mL", "SDp_mm", "MP_H", "Hy_Me", "Hy_Ba_dist", "Hy_sup", "Hy_ant", "L1_MP_deg", "Condyle_Axial_mean", "Condyle_Lat_mean",
             "lever_Pog_mm", "lever_Hy_mm", "lever_arm_Pog_sup"]
    names += [c for c in d if c.startswith(("EFA_", "MAND_rPC", "MAX_rPC", "CB_rPC", "res_", "F1_PC", "F2_PC", "F2MM_PC"))]+["F2_logCS", "F2MM_logCS"]
    names += [c for c in d if c.startswith("roi_")]
    return {k: None for k in dict.fromkeys(names) if k in d and k not in skip}


def data():
    d = A.data()
    for k in ROH: d[BCOL[k]] = d[MCOL[k]].fillna(d[XCOL[k]])
    d.attrs["filled"] = {k: int(d[MCOL[k]].isna().sum()) for k in ROH}
    assert d[[BCOL[k] for k in ROH]+[XCOL[k] for k in ROH]+B0].notna().all().all() and len(d) == 83
    return d


def fit(u, y, cols):
    X = sm.add_constant(u[cols]); f = sm.OLS(u[y], X).fit(); sy = u[y].std()
    vf = {t: float(VIF(X.to_numpy(float), i)) for i, t in enumerate(X.columns) if t != "const"}
    terms = {t: {"B": float(f.params[t]), "beta": None if t == "const" else float(f.params[t]*u[t].std()/sy), "p": float(f.pvalues[t]), "vif": vf.get(t)} for t in X.columns}
    return f, terms


def model_json(f, terms, n):
    return {"n": int(n), "r2": float(f.rsquared), "ar2": float(f.rsquared_adj), "terms": [{"t": t, **r} for t, r in terms.items()]}


def step1(d):
    out = {}
    for ver, cols in (("엑셀", [XCOL[k] for k in ROH]), ("복원값", [BCOL[k] for k in ROH])):
        for y in YS:
            f, terms = fit(d, y, B0+cols); out[(ver, y)] = {"n": len(d), "r2": f.rsquared, "ar2": f.rsquared_adj, "terms": terms}
    return out


def newfactor(d, cands, sid):
    base = B0+[BCOL[k] for k in ROH]; rows, eq = [], {}
    for x, pair in cands.items():
        if x not in d or d[x].notna().sum() < 30 or d[x].std() == 0: continue
        u = d.dropna(subset=[x]); eq[x] = {}
        for y in YS:
            f0, t0 = fit(u, y, base); fa, ta = fit(u, y, base+[x]); e = {"base": model_json(f0, t0, len(u)), "add": model_json(fa, ta, len(u))}
            row = {"step": sid, "y": y, "var": x, "n": len(u), "pair": pair[0] if pair else None, "meaning": pair[1] if pair else None,
                   "r_pair": u[[x, BCOL[pair[0]]]].corr("spearman").iloc[0, 1] if pair else np.nan,
                   "r2_base": f0.rsquared, "add_beta": ta[x]["beta"], "add_p": ta[x]["p"], "add_dR2": fa.rsquared-f0.rsquared, "add_vif": ta[x]["vif"]}
            if pair:
                fs, ts = fit(u, y, [c for c in base if c != BCOL[pair[0]]]+[x]); e["sub"] = model_json(fs, ts, len(u))
                row |= {"sub_beta": ts[x]["beta"], "sub_p": ts[x]["p"], "sub_r2": fs.rsquared, "sub_dR2": fs.rsquared-f0.rsquared}
            rows.append(row); eq[x][y] = e
    t = pd.DataFrame(rows)
    for y in YS:
        k = t.index[t.y == y]; t.loc[k, "add_q"] = multipletests(t.loc[k, "add_p"], method="fdr_bh")[1]
    return t, eq


# ------------------------------------------------------------ 그림·HTML 조각
def heatmap(d, names, sid):
    lab = {**{BCOL[k]: f"[Step 1] {k}" for k in ROH}, **{n: DISP.get(n, n) for n in names}}
    r_new = d[names].corr("spearman").fillna(0)
    order = [names[i] for i in leaves_list(linkage(1-np.abs(r_new.to_numpy()[np.triu_indices(len(names), 1)]), "average"))] if len(names) > 2 else names
    cols = [BCOL[k] for k in ROH]+order; R = d[order+[BCOL[k] for k in ROH]].corr("spearman").loc[order, cols]
    n, m = R.shape; fs = 7 if n < 40 else 5
    fig, ax = plt.subplots(figsize=(max(6, .16*m+3), max(4, .16*n+2)))
    im = ax.imshow(R.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(m), [lab[c] for c in cols], rotation=90, fontsize=fs); ax.set_yticks(range(n), [lab[c] for c in order], fontsize=fs)
    ax.axvline(len(ROH)-.5, color="black", lw=1.5)
    if sid == "3":
        for tl in ax.get_yticklabels()+ax.get_xticklabels():
            v = next((c for c in names if DISP.get(c, c) == tl.get_text()), None)
            if v in KIND3: tl.set_color(KIND_LAB[KIND3[v]][1]); tl.set_fontweight("bold")
    if n <= 40:
        for i in range(n):
            for j in range(m): ax.text(j, i, f"{R.iat[i, j]:.1f}", ha="center", va="center", fontsize=4.5, color="white" if abs(R.iat[i, j]) > .6 else "black")
    fig.colorbar(im, ax=ax, shrink=.5, label="Spearman r"); ax.set_title("왼쪽 5열: Step 1 인자(복원값) · 오른쪽: 신규 인자끼리 (비슷한 변수끼리 모이도록 정렬)", fontsize=8)
    fig.tight_layout(); f = FIG/f"heat_s{sid}.png"; fig.savefig(f, dpi=110); plt.close(fig); return f


def pf(p): return "" if p is None or pd.isna(p) else ("&lt;0.001" if p < .001 else f"{p:.3f}")
def nf(v, f="+.2f"): return "" if v is None or pd.isna(v) else format(v, f)
def thumb(v, figs): return f'<a href="figs/{figs[v].name}" target="_blank"><img class="thumb" src="figs/{figs[v].name}" alt="{escape(v)}"></a>' if v in figs else ""


def describe(v, cat):
    r = detail(v)
    return r if r else cat.get(v, ("(정의 없음)", ""))


def step1_html(S1, cat, figs, filled):
    divs = []
    for i, y in enumerate(YS):
        for ver in ("엑셀", "복원값"):
            m = S1[(ver, y)]; body = []
            for t, r in m["terms"].items():
                key = next((MCOL[k] for k in ROH if BCOL[k] == t), t)
                de, so = cat.get(key, ("", "")) if t != "const" else ("절편", "")
                k5 = next((k for k in ROH if BCOL[k] == t), None)
                fill = f"<br><span class='note'>결측 {filled[k5]}명은 엑셀 값으로 채움</span>" if k5 and filled[k5] else ""
                sig = " class=sig" if r["p"] < .05 and t != "const" else ""
                body.append(f'<tr><td><code>{escape(TERM.get(t, t))}</code>{fill}</td><td>{escape(de)}</td><td>{escape(so)}</td><td>{thumb(key, figs)}</td><td>{m["n"]}</td>'
                            f'<td>{r["B"]:+.3g}</td><td>{nf(r["beta"])}</td><td{sig}>{pf(r["p"])}</td><td>{nf(r["vif"], ".2f")}</td><td>{m["r2"]:.3f}</td><td>{m["ar2"]:.3f}</td></tr>')
            divs.append(f'<div class="yt" data-s="1" data-k="{i}|{ver}"{"" if (i == 0 and ver == "엑셀") else " hidden"}><div class="tw"><table><thead><tr><th>항</th><th>변수 뜻</th><th>만든 과정</th><th>CT 위치</th>'
                        f'<th>n</th><th>B</th><th>β</th><th>p</th><th>VIF</th><th>모델 R²</th><th>수정 R²</th></tr></thead><tbody>{"".join(body)}</tbody></table></div></div>')
    opts = "".join(f'<option value="{i}">{escape(YL[y])}</option>' for i, y in enumerate(YS))
    fl = ", ".join(f"{k} {v}명" for k, v in filled.items() if v)
    return (f'<h2 id="s1">Step 1. Roh 2025 다중회귀 재현 (엑셀 값 대 복원값)</h2>'
            f'<p><b>설정:</b> Roh 2025 Table 3과 같은 동시 모델. 83명 전원. 엑셀 값은 결측이 없고, 복원값(CBCT 자동 계측)의 결측({fl})은 엑셀 값으로 채웠다.</p>'
            f'<div class="ctl"><label>결과 변수(Y) <select id="y1" onchange="showY(\'1\')">{opts}</select></label> <label>값 <select id="v1" onchange="showY(\'1\')"><option value="엑셀">엑셀 값 (연구실 수작업)</option><option value="복원값">복원값 (CBCT 자동 계측, 결측은 엑셀)</option></select></label></div>'
            f'<p class="eq">회귀식(하나): Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + PMCA + PAL</p>'
            f'<p class="note">Roh 2025 Table 3 (102명, AHI 원값): ANB β 0.32 (p=.005), MP-FH 0.05 (p=.680), TDI −0.27 (p=.002), PMCA −0.21 (p=.039), PAL −0.03 (p=.727), VIF 1.19~2.01, 수정 R² 0.40.</p>'
            + "".join(divs))


def nf_html(sid, title, intro, t, cat, figs, heat, pcs=()):
    vars_ = list(dict.fromkeys(t["var"]))
    d1 = "".join(f'<tr{kc(sid, v)}><td><code>{escape(DISP.get(v, v))}</code></td><td>{escape(describe(v, cat)[0])}</td><td>{escape(describe(v, cat)[1])}</td><td>{thumb(v, figs)}</td></tr>' for v in vars_)
    p1 = (legend3() if sid == "3" else "") + f'<h3>{sid}-1. 변수 설명</h3><div class="yt"><div class="tw"><table><thead><tr><th>변수명</th><th>변수 뜻</th><th>만든 과정</th><th>CT 위치</th></tr></thead><tbody>{d1}</tbody></table></div></div>'
    pp = ""
    if pcs:
        cards = "".join(f'<figure class="pcf"><a href="figs/pc/{f.name}" target="_blank"><img src="figs/pc/{f.name}" alt="{escape(tl)}"></a><figcaption><b>{escape(tl)}</b><br>{escape(ds)}</figcaption></figure>' for tl, f, ds in pcs)
        pp = (f'<h3>{sid}-2. 주성분·요인 축 결과</h3><p class="note">이 단계의 주성분(PC)·요인·PLS 변수가 무엇을 나타내는지 보여 준다. 누르면 크게 열린다.</p><div class="pcgrid">{cards}</div>')
    n0 = 3 if pcs else 2
    p2 = (f'<h3>{sid}-{n0}. 변수 간 상관 히트맵</h3><p class="note">83명 Spearman 상관. 왼쪽 5열은 Step 1 인자(복원값)와의 상관, 오른쪽은 신규 인자끼리의 상관이다. 빨강 양의 상관, 파랑 음의 상관. 누르면 크게 열린다.</p>'
          f'<a href="figs/{heat.name}" target="_blank"><img class="heat" src="figs/{heat.name}" alt="상관 히트맵"></a>')
    g = t[t.y == YS[0]]
    d3 = "".join(f'<tr{kc(sid, r["var"])}><td><code>{escape(DISP.get(r["var"], r["var"]))}</code></td><td>{r["pair"] if pd.notna(r["pair"]) else "<span class=note>짝 없음</span>"}</td>'
                 f'<td>{escape(r["meaning"]) if pd.notna(r["pair"]) else "Step 1 인자 중 같은 차원이 없어 대체하지 않고 추가 모델만 본다"}</td><td>{nf(r["r_pair"])}</td></tr>' for _, r in g.iterrows())
    p3 = (f'<h3>{sid}-{n0+1}. 변수 간 대체 관계</h3><p class="note">신규 인자가 Step 1 인자 중 어느 것과 같은 차원(같은 해부학적 의미)을 재는지 미리 정했다. 대체 모델은 그 Step 1 인자를 빼고 신규 인자를 넣는다.</p>'
          f'<div class="yt"><div class="tw"><table><thead><tr><th>신규 인자</th><th>대체할 Step 1 인자</th><th>도메인 의미</th><th>짝과의 상관 r</th></tr></thead><tbody>{d3}</tbody></table></div></div>')
    divs = []
    for i, y in enumerate(YS):
        body = []
        for _, r in t[t.y == y].iterrows():
            sub = (f'<td>{nf(r.get("sub_beta"))}</td><td{" class=sig" if pd.notna(r.get("sub_p")) and r.get("sub_p") < .05 else ""}>{pf(r.get("sub_p"))}</td><td>{nf(r.get("sub_r2"), ".3f")}</td><td>{nf(r.get("sub_dR2"), "+.3f")}</td>'
                   if pd.notna(r["pair"]) else '<td></td><td></td><td></td><td></td>')
            k3 = KIND3.get(r["var"], "") if sid == "3" else ""
            body.append(f'<tr class="pick {k3}" data-x="{escape(r["var"])}" onclick="showEq(\'{sid}\', this.dataset.x)"><td><code>{escape(DISP.get(r["var"], r["var"]))}</code></td><td>{r["n"]}</td>'
                        f'<td>{r["pair"] if pd.notna(r["pair"]) else ""}</td><td>{r["r2_base"]:.3f}</td>{sub}<td>{nf(r["add_beta"])}</td><td{" class=sig" if r["add_p"] < .05 else ""}>{pf(r["add_p"])}</td>'
                        f'<td>{pf(r["add_q"])}</td><td>{nf(r["add_dR2"], "+.3f")}</td><td>{nf(r["add_vif"], ".2f")}</td></tr>')
        head = ('<tr><th rowspan=2>변수명</th><th rowspan=2>n</th><th rowspan=2>대체 짝</th><th rowspan=2>Step 1 모델 R²</th>'
                '<th colspan=4 class="g2">대체 모델 (짝 자리에 X)</th><th colspan=5 class="g3">추가 모델 (Step 1 + X)</th></tr><tr>'
                '<th class="g2">β</th><th class="g2">p</th><th class="g2">R²</th><th class="g2">ΔR²</th><th class="g3">β</th><th class="g3">p</th><th class="g3">FDR q</th><th class="g3">ΔR²</th><th class="g3">VIF</th></tr>')
        divs.append(f'<div class="yt" data-s="{sid}" data-k="{i}|"{"" if i == 0 else " hidden"}><div class="tw"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div></div>')
    opts = "".join(f'<option value="{i}">{escape(YL[y])}</option>' for i, y in enumerate(YS))
    p4 = (f'<h3>{sid}-{n0+2}. 회귀 결과</h3><div class="ctl"><label>결과 변수(Y) <select id="y{sid}" onchange="showY(\'{sid}\')">{opts}</select></label>'
          f'<span class="note">X 전체 요약. 행을 누르면 아래에 그 X의 회귀식 표(Step 1 모델 · 대체 모델 · 추가 모델)가 열린다.</span></div>'
          '<p class="eq">대체: Y ~ 성별 + 나이 + BMI + Step 1 인자(짝 제외 4개) + X · 추가: Y ~ 성별 + 나이 + BMI + Step 1 인자 5개 + X</p>'
          + "".join(divs) + f'<div class="eqbox" id="eq{sid}"><p class="note">요약표의 행을 누르면 회귀식 표가 여기에 열립니다.</p></div>')
    return f'<h2 id="s{sid}">{escape(title)}</h2><p>{intro}</p>{p1}{pp}{p2}{p3}{p4}'


JS = """const fmt=(v,k)=>v===null||v===undefined?'':(k==='p'?(v<0.001?'<0.001':v.toFixed(3)):k==='v'?v.toFixed(2):k==='b'?((v>=0?'+':'')+v.toFixed(2)):((v>=0?'+':'')+v.toPrecision(3)));
function tt(m,focus,title,eqn){let h=`<div class="eqc"><p><b>${title}</b><br><span class="eq">${eqn}</span></p><table class="reg"><thead><tr><th>항</th><th>B</th><th>β</th><th>p</th><th>VIF</th></tr></thead><tbody>`;
 for(const t of m.terms){h+=`<tr${t.t===focus?' class="focus"':''}><td>${TERM[t.t]||t.t}</td><td>${fmt(t.B,'B')}</td><td>${fmt(t.beta,'b')}</td><td class="${t.p<0.05&&t.t!=='const'?'sig':''}">${fmt(t.p,'p')}</td><td>${fmt(t.vif,'v')}</td></tr>`;}
 return h+`</tbody><tfoot><tr><td colspan=5>n ${m.n} · R² ${m.r2.toFixed(3)} · 수정 R² ${m.ar2.toFixed(3)}</td></tr></tfoot></table></div>`;}
function showEq(s,x){const y=YS[+document.getElementById('y'+s).value];const e=EQ[s][x][y];const nm=TERM[x]||x;const pr=PAIR[s][x];
 const B=BASE[s]||['Step 1 모델 (같은 환자)','Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + PMCA + PAL'];let h=`<h4>${nm} · ${YL[y]}</h4><div class="eqrow">`+tt(e.base,null,B[0],B[1]);
 if(e.sub)h+=tt(e.sub,x,`대체 모델 (${pr} → ${nm})`,`Y ~ 성별 + 나이 + BMI + Step 1 인자(${pr} 제외) + ${nm}`);
 h+=tt(e.add,x,'추가 모델',B[1]+` + ${nm}`)+'</div>';
 const b=document.getElementById('eq'+s);b.innerHTML=h;b.dataset.x=x;
 document.querySelectorAll('.yt[data-s="'+s+'"] tr.pick').forEach(r=>r.classList.toggle('on',r.dataset.x===x));}
function showY(s){const y=document.getElementById('y'+s).value;const v=document.getElementById('v'+s);const k=y+'|'+(v?v.value:'');
 document.querySelectorAll('.yt[data-s="'+s+'"]').forEach(e=>{e.hidden=e.dataset.k!==k;});const b=document.getElementById('eq'+s);if(b&&b.dataset.x)showEq(s,b.dataset.x);}
function sortBy(th){const t=th.closest('table'),tb=t.tBodies[0];let i=0;
 if(t.tHead.rows.length>1){const r0=[...t.tHead.rows[0].cells],r1=[...t.tHead.rows[1].cells];const map=[];let k=0;
  r0.forEach(c=>{if(c.rowSpan>1)map.push(c);else{for(let j=0;j<c.colSpan;j++)map.push(r1[k++]);}});i=map.indexOf(th);}
 else i=[...th.parentNode.children].indexOf(th);
 if(i<0)return;const dir=th.classList.contains('asc')?-1:1;t.querySelectorAll('th').forEach(h=>h.classList.remove('asc','desc'));th.classList.add(dir>0?'asc':'desc');
 const val=td=>{const x=td.textContent.trim().replace('<','').replace('+','');const n=parseFloat(x);return x===''?null:(isNaN(n)?x:n);};
 [...tb.rows].sort((a,b)=>{const p=val(a.cells[i]),q=val(b.cells[i]);if(p===null)return 1;if(q===null)return -1;
  return (typeof p==='number'&&typeof q==='number'?p-q:String(p).localeCompare(String(q),'ko'))*dir;}).forEach(r=>tb.appendChild(r));}
document.addEventListener('DOMContentLoaded',()=>document.querySelectorAll('.yt th').forEach(th=>{if(th.colSpan>1)return;th.addEventListener('click',()=>sortBy(th));}));"""


# ------------------------------------------------------------ Step 2: 대표값 대체 (사용자 결정 2026-11-17)
REP_PMCA, REP_PAL = "AWshape_PC1", "H_col"                                           # PMCA → 기도 굵기 대표값, PAL → 기둥 높이(AW_length와 같은 값)


def cluster2(d):
    """히트맵 정렬에서 HD_mm~AWshape_PC1로 묶이는 기도 크기 지표(대표값 AWshape_PC1)."""
    names = [v for v in STEP2 if v in d and d[v].notna().sum() >= 30]
    r = d[names].corr("spearman").fillna(0); order = [names[i] for i in leaves_list(linkage(1-np.abs(r.to_numpy()[np.triu_indices(len(names), 1)]), "average"))]
    i, j = order.index("HD_mm"), order.index(REP_PMCA); return order[min(i, j):max(i, j)+1], r


def step2(d):
    cl, r = cluster2(d); same = [v for v in STEP2 if v != REP_PAL and v in d and np.allclose(d[v], d[REP_PAL], equal_nan=True)]
    base = B0+[BCOL[k] for k in ROH]; u = d.dropna(subset=[REP_PMCA, REP_PAL])
    swap = lambda cols, a, b: [b if c == a else c for c in cols]
    specs = [("Step 1 모델", base), ("PMCA → AWshape_PC1", swap(base, BCOL["PMCA"], REP_PMCA)), ("PAL → H_col", swap(base, BCOL["PAL"], REP_PAL)),
             ("Step 2 모델 (둘 다 대체)", swap(swap(base, BCOL["PMCA"], REP_PMCA), BCOL["PAL"], REP_PAL))]
    models = {y: [(nm, model_json(*fit(u, y, cols), len(u))) for nm, cols in specs] for y in YS}
    base2 = specs[-1][1]; rows, eq = [], {}
    for x in [v for v in STEP2 if v in d and v not in cl and v != REP_PAL and v not in same and d[v].notna().sum() >= 30 and d[v].std() > 0]:
        w = d.dropna(subset=[x, REP_PMCA, REP_PAL]); eq[x] = {}
        for y in YS:
            f0, t0 = fit(w, y, base2); fa, ta = fit(w, y, base2+[x])
            eq[x][y] = {"base": model_json(f0, t0, len(w)), "add": model_json(fa, ta, len(w))}
            rows.append({"step": "2", "y": y, "var": x, "n": len(w), "pair": None, "r2_base": f0.rsquared, "add_beta": ta[x]["beta"], "add_p": ta[x]["p"],
                         "add_dR2": fa.rsquared-f0.rsquared, "add_vif": ta[x]["vif"]})
    t = pd.DataFrame(rows)
    for y in YS:
        k = t.index[t.y == y]; t.loc[k, "add_q"] = multipletests(t.loc[k, "add_p"], method="fdr_bh")[1]
    role = {}
    for v in STEP2:
        if v not in d: continue
        if v == REP_PMCA: role[v] = ("PMCA", "대표값: PMCA 자리에 넣음", "기도 단면 크기(굵기)")
        elif v in cl: role[v] = ("PMCA", f"AWshape_PC1로 대표 (r = {r.at[v, REP_PMCA]:+.2f})", "기도 단면 크기(굵기)")
        elif v == REP_PAL: role[v] = ("PAL", "대표값: PAL 자리에 넣음", "기도 길이(PNS–Eb 높이)")
        elif v in same: role[v] = ("PAL", "H_col과 같은 값 (중복, 제외)", "기도 길이(PNS–Eb 높이)")
        else: role[v] = ("", "Step 2 모델 위에 추가", "Step 1에 같은 차원 없음")
    return {"cluster": cl, "same": same, "n": len(u), "models": models, "t": t, "eq": eq, "role": role, "r": r}


def step2_html(S2, d, cat, figs, heat, pcs):
    vars_ = [v for v in STEP2 if v in d and d[v].notna().sum() >= 30]
    d1 = "".join(f'<tr><td><code>{escape(v)}</code></td><td>{escape(describe(v, cat)[0])}</td><td>{escape(describe(v, cat)[1])}</td><td>{thumb(v, figs)}</td></tr>' for v in vars_)
    p1 = f'<h3>2-1. 변수 설명</h3><div class="yt"><div class="tw"><table><thead><tr><th>변수명</th><th>변수 뜻</th><th>만든 과정</th><th>CT 위치</th></tr></thead><tbody>{d1}</tbody></table></div></div>'
    cards = "".join(f'<figure class="pcf"><a href="figs/pc/{f.name}" target="_blank"><img src="figs/pc/{f.name}" alt="{escape(tl)}"></a><figcaption><b>{escape(tl)}</b><br>{escape(ds)}</figcaption></figure>' for tl, f, ds in pcs)
    p2 = f'<h3>2-2. 주성분 적재량</h3><div class="pcgrid">{cards}</div>' if pcs else ""
    p3 = (f'<h3>2-3. 변수 간 상관 히트맵</h3><p class="note">83명 Spearman 상관. 왼쪽 5열은 Step 1 인자(복원값), 오른쪽은 신규 인자끼리. 누르면 크게 열린다.</p>'
          f'<a href="figs/{heat.name}" target="_blank"><img class="heat" src="figs/{heat.name}" alt="상관 히트맵"></a>')
    rr = d[vars_+[BCOL["PMCA"], BCOL["PAL"]]].corr("spearman")
    d4 = "".join(f'<tr><td><code>{escape(v)}</code></td><td>{S2["role"][v][0]}</td><td>{escape(S2["role"][v][1])}</td><td>{escape(S2["role"][v][2])}</td>'
                 f'<td>{nf(rr.at[v, BCOL["PMCA"]])}</td><td>{nf(rr.at[v, BCOL["PAL"]])}</td></tr>' for v in vars_)
    p4 = (f'<h3>2-4. 변수 간 대체 관계</h3><p class="note">히트맵에서 서로 상관이 높게 묶인 기도 크기 지표({", ".join(S2["cluster"])})는 AWshape_PC1 하나로 대표해 PMCA 자리에 넣었다. '
          f'PAL 자리에는 H_col(PNS–Eb 수직 높이)을 넣었다{" (AW_length는 H_col과 같은 값이라 제외)" if S2["same"] else ""}. 나머지는 Step 2 모델 위에 하나씩 추가한다.</p>'
          f'<div class="yt"><div class="tw"><table><thead><tr><th>변수</th><th>대체 짝</th><th>처리</th><th>도메인 의미</th><th>r (PMCA)</th><th>r (PAL)</th></tr></thead><tbody>{d4}</tbody></table></div></div>')
    eqs = {"Step 1 모델": "Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + PMCA + PAL", "PMCA → AWshape_PC1": "Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + AWshape_PC1 + PAL",
           "PAL → H_col": "Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + PMCA + H_col", "Step 2 모델 (둘 다 대체)": "Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + AWshape_PC1 + H_col"}
    foc = {"PMCA → AWshape_PC1": [REP_PMCA], "PAL → H_col": [REP_PAL], "Step 2 모델 (둘 다 대체)": [REP_PMCA, REP_PAL]}
    divs = []
    for i, y in enumerate(YS):
        cells = []
        for nm, m in S2["models"][y]:
            body = "".join(f'<tr{" class=focus" if t["t"] in foc.get(nm, []) else ""}><td>{escape(TERM.get(t["t"], t["t"]))}</td><td>{t["B"]:+.3g}</td><td>{nf(t["beta"])}</td>'
                           f'<td{" class=sig" if t["p"] < .05 and t["t"] != "const" else ""}>{pf(t["p"])}</td><td>{nf(t["vif"], ".2f")}</td></tr>' for t in m["terms"])
            cells.append(f'<div class="eqc"><p><b>{escape(nm)}</b><br><span class="eq">{escape(eqs[nm])}</span></p><table class="reg"><thead><tr><th>항</th><th>B</th><th>β</th><th>p</th><th>VIF</th></tr></thead>'
                         f'<tbody>{body}</tbody><tfoot><tr><td colspan=5>n {m["n"]} · R² {m["r2"]:.3f} · 수정 R² {m["ar2"]:.3f}</td></tr></tfoot></table></div>')
        divs.append(f'<div class="yt" data-s="2r" data-k="{i}|"{"" if i == 0 else " hidden"}><div class="eqrow">{"".join(cells)}</div></div>')
    opts = "".join(f'<option value="{i}">{escape(YL[y])}</option>' for i, y in enumerate(YS))
    p5 = (f'<h3>2-5. 대체 회귀식 (회귀식 단위)</h3><div class="ctl"><label>결과 변수(Y) <select id="y2r" onchange="showY(\'2r\')">{opts}</select></label>'
          f'<span class="note">같은 환자 {S2["n"]}명 (AWshape_PC1·H_col 값이 있는 환자). 대체한 항은 노란색.</span></div>' + "".join(divs))
    t = S2["t"]; divs = []
    for i, y in enumerate(YS):
        body = "".join(f'<tr class="pick" data-x="{escape(r["var"])}" onclick="showEq(\'2\', this.dataset.x)"><td><code>{escape(r["var"])}</code></td><td>{r["n"]}</td><td>{r["r2_base"]:.3f}</td>'
                       f'<td>{nf(r["add_beta"])}</td><td{" class=sig" if r["add_p"] < .05 else ""}>{pf(r["add_p"])}</td><td>{pf(r["add_q"])}</td><td>{nf(r["add_dR2"], "+.3f")}</td><td>{nf(r["add_vif"], ".2f")}</td></tr>'
                       for _, r in t[t.y == y].iterrows())
        divs.append(f'<div class="yt" data-s="2" data-k="{i}|"{"" if i == 0 else " hidden"}><div class="tw"><table><thead><tr><th>변수명</th><th>n</th><th>Step 2 모델 R²</th>'
                    f'<th class="g3">β</th><th class="g3">p</th><th class="g3">FDR q</th><th class="g3">ΔR²</th><th class="g3">VIF</th></tr></thead><tbody>{body}</tbody></table></div></div>')
    p6 = (f'<h3>2-6. Step 2 모델 위 추가 변수</h3><div class="ctl"><label>결과 변수(Y) <select id="y2" onchange="showY(\'2\')">{opts}</select></label>'
          f'<span class="note">행을 누르면 아래에 회귀식 표(Step 2 모델 · 추가 모델)가 열린다.</span></div>'
          '<p class="eq">추가: Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + AWshape_PC1 + H_col + X</p>' + "".join(divs) +
          '<div class="eqbox" id="eq2"><p class="note">요약표의 행을 누르면 회귀식 표가 여기에 열립니다.</p></div>')
    return (f'<h2 id="s2">Step 2. 기도 신규 인자</h2><p>Step 1의 기도 인자(PMCA, PAL)를 신규 기도 지표의 대표값으로 대체한 Step 2 모델을 만들고, 나머지 기도·연구개 지표가 그 위에 더하는지 본다.</p>'
            f'{p1}{p2}{p3}{p4}{p5}{p6}')


# ------------------------------------------------------------ Step 3: 대표값 대체 (사용자 결정 2026-11-17, Step 2 방식)
from scipy.cluster.hierarchy import fcluster
BASE2 = B0+[BCOL["ANB"], BCOL["MP-FH"], BCOL["TDI"], "AWshape_PC1", "H_col"]             # Step 2 모델


def step3(d):
    names = [v for v in STEP3 if v in d and d[v].notna().sum() >= 30 and d[v].std() > 0]
    r = d[names+[BCOL[k] for k in ROH]].corr("spearman")
    Z = linkage(1-np.abs(r.loc[names, names].fillna(0).to_numpy()[np.triu_indices(len(names), 1)]), "average")
    lab = dict(zip(names, fcluster(Z, t=0.3, criterion="distance")))                     # 묶음: 평균 |r| ≳ 0.7
    groups = {}
    for v, c in lab.items(): groups.setdefault(c, []).append(v)
    subs = {}
    for k in ("ANB", "MP-FH", "TDI"):                                                     # Step 1 인자와 가장 많이 겹치는 같은 차원 변수
        cand = [v for v in names if STEP3.get(v) and STEP3[v][0] == k]
        if cand: subs[k] = max(cand, key=lambda v: abs(r.at[v, BCOL[k]]))
    rep, role = {}, {}
    for c, mem in groups.items():
        sv = [v for v in mem if v in subs.values()]
        rp = sv[0] if sv else max(mem, key=lambda v: r.loc[v, mem].abs().mean())
        for v in mem: rep[v] = rp
    for v in names:
        k = next((k for k, x in subs.items() if x == v), None)
        if k: role[v] = (k, f"대표값: {k} 자리에 넣음 (r = {r.at[v, BCOL[k]]:+.2f})")
        elif rep[v] != v: role[v] = (STEP3[v][0] if STEP3.get(v) else "", f"{DISP.get(rep[v], rep[v])}로 대표 (r = {r.at[v, rep[v]]:+.2f}, 제외)")
        elif len(groups[lab[v]]) > 1: role[v] = (STEP3[v][0] if STEP3.get(v) else "", "묶음 대표값: Step 3 모델 위에 추가")
        else: role[v] = (STEP3[v][0] if STEP3.get(v) else "", "Step 3 모델 위에 추가")
    swap = lambda cols, a, b: [b if c == a else c for c in cols]
    base3 = BASE2
    for k, v in subs.items(): base3 = swap(base3, BCOL[k], v)
    u = d.dropna(subset=BASE2+list(subs.values()))
    specs = [("Step 2 모델", BASE2)]+[(f"{k} → {DISP.get(v, v)}", swap(BASE2, BCOL[k], v)) for k, v in subs.items()]+[("Step 3 모델 (모두 대체)", base3)]
    models = {y: [(nm, model_json(*fit(u, y, cols), len(u)), cols) for nm, cols in specs] for y in YS}
    adds = [v for v in names if rep[v] == v and v not in subs.values()]; rows, eq = [], {}
    for x in adds:
        w = d.dropna(subset=base3+[x]); eq[x] = {}
        for y in YS:
            f0, t0 = fit(w, y, base3); fa, ta = fit(w, y, base3+[x])
            eq[x][y] = {"base": model_json(f0, t0, len(w)), "add": model_json(fa, ta, len(w))}
            rows.append({"step": "3", "y": y, "var": x, "n": len(w), "pair": None, "r2_base": f0.rsquared, "add_beta": ta[x]["beta"], "add_p": ta[x]["p"],
                         "add_dR2": fa.rsquared-f0.rsquared, "add_vif": ta[x]["vif"]})
    t = pd.DataFrame(rows)
    for y in YS:
        k = t.index[t.y == y]; t.loc[k, "add_q"] = multipletests(t.loc[k, "add_p"], method="fdr_bh")[1]
    return {"subs": subs, "base3": base3, "n": len(u), "models": models, "t": t, "eq": eq, "role": role, "r": r, "names": names, "groups": groups, "lab": lab}


def eqstr(cols):
    return "Y ~ " + " + ".join(TERM.get(c, c).replace(" (복원값)", "") for c in cols)


def step3_html(S3, d, cat, figs, heat, pcs, intro):
    names = S3["names"]; r = S3["r"]
    d1 = "".join(f'<tr{kc("3", v)}><td><code>{escape(DISP.get(v, v))}</code></td><td>{escape(describe(v, cat)[0])}</td><td>{escape(describe(v, cat)[1])}</td><td>{thumb(v, figs)}</td></tr>' for v in names)
    p1 = legend3()+f'<h3>3-1. 변수 설명</h3><div class="yt"><div class="tw"><table><thead><tr><th>변수명</th><th>변수 뜻</th><th>만든 과정</th><th>CT 위치</th></tr></thead><tbody>{d1}</tbody></table></div></div>'
    cards = "".join(f'<figure class="pcf"><a href="figs/pc/{f.name}" target="_blank"><img src="figs/pc/{f.name}" alt="{escape(tl)}"></a><figcaption><b>{escape(tl)}</b><br>{escape(ds)}</figcaption></figure>' for tl, f, ds in pcs)
    p2 = f'<h3>3-2. 주성분·요인 축 적재량</h3><div class="pcgrid">{cards}</div>' if pcs else ""
    p3 = (f'<h3>3-3. 변수 간 상관 히트맵</h3><p class="note">83명 Spearman 상관. 왼쪽 5열은 Step 1 인자(복원값), 오른쪽은 신규 인자끼리. 글자 색 = 변수 출처. 누르면 크게 열린다.</p>'
          f'<a href="figs/{heat.name}" target="_blank"><img class="heat" src="figs/{heat.name}" alt="상관 히트맵"></a>')
    order = sorted(names, key=lambda v: (S3["lab"][v], v != S3["role"] and 0))
    d4 = "".join(f'<tr{kc("3", v)}><td><code>{escape(DISP.get(v, v))}</code></td><td>{S3["lab"][v]}</td><td>{S3["role"][v][0]}</td><td>{escape(S3["role"][v][1])}</td>'
                 + "".join(f'<td class="{"hi" if abs(r.at[v, BCOL[k]]) >= .5 else ""}">{nf(r.at[v, BCOL[k]])}</td>' for k in ("ANB", "MP-FH", "TDI")) + "</tr>" for v in order)
    sb = ", ".join(f"{k} → {DISP.get(v, v)}" for k, v in S3["subs"].items())
    p4 = (f'<h3>3-4. 변수 간 대체 관계</h3><p class="note">히트맵에서 서로 상관이 높은(평균 |r| ≳ 0.7) 변수를 같은 묶음 번호로 묶었다. Step 1 인자마다, 같은 차원으로 짝지은 변수 중 그 인자와 가장 많이 겹치는(|r| 최대) 변수 하나를 대표 대체값으로 썼다: <b>{escape(sb)}</b>. '
          f'대체값과 같은 묶음의 나머지는 대체값으로 대표하고, 다른 묶음은 묶음 대표값 하나만 Step 3 모델 위에 추가한다. |r| ≥ 0.5는 색으로 표시.</p>'
          f'<div class="yt"><div class="tw"><table><thead><tr><th>변수</th><th>묶음</th><th>대체 짝</th><th>처리</th><th>r (ANB)</th><th>r (MP-FH)</th><th>r (TDI)</th></tr></thead><tbody>{d4}</tbody></table></div></div>')
    foc = {nm: [c for c in cols if c not in BASE2] for nm, _, cols in S3["models"][YS[0]]}
    divs = []
    for i, y in enumerate(YS):
        cells = []
        for nm, m, cols in S3["models"][y]:
            body = "".join(f'<tr{" class=focus" if t["t"] in foc[nm] else ""}><td>{escape(TERM.get(t["t"], DISP.get(t["t"], t["t"])))}</td><td>{t["B"]:+.3g}</td><td>{nf(t["beta"])}</td>'
                           f'<td{" class=sig" if t["p"] < .05 and t["t"] != "const" else ""}>{pf(t["p"])}</td><td>{nf(t["vif"], ".2f")}</td></tr>' for t in m["terms"])
            cells.append(f'<div class="eqc"><p><b>{escape(nm)}</b><br><span class="eq">{escape(eqstr(cols))}</span></p><table class="reg"><thead><tr><th>항</th><th>B</th><th>β</th><th>p</th><th>VIF</th></tr></thead>'
                         f'<tbody>{body}</tbody><tfoot><tr><td colspan=5>n {m["n"]} · R² {m["r2"]:.3f} · 수정 R² {m["ar2"]:.3f}</td></tr></tfoot></table></div>')
        divs.append(f'<div class="yt" data-s="3r" data-k="{i}|"{"" if i == 0 else " hidden"}><div class="eqrow">{"".join(cells)}</div></div>')
    opts = "".join(f'<option value="{i}">{escape(YL[y])}</option>' for i, y in enumerate(YS))
    p5 = (f'<h3>3-5. 대체 회귀식 (회귀식 단위)</h3><div class="ctl"><label>결과 변수(Y) <select id="y3r" onchange="showY(\'3r\')">{opts}</select></label>'
          f'<span class="note">Step 2 모델에서 출발해 Step 1 골격 인자를 하나씩, 그리고 모두 대체. 같은 환자 {S3["n"]}명. 대체한 항은 노란색.</span></div>' + "".join(divs))
    t = S3["t"]; divs = []
    for i, y in enumerate(YS):
        body = "".join(f'<tr class="pick {KIND3.get(r_["var"], "")}" data-x="{escape(r_["var"])}" onclick="showEq(\'3\', this.dataset.x)"><td><code>{escape(DISP.get(r_["var"], r_["var"]))}</code></td><td>{r_["n"]}</td><td>{r_["r2_base"]:.3f}</td>'
                       f'<td>{nf(r_["add_beta"])}</td><td{" class=sig" if r_["add_p"] < .05 else ""}>{pf(r_["add_p"])}</td><td>{pf(r_["add_q"])}</td><td>{nf(r_["add_dR2"], "+.3f")}</td><td>{nf(r_["add_vif"], ".2f")}</td></tr>'
                       for _, r_ in t[t.y == y].iterrows())
        divs.append(f'<div class="yt" data-s="3" data-k="{i}|"{"" if i == 0 else " hidden"}><div class="tw"><table><thead><tr><th>변수명</th><th>n</th><th>Step 3 모델 R²</th>'
                    f'<th class="g3">β</th><th class="g3">p</th><th class="g3">FDR q</th><th class="g3">ΔR²</th><th class="g3">VIF</th></tr></thead><tbody>{body}</tbody></table></div></div>')
    p6 = (f'<h3>3-6. Step 3 모델 위 추가 변수</h3><div class="ctl"><label>결과 변수(Y) <select id="y3" onchange="showY(\'3\')">{opts}</select></label>'
          f'<span class="note">묶음 대표값과 단독 변수. 행을 누르면 아래에 회귀식 표(Step 3 모델 · 추가 모델)가 열린다.</span></div>'
          f'<p class="eq">추가: {escape(eqstr(S3["base3"]))} + X</p>' + "".join(divs) + '<div class="eqbox" id="eq3"><p class="note">요약표의 행을 누르면 회귀식 표가 여기에 열립니다.</p></div>')
    return f'<h2 id="s3">Step 3. 3D 골격 신규 인자</h2><p>{intro}</p>{p1}{p2}{p3}{p4}{p5}{p6}'


def main():
    import osa_var_figs as VF
    d = data(); cat = A.catalog(); filled = d.attrs["filled"]; S1 = step1(d)
    cands = {"2": STEP2, "3": STEP3, "4": step4(d)}; T, EQ = [], {}
    for s, c in cands.items():
        if s == "2": S2 = step2(d); t, eq = S2["t"], S2["eq"]
        elif s == "3": S3 = step3(d); t, eq = S3["t"], S3["eq"]
        else: t, eq = newfactor(d, c, s)
        T.append(t); EQ[s] = eq
    T = pd.concat(T); T.round(5).to_csv(OUT/"step1_4.csv", index=False, encoding="utf-8-sig")
    names = list(dict.fromkeys(list(MCOL.values())+list(XCOL.values())+list(STEP2)+list(STEP3)+list(T["var"]))); ex, figs = VF.build(d, names)
    s1 = pd.DataFrame([{"ver": v, "y": y, "term": t, **r, "n": m["n"], "R2": m["r2"], "adjR2": m["ar2"]} for (v, y), m in S1.items() for t, r in m["terms"].items()])
    s1.round(5).to_csv(OUT/"step1_roh.csv", index=False, encoding="utf-8-sig")
    intro = {"2": "기도 신규 인자. 단면 관련 지표는 PMCA, 길이·높이 지표는 PAL과 같은 차원으로 보았다. 연구개·비강·위치·모양 지표는 Step 1에 같은 차원이 없다.",
             "3": "3D 골격 신규 인자. 앞뒤 관계 → ANB, 수직 패턴 → MP-FH, 횡방향 폭 → TDI와 같은 차원으로 보았다. Step 2 모델에서 출발해, Step 1 골격 인자를 같은 차원의 대표값으로 대체한 Step 3 모델을 만들고, 나머지 묶음 대표값·단독 변수가 그 위에 더하는지 본다.",
             "4": "연조직·연동·형상 탐색 인자(혀, 저작근, 부비동, 설골, 구강 공간, 요인 점수, 지렛대, 잔여 형상, 공동 축, 부위별 크기·위치). Step 1에 같은 차원이 없어 추가 모델로 판단한다."}
    titles = {"2": "Step 2. 기도 신규 인자", "3": "Step 3. 3D 골격 신규 인자", "4": "Step 4. 연조직·연동·형상 탐색"}
    import osa_pc_figs as PCF
    PCS = PCF.build({"2": list(STEP2), "3": list(STEP3), "4": list(T[T.step == "4"]["var"].unique())})
    secs = [step1_html(S1, cat, figs, filled)]
    for s in ("2", "3", "4"):
        t = T[T.step == s]
        if s == "3": intro[s] += " 표의 행 색과 히트맵 글자 색으로 변수 출처를 구분했다(맨 위 범례)."
        if s == "2": secs.append(step2_html(S2, d, cat, figs, heatmap(d, [v for v in STEP2 if v in d and d[v].notna().sum() >= 30], s), PCS.get(s, []))); continue
        if s == "3": secs.append(step3_html(S3, d, cat, figs, heatmap(d, S3["names"], s), PCS.get(s, []), intro[s])); continue
        heat = heatmap(d, list(dict.fromkeys(t["var"])), s)
        secs.append(nf_html(s, titles[s], intro[s], t, cat, figs, heat, PCS.get(s, [])))
    pair = {s: {x: (c[x][0] if c.get(x) and s not in ("2", "3") else None) for x in EQ[s]} for s, c in cands.items()}
    term = {**TERM, **{x: DISP.get(x, x) for s in EQ for x in EQ[s]}, REP_PMCA: "AWshape_PC1 (기도 굵기)", REP_PAL: "H_col (기도 높이)"}
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>83명 단계별 회귀</title>
<style>{CSS}td.sig{{font-weight:700;color:var(--accent)}}.yt table,table.reg{{border-collapse:collapse}}.yt th,.yt td,table.reg th,table.reg td{{border:1px solid #b8c2cc;padding:4px 6px}}
.yt th{{cursor:pointer;user-select:none;white-space:nowrap}}.yt th[colspan]{{cursor:default}}.yt th:not([colspan]):after{{content:' ⇅';color:#aab}}.yt th.asc:after{{content:' ▲';color:inherit}}.yt th.desc:after{{content:' ▼';color:inherit}}
th.g2{{background:#eefaf0}}th.g3{{background:#fff4ec}}td code{{font-size:.8rem}}img.thumb{{width:120px;margin:0;border-radius:4px}}img.heat{{width:100%;max-width:1400px}}
tr.pick{{cursor:pointer}}tr.pick:hover td{{background:#f3f6fa}}tr.pick.on td{{background:#e3edff}}tr.focus td{{background:#fff3cd}}table.reg tfoot td{{color:var(--muted);font-size:.85rem}}
.eqbox{{border:1px solid #b8c2cc;border-radius:6px;padding:8px 12px;margin:10px 0}}.eqrow{{display:flex;gap:16px;flex-wrap:wrap}}.eqc{{flex:1 1 300px}}
tr.c1 td{{background:#e6efff}}tr.c1 td:first-child{{border-left:4px solid #2563eb}}.legend .c1{{background:#e6efff;border-left:4px solid #2563eb;padding:2px 8px;margin-right:8px;display:inline-block}}tr.c2 td{{background:#f1e9ff}}tr.c2 td:first-child{{border-left:4px solid #7c3aed}}.legend .c2{{background:#f1e9ff;border-left:4px solid #7c3aed;padding:2px 8px;margin-right:8px;display:inline-block}}tr.c3 td{{background:#e3f7ee}}tr.c3 td:first-child{{border-left:4px solid #059669}}.legend .c3{{background:#e3f7ee;border-left:4px solid #059669;padding:2px 8px;margin-right:8px;display:inline-block}}tr.c4 td{{background:#fff1dc}}tr.c4 td:first-child{{border-left:4px solid #d97706}}.legend .c4{{background:#fff1dc;border-left:4px solid #d97706;padding:2px 8px;margin-right:8px;display:inline-block}}.pcgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}.pcf{{margin:0;border:1px solid #b8c2cc;border-radius:6px;padding:6px}}.pcf img{{width:100%}}.pcf figcaption{{font-size:.82rem}}
.ctl{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:8px 0}}.ctl select{{font-size:.95rem;padding:3px 6px;max-width:100%}}.eq{{font-family:monospace;font-size:.85rem;margin:6px 0}}</style></head><body><main>
<h1>83명 단계별 회귀 결과</h1><p class="sub">2026-11-17 · 분할 없음 (계수 유의성 판단용) · 대상 83명 (19세 미만 2명, CBCT 없음 2명 제외, Class III 3명 포함)</p>
<nav><a href="#s1">Step 1</a><a href="#s2">Step 2</a><a href="#s3">Step 3</a><a href="#s4">Step 4</a></nav>
<div class="box"><b>읽는 법</b><ul>
<li><b>Step 1</b>은 Roh 2025 Table 3과 같은 동시 다중회귀다. 성별·나이·BMI와 ANB·MP-FH·TDI·PMCA·PAL을 한 모델에 넣었고, '값' 상자로 엑셀 값과 복원값을 바꾼다.</li>
<li><b>Step 2~4</b>는 같은 순서로 정리했다. <b>1. 변수 설명</b>(무엇을 재는지, 원자료부터 계산까지) → <b>주성분·요인 축 결과</b>(해당 변수가 있는 단계만) → <b>상관 히트맵</b>(Step 1 인자와, 신규 인자끼리) → <b>대체 관계</b>(같은 차원의 Step 1 인자) → <b>회귀 결과</b>(X 전체 요약표, 행을 누르면 회귀식 단위 표).</li>
<li><b>대체 모델</b>: 같은 차원의 Step 1 인자 자리에 X를 넣는다. ΔR²(Step 1 모델 대비)가 비슷하면 같은 정보, 크면 더 나은 측정이다. <b>추가 모델</b>: Step 1 인자 5개를 모두 둔 채 X를 더한다. 유의하면 Step 1 인자가 보지 못한 새로운 영향이다.</li>
<li>Step 2~4의 Step 1 인자는 복원값 판(결측은 엑셀 값)이며, X 값이 있는 환자만으로 세 모델을 같은 환자에서 비교했다.</li>
<li><b>B</b>: 비표준화 계수, <b>β</b>: 표준화 계수(1 SD 증가 시 Y의 SD 변화), <b>p</b>: t 검정(보정 전), <b>FDR q</b>: 단계 안 다중검정 보정, <b>VIF</b>: 다중공선성(5 이상 주의), <b>R²</b>: 모델이 설명하는 Y 분산 비율, <b>수정 R²</b>: 변수 수를 감안한 R². p &lt; 0.05는 굵게. 열 제목을 누르면 정렬된다.</li>
<li><b>CT 위치</b>: CT에서 바로 잰 변수는 예시 환자(P_{ex})의 CBCT에 위치를 그렸다(누르면 크게). Roh 지표 그림은 기존 CBCT 보고서의 용어 그림(다른 예시 환자)이다. 계산식으로 만든 변수는 그림이 없다.</li></ul></div>
{"".join(secs)}
<script>const YS={json.dumps(YS)};const YL={json.dumps(YL, ensure_ascii=False)};const TERM={json.dumps(term, ensure_ascii=False)};const PAIR={json.dumps(pair, ensure_ascii=False)};const BASE={json.dumps({"2": ["Step 2 모델 (같은 환자)", "Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + AWshape_PC1 + H_col"], "3": ["Step 3 모델 (같은 환자)", eqstr(S3["base3"])]}, ensure_ascii=False)};
const EQ={json.dumps(EQ, ensure_ascii=False, separators=(",", ":"))};
{JS}</script></main></body></html>"""
    check(html, "83명 단계별 회귀"); f = OUT/"전체_83명_단계별_회귀.html"; f.write_text(apply(html), encoding="utf-8"); print("saved", f, round(len(html)/1e6, 1), "MB | 채움", filled)
    miss = [v for v in T["var"].unique() if detail(v) is None and v not in cat]; print("설명 없음:", miss)


if __name__ == "__main__":
    main()
