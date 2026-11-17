"""교수님 보고용 OSA 관련성 보고서 (논문 구성: 요약·서론·대상과 방법·결과·고찰·결론·부록). 로컬 HTML.
그림 함수는 osa_relevance_report에서 가져오고, 대상자 특성표·회귀 계수표·교차검증 모식도·자세별 그림만 새로 만든다.
출력: Result_py/report_85/OSA_관련성_보고서_2026-11-17.html (+ figs_osa/)
"""
import re
from html import escape
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, fisher_exact
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from report_terms import check
from report_editor import apply
from cbct_final_report import CSS, style, tab, R, C_DEV, C_VAL, INK, MUTED
import osa_experiments as X
import osa_validate_holdout as VH
import osa_relevance_report as OR

OUT = OR.DEST/"OSA_관련성_보고서_2026-11-17.html"; FIG = OR.FIG
PRED = {"Age": "나이 (세)", "Male": "남성", "BMI": "BMI (kg/m²)", "m_ANB": "ANB (°)", "Pog_Ba_ant": "턱끝–대후두공 앞뒤 거리 (mm)"}


def cohort():
    """최종 80명(Class III·19세 미만 제외)과 발굴/검증 구분. 결과 변수는 리포트 기준 복원값."""
    d, f = VH.load_frozen(); o = pd.read_csv(R/"phase1/psg_restored_own.csv", encoding="utf-8-sig", index_col=0)
    ids = [p for p in d.index if p in o.index and o.at[p, "Include_own"]]
    s = d.loc[ids].copy(); s["AHI"], s["AHI_SUP"] = o.loc[ids, "AHI_r"], o.loc[ids, "AHI_SUP_r"]
    s["logAHI"], s["logAHI_SUP"] = np.log1p(s.AHI), np.log1p(s.AHI_SUP); s["grp"] = np.where(s.index.isin(f["discovery_ids"]), "발굴", "검증")
    return s


def char_table(s):
    def cont(v, mean=True): return f"{v.mean():.1f} ± {v.std():.1f}" if mean else f"{v.median():.1f} [{v.quantile(.25):.1f}–{v.quantile(.75):.1f}]"
    D, V = s[s.grp == "발굴"], s[s.grp == "검증"]; rows = []
    for c, lab, mean in (("Age", "나이 (세)", True), ("BMI", "BMI (kg/m²)", True), ("AHI", "AHI (회/시간)", False), ("AHI_SUP", "앙와위 AHI (회/시간)", False),
                         ("m_ANB", "ANB (°)", True), ("Pog_Ba_ant", "턱끝–대후두공 앞뒤 거리 (mm)", True)):
        p = mannwhitneyu(D[c].dropna(), V[c].dropna())[1]
        rows.append([lab, cont(D[c], mean), cont(V[c], mean), cont(s[c], mean), f"{p:.2f}"])
    m = lambda g: f"{int(g.Male.sum())} ({100*g.Male.mean():.0f}%)"
    rows.insert(1, ["남성, 명 (%)", m(D), m(V), m(s), f"{fisher_exact(pd.crosstab(s.grp, s.Male))[1]:.2f}"])
    sev = lambda g: f"{int((g.AHI >= 30).sum())} ({100*(g.AHI >= 30).mean():.0f}%)"
    rows.append(["중증 (AHI ≥ 30), 명 (%)", sev(D), sev(V), sev(s), f"{fisher_exact(pd.crosstab(s.grp, s.AHI >= 30))[1]:.2f}"])
    return pd.DataFrame(rows, columns=["항목", f"발굴 (n={len(D)})", f"검증 (n={len(V)})", f"전체 (n={len(s)})", "p (발굴 대 검증)"])


def coef_table(u):
    cols = X.B0+X.CONFIRMED; out = {}; foot = {}
    for y, lab in (("logAHI", "AHI"), ("logAHI_SUP", "앙와위 AHI")):
        v = u.dropna(subset=[y]+cols); fit = sm.OLS(v[y], sm.add_constant(v[cols])).fit(); ci = fit.conf_int()
        beta = fit.params[cols]*v[cols].std()/v[y].std()
        out[lab] = {c: (f"{fit.params[c]:+.3f} ({ci.at[c, 0]:+.3f}, {ci.at[c, 1]:+.3f})", f"{beta[c]:+.2f}", f"{fit.pvalues[c]:.3f}" if fit.pvalues[c] >= .001 else "<0.001") for c in cols}
        cv = X.compare(v, y, X.B0, X.CONFIRMED)
        foot[lab] = (len(v), fit.rsquared, fit.rsquared_adj, cv["R2cv"], cv["R2cv_ref"])
    rows = [[PRED[c]]+[x for lab in out for x in out[lab][c]] for c in cols]
    t = pd.DataFrame(rows, columns=["변수"]+[f"{lab} {k}" for lab in out for k in ("B (95% CI)", "β", "p")])
    ft = "; ".join(f"{lab}: n={n}, R² {r2:.2f}, 수정 R² {a:.2f}, 교차검증 R² {cv:.2f} (나이·성별·BMI만 {cv0:.2f})" for lab, (n, r2, a, cv, cv0) in foot.items())
    return t, ft


def fig_cv():
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axis("off")
    for k in range(5):
        for j in range(5):
            test = j == k
            ax.add_patch(Rectangle((j*1.05, 4-k), 1, .8, color=C_VAL if test else "#d8dee6"))
            ax.text(j*1.05+.5, 4.4-k, "예측" if test else "학습", ha="center", va="center", fontsize=8, color="white" if test else INK)
        ax.text(-.15, 4.4-k, f"{k+1}회차", ha="right", va="center", fontsize=8, color=MUTED)
    ax.text(5.6, 2.4, "59명을 무작위로 5묶음(약 12명)으로 나눔\n→ 4묶음으로 회귀식을 만들고\n   남은 1묶음의 AHI를 예측\n→ 5회차가 끝나면 모든 환자가\n   한 번씩 '처음 보는 환자'로 예측됨\n→ 예측값과 실제값으로 R² 계산\n\n묶음을 새로 나누어 50번 반복 → 평균",
            fontsize=8.5, va="center", color=INK, linespacing=1.7)
    ax.set_xlim(-1, 11); ax.set_ylim(-.2, 5); fig.tight_layout(); f = FIG/"cv_scheme.png"; fig.savefig(f, dpi=90); plt.close(fig); return f


def fig_position(s):
    t = pd.read_csv(R/"analysis/positional/positional.csv", encoding="utf-8-sig", index_col=0)
    v = pd.read_csv(R/"psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
    t = t[(v.SameExam.reindex(t.index) == True) & (t.index != "P_75") & t.POS_ratio.notna()]
    u = s.join(t[["AHI_sup_r", "AHI_non_r"]], how="inner").dropna(subset=["m_ANB", "AHI_sup_r", "AHI_non_r"])
    r = pd.read_csv(R/"analysis/positional/positional_tests.csv", encoding="utf-8-sig")
    r = r[(r["sample"] == "최종 80명") & (r.x == "m_ANB") & (r.adj == "-")].set_index("y")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, (c, y, lab) in zip(ax, (("AHI_sup_r", "logSUP", "앙와위 AHI"), ("AHI_non_r", "logNON", "비앙와위 AHI"))):
        style(a); Z = sm.add_constant(u[X.B0]); res = lambda v: v-sm.OLS(v, Z).fit().fittedvalues      # 나이·성별·BMI를 뺀 잔차끼리 (보정 회귀의 기울기와 같음)
        xr, yr = res(u.m_ANB), res(np.log1p(u[c])); a.scatter(xr, yr, s=18, color=C_DEV if y == "logSUP" else C_VAL, alpha=.8)
        b = np.polyfit(xr, yr, 1); xs = np.linspace(xr.min(), xr.max(), 10); a.plot(xs, np.polyval(b, xs), color=INK, lw=1)
        pv = r.at[y, "p"]; a.set_title(f"{lab}: β {r.at[y, 'beta_std']:+.2f}, p{'<0.001' if pv < .001 else f'={pv:.3f}'} (n={int(r.at[y, 'n'])})", fontsize=9, color=INK)
        a.set_xlabel("ANB (나이, 성별, BMI 보정 잔차, °)", fontsize=8, color=MUTED)
    ax[0].set_ylabel("log(AHI + 1) (보정 잔차)", fontsize=8, color=MUTED); fig.tight_layout(); f = FIG/"position_anb.png"; fig.savefig(f, dpi=90); plt.close(fig); return f


def appendix():
    """docs/실험_가설_설계_결과.md의 실험 표를 그대로 부록으로. 파일 이름·내부 표현은 지운다."""
    md = (R.parent/"docs/실험_가설_설계_결과.md").read_text(encoding="utf-8"); out = []; sec = None
    fix = [(r"`[^`]*\.(csv|json|py|md)[^`]*`", ""), (r"\[([^\]]+)\]\([^)]+\)", r"\1"), (r"\*\*([^*]+)\*\*", r"<b>\1</b>"), (r"`([^`]+)`", r"\1"),
           ("동결", "고정"), ("yao4", "4점 수평면"), ("jung", "3점 수평면"), ("Track", "트랙"), ("compare_all", ""), ("전문가 확인", "추가 확인"), (r"\bM0 ", "확정 모델 "), (r"\bM1 ", "확정 + 크기 축 ")]
    for line in md.splitlines():
        if line.startswith("## ") and not line.startswith("## 종합") and not line.startswith("## 공통"): sec = line[3:].strip(); continue
        if line.startswith("## "): sec = None; continue
        if sec and line.startswith("| ") and not line.startswith("| ID") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) != 5 or c[0].startswith("V-dry") or c[0].startswith("V0"): continue
            for a, b in fix: c = [re.sub(a, b, x) for x in c]
            c = [escape(x).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>") for x in c]
            out.append(f"<tr><td>{escape(sec.split('.')[0])}</td>"+"".join(f"<td>{x}</td>" for x in c)+"</tr>")
    return '<div class="tw"><table class="apx"><thead><tr><th>단계</th><th>실험</th><th>가설</th><th>설계</th><th>결과</th><th>판정</th></tr></thead><tbody>'+"".join(out)+"</tbody></table></div>"


def hyp_box(rows):
    li = "".join(f'<tr><td>{n}<br><span class="note">{i}</span></td><td>{h}</td><td>{d}</td><td><b>{v}</b></td></tr>' for n, (i, h, d, v) in enumerate(rows, 1))
    return f'<details class="hyp" open><summary>검정한 가설</summary><div class="tw"><table><thead><tr><th>번호</th><th>가설</th><th>설계</th><th>판정</th></tr></thead><tbody>{li}</tbody></table></div></details>'


HYPS = {
 "r2": [("S1", "11개 지표 각각 엑셀 값의 AHI 결과가 자동 측정값으로 재현된다", "지표마다 기준 모델 + 엑셀 값 대 기준 모델 + 자동 값, 같은 환자", "ANB만 재현"),
        ("S1b", "턱끝–대후두공 앞뒤 거리가 Nperp-Pog의 정보를 대체한다", "단독·조합 비교, 크기·입 벌림 보정", "채택"),
        ("S1c", "MP-FH 신호를 FH 없는 수직 지표(SN-MP, PP-MP)로 살릴 수 있다", "단독과 ANB 위 비교", "기각"),
        ("S1d", "엑셀–자동 불일치의 원인이 특정 기준점에 있다", "엑셀 − 자동 차이의 지표 간 상관", "기각: 공통 원인은 FH 수평면 방향"),
        ("S1e", "자동값으로 Roh 분류(시상·수직)를 재현할 수 있다", "분류 일치율·κ", "부분 (75%)")],
 "r3": [("S1", "Roh 논문의 골격 지표 중 실제로 AHI를 설명하는 것이 있다", "기준 모델 위 ANB / ANB·MP-FH / 골격 3개 / + PMCA·PAL", "ANB 채택"),
        ("S3 (사전 등록)", "3D 골격 지표가 ANB를 넘어 AHI를 설명한다", "기준 모델 위와 ANB 위", "턱끝–대후두공 거리만 작게")],
 "r4": [("S2", "골격 지표가 기도(PMCA·PAL·PAV)를 설명한다", "기준 모델 위·ANB 위, FDR", "PAV만, 머리 크기로 설명"),
        ("S2c", "골격의 AHI 효과는 기도 단면적을 거친다", "매개 분석, 간접효과 부트스트랩 2000회", "기각"),
        ("A2", "기도 특징(구간별 최소·평균 단면, 최소 위치)이 확정 인자에 더한다", "13개, 확정 인자 위", "기각")],
 "r5": [("B1·B2 (사전 등록)", "턱끝–대후두공 거리의 신호는 하악 위치와 두개저 깊이 중 한쪽, ANB는 SNA·SNB 중 한쪽이 주도한다", "항등식 분해, SNA + SNB 대 ANB", "하악 위치 / 두 턱의 차이 자체"),
        ("D1 (사전 등록)", "하악 후퇴는 하악체 길이·하악지 회전·하악각 중 특정 형태에서 오며 크기 때문이 아니다", "하악 표면 2048점 회귀, 순열 999회, 크기 보정", "채택: 짧은 하악체"),
        ("P1 (사전 등록)", "ANB가 클수록 자세 의존성(앙와위 ÷ 비앙와위 AHI)이 크다", "판독 파일 자세 정보로 직접 계산, 기준 모델 위", "기각"),
        ("P2 (사전 등록)", "ANB는 앙와위 AHI와 관련되고 비앙와위 AHI와는 약하다", "같은 환자에서 두 결과에 따로 회귀", "채택")],
 "r6": [("S5 (사전 등록)", "머리·턱 구조가 작을수록 ANB와 무관하게 AHI가 높다", "부위 크기 12개 첫 주성분(구조 크기 축)", "후보 → 새 환자에서 기각"),
        ("G (사전 등록)", "세 방법에서 반복된 수직 배치를 하나로 묶으면 확정 인자 위에서 AHI를 설명한다", "세 성분 z 평균, 계산 방식 고정", "후보"),
        ("A4", "후보 103개 중 확정 인자에 더할 안정적 묶음이 있다", "elastic net, 안정성 선택 200회", "기각"),
        ("C (사전 등록)", "후보 9개 중 서로 흡수되지 않고 독립적으로 남는 것이 있다", "후보 간 상관, 함께 넣은 모델", "크기 축만 독립"),
        ("E (사전 등록)", "하악·상악·두개저 표면 대응점을 이은 공동 축이 확정 인자에 더한다", "형태·배치 PCA", "넷째 축(수직 배치)만")],
 "r7": [("V1 (사전 등록)", "확정 모델 + 크기 축, 확정 모델 + 수직 배치 지표가 새 환자에서 확정 모델보다 낫다", "발굴 59명 계수를 고정해 새 환자 21명에 적용. R² 증가 &gt; 0, 같은 부호, 단측 p &lt; 0.05", "크기 기각, 수직 배치 미달 (같은 방향)"),
        ("V2", "같음 (Class III 포함)", "최종 83명, 검증 24명", "결론 같음")],
}


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; FIG.mkdir(parents=True, exist_ok=True)
    s = cohort(); ct = char_table(s); D = s[s.grp == "발굴"]
    ctD, ftD = coef_table(D); ctA, ftA = coef_table(s)
    f_cv = fig_cv(); f_pos = fig_position(s); f_anb, _ = OR.fig_anb(); f_steps, steps = OR.model_steps(); f_cand = OR.fig_candidates()
    f_d1 = OR.copy_fig(OR.SM/"D1_mandible_size/m_ANB_form.png", "mandible_anb.png"); f_ax = OR.copy_fig(OR.SM/"E_joint_axes/F1_PC1.png", "joint_axis1.png")
    stp = steps.pivot(index="모델", columns="결과", values="교차검증 R²").reindex(list(steps.모델.unique())).round(3).reset_index()
    MN = {"M0 확정": "확정 모델", "M1 = M0 + SIZE_PC1": "확정 + 구조 크기 축", "M0 + VERT_Z": "확정 + 수직 배치 지표"}
    def vt(f):
        v = pd.read_csv(OR.VAL/f, encoding="utf-8-sig"); v = v[v.model.isin(MN)]
        v = v.assign(결과=v.hyp.map(OR.HYP), 모델=v.model.map(MN))[["결과", "모델", "n_val", "R2_val", "dR2_val", "p_one_sided"]].round(3)
        v.columns = ["결과", "모델", "검증 환자 수", "새 환자 R²", "R² 증가", "단측 p"]; return v
    val, val3 = vt("validation_own_2026-11-16.csv"), vt("validation_own_c3_2026-11-16.csv")
    Dz = D.dropna(subset=["logAHI_SUP"]+X.B0+X.CONFIRMED); anb_b = float(sm.OLS(Dz.logAHI_SUP, sm.add_constant(Dz[X.B0+X.CONFIRMED])).fit().params["m_ANB"])

    s1 = pd.DataFrame([("ANB", "있음", "재현됨 (엑셀 이상)", "자동 측정값으로 대체"),
                       ("Nperp-Pog", "있음", "재현 안 됨", "턱끝–대후두공 앞뒤 거리로 대체"),
                       ("MP-FH", "약함", "재현 안 됨", "모델에서 제외 (ANB와 겹치는 정보)"),
                       ("Nperp-A, TDI, PHI, NCWR, PAL, PAV, NAV, PMCA", "없음", "없음", "예측력으로는 판정 불가, 일치도로만 평가")],
                      columns=["지표", "엑셀 값의 AHI 관련", "자동 값의 AHI 관련", "처리"])
    var = pd.DataFrame([
        ("결과", "AHI", "수면 1시간당 무호흡·저호흡 횟수. 분석에는 log(AHI + 1)", "PSG 리포트"),
        ("결과", "앙와위 AHI", "똑바로 누워 잔 시간만의 AHI. 분석에는 log(앙와위 AHI + 1)", "PSG 리포트"),
        ("결과(보조)", "비앙와위 AHI", "옆·엎드린 자세 수면시간만의 AHI", "판독 파일에서 직접 계산"),
        ("확정 인자", "ANB", "N–A선과 N–B선 사이 각을 정중 시상면에 투영. 클수록 하악이 상악보다 뒤", "자동 기준점"),
        ("확정 인자", "턱끝–대후두공 앞뒤 거리", "턱끝(Pog)의 앞뒤 위치 − 대후두공 앞점(Ba)의 앞뒤 위치. 작을수록 하악 후퇴. Roh 논문 Nperp-Pog 대체", "자동 기준점"),
        ("후보", "수직 배치 지표", "세 성분 z 점수 평균: 아래 앞얼굴 비율(ANS–Me ÷ N–Me), 하악 잔여 형상의 수직 성분, 공동 축 넷째 축. 클수록 하악이 상악보다 아래", "자동 기준점, 자동 분할"),
        ("후보(기각)", "구조 크기 축", "하악·상악·두개저 12부위 크기의 첫 주성분(분산 69%). 클수록 큰 머리", "자동 분할"),
        ("기도", "PAV, PMCA, PAL", "Roh 논문 정의를 따른 인두 부피, 최소 단면적(중심선 수직), 길이. 공기 −1200~−400 HU", "밝기 기준 분할"),
        ("형태", "하악 표면 대응점, 공동 축", "자동 분할한 하악(·상악·두개저) 표면의 2048개 대응점과 그 주성분", "자동 분할")],
        columns=["구분", "변수", "정의", "원천"])
    roh = pd.DataFrame([
        ("대상·계측", "102명, 수작업 계측 (Anatomage, Mimics)", "80명, CBCT 자동 계측 (기준점 검출, 구조 분할)"),
        ("분석", "표현형 그룹 비교 + 다중회귀 (성별·나이·BMI 보정), p &lt; 0.05", "다중회귀 (같은 보정) + 반복 교차검증 + 새 환자 검증, 사전 등록"),
        ("ANB", "AHI·앙와위 AHI 등 6개 PSG 변수의 유의 예측 인자", "재현. 자동값이 엑셀값보다 예측력이 크고, 앙와위 AHI에서 더 강하며, 새 환자에서도 재현"),
        ("TDI (횡방향)", "유의 예측 인자, CSP3D가 가장 중증", "이 자료에서는 엑셀 값으로도 AHI 관련이 재현되지 않음"),
        ("MP-FH (수직)", "고각형이 정상각형보다 중증", "엑셀 값에서 약함. 자동 값은 FH 수평면 오차로 재현되지 않음. FH 없는 수직 배치 지표가 후보"),
        ("기도 (PMCA·PAL)", "CSP에서 PMCA 작고 PAL 김", "골격 위에 더하는 정보 없음, 골격 → AHI의 매개 아님"),
        ("Nperp-Pog", "성별 차이 보고", "턱끝–대후두공 앞뒤 거리로 대체, 하악 위치 성분")], columns=["항목", "Roh 2025", "이 연구"])
    hb = {k: hyp_box(v) for k, v in HYPS.items()}

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CBCT 골격과 OSA 중증도</title><style>{CSS}.hyp{{border:1px solid #d8dee6;border-radius:6px;padding:6px 12px;margin:8px 0 14px}}.hyp summary{{cursor:pointer;font-weight:600}}.hyp td:first-child{{white-space:nowrap}}
.flow{{display:flex;flex-direction:column;align-items:center;gap:4px;margin:12px 0}}.flow div{{border:1px solid #b8c2cc;border-radius:6px;padding:6px 14px;text-align:center;min-width:280px}}.flow .ex{{font-size:.85rem;color:var(--muted);border-style:dashed}}.flow .row{{display:flex;gap:16px;border:none;padding:0}}
.abs dt{{font-weight:700;margin-top:6px}}.abs dd{{margin:2px 0 0 0}}table.apx td{{font-size:.82rem;vertical-align:top}}</style></head><body><main>
<h1>CBCT 자동 계측 골격 인자와 폐쇄성 수면무호흡 중증도의 관련성</h1>
<p class="sub">2026-11-17 · 초안 · 최종 분석 대상 80명 (발굴 59명, 새 환자 검증 21명)</p>
<nav><a href="#abs">요약</a><a href="#s1">1. 서론</a><a href="#s2">2. 대상과 방법</a><a href="#s3">3. 결과</a><a href="#s4">4. 고찰</a><a href="#s5">5. 결론</a><a href="#apx">부록</a></nav>

<h2 id="abs">요약</h2><dl class="abs box">
<dt>목적</dt><dd>Roh 등(2025)이 수작업 CBCT 계측으로 보고한 골격 인자와 OSA 중증도의 관련이 CBCT 자동 계측으로 재현되는지 확인하고, 기존 인자에 더해 OSA 중증도를 설명하는 3D 골격 인자가 있는지 찾는다.</dd>
<dt>대상과 방법</dt><dd>성인 OSA 환자 80명(Class III·19세 미만 제외)의 CBCT에서 기준점과 골격·기도를 자동으로 측정했다. 결과 변수는 log 변환한 AHI와 앙와위 AHI이고, 나이·성별·BMI를 보정한 다중회귀를 썼다. 예측력은 5겹 교차검증 50회 반복으로 평가했고, 가설은 결과를 보기 전에 기록했다. 발굴 59명에서 정한 모델을 나머지 21명에 그대로 적용해 검증했다.</dd>
<dt>결과</dt><dd>자동 계측 ANB는 엑셀 ANB의 결과를 재현했고 예측력은 더 컸다. FH 수평면을 쓰는 지표(MP-FH, Nperp-A, Nperp-Pog)는 재현되지 않았고, Nperp-Pog는 턱끝–대후두공 앞뒤 거리로 대체했다. 확정 모델(나이·성별·BMI + ANB + 턱끝–대후두공 거리)에서 ANB는 앙와위 AHI와 가장 강하게 관련됐고, 새 환자에서도 앙와위 AHI 예측력이 0.05에서 0.25로 올랐다. 판독 파일로 계산한 자세별 AHI에서 ANB는 앙와위 AHI와 관련되고 비앙와위 AHI와는 약했다. 기도 크기, 3D 형태, 연조직, 고전 비율은 확정 인자에 정보를 더하지 않았다. 하악 후퇴의 형태적 실체는 크기와 무관하게 짧은 하악체였다. 추가 후보 중 구조 크기 축은 새 환자에서 기각됐고, 수직 배치 지표는 같은 방향이었으나 기준을 넘지 못했다.</dd>
<dt>결론</dt><dd>OSA 중증도와 관련된 골격 정보는 상악–하악 앞뒤 부조화(ANB) 하나로 모이며, 이는 누운 자세의 호흡 사건과 가장 밀접하다. CBCT 자동 계측으로 이 관계를 재현할 수 있다.</dd></dl>

<h2 id="s1">1. 서론</h2>
<p>두개안면 골격은 상기도의 크기와 모양을 정해 OSA의 해부학적 위험 인자가 된다. Roh 등(2025)은 성인 OSA 환자 102명의 CBCT를 시상·수직·횡방향으로 분류해, Class II 고각형 골격(CSP) 환자가 더 젊고 덜 비만함에도 OSA가 더 심하며, 다중회귀에서 ANB와 TDI가 AHI·앙와위 AHI 등 여러 PSG 변수의 유의 예측 인자임을 보고했다.</p>
<p>그러나 이 결과는 숙련자의 수작업 계측에 기반하고, 그룹 비교와 같은 표본 안의 유의성으로 판단해 새 환자에서의 예측력은 확인되지 않았다. 또 3D 영상이 담고 있는 형태 정보 대부분은 몇 개의 각도·거리로 줄여 쓰였다.</p>
<p>이 연구의 목적은 두 가지다. <b>첫째</b>, 같은 기관 자료에서 CBCT 자동 계측으로 엑셀(수작업) 값의 결과가 재현되는지 확인한다. <b>둘째</b>, 재현된 인자에 더해 OSA 중증도를 설명하는 3D 골격 인자(위치, 크기, 비율, 형태)가 있는지 찾는다. 이를 위해 다음 질문을 차례로 검정했다.</p>
<ol><li>자동 계측값이 엑셀 값을 대체할 수 있는가 (3-2)</li><li>어떤 골격 인자가 AHI·앙와위 AHI를 설명하는가 (3-3)</li><li>골격과 AHI 사이에서 기도는 어떤 역할을 하는가 (3-4)</li>
<li>하악 후퇴의 형태적 실체는 무엇이며, 자세와 어떻게 관련되는가 (3-5)</li><li>확정 인자에 더할 3D 인자가 있는가 (3-6)</li><li>결과가 새 환자에서 재현되는가 (3-7)</li></ol>

<h2 id="s2">2. 대상과 방법</h2>
<h3>2-1. 대상</h3>
<p>Roh 등(2025)과 같은 기관의 성인 OSA 환자 자료(엑셀, 87명)를 대상으로 했다. 포함·제외 기준은 Roh 논문을 따라 Class III와 19세 미만을 제외했다.</p>
<div class="flow"><div>엑셀 등록 환자 87명</div><div class="ex">CBCT 제외 2명: 본인 영상 없음 1명, 촬영 범위 부족(10 cm) 1명</div>
<div>CBCT 자동 측정 85명</div><div class="ex">제외 5명: Class III 3명, 검사 당시 19세 미만 2명</div>
<div><b>최종 분석 대상 80명</b></div>
<div class="row"><div>발굴 59명<br><span class="note">PSG 값 확인을 먼저 마친 환자. 모든 가설 검정</span></div><div>새 환자 검증 21명<br><span class="note">발굴에서 정한 모델을 그대로 적용</span></div></div></div>
<p>발굴군은 분석 시작 시점에 PSG 값 확인이 끝난 환자이고, 검증군은 이후 확인한 환자다. 검증군의 PSG 값은 공식 리포트 기준으로 복원했으며, 엑셀에 이전 검사 값이 잘못 기록된 3명은 해당 검사의 리포트 값으로 바로잡았다. 민감도 분석으로 Class III 3명을 포함한 83명(검증 24명)도 분석했다.</p>

<h3>2-2. PSG 변수</h3>
<p>주 결과 변수는 <b>AHI</b>(수면 1시간당 무호흡·저호흡 횟수)와 <b>앙와위 AHI</b>(똑바로 누워 잔 시간만의 AHI)다. 분포가 오른쪽으로 길어 log(값 + 1)로 변환했다. 자세별 분석(3-5)에서는 판독 파일(Embla 사건·수면단계 파일, Nox 판독 데이터베이스)의 자세 정보로 앙와위·비앙와위 AHI와 앙와위 수면시간을 직접 계산했다. 계산값은 같은 검사의 리포트와 총 수면시간 상관 1.00, AHI 0.99, 앙와위 AHI 0.92로 일치했다.</p>

<h3>2-3. CBCT 자동 계측</h3>
<p>기준점은 자동 검출 모델로, 하악·상악·두개저와 기도는 자동 분할로 얻었다. 측정 방법과 정확도는 별도의 CBCT 종합 보고서에 정리했다. 기준점이 아닌 3D 위치는 귓구멍 점을 쓰지 않는 정중 좌표계(원점 PNS, ANS–PNS 방향이 앞)에서 쟀다.</p>
<div class="tw">{tab(var)}</div>

<h3>2-4. 통계 분석</h3>
<p><b>기준 모델과 회귀:</b> Roh 논문과 같이 나이·성별·BMI를 보정한 다중 선형회귀를 썼다(기준 모델). 골격 인자는 기준 모델 위에 더해 계수와 p를 보고, 새 인자는 확정 인자 위에 더했다.</p>
<p><b>예측력 (반복 교차검증):</b> 같은 표본에 맞춘 R²는 변수를 더할수록 커지므로, 학습에 쓰지 않은 환자의 예측력을 따로 평가했다. 발굴 59명을 무작위로 5묶음으로 나눠 4묶음으로 회귀식을 만들고 남은 1묶음을 예측하는 일을 5번 하면 모든 환자가 한 번씩 예측된다. 예측값과 실제값으로 R²를 구하고, 묶음을 새로 나눠 50번 반복한 평균을 <b>교차검증 R²</b>로 썼다.</p>
<img src="{OR.rel(f_cv)}" alt="반복 교차검증 모식도"><p class="note">5겹 교차검증 한 번의 구성. 주황 칸이 그 회차에 예측하는 환자 묶음이다.</p>
<p>두 모델의 비교는 같은 환자, 같은 묶음 나눔에서 계산한 교차검증 R²의 차이(<b>R² 증가</b>)와, 50번 중 더한 모델이 나았던 비율(<b>반복 개선 비율</b>)로 했다. 함께 중첩 모델 F 검정 p를 보고했다.</p>
<p><b>판정 기준:</b> 결과를 보기 전에 가설·변수·기준을 기록했다(사전 등록). 사전 가설은 Bonferroni, 탐색은 FDR로 다중검정을 보정했다. 확정 인자 위에서 p &lt; 0.10, R² 증가 &gt; 0, 반복 개선 70% 이상, 85명에서도 같은 방향이면 <b>후보</b>, 새 환자 검증까지 통과하면 <b>확정</b>으로 했다.</p>
<p><b>새 환자 검증:</b> 발굴 명단과 후보 변수의 계산 방식을 먼저 고정한 뒤, 발굴 59명으로 정한 회귀 계수를 검증 환자에 그대로 적용했다. 검증 환자의 실제값으로 R²를 계산하고(<b>새 환자 R²</b>), R² 증가 &gt; 0, 발굴과 같은 부호, 단측 p &lt; 0.05를 통과 기준으로 했다.</p>
<p><b>형태 분석:</b> 하악 표면 2048개 대응점 좌표를 골격 인자에 회귀해(나이·성별·BMI·크기 보정) 인자가 클 때 하악 표면이 움직이는 방향을 지도로 나타냈고, 순열검정 999회로 유의성을 봤다.</p>

<h2 id="s3">3. 결과</h2>
<h3>3-1. 대상자 특성</h3>
<div class="tw">{tab(ct)}</div>
<p class="note">평균 ± SD, AHI는 중앙값 [사분위]. p는 Mann–Whitney 검정, 성별·중증 비율은 Fisher 정확 검정.</p>

<h3>3-2. 자동 계측의 엑셀 값 재현</h3>{hb['r2']}
<img src="{OR.rel(f_anb)}" alt="ANB 엑셀 대 자동"><p class="note">나이·성별·BMI 위에 ANB를 더했을 때의 교차검증 R² 증가. 같은 환자에서 엑셀 값과 자동 값을 비교했다.</p>
<div class="tw">{tab(s1)}</div>
<p>재현되지 않은 Nperp-A·Nperp-Pog·MP-FH는 모두 귓구멍·눈밑점으로 정하는 FH 수평면을 쓴다. 세 지표의 엑셀–자동 차이는 서로 같이 움직였고(상관 +0.58, −0.52), FH를 쓰지 않는 ANB만 독립적이었다. 공통 원인은 FH 수평면 방향의 차이다. 같은 이유로 자동 값의 수직 분류는 엑셀과 75%만 일치했다(κ 0.61).</p>
<p class="sum">요점: ANB는 자동 계측으로 대체되고, Nperp-Pog는 턱끝–대후두공 앞뒤 거리로 바꾼다. FH 수평면을 쓰는 지표는 재현되지 않는다.</p>

<h3>3-3. 확정 인자와 모델</h3>{hb['r3']}
<p><b>표. 확정 모델의 회귀 계수 (발굴 59명)</b></p><div class="tw">{tab(ctD)}</div>
<p class="note">결과 변수는 log(값 + 1). B는 비표준화 계수와 95% 신뢰구간, β는 표준화 계수. {escape(ftD)}</p>
<p><b>표. 같은 모델, 최종 80명</b></p><div class="tw">{tab(ctA)}</div>
<p class="note">{escape(ftA)}</p>
<img src="{OR.rel(f_steps)}" alt="모델 단계별 교차검증 R²"><p class="note">인자를 하나씩 더할 때의 교차검증 R² (발굴군, 모든 값이 있는 환자).</p>
<p>ANB는 나이·성별·BMI와 독립적으로 AHI와 앙와위 AHI를 설명했고, 앙와위 AHI에서 더 강했다. ANB 1° 증가는 (앙와위 AHI + 1)의 약 {100*(np.exp(anb_b)-1):.0f}% 증가에 해당한다. 턱끝–대후두공 거리는 ANB와 같은 하악 후퇴 정보를 담아, 더하는 예측력은 작았다.</p>
<p class="sum">요점: 확정 모델은 나이·성별·BMI + ANB + 턱끝–대후두공 앞뒤 거리이며, 핵심은 ANB다.</p>

<h3>3-4. 기도의 역할</h3>{hb['r4']}
<p>자동 분할한 기도 부피(PAV)는 골격 인자와 관련됐지만, 그 관련은 기도 길이가 아니라 단면적, 곧 머리 크기에서 왔다. 크기 → 기도 단면적은 강했으나(표준화 0.63) 단면적 → AHI는 0이어서, 기도는 골격과 AHI를 잇는 통로가 아니었다. 최소 단면적·기도 길이를 ANB 위에 더하면 교차검증 R²가 오히려 줄었다(−0.04). 기도가 가장 좁은 위치는 골격이 아니라 연구개 길이로 정해졌다.</p>
<p class="sum">요점: 깨어 있는 상태에서 찍은 정적 기도 크기는 골격 위에 정보를 더하지 않았다.</p>

<h3>3-5. 하악 후퇴의 실체와 자세</h3>{hb['r5']}
<img src="{OR.rel(f_d1)}" alt="하악 형상 지도"><p class="note">ANB가 1 SD 클 때 하악 표면이 움직이는 방향(화살표)과 크기(색). 나이·성별·BMI·크기 보정. 턱끝이 뒤로 약 2.8 mm 움직이고 과두는 거의 그대로다.</p>
<p>턱끝–대후두공 거리의 신호는 두개저 위치가 아니라 상악 기준 하악 위치에서 왔고, ANB는 SNA나 SNB 한쪽이 아니라 두 턱의 앞뒤 차이 자체로 작동했다. 형태 지도에서 하악 후퇴는 크기와 무관하게 짧은 하악체에서 비롯됐다.</p>
<img src="{OR.rel(f_pos)}" alt="ANB와 자세별 AHI"><p class="note">판독 파일로 계산한 자세별 AHI (앙와위·비앙와위 수면이 각각 30분 이상인 최종군 환자). 두 축 모두 나이·성별·BMI의 영향을 뺀 값이라, 선의 기울기가 보정 회귀의 ANB 효과다. β와 p는 보정 회귀 결과.</p>
<p>ANB는 앙와위 AHI를 강하게 설명했고(앙와위 수면시간 보정 후에도 유지) 비앙와위 AHI는 약하게 설명했다. 다만 두 값의 비(자세 의존성) 자체는 유의하지 않았다.</p>
<p class="sum">요점: 하악 후퇴는 짧은 하악체에서 오며, 누운 자세의 호흡 사건과 가장 밀접하다.</p>

<h3>3-6. 확정 인자에 더할 후보</h3>{hb['r6']}
<img src="{OR.rel(f_cand)}" alt="후보별 R² 증가"><p class="note">확정 인자 위에 하나씩 더했을 때의 교차검증 R² 증가. 동그라미 발굴군, 세모 측정 85명.</p>
<ul><li><b>수직 배치 지표:</b> 하악이 상악보다 아래로 놓이고 아래 얼굴이 긴 배치. 세 방법에서 반복돼 하나로 묶었다. 확정 인자 위 AHI R² 증가 +0.069 (p=0.021, 반복 100% 개선), ANB·크기와 거의 무관하다. 탐색 분석에서 비앙와위 AHI와 관련됐다(β +0.42, p=0.002).</li>
<li><b>구조 크기 축:</b> 하악·상악·두개저 12부위 크기의 공통 축. 같은 성별 안에서 작을수록 AHI가 높았으나 새 환자에서 반대 방향이라 기각했다.</li>
<li><b>영향이 없었던 항목:</b> 하악 형태 주성분, 골격 요인분석, 혀·저작근·부비동 부피, 혀·설골 위치, 고전 비율(Jarabak, McNamara), 두개저 각, 구조별 잔여 형상은 모두 하악 후퇴 축이나 크기 축을 다시 잰 것이었다.</li></ul>
<img src="{OR.rel(f_ax)}" alt="공동 축 1"><p class="note">계측 없이 하악·상악·두개저 표면 대응점으로 만든 첫째 공동 축(분산 22%)의 ±2 SD. 하악이 상악에 비해 뒤·위로 놓이는 배치로, ANB(0.61)·턱끝–대후두공 거리(−0.89)와 같은 축이다.</p>
<p class="sum">요점: 3D 형태 정보는 대부분 하악 후퇴로 수렴했고, 가장 일관된 추가 후보는 수직 배치다.</p>

<h3>3-7. 새 환자 검증</h3>{hb['r7']}
<div class="tw">{tab(val)}</div>
<p>확정 모델은 새 환자에서 앙와위 AHI 예측력을 기준 모델의 0.05에서 0.25로 올렸고, 검증 환자만으로 다시 적합해도 ANB가 유의했다(단측 p=0.049). 전체 AHI에서는 개선이 작았다. 구조 크기 축은 두 결과 모두 반대 방향이었고, 수직 배치 지표는 같은 방향으로 예측력을 올렸지만(+0.08~0.09) 기준(p &lt; 0.05)을 넘지 못했다.</p>
<p><b>Class III 포함 (83명, 검증 24명):</b> Class III를 넣으면 ANB 범위가 넓어져 확정 모델이 새 환자의 전체 AHI도 예측했다(R² 0.02 → 0.22). 후보에 대한 결론은 같았다.</p><div class="tw">{tab(val3)}</div>
<p class="sum">요점: ANB → 앙와위 AHI는 새 환자에서 재현됐다.</p>

<h2 id="s4">4. 고찰</h2>
<h3>Roh 등(2025)과의 비교</h3><div class="tw">{tab(roh)}</div>
<p>ANB의 결과는 자동 계측으로도 재현됐고, 예측력은 오히려 컸다. 자동 계측이 두 번 측정의 평균처럼 측정 잡음을 줄였을 가능성이 있다. 반면 TDI는 같은 기관 자료임에도 엑셀 값에서조차 AHI 관련이 재현되지 않았다. Roh 논문의 TDI 효과는 CSP 그룹 비교에서 두드러졌는데, 연속 변수 회귀와 교차검증에서는 표본에 따라 쉽게 사라지는 크기였을 수 있다. FH 수평면을 쓰는 지표가 재현되지 않은 것은 자동 계측의 한계이자, 수평면 정의에 민감한 지표의 일반적 약점이다.</p>
<h3>기전 해석</h3>
<p>ANB가 앙와위 AHI와 가장 밀접하고 비앙와위 AHI와는 약한 것은, 누운 자세에서 중력으로 하악과 혀가 뒤로 처질 때 짧은 하악체가 혀를 받쳐 줄 공간을 줄인다는 해석과 맞는다. 정적 기도 크기가 골격 위에 정보를 더하지 않은 것은, 깨어서 앉은 자세로 찍은 CBCT가 수면 중 동적 허탈을 담지 못한다는 해석과 일치한다. 다만 기도 계측 오차, 촬영 순간의 호흡, CBCT–PSG 간격(중앙값 59일)도 같은 결과를 낼 수 있어, "이 자료에서 골격 위에 추가 정보가 없었다"로 한정한다.</p>
<h3>임상적 의미</h3>
<p>CBCT에서 자동으로 얻은 ANB 하나로 앙와위 AHI의 상당 부분을 설명할 수 있어, 교정 진료에서 OSA 위험 선별에 활용할 여지가 있다. 하악 후퇴 환자에서 자세 치료나 하악 전진 장치의 효과를 기대할 근거와도 이어진다.</p>
<h3>한계</h3><ol>
<li>단일 기관 후향 연구이며, 발굴 59명·검증 21명으로 표본이 작다. 수직 배치 지표를 확정하려면 더 많은 환자가 필요하다.</li>
<li>85명 모두 자동 측정 규칙을 만들 때 쓴 자료라, 측정 단계에서는 독립 검증군이 아니다.</li>
<li>검증군의 PSG 값은 공식 리포트 기준으로 복원한 값이다.</li>
<li>CBCT는 깨어서 앉은 자세로 촬영해 수면 중 상기도를 직접 반영하지 않는다.</li>
<li>자세별 AHI는 앙와위·비앙와위 수면이 각각 30분 이상인 40~44명에서만 판정할 수 있었다.</li>
<li>키 자료가 없어 구조 크기 축이 키를 대신 잰 것인지 구분할 수 없다.</li></ol>

<h2 id="s5">5. 결론</h2>
<p>CBCT 자동 계측으로 Roh 등(2025)의 핵심 결과인 ANB와 OSA 중증도의 관련을 재현했다. OSA 중증도와 관련된 골격 정보는 상악–하악 앞뒤 부조화 하나로 모였으며, 그 형태적 실체는 짧은 하악체이고, 누운 자세의 호흡 사건과 가장 밀접했다. 이 관계는 새 환자에서도 재현됐다. 기도 크기와 그 밖의 3D 형태 정보는 이를 넘어서는 정보를 더하지 않았고, 수직 배치 지표는 추가 검증이 필요한 후보로 남았다.</p>

<h2 id="apx">부록. 전체 실험의 가설·설계·결과</h2>
<p class="note">본문의 "검정한 가설" 표 번호 아래 회색 글자가 이 표의 실험 이름이다.</p>
{appendix()}
<p class="note">측정 방법과 정확도: {OR.link(OR.DEST/"CBCT_종합보고서_2026-11-15.html", "CBCT 종합 보고서")}</p>
</main></body></html>"""
    check(html, "OSA 관련성 보고서 (교수님용)"); OUT.write_text(apply(html), encoding="utf-8"); print("saved", OUT)


if __name__ == "__main__":
    main()
