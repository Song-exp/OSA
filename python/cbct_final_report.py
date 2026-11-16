"""85명 CBCT 측정·형태 정보·PCA·회귀 종합 보고서 (검수용, 로컬 HTML). 기존 결과 파일만 읽는다.

표현 규칙: 우리 작업 안에서만 통하는 이름은 쓰지 않는다(report_terms.check로 확인). 데이터 분석 방법론 용어는 그대로 둔다.
출력: Result_py/report_85/CBCT_종합보고서_2026-11-15.html (+ figs/)
"""
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report_terms import check
from report_editor import apply

ROOT = Path(__file__).resolve().parents[1]; R = ROOT/"Result_py"; V2 = R/"CBCT/patient_pipeline/v2"; EXT = V2/"restore_ext/v1"
DEST = R/"report_85"; FIG = DEST/"figs"; OUT = DEST/"CBCT_종합보고서_2026-11-15.html"
C_DEV, C_VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"    # 참조 팔레트 1·2번, 텍스트 토큰
ORDER = ["ANB", "Nperp-A", "Nperp-Pog", "MP-FH", "TDI", "PHI", "NCWR", "PAL", "PAV", "NAV", "PMCA"]
# 데이터 속 내부 이름 → 보고서 표현
STATUS = {"분석 명단": "PSG 확인 완료", "계측만(PSG 회신 대기)": "PSG 자료 확인 중", "등록번호 일치 기준 포함": "등록번호로 본인 확인"}
CONFIG = {"기준선": "자동 측정값", "개선 설정": "보정한 자동 측정값", "개선+엑셀 환산": "보정 + 엑셀 방식 환산"}
SETS = {"dev": "개발군", "holdout": "검증군"}
SAMPLE = {"주 분석: 분석 명단": "주 분석 (PSG 확인 완료 환자)", "민감도: 복원 85명": "민감도 분석 (측정한 85명 전체)"}
MODEL = {"M0": "기본 모델 (나이·성별·BMI + 엑셀 Roh 5지표)", "M0r": "Roh 5지표를 자동 측정값으로 바꾼 모델", "M1": "+ 하악 전체 형태 (SSM 주성분 3개)",
         "M2": "+ 하악 부위별 형태 (부위 3곳 주성분)", "M3": "+ 기도·연조직 4개", "M4": "+ 치열궁 3개"}
ALIGN = {"M0": "자동 측정값 그대로", "M1": "직선 환산 (Huber)", "M2": "여러 계산값 조합 (Ridge)"}
LMK = {"N": "코뿌리점", "S": "터키안장", "Ba": "대후두공 앞점", "RPo": "오른쪽 귓구멍 점", "LPo": "왼쪽 귓구멍 점", "ROr": "오른쪽 눈밑점", "LOr": "왼쪽 눈밑점",
       "ANS": "앞코가시", "PNS": "뒤코가시", "A": "상악 A점", "B": "하악 B점", "Pog": "턱끝 앞점", "Me": "턱끝 아래점", "Gn": "턱끝점", "RGo": "오른쪽 하악각점", "LGo": "왼쪽 하악각점"}
AX = {"left": "좌우", "ant": "앞뒤", "sup": "위아래"}


def rel(p): return quote(os.path.relpath(p, DEST).replace("\\", "/"))
def link(p, label): return f'<a href="{rel(p)}" target="_blank">{escape(label)}</a>' if Path(p).exists() else f"{escape(label)} (파일 없음)"
def tab(df, idx=False): return df.to_html(index=idx, border=0, na_rep="—", classes="t", escape=True)


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)


def fig_scatter(ca, dev):
    fig, axes = plt.subplots(3, 4, figsize=(15, 11)); unit = ca.drop_duplicates("metric").set_index("metric").unit
    for a, m in zip(axes.flat, ORDER):
        g = ca[(ca.metric == m) & ca.paired]; style(a)
        for name, c, mk, sel in (("개발군 54명 (규칙을 만들 때 사용)", C_DEV, "o", g.Patient.isin(dev)), ("검증군 31명 (규칙을 정한 뒤 측정)", C_VAL, "^", ~g.Patient.isin(dev))):
            h = g[sel]; a.scatter(h.excel, h.restored, s=26, c=c, marker=mk, edgecolors="white", linewidths=.8, label=f"{name}, n={len(h)}")
        lo, hi = np.nanmin([g.excel.min(), g.restored.min()]), np.nanmax([g.excel.max(), g.restored.max()])
        a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--"); mae = (g.restored-g.excel).abs()
        a.set_title(f"{m} ({unit[m]}) · MAE 개발군 {mae[g.Patient.isin(dev)].mean():.2f} / 검증군 {mae[~g.Patient.isin(dev)].mean():.2f}", fontsize=9, color=INK)
        a.set_xlabel("엑셀 (사람이 잰 값)", fontsize=8, color=MUTED); a.set_ylabel("자동 측정값", fontsize=8, color=MUTED)
    axes.flat[0].legend(fontsize=7.5, frameon=False); axes.flat[-1].axis("off")
    fig.tight_layout(); f = FIG/"restore_scatter.png"; fig.savefig(f, dpi=80); plt.close(fig); return f


def fig_forest(r):
    d = r[(r.y == "AHI") & (r.model != "M0")]; labels = {k: MODEL[k] for k in ("M0r", "M1", "M2", "M3", "M4")}
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); style(ax); y = 0; ticks, names = [], []
    for m in labels:
        for s, c, mk, off in (("주 분석: 분석 명단", C_DEV, "o", -.15), ("민감도: 복원 85명", C_VAL, "^", .15)):
            q = d[(d.model == m) & (d["sample"] == s)].iloc[0]
            ax.plot([q.dR2_lo, q.dR2_hi], [y+off]*2, color=c, lw=2); ax.plot(q.dR2_vs_M0, y+off, mk, color=c, ms=8, mec="white", label=SAMPLE[s] if m == "M0r" else None)
        ticks.append(y); names.append(labels[m]); y += 1
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(ticks, names, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("기본 모델 대비 교차검증 R² 변화 (점: 평균, 선: 95% 범위)", fontsize=9, color=MUTED); ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); f = FIG/"regression_forest.png"; fig.savefig(f, dpi=90); plt.close(fig); return f


def fig_pca(summ, sc, d):
    var = np.array(summ["landmark_pca_var"]); fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    style(ax[0]); ax[0].bar(np.arange(1, len(var)+1), var*100, color=C_DEV, width=.6); ax[0].set_xlabel("주성분", color=MUTED); ax[0].set_ylabel("설명분산 (%)", color=MUTED)
    ax[0].set_title("기준점 좌표 PCA의 설명분산", fontsize=10, color=INK)
    style(ax[1]); j = sc.join(d[["MPFH_r"]]); j = j[j.in_pca_axes]; s = ax[1].scatter(j.PC1, j.PC2, c=j.MPFH_r, cmap="Blues", s=30, edgecolors=MUTED, linewidths=.4)
    fig.colorbar(s, ax=ax[1], label="MP-FH 자동 측정값 (°)"); ax[1].set_xlabel("PC1", color=MUTED); ax[1].set_ylabel("PC2", color=MUTED)
    ax[1].set_title("PC1–PC2 (색 = MP-FH)", fontsize=10, color=INK)
    fig.tight_layout(); f = FIG/"pca.png"; fig.savefig(f, dpi=85); plt.close(fig); return f


def pc_text(corr, top):
    nm = lambda c: c.replace("_r", "").replace("MPFH", "MP-FH")
    coord = lambda c: f"{LMK.get(c.rsplit('_', 1)[0], c.rsplit('_', 1)[0])}({c.rsplit('_', 1)[0]}) {AX.get(c.rsplit('_', 1)[1], '')} 위치"
    out = []
    for pc in ("PC1", "PC2", "PC3"):
        r = corr.loc[pc].drop("AHI"); rel_ = r[r.abs() >= .3].sort_values(key=abs, ascending=False)
        out.append(f"<li><b>{pc}</b>: 가장 크게 움직이는 좌표는 {', '.join(coord(c) for c in top[pc][:4])}입니다. "
                   + (f"관련 지표: {', '.join(f'{nm(k)}({v:+.2f})' for k, v in rel_.items())}." if len(rel_) else "상관 0.3 이상인 지표는 없습니다.") + "</li>")
    a = corr.AHI.abs(); return "<ul>" + "".join(out) + f"</ul><p>AHI와의 단순 상관은 가장 큰 것이 {corr.AHI[a.idxmax()]:+.2f}({a.idxmax()})로 약합니다.</p>"


CSS = """:root{--bg:#fcfcfb;--card:#fff;--fg:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6;--warn:#8a5a00;--warnbg:#fbf0d9;--sum:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#222220;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--warn:#e3b65c;--warnbg:#3a2e14;--sum:#1d2633}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 "Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.25rem;margin:40px 0 10px;padding-top:8px;border-top:2px solid var(--accent)}
h3{font-size:1.02rem;margin:18px 0 6px}.sub,.note{color:var(--muted)}.note{font-size:.88rem}a{color:var(--accent)}
nav{display:flex;flex-wrap:wrap;gap:4px 14px;margin:10px 0}table.t{border-collapse:collapse;width:100%;font-size:.86rem;background:var(--card);margin:6px 0}
table.t th,table.t td{border:1px solid var(--line);padding:4px 8px;text-align:left}table.t th{background:var(--bg)}
.tw{overflow-x:auto}img{max-width:100%;border:1px solid var(--line);background:#fff}.box,.sum{border-radius:6px;padding:10px 14px;margin:12px 0}.box{background:var(--card);border:1px solid var(--line)}
.sum{background:var(--sum)}.warn{background:var(--warnbg);color:var(--warn);border-radius:6px;padding:8px 12px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.kpi b{display:block;font-size:1.6rem}.kpi span{font-size:.84rem;color:var(--muted)}
details{margin:10px 0}summary{cursor:pointer;color:var(--accent)}ol.check li{margin:4px 0}"""


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; FIG.mkdir(parents=True, exist_ok=True)
    ca = pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig"); dev = set(pd.read_csv(V2/"fh_repair/v2/comparison.csv", encoding="utf-8-sig").Patient)
    fin = pd.read_csv(EXT/"experiments/final_comparison.csv", encoding="utf-8-sig")
    cvr = pd.read_csv(EXT/"alignment/cv_results.csv", encoding="utf-8-sig"); ch = json.loads((EXT/"alignment/choices.json").read_text(encoding="utf-8"))
    val = json.loads((R/"shape/validation/summary.json").read_text(encoding="utf-8"))
    reg = pd.read_csv(R/"analysis/regression_cv.csv", encoding="utf-8-sig"); ins = pd.read_csv(R/"analysis/regression_insample.csv", encoding="utf-8-sig")
    ps = json.loads((R/"analysis/pca_summary.json").read_text(encoding="utf-8")); corr = pd.read_csv(R/"analysis/pca_landmark_corr.csv", encoding="utf-8-sig", index_col=0)
    sc = pd.read_csv(R/"analysis/pca_landmark_scores.csv", encoding="utf-8-sig", index_col=0)
    import shape_regression as SR
    d = SR.table(); f1, f2, f3 = fig_scatter(ca, dev), fig_forest(reg), fig_pca(ps, sc, d)

    st = ca.drop_duplicates("Patient").status.map(STATUS).value_counts()
    kpi = [("85 / 87", "CBCT 측정 완료 (P_39·P_78 불가)"), (str(st.get("PSG 확인 완료", 0)), "PSG 자료 확인 완료"), (str(st.get("PSG 자료 확인 중", 0)), "PSG 자료 확인 중 (CBCT는 측정함)"),
           (str(st.get("등록번호로 본인 확인", 0)), "성별·생년월일 일부가 달랐으나 등록번호로 본인 확인"), ("31", "검증군 (규칙을 정한 뒤 새로 측정)")]
    kh = '<div class="kpis">'+"".join(f'<div class="kpi"><b>{a}</b><span>{escape(b)}</span></div>' for a, b in kpi)+"</div>"

    acc = fin.pivot_table(index="metric", columns=["set", "config"], values="MAE").reindex(ORDER).round(2)
    acc.columns = [f"{SETS[s]} · {CONFIG[c]}" for s, c in acc.columns]
    icc = fin[fin.config == "기준선"].pivot_table(index="metric", columns="set", values="ICC").reindex(ORDER).round(2); hv = icc["holdout"]; icc.columns = [f"ICC {SETS[c]}" for c in icc.columns]
    al = cvr.pivot_table(index="metric", columns="model", values="MAE_mean").reindex(ORDER).round(2); al.columns = [ALIGN[c] for c in al.columns]
    al["채택"] = [ALIGN[ch["metrics"][m]["model"]] for m in al.index]; al["학습 환자 수"] = [ch["metrics"][m]["n_train"] for m in al.index]
    al["엑셀 값이 의심돼 뺀 수"] = [ch["metrics"][m]["n_excluded"] for m in al.index]

    inv = pd.DataFrame([
        ("기준점 16개 + 제1대구치 점 (머리 좌표계 좌표)", "85"), ("Roh 지표 11개 (자동 측정값·엑셀 방식 환산값)", "85 (지표별 결측 있음)"),
        ("엑셀 추가 변수 (턱끝·비중격 편위, 비강저 경사, 교합평면 각)", "85"), ("과두 측면각·축상각", "85"),
        ("구조 전체 형태 SSM (하악·두개저·상악·기도)", "85"), ("부위별 형태 12곳 (모양 주성분·크기·위치)", "85"),
        ("기도 단면적 프로파일 (후두개곡 위 5 mm ~ 뒤코가시)", "81 (4명은 후두개곡이 영상 밖)"), ("비강 단면적 프로파일 (뒤코가시 ~ 앞코가시)", "85"),
        ("상대 위치 (설골, 설골–하악평면 거리 등)", "85"), ("연조직 (연구개·경구개·인두 3구간)", "85"), ("치아 32개와 치열궁", "85 (라벨 오류 치아 표시)")], columns=["항목", "환자 수"])
    vt = pd.DataFrame([
        ("인두 영역: TotalSegmentator vs AMASSS 상기도", f"두 프로그램이 같은 곳을 칠한 정도 Dice {val['pharynx_TS_vs_AMASSS']['dice']['50%']:.2f}, 경계 차이 {val['pharynx_TS_vs_AMASSS']['surf_mean_mm']['50%']:.2f} mm", "믿을 만함"),
        ("인두 부피: TotalSegmentator vs 공기 밝기 기준 (같은 범위)", f"ICC {val['pharynx_volume_same_range_vs_threshold']['ICC']:.3f}, TotalSegmentator가 평균 {val['pharynx_volume_same_range_vs_threshold']['bias']:.1f} mL 큼", "믿을 만함 (일정한 차이)"),
        ("기도 단면적 곡선 모양", f"두 곡선 상관 중앙값 {val['airway_profile_curve_r']['50%']:.2f}, 5명은 0.7 미만", "대체로 믿을 만함"),
        ("치아 위치: 치아 분할 vs ALI 치아 점", f"차이 중앙값 {val['molar_seg_vs_ALI_mm']['50%']:.1f} mm, 10 mm 넘는 14명은 라벨 오류로 처리", "주의"),
        ("연구개", f"뒤코가시에서 떨어진 거리 중앙값 {val['softpalate_PNS_gap_mm']['50%']:.1f} mm, 이상치 {val['softpalate_flags']}명", "길이·각도만 권장"),
        ("형태 SSM", "하악·상악·기도는 정상 범위, 두개저는 표면 맞추기 문제로 19명 이상치", "두개저는 빼는 것을 권장")], columns=["비교", "결과", "판단"])
    rt = reg[reg.y == "AHI"].copy(); rt["표본"] = rt["sample"].map(SAMPLE); rt["모델"] = rt.model.map(MODEL)
    rt["ΔR² (95% 범위)"] = [("—" if m == "M0" else f"{a:+.3f} ({lo:+.3f} ~ {hi:+.3f})") for m, a, lo, hi in zip(rt.model, rt.dR2_vs_M0, rt.dR2_lo, rt.dR2_hi)]
    main_t = rt[["표본", "모델", "n", "R2_cv", "ΔR² (95% 범위)"]].rename(columns={"n": "환자 수", "R2_cv": "교차검증 R²"}).round(3)
    full = rt.merge(ins[ins.y == "AHI"][["sample", "model", "adjR2", "F_p_vs_M0"]], on=["sample", "model"])
    full = full[["표본", "모델", "n", "R2_cv", "M0_R2_cv_same_n", "dR2_vs_M0", "dR2_lo", "dR2_hi", "share_better", "adjR2", "F_p_vs_M0"]].round(3)
    full.columns = ["표본", "모델", "환자 수", "교차검증 R²", "같은 환자로 본 기본 모델 R²", "ΔR²", "ΔR² 2.5%", "ΔR² 97.5%", "기본 모델보다 나은 반복 비율", "adj R² (전체 환자로 적합)", "F 검정 p (기본 모델 대비)"]
    ssm = pd.DataFrame({k: v for k, v in ps["ssm_procrustes_cumvar_PC1_5"].items()}, index=[f"PC1~{i}" for i in range(1, 6)]).T
    ssm.index = ssm.index.map({"MAND": "하악", "CB": "두개저", "MAX": "상악", "UAW": "상기도"}); ssm = ssm.rename_axis("구조").reset_index()
    corr_t = corr.round(2).rename(columns=lambda c: c.replace("_r", "").replace("MPFH", "MP-FH")).reset_index().rename(columns={"index": "주성분"})

    files = [(EXT/"compare_all.csv", "85명 × 11개 지표의 자동 측정값·엑셀 값"), (EXT/"experiments/final_comparison.csv", "개발군·검증군 정확도 전체 지표"),
             (EXT/"experiments/experiment_log.json", "정확도 개선 실험 단계별 선택 기록"), (EXT/"alignment/aligned_metrics.csv", "엑셀 방식 환산값"),
             (EXT/"alignment/cv_results.csv", "엑셀 방식 환산 교차검증 결과"), (EXT/"experiments/export/skeletal_landmarks_head.csv", "기준점 좌표(머리 좌표계)"),
             (R/"shape/validation/summary.json", "수치 검증 요약"), (R/"analysis/regression_cv.csv", "회귀 교차검증 결과(AHI·log AHI)"),
             (R/"analysis/regression_insample.csv", "회귀 전체 환자 적합 결과"), (R/"analysis/pca_landmark_scores.csv", "환자별 주성분 점수")]
    fl = "".join(f"<li>{link(p, t)}</li>" for p, t in files)
    va = ps["landmark_pca_var"]
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CBCT 종합 보고서</title><style>{CSS}</style></head><body><main>
<h1>CBCT 자동 측정·형태 정보·PCA·회귀 종합 보고서</h1>
<p class="sub">2026-11-15 · 등록 87명 중 85명 측정 · 탐색적 자동 분석 · 내부 검수용</p>
<p class="note">프로그램별 측정 기준(기준점·구조 분할·좌표계·지표 정의)은 별도 보고서에 있습니다: {link(DEST/"CBCT_측정기준_보고서_2026-11-15.html", "CBCT 측정 기준 보고서")}</p>
<nav><a href="#s1">1. 대상</a><a href="#s2">2. 정확도</a><a href="#s3">3. 정확도 개선 실험</a><a href="#s4">4. 엑셀 방식 환산</a><a href="#s5">5. 형태 정보</a>
<a href="#s6">6. 수치 검증</a><a href="#s7">7. PCA</a><a href="#s8">8. 회귀</a><a href="#s9">9. 검수 목록</a><a href="#s10">자료 위치</a></nav>
<div class="box"><b>요약</b><ul>
<li><b>87명 중 85명을 자동으로 측정했습니다.</b> P_39는 본인 CBCT가 없고, P_78은 촬영 범위가 좁아(102 mm) 잴 수 없었습니다.</li>
<li><b>실제 정확도는 규칙을 정한 뒤 새로 측정한 31명(검증군)으로 봐야 합니다.</b> 규칙을 만들 때 쓴 54명보다 오차가 큽니다(예: ANB 0.87° → 1.67°).</li>
<li><b>자동 측정값으로 엑셀 값을 대신해도 AHI 예측력은 거의 같습니다.</b></li>
<li><b>형태 정보를 더해도 AHI 예측이 나아지지 않았습니다.</b> 다만 표본이 작아 "효과가 없다"가 아니라 "이 크기로는 확인되지 않았다"로 봐야 합니다.</li></ul></div>
<h2 id="s1">1. 대상</h2>{kh}
<div class="sum">한 줄 요약: CBCT는 85명 모두 측정했고, 그중 PSG 자료까지 확인이 끝난 환자가 회귀 분석의 주 대상입니다.</div>
<h2 id="s2">2. 자동 측정의 정확도 (엑셀 값과 비교)</h2>
<p>엑셀 값은 사람이 직접 잰 값이라 비교 기준으로 씁니다. 환자를 두 무리로 나눠 봅니다. <b>개발군</b>(54명)은 측정 규칙을 만들고 다듬을 때 본 환자라 오차가 작게 나오기 쉽고, <b>검증군</b>(31명)은 규칙을 다 정한 뒤 처음 측정한 환자라 실제 성능에 가깝습니다.
MAE는 평균적으로 몇 mm(또는 °) 틀리는지, ICC는 두 값이 얼마나 일치하는지(1에 가까울수록 일치)를 뜻합니다.</p>
<img src="{rel(f1)}" alt="11개 지표 산점도"><p class="note">점선 위에 있을수록 엑셀과 같습니다.</p>
<div class="tw">{tab(acc.join(icc).reset_index().rename(columns={"metric": "지표"}))}</div>
<div class="sum">한 줄 요약 (검증군 ICC 기준): 일치도가 높은 지표는 {", ".join(f"{m}({v:.2f})" for m, v in hv.sort_values(ascending=False).items() if v >= .7)}, 낮은 지표는 {", ".join(f"{m}({v:.2f})" for m, v in hv.sort_values().items() if v < .6)}입니다. 기도 지표 NAV·PMCA는 엑셀과 늘 같은 방향으로 차이가 나서 4절의 환산이 효과적입니다.</div>
<h2 id="s3">3. 정확도 개선 실험</h2>
<p><b>문제:</b> 귓구멍 점이 틀리면 수평면이 기울어 MP-FH·Nperp가 틀어지고, 턱끝 앞점이 연조직에 찍히면 Nperp-Pog가 틀어집니다.</p>
<p><b>시도와 결과</b> (개발군에서 고르고 검증군에서 한 번만 확인)</p>
<ul><li><b>수평면을 4점으로 잡기</b>(좌우 귓구멍·좌우 눈밑점): MP-FH·Nperp-A에서 채택. 검증군 MP-FH 오차 2.95° → 2.66°.</li>
<li><b>턱끝 앞점을 하악 윤곽에서 다시 잡기:</b> Nperp-Pog에서 채택. 검증군 오차 5.44 → 5.02 mm.</li>
<li><b>영상을 좌우로 뒤집어 한 번 더 측정해 평균 내기:</b> 오히려 나빠져 쓰지 않았습니다. 프로그램이 좌우를 대칭으로 다루지 않아 뒤집은 결과가 평균 3.9 mm 어긋났습니다.</li>
<li><b>같은 지표의 다른 계산 방식 고르기:</b> PHI에서 개발군 차이가 2%에 불과한 방식이 골라졌다가 검증군에서 나빠졌습니다. 그래서 이후에는 "오차가 5% 이상 줄고, 반복 시험의 80% 이상에서 나을 때만" 바꾸기로 했습니다.</li></ul>
<div class="sum">한 줄 요약: 수평면을 4점으로 잡고 턱끝 점을 뼈 윤곽으로 옮긴 두 가지만 효과가 있었습니다.</div>
<h2 id="s4">4. 엑셀 측정 방식에 맞춘 환산</h2>
<p>우리 값과 엑셀 값 사이에 늘 같은 방향의 차이가 있다면, 그 차이를 맞춰 엑셀 측정자의 방식으로 옮겨 읽을 수 있습니다. <b>측정을 고친 것이 아니라 엑셀의 자로 바꿔 읽은 값</b>입니다. 85명 반복 교차검증(5겹 × 20회)으로 학습에 쓰지 않은 환자에서 오차를 쟀고, 엑셀 값이 0이거나 오차가 유난히 큰 사례(42건)는 학습에서 뺐습니다.</p>
<div class="tw">{tab(al.reset_index().rename(columns={"metric": "지표"}))}</div>
<div class="sum">한 줄 요약: PMCA·NAV는 오차가 절반으로, Nperp-A·Nperp-Pog·PHI는 16~30% 줄었습니다. 나머지는 환산 효과가 작아 그대로 둡니다.</div>
<h2 id="s5">5. 추출한 형태 정보</h2>
<p>지표 11개 외에, 뼈·기도·연조직·치아의 모양을 숫자로 바꾼 정보를 85명 모두 뽑았습니다. SSM은 같은 구조의 표면을 환자끼리 점 대 점으로 맞춘 뒤 모양 차이를 주성분으로 요약한 것입니다.</p>
<div class="tw">{tab(inv)}</div>
<h3>구조별 SSM의 누적 설명분산 (위치·크기를 뺀 모양만)</h3><div class="tw">{tab(ssm)}</div>
<p class="note">상악은 뚜렷한 모양 축이 약하고(첫 주성분 9%), 두개저는 표면 맞추기 품질 문제로 첫 주성분이 실제 모양이 아닐 수 있습니다.</p>
<div class="sum">한 줄 요약: 형태 정보는 모두 준비됐고, 하악·기도는 쓸 만하며 상악·두개저는 해석에 주의가 필요합니다.</div>
<h2 id="s6">6. 수치 검증: 서로 다른 프로그램끼리 맞춰 보기</h2>
<p>사람이 모두 보기 전에, 독립적인 두 방법이 같은 답을 내는지 숫자로 먼저 확인했습니다.</p><div class="tw">{tab(vt)}</div>
<div class="sum">한 줄 요약: 공기와 맞닿은 기도·인두 정보는 믿을 만하고, 치아는 일부 라벨 오류를 걸러야 하며, 연구개는 길이·각도만 쓰는 것이 안전합니다.</div>
<h2 id="s7">7. PCA: 기준점 배치의 주된 변화 방향</h2>
<p>기준점 16개의 좌표(머리 좌표계)로 PCA를 해서, 환자마다 기준점 배치가 가장 크게 다른 방향을 찾았습니다. 축은 자동 점검을 통과한 환자로만 만들었습니다. 기준점이 크게 틀린 환자(P_35 등)를 넣으면 그 한 명이 첫 축을 차지하기 때문입니다.
설명분산은 PC1 {va[0]*100:.1f}%, PC2 {va[1]*100:.1f}%, PC3 {va[2]*100:.1f}%입니다.</p>
<img src="{rel(f3)}" alt="PCA">{pc_text(corr, ps["landmark_pca_top_loadings"])}
<div class="tw">{tab(corr_t)}</div><p class="note">주성분 점수와 자동 측정 지표·AHI 사이의 피어슨 상관입니다.</p>
<div class="sum">한 줄 요약: 기준점 배치의 큰 변화는 하악의 수직 패턴(MP-FH)과 턱끝의 앞뒤 위치를 따라가며, 그 자체로는 AHI와 거의 관련이 없습니다.</div>
<h2 id="s8">8. 회귀: 형태 정보가 AHI 예측에 보탬이 되나</h2>
<p><b>질문:</b> 기존 Roh 지표(기본 모델)에 형태 정보를 더하면 AHI를 더 잘 예측하는가?
<b>방법:</b> 반복 5겹 교차검증(50회)으로, 학습에 쓰지 않은 환자의 AHI를 얼마나 맞히는지(교차검증 R²) 봤습니다. 모델마다 필요한 값이 있는 환자로 같은 환자 안에서 기본 모델과 1:1로 비교했습니다(그래서 환자 수가 모델마다 다릅니다). 더할 정보와 개수는 결과를 보기 전에 정했습니다.</p>
<img src="{rel(f2)}" alt="R² 변화"><p class="note">0보다 오른쪽이면 예측이 나아진 것입니다. 선이 0을 지나면 이 표본에서는 차이를 확인할 수 없다는 뜻입니다.</p>
<div class="tw">{tab(main_t)}</div>
<div class="warn"><b>해석:</b> 교차검증 R² 자체가 0.1~0.2로 낮고 범위가 넓습니다. 표본({rt.n.min()}~{rt.n.max()}명)이 작기 때문입니다. 기존 계산상 R²가 0.10 정도 늘어야 이 크기로 검출할 수 있습니다. 하악 부위별 형태는 주 분석에서 전체 환자로 적합한 F 검정이 p=0.035였지만, 교차검증과 민감도 분석에서는 재현되지 않았습니다.</div>
<details><summary>전체 표 보기 (log AHI 결과는 자료 위치의 파일)</summary><div class="tw">{tab(full)}</div></details>
<div class="sum">한 줄 요약: 자동 측정값은 엑셀 값을 대신할 수 있지만, 형태 정보의 추가 예측력은 이 표본 크기로는 확인되지 않았습니다.</div>
<h2 id="s9">9. 검수 목록</h2><p>각 페이지에서 판단을 고르면 파일(JSON)로 저장됩니다.</p>
<ol class="check">
<li><b>엑셀과 오차가 유난히 큰 16명·19건:</b> 자동 기준점이 틀렸는지, 엑셀이 이상한지 판단 — {link(EXT/'alignment/review/review.html', '검토 페이지')}</li>
<li><b>형태 정보 육안 점검 22명:</b> 기도 곡선 불일치 5, 치아 라벨 14, 연구개 4, SSM 모양 3 — {link(R/'shape/validation/visual_review/visual_review.html', '점검 페이지')}</li>
<li><b>연구실 자료 확인 요청서</b> (14건, 40명) — {link(R/'patient_matching/CBCT_PSG_자료확인요청_2026-11-12_v6.docx', '요청서')} · {link(R/'patient_matching/자료확인요청_증거.html', '항목별 증거')}. 5번 P_76 문구 수정 필요</li>
<li><b>결정 필요:</b> P_36·P_59·P_85를 PSG 분석 대상에 넣을지, 두개저 형태 정보를 어떻게 처리할지</li></ol>
<h2 id="s10">자료 위치</h2><ul>{fl}</ul>
<p class="note">생성: python/cbct_final_report.py (그림·표는 결과 파일에서 매번 다시 만듭니다)</p>
</main></body></html>"""
    check(html, "종합 보고서"); OUT.write_text(apply(html), encoding="utf-8"); print("saved", OUT)


if __name__ == "__main__":
    main()
