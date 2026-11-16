"""OSA 관련성 종합 보고서 (검수용, 로컬 HTML). 기존 결과 파일만 읽고, 모델 단계별 교차검증 R²만 새로 계산한다.

표현 규칙은 cbct_final_report.py와 같다(report_terms.check로 확인, report_editor.apply로 편집 기능).
출력: Result_py/report_85/OSA_관련성_종합보고서_2026-11-16.html (+ figs_osa/)
"""
from html import escape
import shutil
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from report_terms import check
from report_editor import apply
from cbct_final_report import CSS, style, tab, R, C_DEV, C_VAL, INK, MUTED, GRID
import cbct_final_report as CF
import osa_experiments as X
import osa_validate_holdout as VH

DEST = R/"report_85"; FIG = DEST/"figs_osa"; OUT = DEST/"OSA_관련성_종합보고서_2026-11-16.html"
A = R/"analysis"; F2 = A/"osa_experiments_v2_final"; SM = A/"shape_map"; VAL = A/"validation"
CF.DEST = DEST                                                          # rel()·link()가 이 보고서 위치 기준으로 경로를 만들게
rel, link = CF.rel, CF.link
NAME = {"m_ANB": "ANB", "Pog_Ba_ant": "턱끝–대후두공 앞뒤 거리", "VERT_Z": "수직 배치 지표", "SIZE_PC1": "구조 크기 축"}
CAND = {"SIZE_PC1": "구조 크기 축", "oropharynx_min_area": "구인두 최소 단면적", "soft_palate_thickness": "연구개 두께", "ANBxMale": "ANB × 성별",
        "ANBxTongue": "ANB × 혀 위치", "Co_Gn": "하악 길이", "S_Ba": "뒤 두개저 길이", "MAND_rPC4": "하악 수직 형상 성분", "LAFH_ratio": "아래 앞얼굴 비율"}
HYP = {"H1": "AHI", "H2": "앙와위 AHI"}


def fig_anb():
    rows = []
    for lab, f in (("현재 값 (3점 수평면)", A/"osa_experiments_v2/S1_excel_vs_auto_per_metric.csv"), ("최종표 (4점 수평면)", F2/"S1_excel_vs_auto_per_metric.csv"),
                   ("최종표 · 확실한 값만", A/"osa_experiments_v2_final_AB/S1_excel_vs_auto_per_metric.csv")):
        t = pd.read_csv(f, encoding="utf-8-sig"); t = t[(t.metric == "ANB") & (t["sample"] == "main59")]
        for _, r in t.iterrows(): rows.append({"판": lab, "결과": HYP[r.hyp], "엑셀": r.dR2_excel, "자동": r.dR2_auto, "n": int(r.n)})
    t = pd.DataFrame(rows); fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, (h, g) in zip(ax, t.groupby("결과", sort=False)):
        x = np.arange(len(g)); style(a)
        a.bar(x-.18, g.엑셀, .34, color=C_VAL, label="엑셀 ANB"); a.bar(x+.18, g.자동, .34, color=C_DEV, label="자동 ANB")
        a.set_xticks(x, [f"{p}\n(n={n})" for p, n in zip(g.판, g.n)], fontsize=8); a.set_title(h, fontsize=10, color=INK); a.axhline(0, color=MUTED, lw=.8)
    ax[0].set_ylabel("나이, 성별, BMI 대비 교차검증 R² 증가", fontsize=8, color=MUTED); ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); f = FIG/"anb_excel_auto.png"; fig.savefig(f, dpi=90); plt.close(fig); return f, t


def model_steps():
    d, fr = VH.load_frozen(); s = d[d.index.isin(fr["discovery_ids"])]
    steps = [("나이, 성별, BMI", []), ("+ ANB", ["m_ANB"]), ("+ 턱끝–대후두공 거리", ["m_ANB", "Pog_Ba_ant"]),
             ("+ 수직 배치 지표", ["m_ANB", "Pog_Ba_ant", "VERT_Z"]), ("+ 구조 크기 축", ["m_ANB", "Pog_Ba_ant", "VERT_Z", "SIZE_PC1"])]
    rows = []
    for h, y in X.H.items():
        u = s.dropna(subset=[y]+X.B0+steps[-1][1])
        for lab, add in steps:
            r2 = X.compare(u, y, X.B0, add)["R2cv"] if add else X.compare(u, y, X.B0, ["m_ANB"])["R2cv_ref"]
            rows.append({"결과": HYP[h], "모델": lab, "교차검증 R²": r2, "n": len(u)})
    t = pd.DataFrame(rows); fig, ax = plt.subplots(figsize=(10, 4)); style(ax); w = .38
    for i, (h, g) in enumerate(t.groupby("결과", sort=False)):
        x = np.arange(len(g)); ax.bar(x+(i-.5)*w, g["교차검증 R²"], w, color=[C_DEV, C_VAL][i], label=f"{h} (n={g.n.iloc[0]})")
        for xx, v in zip(x, g["교차검증 R²"]): ax.text(xx+(i-.5)*w, v+.005, f"{v:.2f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(np.arange(len(steps)), [a for a, _ in steps], fontsize=8); ax.set_ylabel("교차검증 R² (새 환자 예측력)", fontsize=8, color=MUTED); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); f = FIG/"model_steps.png"; fig.savefig(f, dpi=90); plt.close(fig); return f, t


def fig_candidates():
    c = pd.read_csv(F2/"C_candidates/C_results.csv", encoding="utf-8-sig"); c = c[c.part == "단독(기존 인자 위)"].copy(); c["name"] = c.cand.map(CAND)
    g = pd.read_csv(SM/"G_vertical_index.csv", encoding="utf-8-sig"); g = g[g.part == "확정 위"].assign(name="수직 배치 지표 (묶음)")
    t = pd.concat([c[["name", "sample", "hyp", "dR2", "F_p", "share_better"]], g[["name", "sample", "hyp", "dR2", "F_p", "share_better"]]])
    order = ["수직 배치 지표 (묶음)"]+list(CAND.values()); fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for a, (h, hg) in zip(ax, t.groupby("hyp")):
        style(a); y = np.arange(len(order))
        for k, (sn, col, mk) in enumerate((("main59", C_DEV, "o"), ("sens85", C_VAL, "^"))):
            v = hg[hg["sample"] == sn].set_index("name").reindex(order)
            a.scatter(v.dR2, y+(k-.5)*.25, c=col, marker=mk, s=34, label="주 분석 (PSG 확인 완료)" if k == 0 else "측정 85명 전체")
        a.axvline(0, color=MUTED, lw=.8); a.set_yticks(y, order, fontsize=8); a.invert_yaxis(); a.set_title(HYP[h], fontsize=10, color=INK)
        a.set_xlabel("ANB, 턱끝–대후두공 거리 위에 더했을 때 교차검증 R² 증가", fontsize=8, color=MUTED)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .94)); f = FIG/"candidates.png"; fig.savefig(f, dpi=90); plt.close(fig); return f


def fig_corr():
    c = pd.read_csv(F2/"C_candidates/C1_corr.csv", encoding="utf-8-sig", index_col=0)
    nm = {**CAND, "tongue_c_ant": "혀 무게중심 앞뒤", "m_ANB": "ANB", "Pog_Ba_ant": "턱끝–대후두공 거리"}; c = c.rename(index=nm, columns=nm)
    fig, ax = plt.subplots(figsize=(8.5, 7)); im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(c)), c.index, fontsize=8)
    for i in range(len(c)):
        for j in range(len(c)): ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=INK if abs(c.iat[i, j]) < .6 else "white")
    fig.colorbar(im, ax=ax, shrink=.7, label="Spearman 상관"); fig.tight_layout(); f = FIG/"candidate_corr.png"; fig.savefig(f, dpi=90); plt.close(fig); return f


def copy_fig(src, name):
    dst = FIG/name; shutil.copy(src, dst); return dst


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; FIG.mkdir(parents=True, exist_ok=True)
    f_anb, anb = fig_anb(); f_steps, steps = model_steps(); f_cand = fig_candidates(); f_corr = fig_corr()
    f_d1 = copy_fig(SM/"D1_mandible_size/m_ANB_form.png", "mandible_anb.png"); f_ax = copy_fig(SM/"E_joint_axes/F1_PC1.png", "joint_axis1.png")
    f_ax4 = copy_fig(SM/"E_joint_axes/F1_PC4.png", "joint_axis4.png") if (SM/"E_joint_axes/F1_PC4.png").exists() else None
    dry = pd.read_csv(VAL/"validation_own_2026-11-16.csv", encoding="utf-8-sig")
    MN = {"M0 확정": "확정 모델", "M1 = M0 + SIZE_PC1": "확정 + 구조 크기 축", "M1 + 구인두 최소 단면적": "확정 + 크기 + 구인두 최소 단면적",
          "M1 + 아래 앞얼굴 비율": "확정 + 크기 + 아래 앞얼굴 비율", "M1 + ANB × 혀 위치": "확정 + 크기 + ANB × 혀 위치",
          "M0 + VERT_Z": "확정 + 수직 배치 지표", "M1 + VERT_Z": "확정 + 크기 + 수직 배치 지표"}
    dt = dry.assign(결과=dry.hyp.map(HYP), 모델=dry.model.map(MN))[["결과", "모델", "n_val", "R2_val", "dR2_val", "p_one_sided", "same_sign"]]
    dt.columns = ["결과", "모델", "검증 환자 수", "새 환자 R²", "R² 증가", "단측 p", "발굴 때와 같은 방향"]; dt = dt.round(3)
    c3 = pd.read_csv(VAL/"validation_own_c3_2026-11-16.csv", encoding="utf-8-sig")
    c3 = c3[c3.model.isin(["M0 확정", "M1 = M0 + SIZE_PC1", "M0 + VERT_Z"])].assign(결과=lambda x: x.hyp.map(HYP), 모델=lambda x: x.model.map(MN))
    c3t = c3[["결과", "모델", "n_val", "R2_val", "dR2_val", "p_one_sided"]].round(3); c3t.columns = ["결과", "모델", "검증 환자 수", "새 환자 R²", "R² 증가", "단측 p"]
    stp = steps.pivot(index="모델", columns="결과", values="교차검증 R²").reindex([m for m in steps.모델.unique()]).round(3).reset_index()

    kpi = [("85 / 87", "CBCT 측정 완료"), ("59", "PSG 자료 확인 완료 (주 분석)"), ("26", "PSG 자료 확인 중 (민감도·검증 대기)"),
           ("2", "확정 골격 인자 (ANB, 턱끝–대후두공 거리)"), ("1", "유지 후보 (수직 배치). 구조 크기 축은 검증에서 기각")]
    def H(rows):   # 절 앞 가설 상자: (실험 ID, 가설, 설계, 판정)
        li = "".join(f'<tr><td>{i}</td><td>{h}</td><td>{d}</td><td><b>{v}</b></td></tr>' for i, h, d, v in rows)
        return f'<details class="hyp" open><summary>이 절의 가설과 설계</summary><div class="tw"><table><thead><tr><th>실험</th><th>가설</th><th>설계</th><th>판정</th></tr></thead><tbody>{li}</tbody></table></div></details>'
    HY = {
     "s2": H([("S1 지표별", "11개 지표 각각 엑셀 값의 AHI 결과가 자동 측정값으로 재현된다", "지표마다 B0 + 엑셀 대 B0 + 자동, 같은 환자, 엑셀 0값 제외", "ANB만 재현"),
              ("S1b", "턱끝–대후두공 앞뒤 거리(Pog_Ba)가 Nperp-Pog의 정보를 대체한다", "엑셀·자동 Nperp-Pog와 Pog_Ba 단독·조합, 크기·입 벌림 보정", "채택"),
              ("S1c", "MP-FH 신호를 FH 없는 수직 지표(SN-MP, PP-MP)로 살릴 수 있다", "단독과 ANB 위 비교", "기각"),
              ("S1d", "엑셀–자동 불일치의 원인이 특정 기준점에 있다", "환산 전 자동값으로 엑셀 − 자동 차이, 지표 간 차이 상관", "기각: 공통 원인은 FH 수평면 방향"),
              ("S1e", "자동값으로 Roh 분류(시상·수직·CSP)를 재현할 수 있다", "분류 일치율·κ", "부분(75%)"),
              ("S1f", "85명에서 엑셀이 더 강한 것은 측정 방법 차이다", "엑셀–자동 차이가 큰 환자 제외", "채택: 7명이 좌우")]),
     "s3": H([("S1 기준선", "기존 Roh 지표 중 실제로 AHI를 설명하는 것이 있다", "B0 위 ANB / ANB·MP-FH / Roh 골격 3개 / + PMCA·PAL", "ANB 채택"),
              ("S3 (사전 등록)", "3D 골격 피처(Pog_Ba, Overjet, PLS, 혀 MMV)가 ANB를 넘어 AHI를 설명한다", "B0 위와 ANB 위, AHI·앙와위 AHI", "Pog_Ba만 작게"),
              ("S5 (사전 등록)", "머리·턱 구조가 작을수록 ANB와 무관하게 AHI가 높다", "부위 크기 12개 첫 주성분(SIZE_PC1), 최종 ANB 위", "후보 → V1에서 기각")]),
     "s4": H([("S2", "골격 피처가 기도(PMCA·PAL·PAV)를 설명한다", "85명 + Eb 직접 검출 45명, B0 위·ANB 위, FDR", "PAV만 크기로 설명"),
              ("S2b", "PAV의 골격 관련은 기둥 길이(기하) 효과다", "PAV를 높이와 평균 단면적으로 분해", "기각: 단면적(머리 크기)"),
              ("S2c", "크기 축의 AHI 효과는 기도 단면적을 거친다", "매개 분석, 간접효과 부트스트랩 2000회", "기각"),
              ("A2", "크기가 아닌 기도 특징(구간별 최소·평균 단면, 최소 위치)이 기존 인자에 더한다", "13개, 기존 인자 위 / + 크기 위", "기각(구인두 최소는 정의 의존)"),
              ("D8 (사전 등록)", "하악 후퇴가 클수록 기도 최소 위치가 혀 뒤(낮은 쪽)다", "최소 위치 ~ ANB·Pog_Ba·VERT_Z·연구개 길이·설골 높이", "기각: 연구개 길이가 결정")]),
     "s5": H([("B1·B2 (사전 등록)", "Pog_Ba 신호는 하악 위치와 두개저 깊이 중 한쪽, ANB는 SNA·SNB 중 한쪽이 주도한다", "Pog_Ba = Pog–PNS + PNS–Ba 항등식 분해, SNA + SNB 대 ANB", "하악 위치 / 차이 자체"),
              ("D1·D1s", "하악 후퇴는 하악체 길이·하악지 회전·하악각 중 특정 형태에서 오며, 크기 때문이 아니다", "하악 2048 대응점 회귀, 순열 999회, 크기 보정판", "채택: 짧은 하악체"),
              ("P1 (사전 등록)", "ANB가 클수록 자세 의존성(앙와위 ÷ 비앙와위 AHI)이 크다", "판독 파일 자세 정보로 계산(리포트와 상관 0.92~1.00), B0 위", "기각"),
              ("P2 (사전 등록)", "ANB는 앙와위 AHI와 관련되고 비앙와위 AHI와는 약하다", "같은 환자에서 두 결과에 따로 회귀", "채택")]),
     "s6": H([("A1·A3·A5 (사전 등록)", "크기 축은 나이·성별 교란이 아니고, 잔차 피처·교호작용이 기존 인자에 더한다", "기존 인자 위, 교란 항·잔차 6개·교호작용", "크기 유지, ANB × 혀 약함"),
              ("A4", "후보 103개 중 기존 인자에 더할 안정적 묶음이 있다", "기존 인자 고정 elastic net, 안정성 선택 200회", "기각"),
              ("A6", "골격이 자세 의존성(앙와위 AHI − 전체 AHI)을 설명한다", "B0 위 ANB·Pog_Ba·크기", "대리 지표에서 채택 → P1에서 약화"),
              ("C1~C4 (사전 등록)", "후보 9개 중 서로 흡수되지 않고 독립적으로 남는 것이 있다", "후보 간 상관, 함께 넣은 모델, 교차검증", "크기 축만 독립"),
              ("G (사전 등록)", "세 방법에서 반복된 수직 배치를 하나로 묶으면 독립적으로 AHI를 설명한다", "수직 배치 지표 = 세 성분 z 평균, 계산 방식 고정", "채택(후보)"),
              ("D3·D4·D5·D7 (사전 등록)", "설골 위치, 구강 공간 대비 연조직, 골성 비인두, 상대성장 모양이 더한다", "기존 인자(+ VERT_Z) 위", "비인두 → 앙와위 초과만")]),
     "s7": H([("E (사전 등록)", "AMASSS 대응점을 이어 붙인 공동 축이 구조 간 배치 차이를 담고 기존 인자에 더한다", "하악·상악·두개저 대응점 연결, 형태 / GPA / 크기 포함 PCA", "배치는 담음, 추가는 F1_PC4만"),
              ("F (사전 등록)", "전형성 거리, 성장 축(Y축)이 기존 인자에 더한다", "마할라노비스 D², Procrustes 거리, Y축", "기각")]),
     "s8": H([("E0", "엑셀 기본 모델에 3D 형태 블록을 더하면 설명력이 오른다", "하악 SSM·부위 형태·기도·치열궁 블록 추가", "기각"),
              ("S4E", "골격 연동 구조(EFA 요인)가 확정 인자 위에 더한다", "요인 전부 정렬 검정 + 묶음, fold 안 재적합", "기각"),
              ("S6 (사전 등록)", "혀뿌리가 뒤에 있을수록, 하악 대비 혀가 클수록 AHI가 높다", "확정 인자 위, p &lt; 0.0125", "기각"),
              ("B3 (사전 등록)", "크기와 무관한 길이·각도 비율이 기존 인자에 더한다", "고전 비율 15개, FDR", "기각"),
              ("D-M5·D6 (사전 등록)", "기존 인자로 설명되지 않는 잔여 형상, FH 없는 수직 패턴이 더한다", "잔차 형상 PCA, 수직 계측 9개", "수직 성분만 → G로 통합")]),
     "s9": H([("V1 (사전 등록)", "확정 + 크기 축, 확정 + 수직 배치 지표가 새 환자에서 확정 모델보다 낫다", "발굴 59명 고정 계수를 새 환자 21명에 적용, ΔR² &gt; 0·같은 부호·단측 p &lt; 0.05", "크기 기각, 수직 배치 미달(같은 방향)"),
              ("V2", "같음 (Class III 포함)", "최종 83명, 검증 24명", "결론 같음")]),
    }
    kh = '<div class="kpis">'+"".join(f'<div class="kpi"><b>{a}</b><span>{escape(b)}</span></div>' for a, b in kpi)+"</div>"
    s1 = pd.DataFrame([("ANB", "있음", "재현됨 (엑셀 이상)", "자동 측정값으로 대체"),
                       ("Nperp-Pog", "있음", "재현 안 됨", "턱끝–대후두공 앞뒤 거리로 대체 (예측력은 엑셀보다 큼)"),
                       ("MP-FH", "약함", "재현 안 됨", "모델에서 제외 (엑셀에서도 ANB와 겹치는 정보)"),
                       ("Nperp-A, TDI, PHI, NCWR, PAL, PAV, NAV, PMCA", "없음", "없음", "예측력으로는 판정 불가, 일치도로만 평가")],
                      columns=["지표", "엑셀 값의 AHI 관련", "자동 값의 AHI 관련", "처리"])
    mech = pd.DataFrame([("기도 크기는 무엇이 정하나", "머리 크기(단면적)가 정합니다. 기도 길이가 아니라 단면이 넓어지는 효과입니다"),
                         ("기도 크기가 골격 → AHI의 통로인가", "아닙니다. 크기 → 단면적은 강하지만(표준화 0.63), 단면적 → AHI는 0입니다"),
                         ("하악 후퇴는 어디에서 오나", "크기와 무관하게 하악체가 짧은 것이 주이고, 하악지 기울기가 다음입니다. 하악각 회전과는 무관합니다"),
                         ("ANB는 무엇을 재나", "상악이 앞인지 하악이 뒤인지가 아니라 두 턱의 앞뒤 차이 자체입니다"),
                         ("턱끝–대후두공 거리는 무엇을 재나", "두개저(대후두공) 위치가 아니라 상악 기준 하악 위치 성분이 일합니다"),
                         ("누운 자세와의 관계", "ANB는 앙와위 의존성(앙와위 AHI − 전체 AHI)을 설명합니다. 비인두 공간이 좁을수록 앙와위 의존이 큽니다(후보)")],
                        columns=["질문", "답"])
    rej = pd.DataFrame([("TDI (횡방향 부조화, Roh 논문의 핵심)", "엑셀 값에서도 재현 안 됨"), ("최소 단면적·기도 길이 (ANB 위 추가)", "더하지 않음, 오히려 가림"),
                        ("하악 형태 SSM, 요인분석, PLS, 지렛대", "하악 후퇴 축을 다시 잰 것"), ("혀·저작근·부비동 부피", "크기 축(+성별)을 다시 잰 것"),
                        ("혀 위치, 설골 높이", "하악 후퇴·성별로 설명됨"), ("고전 비율 (Jarabak, McNamara, 하악지/하악체), 두개저 각", "신호 없음"),
                        ("구조별 잔여 모양, 전형성 거리, 성장 축", "기존 인자에 흡수")], columns=["항목", "결과"])
    files = [(A/"osa_experiments_v2_final", "최종표 판 실험 결과 폴더"), (F2/"C_candidates", "후보 상호 독립성 결과"),
             (SM/"D1_mandible_size/index.html", "하악 형상 지도 (크기 보정)"), (SM/"E_joint_axes/index.html", "AMASSS 공동 축"),
             (SM/"D6_vertical/index.html", "수직 패턴"), (SM/"D3D4/index.html", "설골·혀"), (VAL, "검증 준비 (고정 명단·정의, 시험 실행)"),
             (R.parent/"docs/실험_가설_설계_결과.md", "실험 요약표 (가설·설계·결과)"), (R.parent/"docs/OSA_관련성_실험기록.md", "시간순 실험 기록")]
    fl = "".join(f"<li>{link(p, t)}</li>" for p, t in files)

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OSA 관련성 종합 보고서</title><style>{CSS}.hyp{{border:1px solid var(--grid,#ddd);border-radius:6px;padding:6px 12px;margin:8px 0 14px}}.hyp summary{{cursor:pointer;font-weight:600}}.hyp td:first-child{{white-space:nowrap}}</style></head><body><main>
<h1>CBCT 골격 정보와 OSA 중증도 관련성 종합 보고서</h1>
<p class="sub">2026-11-16 · 최종 분석 대상 80명(Class III 포함 시 83명) · 발굴 59명 · 새 환자 검증 21명 · 탐색적 분석 · 내부 검수용</p>
<p class="note">측정 방법과 정확도는 별도 보고서에 있습니다: {link(DEST/"CBCT_종합보고서_2026-11-15.html", "CBCT 종합 보고서")} · {link(DEST/"CBCT_측정기준_보고서_2026-11-15.html", "CBCT 측정 기준 보고서")}</p>
<nav><a href="#s1">1. 대상과 방법</a><a href="#s2">2. 엑셀 값 대체</a><a href="#s3">3. 확정 인자</a><a href="#s4">4. 기도의 역할</a><a href="#s5">5. 하악 후퇴의 실체</a>
<a href="#s6">6. 추가 후보</a><a href="#s7">7. 얼굴 구조 수치화</a><a href="#s8">8. 영향 없던 항목</a><a href="#s9">9. 새 환자 검증</a><a href="#s10">10. 한계와 남은 일</a><a href="#s11">자료 위치</a></nav>
<div class="box"><b>요약</b><ul>
<li><b>CBCT 자동 측정 ANB는 엑셀 ANB를 대신할 수 있고, 예측력은 오히려 큽니다.</b> 엑셀의 Nperp-Pog 자리는 턱끝–대후두공 앞뒤 거리로 바꿨습니다.</li>
<li><b>OSA 중증도와 관련된 골격 정보는 사실상 "상악–하악 앞뒤 부조화" 하나입니다.</b> 그 실체는 크기와 무관하게 짧은 하악체이고, 누운 자세에서 더 강하게 나타납니다.</li>
<li><b>CBCT에서 잰 기도 크기는 골격과 AHI 사이의 통로가 아니었습니다.</b> 깨어서 앉은 자세의 정적 영상이 수면 중 동적 허탈을 담지 못한다는 해석과 맞습니다.</li>
<li><b>3D 형태·연조직·고전 비율은 대부분 위 정보를 다시 잰 것이었습니다.</b> 추가 후보였던 <b>수직 배치 지표</b>(하악이 상악보다 아래로 놓임)는 새 환자에서도 같은 방향이었지만 17명으로는 기준을 넘지 못했고, <b>구조 크기 축</b>은 새 환자에서 반대 방향이라 기각했습니다.</li>
<li><b>새 환자 검증에서 가장 견고하게 재현된 것은 "ANB → 앙와위 AHI"입니다.</b></li></ul></div>

<h2 id="s1">1. 대상과 방법</h2>{kh}
<p><b>결과 변수:</b> AHI와 앙와위 AHI(둘 다 log 변환)를 주 결과로 봤습니다. <b>기준 모델</b>은 나이·성별·BMI입니다.
<b>성능:</b> 5겹 교차검증을 50회 반복해, 학습에 쓰지 않은 환자의 결과를 얼마나 맞히는지(교차검증 R²) 봤습니다. 모델끼리는 값이 모두 있는 같은 환자 안에서 1:1로 비교했습니다.</p>
<p><b>판정:</b> 결과를 보기 전에 가설과 기준을 기록하고 검정했습니다. 다중검정은 사전 가설에 Bonferroni, 탐색에 FDR을 썼습니다.
"후보"는 기존 인자 위에서 p &lt; 0.10, 교차검증 R² 증가, 반복의 70% 이상에서 개선, 85명 전체에서도 같은 방향일 때입니다. "확정"은 사전 기준과 추가 환자 검증을 모두 통과한 것입니다.</p>
<div class="warn"><b>주의:</b> 85명 모두 측정 규칙을 만들 때 쓴 자료라 독립 검증군이 아닙니다. 주 분석은 59명으로 작아, R² 0.05 안팎의 차이는 환자 몇 명에 좌우될 수 있습니다.</div>
<div class="sum">한 줄 요약: 59명에서 찾고 85명에서 방향을 확인했으며, PSG 확인 중인 26명은 검증용으로 남겨 두었습니다.</div>

<h2 id="s2">2. CBCT 자동 측정으로 엑셀 값을 대체할 수 있나</h2>{HY['s2']}
<p>엑셀 값으로 나온 AHI 관련 결과가 자동 측정값으로도 나오는지 지표마다 봤습니다.</p>
<img src="{rel(f_anb)}" alt="ANB 엑셀 대 자동"><p class="note">나이·성별·BMI 위에 ANB 하나를 더했을 때 교차검증 R²가 늘어난 양입니다. 같은 환자에서 비교했습니다.</p>
<div class="tw">{tab(s1)}</div>
<p><b>재현되지 않은 이유:</b> Nperp-A·Nperp-Pog·MP-FH는 모두 귓구멍·눈밑점으로 잡는 수평면을 씁니다. 세 지표의 엑셀–자동 차이가 서로 같이 움직이고(상관 +0.58, −0.52), 수평면을 쓰지 않는 ANB만 따로 놉니다. 공통 원인은 수평면 방향 차이이며, 4점 수평면으로 바꿔도 줄지 않았습니다.</p>
<p><b>골격 분류:</b> 같은 이유로 자동 값으로 나눈 수직 분류(고각형 등)는 엑셀과 75%만 일치합니다(κ 0.61). 고각형·CSP 같은 그룹 비교는 엑셀 값으로 분류해야 합니다.</p>
<div class="sum">한 줄 요약: ANB는 자동으로 대체되고, Nperp-Pog는 턱끝–대후두공 거리로 바꾸며, 수평면을 쓰는 나머지 지표는 수평면 오차 때문에 재현되지 않습니다.</div>

<h2 id="s3">3. 확정 인자와 모델</h2>{HY['s3']}
<img src="{rel(f_steps)}" alt="모델 단계별 R²"><p class="note">PSG 확인 완료 환자 중 모든 값이 있는 환자에서, 인자를 하나씩 더할 때의 교차검증 R²입니다.</p>
<div class="tw">{tab(stp)}</div>
<p><b>확정 모델:</b> 나이·성별·BMI + ANB + 턱끝–대후두공 앞뒤 거리. ANB 하나로 AHI 예측력이 크게 오르고(앙와위 AHI에서 더 큼), 턱끝–대후두공 거리는 ANB와 같은 하악 후퇴 정보라 더하는 양은 작습니다.
PSG 확인 중인 환자 22명에 발굴 때의 계수를 그대로 적용해도 예측력이 유지됩니다(R² 0.17 / 0.40).</p>
<div class="sum">한 줄 요약: 확정 모델은 ANB 중심이며, 새 환자에서도 예측력이 유지됩니다.</div>

<h2 id="s4">4. 기도의 역할</h2>{HY['s4']}<div class="tw">{tab(mech.iloc[:2])}</div>
<p>최소 단면적·기도 길이를 ANB 위에 더하면 오히려 예측이 나빠졌습니다(−0.04). 기도가 가장 좁은 위치도 골격이 아니라 연구개 길이로 정해졌습니다.</p>
<div class="warn"><b>해석:</b> "정적 기도 크기가 수면 중 동적 허탈을 반영하지 못한다는 해석과 일치한다"까지가 데이터가 보여 준 범위입니다. 기도 계측 오차(검증군 ICC 0.62~0.78), 촬영 순간의 호흡·삼킴, CBCT–PSG 간격(중앙값 59일), 표본 크기도 같은 결과를 낼 수 있습니다. Roh 논문은 최소 단면적이 유의하다고 보고했으므로 "이 자료에서 골격 위에 추가 정보가 없었다"로 한정합니다.</div>
<div class="sum">한 줄 요약: 기도 크기는 머리 크기를 따라가며, 골격과 AHI를 잇는 통로는 아니었습니다.</div>

<h2 id="s5">5. 하악 후퇴의 실체</h2>{HY['s5']}<div class="tw">{tab(mech.iloc[2:])}</div>
<img src="{rel(f_d1)}" alt="하악 형상 지도"><p class="note">ANB가 1 SD 클 때 하악 표면이 움직이는 방향(빨간 화살표)과 크기(색)입니다. 나이·성별·BMI·크기를 보정했습니다. 턱끝에서 가장 크게(뒤로 약 2.8 mm) 움직이고 과두 쪽은 거의 그대로입니다.</p>
<p><b>자세별 AHI(P 단계):</b> 판독 파일의 자세 정보로 앙와위·비앙와위 AHI를 직접 계산했습니다(리포트와 상관 0.92~1.00). ANB는 앙와위 AHI를 강하게 설명했지만(표준화 계수 +0.38, p=0.001, 앙와위 수면시간 보정 후 유지) 비앙와위 AHI는 약했고(+0.22, p=0.23), 둘의 비(자세 의존성)는 유의하지 않았습니다. 탐색에서는 수직 배치 지표가 비앙와위 AHI와 관련됐습니다(+0.42, p=0.002).</p>
<div class="sum">한 줄 요약: 하악 후퇴는 짧은 하악체에서 오고, 그 정보가 ANB·턱끝–대후두공 거리에 이미 담겨 있습니다.</div>

<h2 id="s6">6. 기존 인자에 더할 후보</h2>{HY['s6']}
<img src="{rel(f_cand)}" alt="후보별 R² 증가"><p class="note">확정 인자(ANB + 턱끝–대후두공 거리) 위에 하나씩 더했을 때입니다. 동그라미는 주 분석, 세모는 측정 85명 전체입니다.</p>
<img src="{rel(f_corr)}" alt="후보 상관"><p class="note">후보끼리의 상관입니다. 하악 길이·뒤 두개저 길이는 구조 크기 축과 크게 겹칩니다.</p>
<ul><li><b>수직 배치 지표 (주 후보):</b> 하악이 상악보다 아래로 놓이고 아래 얼굴이 긴 배치입니다. 아래 앞얼굴 비율, 하악 잔여 형상, AMASSS 공동 축 세 가지 방법에서 반복돼 하나로 묶었습니다. 확정 인자 위 AHI +0.069(p=0.021, 반복 100% 개선)이고, ANB·크기와 거의 무관합니다. 앙와위 AHI에서는 약합니다.</li>
<li><b>구조 크기 축 (검증에서 기각):</b> 하악·상악·두개저 12부위 크기의 공통 축입니다. 같은 성별 안에서 작을수록 AHI가 높고, 다른 후보에 흡수되지 않는 유일한 후보였습니다. 다만 AHI와의 단순 상관은 0이고 성별과 겹치며(0.68), 85명 전체에서 약해집니다.</li>
<li><b>앙와위 의존성 후보:</b> 비인두 공간이 좁을수록 앙와위 AHI가 전체 AHI보다 더 높습니다(두 표본 모두 p &lt; 0.05). 뼈(비인두 깊이)는 비인두 크기를 정하지만 앙와위 의존성과는 무관해, 신호는 공간·연조직 쪽입니다.</li>
<li><b>약한 후보:</b> ANB × 혀 위치·설골 후방 위치(85명에서만). 구인두 최소 단면적은 측정 방식에 따라 값이 크게 달라지고(우리 기도 최소 단면적과 상관 0.21, 가장 좁은 위치가 평균 26 mm 다름) 새 환자에서도 재현되지 않아 후보에서 내렸습니다.</li></ul>
<div class="warn"><b>주의:</b> 후보를 9개 모두 넣으면 같은 표본 안에서는 유의하지만 새 환자 예측은 나빠졌습니다(교차검증 R² −0.08). 후보는 하나씩 검증해야 합니다.</div>
<div class="sum">한 줄 요약: 기존 인자 위에서 가장 일관된 추가 후보는 수직 배치이고, 크기 축이 다음입니다. 둘 다 확정 전입니다.</div>

<h2 id="s7">7. 얼굴 구조 수치화: AMASSS 공동 축</h2>{HY['s7']}
<p>하악·상악·두개저의 표면 대응점을 한 벡터로 이어 주성분 축을 만들었습니다. 사람이 정한 계측 없이 구조 사이의 배치 차이를 그대로 담습니다.</p>
<img src="{rel(f_ax)}" alt="공동 축 1"><p class="note">첫째 축(분산 22%)의 ±2 SD입니다. 하악이 상악에 비해 뒤·위로 놓이는 배치이며, ANB(0.61)·턱끝–대후두공 거리(−0.89)와 같은 축입니다.</p>
{f'<img src="{rel(f_ax4)}" alt="공동 축 4"><p class="note">넷째 축(7%): 하악이 상악보다 아래로 놓이는 수직 배치입니다. 아래 앞얼굴 비율(0.48)과 관련되며, 수직 배치 지표의 한 성분입니다.</p>' if f_ax4 else ""}
<div class="sum">한 줄 요약: 계측 없이 만든 축이 하악 후퇴(1위)와 크기(2위)를 다시 찾아내, 계측 기반 결론을 형태학적으로 뒷받침합니다.</div>

<h2 id="s8">8. 영향이 없었던 항목</h2>{HY['s8']}<div class="tw">{tab(rej)}</div>
<div class="sum">한 줄 요약: 대부분의 3D 정보는 하악 후퇴 축이나 크기 축을 다른 방식으로 잰 것이었습니다.</div>

<h2 id="s9">9. 새 환자 검증</h2>{HY['s9']}
<p>발굴에 쓴 59명 명단과 크기 축·수직 배치 지표의 계산 방식을 고정한 뒤, 나머지 환자에 발굴 때의 계수를 그대로 적용했습니다. 이 환자들의 PSG 값은 연구실 회신을 기다리지 않고 공식 리포트 기준으로 우리가 복원했습니다. 엑셀에 이전 검사 값이 잘못 들어간 3명은 리포트 값으로 바로잡았고, Class III와 검사 당시 19세 미만은 발굴과 같이 뺐습니다.</p>
<div class="tw">{tab(dt)}</div>
<p>ANB만 더한 모델은 새 환자에서 앙와위 AHI 예측력을 0.05에서 0.25로 올렸고, 새 환자만으로 다시 적합해도 ANB가 유의했습니다(단측 p=0.049). 전체 AHI에서는 이런 개선이 거의 없었습니다.</p>
<h3>Class III를 포함한 판 (최종 대상 83명, 검증 24명)</h3><p>Roh 논문처럼 Class III를 뺀 위 결과가 주 결과이고, 아래는 Class III 3명을 검증군에 넣은 민감도 결과입니다. 발굴 표본에는 Class III가 없어 Class I·II에서 정한 관계를 그대로 적용한 것입니다.</p><div class="tw">{tab(c3t)}</div>
<p>Class III를 넣으면 ANB 범위가 넓어져 확정 모델이 새 환자의 전체 AHI도 예측합니다(R² 0.02 → 0.22). 후보에 대한 결론은 같습니다.</p>
<div class="warn"><b>해석:</b> 구조 크기 축은 새 환자에서 두 결과 모두 반대 방향이라 기각했습니다. 수직 배치 지표는 같은 방향으로 예측력을 올렸지만(+0.08~0.09) 17명으로는 미리 정한 기준(p &lt; 0.05)을 넘지 못했습니다. 검증 값은 연구실 확인 전 복원값입니다.</div>
<div class="sum">한 줄 요약: 새 환자에서도 ANB의 앙와위 효과는 재현됐고, 크기 축은 기각, 수직 배치는 유지 후보입니다.</div>

<h2 id="s10">10. 한계와 남은 일</h2><ol class="check">
<li><b>표본:</b> 주 분석 59명, 새 환자 검증 17명. 수직 배치 지표를 확정하려면 더 많은 환자가 필요합니다. 연구실 회신으로 복원 규칙이 바뀌면 검증을 다시 실행합니다.</li>
<li><b>앙와위 의존성:</b> 판독 파일로 직접 계산했으나 판정 가능 환자가 40~44명이라, ANB의 자세 의존성 자체와 수직 배치 → 비앙와위 AHI(탐색)는 다음 자료에서 확인해야 합니다.</li>
<li><b>키 자료 없음:</b> 크기 축이 키를 대신 잰 것인지 구분할 수 없습니다.</li>
<li><b>엑셀 검토:</b> 엑셀–자동 ANB 차이가 3°를 넘는 7명(5명은 엑셀 검토 대상)이 85명 결과를 끌어내립니다.</li>
<li><b>영상 확인:</b> 대후두공 앞점 10명 판정이 남았습니다. 위치가 크게 벗어났던 환자 1명은 형상이 아닌 FH 머리 좌표계 오류로 확인했고 결론에 영향이 없습니다.</li></ol>

<h2 id="s11">자료 위치</h2><ul>{fl}</ul>
<p class="note">생성: python/osa_relevance_report.py (그림·표는 결과 파일에서 매번 다시 만듭니다)</p>
</main></body></html>"""
    check(html, "OSA 관련성 보고서"); OUT.write_text(apply(html), encoding="utf-8"); print("saved", OUT)


if __name__ == "__main__":
    main()
