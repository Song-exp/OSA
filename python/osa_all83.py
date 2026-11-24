"""83명 전체(19세 미만 2명·CBCT 없음 2명 제외, Class III 포함) 단계별 회귀 재실행 (분할 없음, 계수 유의성 판단용).
변수 추출 기준·정의는 기존 그대로. 결과 변수는 리포트 기준 복원 PSG(psg_restored_own.csv).
단계마다 원래 실험 설정(보정 변수)으로 y ~ 보정 + 변수 하나 OLS → B, 표준화 β, p, n. 단계·결과 변수 안 FDR q는 CSV에만.
출력: Result_py/analysis/all83/ (steps.csv, patient_values.csv, 전체_83명_기존단계별_회귀.html)
"""
import re, json, warnings; warnings.filterwarnings("ignore")
from html import escape
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import osa_experiments as X, osa_validate_holdout as VH, osa_residual_shape as RS, osa_joint_axes as JA, osa_shape_map as SM
from osa_addition import A2_FEATS, A3_FEATS
from osa_proportions import LENGTHS, RATIOS
from osa_vertical import VERT
from shape_ssm import gpa
from cbct_final_report import CSS, tab
from report_terms import check

R = X.R_; OUT = R/"analysis/all83"; OUT.mkdir(parents=True, exist_ok=True)
B0, C = X.B0, X.B0+X.CONFIRMED
YH = {"logAHI": "AHI", "logAHI_SUP": "앙와위 AHI"}


# ------------------------------------------------------------ 자료
def data():
    d, _ = VH.load_frozen(); d = X.airway_raw(d)
    o = pd.read_csv(R/"phase1/psg_restored_own.csv", encoding="utf-8-sig", index_col=0)
    d = d.loc[[p for p in d.index if p in o.index and o.at[p, "Include_own_c3"]]].copy()
    d["AHI"], d["AHI_SUP"] = o.loc[d.index, "AHI_r"], o.loc[d.index, "AHI_SUP_r"]
    d["logAHI"], d["logAHI_SUP"] = np.log1p(d.AHI), np.log1p(d.AHI_SUP); d["SUP_excess"] = d.logAHI_SUP-d.logAHI
    d["ClassIII"] = o.loc[d.index, "ClassIII"]
    d["H_col"] = -d.c_Eb_sup; d["a_PAV_h"] = d.a_PAV*1000/d.H_col
    sm_ = R/"analysis/shape_map"
    d = d.join(pd.read_csv(sm_/"E_joint_axes/joint_axes.csv", encoding="utf-8-sig", index_col=0), rsuffix="_j")
    d = d.join(pd.read_csv(sm_/"F_composite/F_values.csv", encoding="utf-8-sig", index_col=0)[["Atyp_D2", "Atyp_Procrustes", "Yaxis", "FacialAxis_proxy"]])
    full, _ = VH.load_frozen()                                                      # 잔여 형상 PC는 기존대로 85명 기준
    for st in ("MAND", "MAX", "CB"):
        sc = RS.residual_pcs(full, st)[0]; d = d.join(sc[[c for c in sc if c not in d]])
    rel = pd.read_csv(R/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
    st = pd.read_csv(R/"shape/fine/soft_tissue.csv", encoding="utf-8-sig").set_index("Patient")
    for c in ("MP_H", "Hy_Me"):
        if c not in d: d[c] = rel[c].reindex(d.index)
    d["Hy_sup"], d["Hy_ant"] = d.c_Hy_sup, d.c_Hy_ant
    d["SoftLoad_MMV"] = (d.tongue_mL+st.soft_palate_volume.reindex(d.index))/d.MMV_mL; d["tongue_MMVhull"] = d.tongue_mL/d.MMV_hull_mL
    g = SM.angles(full); full = full.join(g)                                         # 하악체 길이 잔차(크기·성별), 85명 기준
    u = full.dropna(subset=["body_len", "SIZE_PC1", "Male"]); d["body_len"] = full.body_len
    d["body_len_res"] = sm.OLS(u.body_len, sm.add_constant(u[["SIZE_PC1", "Male"]])).fit().resid.reindex(d.index)
    aw = [f"AW_p{i:02d}" for i in range(20)]; P = full[aw].dropna(); Z = (P-P.mean())/P.std(); _, _, vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    for i in range(3):
        if vt[i, np.argmax(np.abs(vt[i]))] < 0: vt[i] *= -1
        d[f"AWshape_PC{i+1}"] = pd.Series(Z.to_numpy()@vt[i], index=P.index).reindex(d.index)
    ids, Cc, _, _ = JA.combined(("MAND", "MAX", "CB")); G, cs = gpa(Cc); Y = G.reshape(len(G), -1); Y = Y-Y.mean(0)
    sex = full.Male.reindex(ids).to_numpy(float); lcs = np.log(cs); B = np.linalg.lstsq(np.c_[np.ones(len(ids)), lcs, sex], Y, rcond=None)[0]
    d["AlloShape"] = pd.Series(Y@(B[1]/np.linalg.norm(B[1])), index=ids).reindex(d.index); d["logCS_all"] = pd.Series(lcs, index=ids).reindex(d.index)
    t = pd.read_csv(R/"analysis/positional/positional.csv", encoding="utf-8-sig", index_col=0); v = pd.read_csv(R/"psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
    t = t[(v.SameExam.reindex(t.index) == True) & (t.index != "P_75") & t.POS_ratio.notna()]
    d["logSUP_pos"], d["logNON_pos"] = np.log1p(t.AHI_sup_r).reindex(d.index), np.log1p(t.AHI_non_r).reindex(d.index)
    d["POS_ratio"], d["sup_frac"] = t.POS_ratio.reindex(d.index), t.sup_frac.reindex(d.index)
    assert len(d) == 83, len(d)
    return d


# ------------------------------------------------------------ 변수 설명
def catalog():
    """변수 → (뜻, 만든 과정). feature_catalog.csv, 파생변수_정의서.md 표, 아래 MANUAL 순으로 찾는다."""
    cat = {}
    c = pd.read_csv(R/"features/v1/feature_catalog.csv", encoding="utf-8-sig")
    for _, r in c.iterrows(): cat[r.feature] = (str(r.definition), str(r.source))
    md = (R.parent/"docs/파생변수_정의서.md").read_text(encoding="utf-8")
    for line in md.splitlines():
        if not line.startswith("| `"): continue
        cols = [x.strip() for x in line.strip("|").split("|")]
        if len(cols) < 3: continue
        for nm in re.findall(r"`([^`]+)`", cols[0]):
            cat.setdefault(nm, (re.sub(r"`", "", cols[1]), re.sub(r"`", "", cols[2])))
    cat.update(MANUAL); return cat


MANUAL = {
    "m_ANB": ("ANB 각: N–A선과 N–B선 사이 각을 정중 시상면에 투영. 클수록 하악이 상악보다 뒤", "ALI 자동 기준점, 4점 FH 최종 기준(등급 A~D)"),
    "Pog_Ba_ant": ("턱끝(Pog) 앞뒤 위치 − 대후두공 앞점(Ba) 앞뒤 위치(mm). 작을수록 하악 후퇴", "ALI 자동 기준점, 정중 좌표계(PNS 원점)"),
    "a_PMCA": ("인두 최소 단면적(mm²), 기도 중심선에 수직인 단면", "밝기 규칙(−1200~−400 HU) 기도 분할, Eb+5 mm ~ PNS"),
    "a_PAL": ("인두 기도 길이(mm), PNS–Eb", "밝기 규칙 기도 분할, ALI PNS, Eb 검출"),
    "a_PAV": ("인두 기도 부피(mL), 위 S–PNS면 ~ 아래 Eb 수평면", "밝기 규칙 기도 분할"),
    "a_PAV_h": ("인두 평균 단면적(mm²) = 인두 부피 ÷ 기둥 높이", "a_PAV, H_col에서 계산"),
    "H_col": ("기도 기둥 높이(mm): PNS에서 Eb까지 수직 거리", "정중 좌표계 Eb 높이"),
    "VERT_Z": ("수직 배치 지표: 하악 수직 잔여 형상(MAND_rPC4), 아래 앞얼굴 비율, 공동 축 넷째 축의 z 평균. 클수록 하악이 상악보다 아래", "AMASSS 형상 + ALI 기준점, 계산 방식 고정"),
    "SIZE_PC1": ("구조 크기 축: 하악·상악·두개저 12부위 크기의 첫 주성분(분산 69%). 클수록 큰 머리", "AMASSS 부위 크기, 계산 방식 고정"),
    "body_len_res": ("크기·성별을 뺀 하악체 길이(Pog–Go 중점 거리의 회귀 잔차)", "ALI 기준점, 85명 회귀 잔차"),
    "body_len": ("하악체 길이: Pog–좌우 Go 중점 거리(mm)", "ALI 기준점, 정중 좌표계"),
    "AWshape_PC1": ("기도 단면 프로파일(20구간) 모양 첫 주성분", "밝기 규칙 기도 단면, 85명 표준화 PCA"),
    "AWshape_PC2": ("기도 단면 프로파일 모양 둘째 주성분", "같음"), "AWshape_PC3": ("기도 단면 프로파일 모양 셋째 주성분", "같음"),
    "AlloShape": ("상대성장 모양 점수: 크기(log 중심 크기)에 따라 변하는 하악·상악·두개저 공동 모양 방향으로의 투영", "AMASSS 대응점, 공동 GPA, 성별 보정 회귀"),
    "logCS_all": ("하악·상악·두개저 대응점 전체의 log 중심 크기", "AMASSS 대응점"),
    "SoftLoad_MMV": ("(혀 + 연구개 부피) ÷ 하악 내부 공간 부피", "TotalSegmentator 혀·연구개, MMV"),
    "tongue_MMVhull": ("혀 부피 ÷ 하악 볼록 껍질 부피", "TotalSegmentator 혀, AMASSS 하악"),
    "Hy_sup": ("설골 위아래 위치(정중 좌표계, PNS 기준 mm)", "밝기 규칙 설골 검출"), "Hy_ant": ("설골 앞뒤 위치(정중 좌표계, mm)", "밝기 규칙 설골 검출"),
    "Atyp_D2": ("전형성 거리: 공동 축 점수의 마할라노비스 D²(평균 얼굴에서 얼마나 벗어났는가)", "AMASSS 공동 축"),
    "Atyp_Procrustes": ("평균 형상과의 Procrustes 거리", "AMASSS 공동 GPA"), "Yaxis": ("Y축 각(S–Gn과 S–N 사이)", "ALI 기준점"),
    "FacialAxis_proxy": ("얼굴 축 근사: Ba–N선과 PNS–Gn선 사이 각", "ALI 기준점"),
    "PLS_t1": ("골격→기도 PLS 첫째 점수", "ALI·AMASSS 골격 피처, PLS"),
    "m_ANBxMale": ("ANB × 성별 교호작용", "중심화한 곱"), "m_ANBxSIZE_PC1": ("ANB × 구조 크기 축", "곱"), "Pog_Ba_antxMale": ("턱끝–대후두공 거리 × 성별", "곱"),
    "SIZE_PC1xBMI": ("구조 크기 축 × BMI", "곱"), "SIZE_PC1xMale": ("구조 크기 축 × 성별", "곱"), "SIZE_PC1xAge": ("구조 크기 축 × 나이", "곱"),
    "m_ANBxtongue_c_ant": ("ANB × 혀 무게중심 앞뒤 위치", "중심화한 곱(발굴군 평균)"), "Age2": ("나이²", "공변량"),
    "SIZE_sexz": ("성별 안에서 표준화한 구조 크기 축", "SIZE_PC1"),
}

ROH = {"ANB": "ANB 각(°): N–A선과 N–B선 사이 각", "NperpA": "Nperp-A(mm): N을 지나 FH에 수직인 선에서 A점까지 앞뒤 거리", "NperpPog": "Nperp-Pog(mm): 같은 선에서 Pog까지 앞뒤 거리",
       "MPFH": "MP-FH(°): 하악평면과 FH 평면 사이 각", "TDI": "TDI(mm): 위 치조폭(MxAW) − 아래 치조폭(MnAW), 제1대구치 저항중심 높이 관상면", "PHI": "PHI: 구개 높이 지수(구개 천장–교합 높이 ÷ 저항중심 폭)",
       "NCWR": "NCWR: 비강 폭 ÷ 상악 기저폭", "PAL": "PAL(mm): 인두 기도 길이(PNS–Eb)", "PAV": "PAV(mL): 인두 기도 부피(S–PNS면 ~ Eb면)", "NAV": "NAV(mL): 비강 기도 부피", "PMCA": "PMCA(mm²): 인두 최소 단면적"}
for k, v in ROH.items():
    MANUAL[f"x_{k}"] = (v+" — 엑셀(연구실 수작업 계측)", "Roh 2025 정의, 연구실 Invivo·Mimics 수작업")
    if f"m_{k}" not in MANUAL: MANUAL[f"m_{k}"] = (v+" — 자동 계측", "ALI 기준점·밝기 규칙 자동 계측(엑셀 방식에 맞춘 환산 포함)")
for _k, _t in (("nasopharynx", "비인두"), ("oropharynx", "구인두"), ("hypopharynx", "하인두")):
    MANUAL[f"{_k}_min_area"] = (f"{_t} 최소 단면적(mm²): {_t} 라벨을 1 mm 두께 가로 단면으로 잘라 가장 좁은 넓이(위아래 끝 2 mm 제외)", "TotalSegmentator head_glands_cavities")
    MANUAL[f"{_k}_mean_area"] = (f"{_t} 평균 단면적(mm²): 같은 가로 단면 넓이의 평균", "TotalSegmentator head_glands_cavities")
MANUAL.update({
    "soft_palate_length": ("연구개 길이(mm): 정중 단면에서 연구개의 앞뒤 끝 사이 길이", "TotalSegmentator 연구개"),
    "soft_palate_thickness": ("연구개 최대 두께(mm): 길이 방향에 수직인 두께의 최댓값", "TotalSegmentator 연구개"),
    "soft_palate_angle": ("연구개 기울기(°): 연구개 장축의 앞뒤 축 대비 각", "TotalSegmentator 연구개"),
    "soft_palate_PNS_gap": ("PNS와 연구개 시작점 사이 거리(mm)", "TotalSegmentator 연구개, ALI PNS"),
    "SNA": ("SNA(°): N에서 S 방향과 A 방향 사이 각(정중 시상면 투영)", "ALI 기준점"), "SNB": ("SNB(°): N에서 S 방향과 B 방향 사이 각(정중 시상면 투영)", "ALI 기준점"),
    "S_N": ("앞 두개저 길이: S–N 3D 거리(mm)", "ALI 기준점"), "S_Ba": ("뒤 두개저 길이: S–Ba 3D 거리(mm)", "ALI 기준점"),
    "ANS_PNS": ("구개 길이: ANS–PNS 3D 거리(mm)", "ALI 기준점"), "Go_Me": ("하악체 길이: 좌우 Go 중점–Me 3D 거리(mm)", "ALI 기준점"),
    "N_Me": ("앞 얼굴 높이: N–Me 3D 거리(mm)", "ALI 기준점"), "ANS_Me": ("아래 앞 얼굴 높이: ANS–Me 거리(mm)", "ALI 기준점"),
    "Co_Gn": ("유효 하악 길이: 과두 최상점(좌우 평균)–Gn(Pog·Me 중점) 거리(mm)", "AMASSS 과두, ALI 기준점"), "Co_A": ("유효 중안면 길이: 과두–A점 거리(mm)", "AMASSS 과두, ALI 기준점"),
    "Co_Go": ("하악지 높이: 과두–Go 중점 거리(mm)", "AMASSS 과두, ALI 기준점"), "NSBa": ("두개저 각(°): N–S–Ba", "ALI 기준점"), "gonial": ("하악각(°): 과두–Go–Me", "AMASSS 과두, ALI 기준점"),
    "PP_MP": ("구개평면(ANS–PNS)과 하악평면(Go 중점–Me) 사이 각(°)", "ALI 기준점"), "SN_MP": ("S–N선과 하악평면(Go 중점–Me) 사이 각(°)", "ALI 기준점"),
    "Pog_PNS_ant": ("PNS 기준 Pog의 앞뒤 위치(mm). Pog_Ba_ant = Pog_PNS_ant + PNS_Ba_depth", "ALI 기준점, 정중 좌표계"), "PNS_Ba_depth": ("PNS 기준 Ba의 뒤쪽 깊이(mm)", "ALI 기준점, 정중 좌표계"),
    "MP_H": ("설골에서 하악평면(Go–Me)까지 수직 거리(mm)", "밝기 규칙 설골, ALI 기준점"), "Hy_Me": ("설골–Me 거리(mm)", "밝기 규칙 설골, ALI 기준점"),
    "lever_Pog_mm": ("좌우 과두 축으로 하악을 5° 벌렸을 때 Pog가 뒤로 밀리는 거리(mm)", "AMASSS 과두, ALI 기준점, 강체 회전 계산"),
    "lever_Hy_mm": ("같은 회전에서 설골이 뒤로 밀리는 거리(mm, 강체 값)", "AMASSS 과두, 밝기 규칙 설골"),
    "lever_arm_Pog_sup": ("과두 축과 Pog의 높이 차(mm)", "AMASSS 과두, ALI 기준점"),
    "masseter_mL": ("교근 좌우 합 부피(mL)", "TotalSegmentator head_muscles"), "lat_pterygoid_mL": ("외측익돌근 좌우 합 부피(mL)", "TotalSegmentator head_muscles"),
    "med_pterygoid_mL": ("내측익돌근 좌우 합 부피(mL)", "TotalSegmentator head_muscles"), "chew_mL": ("저작근(교근·외측·내측익돌근) 합 부피(mL)", "TotalSegmentator head_muscles"),
    "sinus_max_mL": ("상악동 좌우 합 부피(mL)", "TotalSegmentator craniofacial_structures"), "sinus_max_air_frac": ("상악동 안 공기(−400 HU 미만) 비율", "TotalSegmentator, CT 밝기"),
    "sinus_frontal_mL": ("전두동 부피(mL)", "TotalSegmentator craniofacial_structures"),
    "tongue_c_ant": ("혀 무게중심의 앞뒤 위치(mm, PNS 원점)", "TotalSegmentator 혀"), "tongue_c_sup": ("혀 무게중심의 위아래 위치(mm)", "TotalSegmentator 혀"),
    "tongue_low_sup": ("혀 최저점 높이(mm)", "TotalSegmentator 혀"), "tongue_len": ("혀 앞뒤 길이(mm)", "TotalSegmentator 혀"), "tongue_height": ("혀 높이(mm)", "TotalSegmentator 혀"),
    "tongue_back_ant": ("혀뿌리(혀 뒤쪽 경계)의 앞뒤 위치(mm)", "TotalSegmentator 혀"), "tongue_mandible_ratio": ("혀 부피 ÷ 하악 부피", "TotalSegmentator 혀, AMASSS 하악"),
    "MMV_mL": ("구개평면 아래 하악·상악 볼록 껍질 안 공간 부피(뼈 제외, mL)", "AMASSS 하악·상악"), "MMV_hull_mL": ("같은 볼록 껍질 전체 부피(뼈 포함, mL)", "AMASSS 하악·상악"),
    "Condyle_Axial_mean": ("과두 축상각 좌우 평균(°)", "AMASSS 하악 과두"), "Condyle_Lat_mean": ("과두 측면각 좌우 평균(°)", "AMASSS 하악 과두"),
    "Jarabak": ("S–Go ÷ N–Me (뒤/앞 얼굴 높이 비)", "계산식"), "McN_diff": ("Co_Gn − Co_A (McNamara 차)", "계산식"), "ramus_body": ("Co_Go ÷ Go_Me", "계산식"),
    "palate_cb": ("ANS_PNS ÷ S_Ba", "계산식"), "mand_cb": ("Co_Gn ÷ S_N", "계산식"), "LAFH_ratio": ("ANS_Me ÷ N_Me (아래 앞얼굴 비율)", "계산식"),
})
MANUAL.update({"Age": ("나이(세)", "엑셀 임상 정보"), "Male": ("성별(남성 = 1)", "엑셀 임상 정보"), "BMI": ("체질량지수(kg/m²)", "엑셀 임상 정보"),
    "res_tongue_c_ant": ("혀 무게중심 앞뒤 위치에서 ANB·턱끝–대후두공 거리·크기 축·성별로 설명되는 부분을 뺀 잔차", "TotalSegmentator 혀, 회귀 잔차"),
    "res_tongue_back_ant": ("혀뿌리 앞뒤 위치의 같은 잔차", "TotalSegmentator 혀, 회귀 잔차"), "res_tongue_mL": ("혀 부피의 같은 잔차", "TotalSegmentator 혀, 회귀 잔차"),
    "res_lat_pterygoid_mL": ("외측익돌근 부피의 같은 잔차", "TotalSegmentator 저작근, 회귀 잔차"), "res_MMV_mL": ("하악·상악 안 공간 부피의 같은 잔차", "AMASSS, 회귀 잔차"),
    "res_L1_MP_deg": ("아래 앞니–하악평면 각의 같은 잔차", "치아 분할, 회귀 잔차")})
for _st in ("MAND", "MAX", "CB"):
    for _k in range(1, 6): MANUAL[f"{_st}_rPC{_k}"] = (f"{ {'MAND': '하악', 'MAX': '상악', 'CB': '두개저'}[_st]} 잔여 형상 {_k}번째 주성분 (ANB·턱끝–대후두공 거리·크기·성별로 설명되지 않는 모양)", "AMASSS 대응점, 잔차 PCA(85명)")
for _p, _lab in (("F1", "형태(모양+배치, 크기 제거 전)"), ("F2", "전체 구성 GPA(배치 보존, 크기 제거)"), ("F2MM", "하악+상악 GPA")):
    for _k in range(1, 11): MANUAL[f"{_p}_PC{_k}"] = (f"하악·상악(·두개저) 대응점을 이은 공동 축 {_k}번째 주성분, {_lab}", "AMASSS 대응점 연결 PCA")
MANUAL["F2_logCS"] = ("공동 GPA 전체 log 중심 크기", "AMASSS"); MANUAL["F2MM_logCS"] = ("하악+상악 GPA log 중심 크기", "AMASSS")
for _k in range(1, 8): MANUAL[f"EFA_with_Eb_F{_k}"] = (f"골격 연동 요인분석(Eb 포함 세트) {_k}번째 요인 점수", "ALI 골격 피처, 탐색적 요인분석")
for _k in range(1, 7): MANUAL[f"EFA_without_Eb_F{_k}"] = (f"골격 연동 요인분석(Eb 제외 세트) {_k}번째 요인 점수", "ALI 골격 피처, 탐색적 요인분석")


# ------------------------------------------------------------ 단계 설정
def steps(d):
    cat = pd.read_csv(R/"features/v1/feature_catalog.csv", encoding="utf-8-sig")
    s4c = [c for c in cat.feature[cat.role.str.startswith("탐색")] if c in d and c not in set(X.CONFIRMED) | {"SIZE_PC1"} and d[c].notna().sum() >= 50 and not c.startswith(("EFA_", "lever"))]
    xm = [("x_ANB", "m_ANB"), ("x_NperpA", "m_NperpA"), ("x_NperpPog", "m_NperpPog"), ("x_MPFH", "m_MPFH"), ("x_TDI", "m_TDI"), ("x_PHI", "m_PHI"),
          ("x_NCWR", "m_NCWR"), ("x_PAL", "a_PAL"), ("x_PAV", "a_PAV"), ("x_NAV", "m_NAV"), ("x_PMCA", "a_PMCA")]
    v = lambda names, extra=(): [(n, list(extra)) for n in names]
    H = list(YH)
    return [
     ("0", "기존 인자 동시 모델", "AHI·앙와위 AHI ~ 나이 + 성별 + BMI + ANB + 턱끝–대후두공 거리 (한 모델의 모든 계수)", "joint", C, H, v(C)),
     ("1", "Step 1. Roh 2025 다중회귀 재현 (엑셀 값 대 복원값)", "Roh 2025 Table 3과 같은 동시 모델: Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + PMCA + PAL. 엑셀 값과 복원값(자동 계측)을 각각 넣고, 두 값이 모두 있는 같은 환자로 맞춤", "roh", B0, H+["AHI", "AHI_SUP"],
      v(ROH5["엑셀"])),
     ("2a", "Step 2-1. 골격 → 기도", "기도 지표 ~ 나이·성별·BMI + 골격 변수 하나", "single", B0, ["a_PAV", "a_PAV_h", "a_PMCA", "a_PAL", "H_col"],
      v(["m_ANB", "Pog_Ba_ant", "SIZE_PC1", "VERT_Z"])),
     ("2b", "Step 2-2. 기도 → AHI", "기존 인자(나이·성별·BMI + ANB + 턱끝–대후두공 거리) 위에 기도 변수 하나", "single", C, H, v(["a_PAV", "a_PAV_h", "a_PMCA", "a_PAL", "H_col"])),
     ("3a", "Step 3-1. 3D 골격 지표 (나이·성별·BMI 위)", "나이·성별·BMI 위에 변수 하나", "single", B0, H, v(["Pog_Ba_ant", "Overjet_mm", "PLS_t1", "tongue_MMV", "SIZE_PC1"])),
     ("3b", "Step 3-2. 3D 골격 지표 (ANB 위)", "나이·성별·BMI + ANB 위에 변수 하나", "single", B0+["m_ANB"], H, v(["Pog_Ba_ant", "Overjet_mm", "PLS_t1", "tongue_MMV", "SIZE_PC1"])),
     ("3c", "Step 3-3. 구조 크기", "기존 인자 위에 크기 변수 하나", "single", C, H, v(["SIZE_PC1", "SIZE_meanz", "SIZE_MAND", "SIZE_MAX", "SIZE_CB"])),
     ("4a", "Step 4-1. 골격 연동 요인", "기존 인자 위에 요인 점수 하나", "single", C, H, v([c for c in d if c.startswith("EFA_")])),
     ("4b", "Step 4-2. 지렛대", "기존 인자 위에 변수 하나", "single", C, H, v(["lever_Pog_mm", "lever_Hy_mm", "lever_arm_Pog_sup"])),
     ("4c", "Step 4-3. 탐색 피처", "기존 인자 위에 변수 하나", "single", C, H, v(s4c)),
     ("S6", "연조직", "기존 인자 위에 변수 하나", "single", C, H, v(X.SOFT_MAIN+X.SOFT_EXP)),
     ("A1", "A1. 크기 축 교란 확인", "기존 인자 + 표시한 보정 위에 변수 하나", "single", C, H,
      [("SIZE_PC1", []), ("SIZE_PC1", ["Age2"]), ("SIZE_sexz", []), ("SIZE_PC1xMale", ["SIZE_PC1"]), ("SIZE_PC1xAge", ["SIZE_PC1"])]),
     ("A2", "A2. 기도 모양", "기존 인자 위에 변수 하나", "single", C, H, v(A2_FEATS)),
     ("A3", "A3. 잔차 피처", "기존 인자 위에 변수 하나", "single", C, H, v(A3_FEATS)),
     ("A5", "A5. 교호작용", "기존 인자 + 주효과 위에 곱 하나", "single", C, H,
      [("m_ANBxSIZE_PC1", ["SIZE_PC1"]), ("m_ANBxMale", []), ("Pog_Ba_antxMale", []), ("SIZE_PC1xBMI", ["SIZE_PC1"]), ("m_ANBxtongue_c_ant", ["tongue_c_ant"])]),
     ("A6", "A6. 앙와위 초과 (log 앙와위 AHI − log AHI)", "결과 = 앙와위 초과. 나이·성별·BMI 위에 변수 하나", "single", B0, ["SUP_excess"], v(["m_ANB", "Pog_Ba_ant", "SIZE_PC1", "VERT_Z"])),
     ("B1", "B1·B2. 분해 (나이·성별·BMI 위)", "나이·성별·BMI 위에 변수 하나", "single", B0, H, v(["Pog_PNS_ant", "PNS_Ba_depth", "SNA", "SNB"])),
     ("B3", "B3. 길이·비율", "기존 인자 위에 변수 하나", "single", C, H, v(LENGTHS+RATIOS)),
     ("D1", "D1. 하악체 길이", "기존 인자 위에 변수 하나", "single", C, H, v(["body_len", "body_len_res"])),
     ("M5", "D-M5. 잔여 형상", "기존 인자 + 구조 크기 축 위에 변수 하나", "single", C+["SIZE_PC1"], H, v([f"{s}_rPC{k}" for s in ("MAND", "MAX", "CB") for k in range(1, 6)])),
     ("D6", "D6. 수직 패턴", "기존 인자 + 구조 크기 축 위에 변수 하나", "single", C+["SIZE_PC1"], H, v(VERT)),
     ("D34", "D3·D4. 설골·구강 공간", "기존 인자 위에 변수 하나", "single", C, H, v(["MP_H", "Hy_Me", "Hy_Ba_dist", "Hy_sup", "Hy_ant", "SoftLoad_MMV", "tongue_MMVhull", "InterGonial", "MMV_mL"])),
     ("D5", "D5. 비인두", "기존 인자 위에 변수 하나", "single", C, H+["SUP_excess"], v(["nasopharynx_min_area", "nasopharynx_mean_area", "PNS_Ba_depth", "ANS_PNS"])),
     ("D78", "D7·D8. 상대성장·기도 프로파일", "기존 인자 위에 변수 하나", "single", C, H, v(["AlloShape", "logCS_all", "AW_min_pos", "AWshape_PC1", "AWshape_PC2", "AWshape_PC3"])),
     ("E", "E. 공동 축", "기존 인자 위에 변수 하나", "single", C, H,
      v([f"{p}_PC{k}" for p in ("F1", "F2", "F2MM") for k in range(1, 11)]+["F2_logCS", "F2MM_logCS"])),
     ("F", "F. 종합 지표", "기존 인자 위에 변수 하나", "single", C, H, v(["Atyp_D2", "Atyp_Procrustes", "Yaxis", "FacialAxis_proxy"])),
     ("G", "G. 수직 배치 지표", "기존 인자 위 / 기존 인자 + 구조 크기 축 위", "single", C, H, [("VERT_Z", []), ("VERT_Z", ["SIZE_PC1"])]),
     ("C", "C. 후보 동시 투입", "기존 인자 + 후보 전부를 한 모델에 (계수는 서로 보정된 값)", "joint", C, H,
      v(C+["SIZE_PC1", "VERT_Z", "oropharynx_min_area", "soft_palate_thickness", "m_ANBxMale", "tongue_c_ant", "m_ANBxtongue_c_ant", "Co_Gn", "S_Ba"])),
     ("P", "P. 자세별 AHI", "결과 = 판독 파일로 계산한 앙와위·비앙와위 AHI(log), 자세 의존 비. 나이·성별·BMI(+표시) 위에 변수 하나", "single", B0,
      ["logSUP_pos", "logNON_pos", "POS_ratio"], [("m_ANB", []), ("m_ANB", ["sup_frac"]), ("Pog_Ba_ant", []), ("VERT_Z", ["m_ANB", "Pog_Ba_ant"])]),
    ]


def fit_rows(d, sid, kind, base, ys, vars_):
    rows = []
    if kind == "roh":
        u = d.dropna(subset=list(ys)+base+ROH5["엑셀"]+ROH5["자동"])
        for k, cols in ROH5.items():
            for y in ys:
                f = sm.OLS(u[y], sm.add_constant(u[base+cols])).fit()
                for c in base+cols: rows.append({"step": sid, "y": y, "var": c, "adj": f"동시 모델({k})", "n": len(u), "B": f.params[c], "beta": f.params[c]*u[c].std()/u[y].std(), "p": f.pvalues[c], "R2": f.rsquared})
        return rows
    if kind == "joint":
        cols = [n for n, _ in vars_]
        for y in ys:
            u = d.dropna(subset=[y]+cols); f = sm.OLS(u[y], sm.add_constant(u[cols])).fit()
            for c in cols: rows.append({"step": sid, "y": y, "var": c, "adj": "같은 모델", "n": len(u), "B": f.params[c], "beta": f.params[c]*u[c].std()/u[y].std(), "p": f.pvalues[c], "R2": f.rsquared})
        return rows
    for n, extra in vars_:
        if n not in d: print("  없음:", n); continue
        cols = list(dict.fromkeys([c for c in base+extra if c != n]+[n]))
        for y in ys:
            u = d.dropna(subset=[y]+cols)
            if len(u) < 20 or u[n].std() == 0: continue
            f = sm.OLS(u[y], sm.add_constant(u[cols])).fit()
            rows.append({"step": sid, "y": y, "var": n, "adj": "+".join(extra) or "-", "n": len(u), "B": f.params[n], "beta": f.params[n]*u[n].std()/u[y].std(), "p": f.pvalues[n], "R2": f.rsquared})
    return rows


YL = {**YH, "AHI": "AHI 원값 (Roh와 같은 척도)", "AHI_SUP": "앙와위 AHI 원값", "a_PAV": "인두 부피", "a_PAV_h": "평균 단면적", "a_PMCA": "최소 단면적", "a_PAL": "기도 길이", "H_col": "기둥 높이", "SUP_excess": "앙와위 초과",
      "logSUP_pos": "앙와위 AHI (계산)", "logNON_pos": "비앙와위 AHI (계산)", "POS_ratio": "자세 의존 비"}
ADJ = {"Age2": "나이²", "SIZE_PC1": "크기 축", "tongue_c_ant": "혀 위치", "sup_frac": "앙와위 수면 비율", "m_ANB": "ANB", "Pog_Ba_ant": "턱끝–대후두공 거리"}


def html_table(t, cat, ys, figs=None):
    figs = figs or {}
    keys = list(dict.fromkeys(zip(t["var"], t.adj))); head = "".join(f'<th>{escape(YL[y])} B</th><th>β</th><th>p</th>' for y in ys)
    body = []
    for v_, a in keys:
        de, so = cat.get(v_, ("(정의 없음)", ""))
        name = v_ + ("" if a in ("-", "같은 모델") else f"<br><span class='note'>+ {escape(', '.join(ADJ.get(x, x) for x in a.split('+')))} 보정</span>")
        cells, n = "", None
        for y in ys:
            r = t[(t["var"] == v_) & (t.adj == a) & (t.y == y)]
            if not len(r): cells += "<td></td><td></td><td></td>"; continue
            r = r.iloc[0]; n = int(r.n); p = r.p; pc = "<0.001" if p < .001 else f"{p:.3f}"
            cells += f'<td>{r.B:+.3g}</td><td>{r.beta:+.2f}</td><td{" class=sig" if p < .05 else ""}>{pc}</td>'
        fg = f'<a href="figs/{figs[v_].name}" target="_blank"><img class="thumb" src="figs/{figs[v_].name}" alt="{escape(v_)}"></a>' if v_ in figs else ""
        body.append(f"<tr><td><code>{name}</code></td><td>{escape(de)}</td><td>{escape(so)}</td><td>{fg}</td><td>{n}</td>{cells}</tr>")
    return f'<div class="tw"><table><thead><tr><th>변수명</th><th>변수 뜻</th><th>만든 과정</th><th>CT 위치</th><th>n</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

# ------------------------------------------------------------ 회귀식당 표 (선택형)
TERM = {"const": "절편", "Age": "나이", "Male": "성별(남=1)", "BMI": "BMI", "m_ANB": "ANB (자동)", "Pog_Ba_ant": "턱끝–대후두공 거리", "SIZE_PC1": "구조 크기 축",
        "Age2": "나이²", "tongue_c_ant": "혀 무게중심 앞뒤", "sup_frac": "앙와위 수면 비율"}
ROH5 = {"엑셀": ["x_ANB", "x_MPFH", "x_TDI", "x_PMCA", "x_PAL"], "자동": ["m_ANB", "m_MPFH", "m_TDI", "a_PMCA", "a_PAL"]}
PAIRS = [("ANB", "x_ANB", "m_ANB"), ("Nperp-A", "x_NperpA", "m_NperpA"), ("Nperp-Pog", "x_NperpPog", "m_NperpPog"), ("MP-FH", "x_MPFH", "m_MPFH"),
         ("TDI", "x_TDI", "m_TDI"), ("PHI", "x_PHI", "m_PHI"), ("NCWR", "x_NCWR", "m_NCWR"), ("PAL", "x_PAL", "a_PAL"), ("PAV", "x_PAV", "a_PAV"),
         ("NAV", "x_NAV", "m_NAV"), ("PMCA", "x_PMCA", "a_PMCA"), ("턱끝–대후두공 거리 (Nperp-Pog 대체)", None, "Pog_Ba_ant"), ("SN-MP", None, "SN_MP"), ("PP-MP", None, "PP_MP")]


def fit_model(u, y, cols):
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    Xc = sm.add_constant(u[cols]); f = sm.OLS(u[y], Xc).fit(); sy = u[y].std()
    vf = {t: (float(vif(Xc.to_numpy(float), i)) if len(cols) > 1 else 1.0) for i, t in enumerate(Xc.columns) if t != "const"}
    terms = [{"t": t, "B": float(f.params[t]), "beta": None if t == "const" else float(f.params[t]*u[t].std()/sy), "p": float(f.pvalues[t]), "vif": vf.get(t)} for t in ["const"]+cols]
    return {"n": int(len(u)), "r2": float(f.rsquared), "ar2": float(f.rsquared_adj), "terms": terms}


def exps_for(d, sid, kind, base, ys, vars_):
    """단계 → 실험(회귀식) 목록. Step 1은 지표별 엑셀/자동 쌍(같은 환자)."""
    if sid == "1":
        u = d.dropna(subset=list(ys)+base+ROH5["엑셀"]+ROH5["자동"])
        return [{"label": "Roh 동시 모델", "versions": {k: {"var": None, "models": {y: fit_model(u, y, base+cols) for y in ys}} for k, cols in ROH5.items()}}]
    if kind == "joint":
        cols = [n for n, _ in vars_]; return [{"label": "한 모델", "var": None, "models": {y: fit_model(d.dropna(subset=[y]+cols), y, cols) for y in ys}}]
    out = []
    for n, extra in vars_:
        if n not in d: continue
        cols = list(dict.fromkeys([c for c in base+extra if c != n]+[n])); ms = {}
        for y in ys:
            u = d.dropna(subset=[y]+cols)
            if len(u) >= 20 and u[n].std() > 0: ms[y] = fit_model(u, y, cols)
        if ms: out.append({"label": n+("" if not extra else " (+ "+", ".join(ADJ.get(x, x) for x in extra)+" 보정)"), "var": n, "models": ms})
    return out


JS = r"""
const fmt=(v,k)=>v===null||v===undefined?'':(k==='p'?(v<0.001?'<0.001':v.toFixed(3)):(Math.abs(v)<0.001&&v!==0?v.toExponential(2):(v>=0?'+':'')+v.toPrecision(3)));
function table(ex,ys){
  const terms=ex.models[ys.find(y=>ex.models[y])].terms.map(t=>t.t);
  let h='<div class="tw"><table class="reg"><thead><tr><th rowspan=2>항</th>'+ys.map(y=>`<th colspan=3>${YL[y]}</th>`).join('')+'</tr><tr>'+ys.map(()=>'<th>B</th><th>β</th><th>p</th>').join('')+'</tr></thead><tbody>';
  for(const t of terms){h+=`<tr${t===ex.focus?' class="focus"':''}><td>${TERM[t]||t}</td>`;
    for(const y of ys){const m=ex.models[y];const r=m?m.terms.find(x=>x.t===t):null;
      h+=r?`<td>${fmt(r.B,'B')}</td><td>${fmt(r.beta,'b')}</td><td class="${r.p<0.05&&t!=='const'?'sig':''}">${fmt(r.p,'p')}</td>`:'<td></td><td></td><td></td>';}
    h+='</tr>';}
  for(const [k,l] of [['n','n'],['r2','R²'],['ar2','수정 R²']]){h+=`<tr class="foot"><td>${l}</td>`+ys.map(y=>{const m=ex.models[y];return `<td colspan=3>${m?(k==='n'?m.n:m[k].toFixed(3)):''}</td>`}).join('')+'</tr>';}
  return h+'</tbody></table></div>';}
function eq(v,ys,base){const yl=ys.map(y=>YL[y]).join(' / ');return `<p class="eq">회귀식: ${yl} ~ ${base}${v?' + <b>'+(TERM[v]||v)+'</b>':''}</p>`;}
function desc(v){if(!v)return '';const c=CAT[v]||['',''];const f=FIGS[v]?`<a href="figs/${FIGS[v]}" target="_blank"><img class="big" src="figs/${FIGS[v]}"></a>`:'<p class="note">계산식으로 만든 변수라 CT 위치 그림이 없습니다.</p>';
  return `<div class="desc"><div><p><b>변수명</b> <code>${v}</code></p><p><b>변수 뜻</b> ${c[0]}</p><p><b>만든 과정</b> ${c[1]}</p></div><div>${f}</div></div>`;}
function render(sid){const S=STEPS[sid];const box=document.getElementById('out'+sid);
  if(S.step1){const i=+document.getElementById('sel'+sid).value;const e=S.exps[i];
    const ver=[...document.getElementsByName('ver'+sid)];
    ver.forEach(r=>{r.disabled=!e.versions[r.value];});let cur=ver.find(r=>r.checked&&!r.disabled);if(!cur){cur=ver.find(r=>!r.disabled);cur.checked=true;}
    const V=e.versions[cur.value];const ex={models:V.models,focus:V.var};
    box.innerHTML=eq(V.var,S.ys,S.base)+table(ex,S.ys)+(Object.keys(e.versions).length>1?'<p class="note">엑셀과 자동은 두 값이 모두 있는 같은 환자로 맞췄습니다.</p>':'')+desc(V.var);return;}
  const i=S.exps.length>1?+document.getElementById('sel'+sid).value:0;const e=S.exps[i];
  box.innerHTML=eq(e.var,S.ys,S.base)+table({models:e.models,focus:e.var},S.ys)+desc(e.var);}
document.addEventListener('DOMContentLoaded',()=>{for(const k in STEPS)render(k);});
"""


def step_html(sid, title, setting, S):
    if S.get("step1"):
        opts = "".join(f'<option value="{i}">{escape(e["label"])}</option>' for i, e in enumerate(S["exps"]))
        ctl = (f'<label>지표 <select id="sel{sid}" onchange="render(\'{sid}\')">{opts}</select></label>'
               f'<span class="radio"><label><input type="radio" name="ver{sid}" value="엑셀" checked onchange="render(\'{sid}\')"> 엑셀 (수작업)</label>'
               f'<label><input type="radio" name="ver{sid}" value="자동" onchange="render(\'{sid}\')"> 자동 계측</label></span>')
    elif len(S["exps"]) > 1:
        opts = "".join(f'<option value="{i}">{escape(e["label"])}</option>' for i, e in enumerate(S["exps"]))
        ctl = f'<label>실험 선택 <select id="sel{sid}" onchange="render(\'{sid}\')">{opts}</select></label> <span class="note">{len(S["exps"])}개 회귀식</span>'
    else: ctl = '<span class="note">회귀식 1개 (변수를 한 모델에 함께 넣음)</span>'
    return f'<h2 id="s{sid}">{escape(title)}</h2><p><b>설정:</b> {escape(setting)}.</p><div class="ctl">{ctl}</div><div id="out{sid}"></div>'


def rows_for(S, y, ver=None):
    """선택한 Y에 대해 X 인자 행 목록: (변수, 보정 표시, 모델 dict, 관심 항)."""
    out = []
    for e in S["exps"]:
        if S.get("step1"):
            for k in ((ver,) if ver else ("엑셀", "자동")):
                v = e["versions"].get(k)
                if v and y in v["models"]: m = v["models"][y]; out += [(t["t"], "같은 모델", m, t["t"]) for t in m["terms"]]
        elif e["var"] is None:                                                    # 한 모델(동시 투입): 항마다 한 행
            m = e["models"].get(y)
            if m: out += [(t["t"], "같은 모델", m, t["t"]) for t in m["terms"] if t["t"] != "const"]
        elif y in e["models"]:
            lab = e["label"][len(e["var"]):].strip(); out.append((e["var"], lab, e["models"][y], e["var"]))
    return out


def ytable(S, y, cat, figs, ver=None):
    body = []
    for v_, lab, m, foc in rows_for(S, y, ver):
        t = next(x for x in m["terms"] if x["t"] == foc); de, so = cat.get(v_, ("(정의 없음)", ""))
        pc = "&lt;0.001" if t["p"] < .001 else f'{t["p"]:.3f}'
        fg = f'<a href="figs/{figs[v_].name}" target="_blank"><img class="thumb" src="figs/{figs[v_].name}" alt="{escape(v_)}"></a>' if v_ in figs else ""
        name = f"<code>{escape(v_)}</code>" + (f"<br><span class='note'>{escape(lab)}</span>" if lab and lab != "같은 모델" else "")
        bt = "" if t["beta"] is None else f'{t["beta"]:+.2f}'; vf = "" if t.get("vif") is None else f'{t["vif"]:.2f}'
        sig = " class=sig" if t["p"] < .05 and v_ != "const" else ""
        body.append(f'<tr><td>{name}</td><td>{escape(de)}</td><td>{escape(so)}</td><td>{fg}</td><td>{m["n"]}</td>'
                    f'<td>{t["B"]:+.3g}</td><td>{bt}</td><td{sig}>{pc}</td><td>{vf}</td><td>{m["r2"]:.3f}</td><td>{m["ar2"]:.3f}</td></tr>')
    return ('<div class="tw"><table><thead><tr><th>변수명</th><th>변수 뜻</th><th>만든 과정</th><th>CT 위치</th><th>n</th><th>B</th><th>β</th><th>p</th><th>VIF</th><th>모델 R²</th><th>수정 R²</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def step_html_y(sid, title, setting, S, cat, figs):
    ys = S["ys"]; opts = "".join(f'<option value="{i}">{escape(YL[y])}</option>' for i, y in enumerate(ys))
    ctl = f'<label>결과 변수(Y) <select id="y{sid}" onchange="showY(\'{sid}\')">{opts}</select></label>'
    if S.get("step1"):
        ctl += f' <label>값 <select id="v{sid}" onchange="showY(\'{sid}\')"><option value="엑셀">엑셀 값 (연구실 수작업)</option><option value="자동">복원값 (CBCT 자동 계측)</option></select></label>'
    eqs = f'<p class="eq">회귀식(하나): Y ~ 성별 + 나이 + BMI + ANB + MP-FH + TDI + PMCA + PAL (아래 행 전부를 한 모델에)</p>' if S.get("step1") else f'<p class="eq">회귀식(행마다 따로): Y ~ {escape(S["base"])} + <b>X(행의 변수)</b></p>' if S["exps"] and S["exps"][0].get("var", 1) is not None else \
          f'<p class="eq">회귀식(하나): Y ~ 아래 행의 변수 전부</p>'
    note = ('<p class="note">엑셀 값과 복원값이 모두 있는 같은 환자로 맞췄다(값을 바꿔도 n이 같다). Roh 2025 Table 3(102명, AHI 원값)의 AHI 결과: ANB β 0.32 (p=.005), MP-FH 0.05 (p=.680), TDI −0.27 (p=.002), PMCA −0.21 (p=.039), PAL −0.03 (p=.727), VIF 1.19~2.01, 수정 R² 0.40.</p>'
            if S.get("step1") else "")
    vers = ("엑셀", "자동") if S.get("step1") else ("",)
    tabs = "".join(f'<div class="yt" data-s="{sid}" data-k="{i}|{v}"{"" if (i == 0 and v == vers[0]) else " hidden"}>{ytable(S, y, cat, figs, v or None)}</div>'
                   for i, y in enumerate(ys) for v in vers)
    return f'<h2 id="s{sid}">{escape(title)}</h2><p><b>설정:</b> {escape(setting)}.</p><div class="ctl">{ctl}</div>{eqs}{note}{tabs}'


def main():
    import osa_var_figs as VF
    d = data(); cat = catalog(); allrows = []; secs = []; STEPS_JS = {}
    names = list(dict.fromkeys(n for st in steps(d) for n, _ in st[6])); ex_pid, figs = VF.build(d, names); print('그림', len(figs), '예시 환자', ex_pid)
    print("83명:", len(d), "| Class III", int(d.ClassIII.sum()))
    for sid, title, setting, kind, base, ys, vars_ in steps(d):
        rows = fit_rows(d, sid, kind, base, ys, vars_); t = pd.DataFrame(rows)
        if not len(t): continue
        if kind == "single":
            for y in ys:
                k = t.index[t.y == y]
                if len(k): t.loc[k, "q_BH"] = multipletests(t.loc[k, "p"], method="fdr_bh")[1]
        allrows.append(t.assign(title=title))
        bn = ", ".join({"Age": "나이", "Male": "성별", "BMI": "BMI", "m_ANB": "ANB", "Pog_Ba_ant": "턱끝–대후두공 거리", "SIZE_PC1": "구조 크기 축"}.get(c, c) for c in base)
        ns = sorted(t.n.unique()); nn = f"{ns[0]}" if len(ns) == 1 else f"{ns[0]}~{ns[-1]}"
        S = {"ys": ys, "base": " + ".join(TERM.get(c, c) for c in base) if kind != "joint" else "아래 항 전부", "exps": exps_for(d, sid, kind, base, ys, vars_), "step1": sid == "1"}
        secs.append(step_html_y(sid, title, setting, S, cat, figs))
    T = pd.concat(allrows); T.round(5).to_csv(OUT/"steps.csv", index=False, encoding="utf-8-sig")
    miss = sorted({v for v in T["var"] if v not in cat}); print("정의 없음:", miss)
    pv = d[["Age", "Male", "BMI", "ClassIII", "AHI", "AHI_SUP", "x_ANB", "m_ANB", "Pog_Ba_ant", "VERT_Z", "SIZE_PC1", "a_PAV", "a_PMCA", "a_PAL"]].copy()
    pv.round(3).to_csv(OUT/"patient_values.csv", encoding="utf-8-sig")
    full = d[[c for c in dict.fromkeys(T["var"]) if c in d]]; full.round(4).to_csv(OUT/"patient_values_all.csv", encoding="utf-8-sig")
    pvh = pv.rename(columns={"Age": "나이", "Male": "남성", "ClassIII": "Class III", "AHI_SUP": "앙와위 AHI", "x_ANB": "ANB (엑셀)", "m_ANB": "ANB (자동)",
                             "Pog_Ba_ant": "턱끝–대후두공 거리", "VERT_Z": "수직 배치 지표", "SIZE_PC1": "구조 크기 축", "a_PAV": "인두 부피", "a_PMCA": "최소 단면적", "a_PAL": "기도 길이"})
    pvh["남성"] = pvh["남성"].map({1.0: "예", 0.0: "아니오"}); pvh["Class III"] = pvh["Class III"].map({True: "예", False: ""})
    nav = "".join(f'<a href="#s{s[0]}">{escape(s[1].split(".")[0])}</a>' for s in steps(d))
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>83명 단계별 회귀</title>
<style>{CSS}td.sig{{font-weight:700;color:var(--accent)}}.yt table{{border-collapse:collapse}}.yt th,.yt td{{border:1px solid #b8c2cc;padding:4px 6px}}.yt th{{cursor:pointer;user-select:none;white-space:nowrap}}.yt th:after{{content:' ⇅';color:#aab}}.yt th.asc:after{{content:' ▲';color:inherit}}.yt th.desc:after{{content:' ▼';color:inherit}}tr.focus td{{background:var(--sum,#eef4ff)}}tr.foot td{{color:var(--muted);font-size:.85rem}}.ctl{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:8px 0}}.ctl select{{font-size:.95rem;padding:3px 6px;max-width:100%}}.radio label{{margin-right:12px}}.eq{{font-family:monospace;font-size:.88rem;margin:6px 0}}.desc{{display:grid;grid-template-columns:1fr 340px;gap:16px;border-top:1px solid #ddd;margin-top:8px;padding-top:8px}}img.big{{width:100%;max-width:340px}}@media (max-width:700px){{.desc{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>83명 전체 단계별 회귀 결과</h1><p class="sub">2026-11-17 · 분할 없음 (계수 유의성 판단용) · 대상 83명 (19세 미만 2명, CBCT 없음 2명 제외, Class III 3명 포함)</p>
<nav>{nav}<a href="#pv">환자별 값</a></nav>
<div class="box"><b>읽는 법</b><ul><li>결과 변수는 log(AHI + 1), log(앙와위 AHI + 1)이며 PSG 값은 공식 리포트 기준이다. 단계마다 원래 실험의 보정 변수를 그대로 썼다.</li>
<li>단계마다 위의 선택 상자로 <b>결과 변수(Y)</b>를 고른다. 표의 <b>한 행이 회귀식 하나</b>로, Y ~ 보정 변수 + 그 행의 X를 따로 적합한 X의 계수다. '동시 모델' 절은 회귀식 하나에 모든 행의 변수를 함께 넣은 결과다. 열 제목을 누르면 오름차순·내림차순으로 정렬된다(빈 값은 맨 아래).</li><li><b>B</b>: 비표준화 회귀계수(변수 1단위 증가 시 log 결과 변화). <b>β</b>: 표준화 계수(변수 1 SD 증가 시 결과의 SD 변화). <b>p</b>: 해당 계수의 t 검정. p &lt; 0.05는 굵게 표시했다(다중검정 보정 전).</li><li><b>Step 1</b>은 Roh 2025 Table 3과 같은 동시 다중회귀다(성별·나이·BMI + ANB·MP-FH·TDI·PMCA·PAL을 한 모델에). '값' 상자로 엑셀 값과 복원값을 바꾸고, Y에서 log 값과 원값(Roh와 같은 척도)을 고른다. <b>VIF</b>는 다중공선성(다른 설명 변수와 겹치는 정도, 5 이상이면 주의)이다.</li>
<li>단계·결과 변수별 FDR q는 결과 파일에 함께 있다. 변수 정의와 추출 기준은 기존과 같다.</li><li><b>CT 위치</b>: CT에서 바로 잰 변수는 예시 환자 1명(P_{ex_pid})의 CBCT에 해당 기준점·선·분할 영역을 그렸다(누르면 크게). 정중 시상 단면은 PNS 원점 정중 좌표계이며, 좌우에 있는 점(Go, 과두)은 정중면에 투영했다. Roh 지표(ANB·Nperp·MP-FH·TDI·PHI·NCWR·기도)는 엑셀과 자동 계측이 같은 정의라 같은 그림을 쓰며, 기존 CBCT 보고서의 용어 그림(다른 예시 환자)을 가져왔다. 주성분·요인·잔차·교호작용·비율처럼 계산식으로 만든 변수는 그림이 없다.</li></ul></div>
{"".join(secs)}
<h2 id="pv">환자별 주요 인자값</h2><p class="note">전체 변수 값은 결과 폴더의 환자별 값 파일에 있다.</p><div class="tw">{tab(pvh.round(2).reset_index())}</div>
<script>function sortBy(th){{const t=th.closest('table'),tb=t.tBodies[0],i=[...th.parentNode.children].indexOf(th);const dir=th.classList.contains('asc')?-1:1;t.querySelectorAll('th').forEach(h=>h.classList.remove('asc','desc'));th.classList.add(dir>0?'asc':'desc');const val=td=>{{const x=td.textContent.trim().replace('<','').replace('+','').replace('−','-');const n=parseFloat(x);return x===''?null:(isNaN(n)?x:n);}};[...tb.rows].sort((r1,r2)=>{{const a=val(r1.cells[i]),b=val(r2.cells[i]);if(a===null)return 1;if(b===null)return -1;return (typeof a==='number'&&typeof b==='number'?a-b:String(a).localeCompare(String(b),'ko'))*dir;}}).forEach(r=>tb.appendChild(r));}}document.addEventListener('DOMContentLoaded',()=>document.querySelectorAll('.yt th').forEach(th=>th.addEventListener('click',()=>sortBy(th))));</script><script>function showY(s){{const y=document.getElementById('y'+s).value;const v=document.getElementById('v'+s);const k=y+'|'+(v?v.value:'');document.querySelectorAll('.yt[data-s="'+s+'"]').forEach(e=>{{e.hidden=e.dataset.k!==k;}});}}</script>
</main></body></html>"""
    check(html, "83명 단계별 회귀"); f = OUT/"전체_83명_기존단계별_회귀.html"; f.write_text(html, encoding="utf-8"); print("saved", f)
    s = T[T.p < .05][["step", "y", "var", "adj", "n", "beta", "p", "q_BH"]]; pd.set_option("display.width", 220); print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
