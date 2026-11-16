"""OSA 관련성 실험 (목적 단계 Step 1~4). 파생변수(build_features.py)로 골격 → 기도(2-A), 골격 → OSA(2-B)를 단계별로 본다.

기록 문서: docs/OSA_관련성_실험기록.md (목적·고정 원칙·단계별 결과·판단)
단계
  S1 기존 지표 기준선: 나이·성별·BMI(B0) + 기존 Roh 지표, 기도 변수 동반 시 가림 효과, 엑셀 값 ↔ 자동 복원값 지표별 1:1
  S2 골격 → 기도(2-A): 새 골격·연동 피처 → PMCA·PAL·HD
  S3 순수 골격 총효과(2-B, 주 가설 H1 = log AHI, H2 = log 앙와위 AHI): 사전 가설 피처를 B0 위·ANB 위에서, 교호작용, 보조 목표(ODI·LSaO2)
  S4 연동·기구학 탐색: 4-A fold 안 VARIMAX EFA(골격+설골, +Eb, +기도 프로파일), 4-B 지렛대, 4-C 탐색 피처 FDR 선별
공통
- 표본: 주 분석 = PSG 확인 완료(cohort Include, 59명), 민감도 = 측정 85명 중 PSG 값이 있는 환자. 2-A는 PSG가 필요 없어 85명 전체.
- 비교는 모델마다 필요한 값이 모두 있는 환자로, 같은 환자 안에서 기준 모델과 1:1.
- 반복 5겹 교차검증 50회: 교차검증 R², 교차검증 RMSE(학습에 쓰지 않은 환자 예측). 범위는 반복 50회의 2.5~97.5 백분위(신뢰구간 아님).
  전체 표본 OLS: 조정 R²(ΔAdj-R²), 중첩 F 검정 p, 표준화 계수.
- PLS·EFA 점수는 fold 학습 환자로만 적합하고 시험 환자에 투영한다. EFA는 fold마다 요인 순서가 바뀌므로,
  같은 표본 전체로 적합한 기준 적재량과 적재량 코사인이 가장 큰 요인을 고르고 부호를 맞춘다(결과 변수는 쓰지 않음).
사용: python osa_experiments.py S1 [S2 ...]   (인자 없으면 전부)
      --source final_v2 [--grades ABCD|AB]: 골격 4개(ANB·Nperp-A·Nperp-Pog·MP-FH)를 develop85/final_criteria_v2 최종 계측표로 바꿔 같은 실험을 반복.
        기본 등급 ABCD(E = 분석 제외 권장 → 결측). AB = 영상으로 확인된 값만. 기도 지표는 최종표도 재확인 전이라 기존 값 유지.
        ANBxBMI·PogBaxMPFH는 바꾼 값으로 다시 계산. 출력 osa_experiments_v2_final[_AB]/
출력: Result_py/analysis/osa_experiments_v2/*.csv, summary.json   (v1 = 2026-11-16 1차 실행, 보존)
"""
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

import build_features as bf
from shape_regression import table

R_ = bf.R_; DEST = R_/"analysis/osa_experiments_v2"
REPEATS, FOLDS = 50, 5
B0 = ["Age", "Male", "BMI"]
ROH_SKEL = ["m_ANB", "m_MPFH", "m_TDI"]; ROH_AIR = ["m_PMCA", "m_PAL"]
H = {"H1": "logAHI", "H2": "logAHI_SUP"}
PRIMARY = [["Pog_Ba_ant"], ["Overjet_mm"], ["PLS"], ["tongue_MMV"]]
METRICS = ["ANB", "Nperp-A", "Nperp-Pog", "MP-FH", "TDI", "PHI", "NCWR", "PAL", "PAV", "NAV", "PMCA"]
SKEL_XY, EFA_SETS, EFA_REF = None, {}, {}
FINAL = bf.OUT/"develop85/final_criteria_v2/final_measurements_85.csv"
SOURCE, GRADES = "current", "ABCD"


def apply_final(d, grades):
    """골격 4개를 최종 계측표 값으로 교체(허용 등급 밖은 결측)."""
    f = pd.read_csv(FINAL, encoding="utf-8-sig").set_index("Patient")
    for k in ("ANB", "Nperp-A", "Nperp-Pog", "MP-FH"):
        v = f[k].where(f[k+"_grade"].isin(list(grades))); d["m_"+k.replace("-", "")] = v.reindex(d.index)
        d["grade_"+k.replace("-", "")] = f[k+"_grade"].reindex(d.index)
    cz = lambda x: x-x.mean()
    d["ANBxBMI"] = cz(d["m_ANB"])*cz(d["BMI"]); d["PogBaxMPFH"] = cz(d["Pog_Ba_ant"])*cz(d["m_MPFH"])
    return d


def data():
    d = table()
    f = pd.read_csv(bf.DEST/"features.csv", encoding="utf-8-sig", index_col=0)
    m = pd.read_csv(R_/"phase1/patient_master.csv", encoding="utf-8-sig").set_index("Patient")
    d = d.join(f, how="left")
    d["logAHI_SUP"], d["logODI"], d["LSaO2"] = np.log1p(m.AHI_SUP), np.log1p(m.ODI), m.LSaO2
    prof = pd.read_csv(R_/"shape/extra/airway_profile.csv", encoding="utf-8-sig").set_index("Patient")
    aw = [f"AW_p{i:02d}" for i in range(20)]; d = d.join(prof[aw])
    xl = pd.read_csv(bf.EXT/"compare_all.csv", encoding="utf-8-sig").pivot(index="Patient", columns="metric", values="excel")
    for k in METRICS: d["x_"+k.replace("-", "")] = xl[k].reindex(d.index)    # 복원과 같은 정의의 엑셀 값
    zs = pd.read_csv(FINAL.parent/"final_values.csv", encoding="utf-8-sig").query("excel_zero_suspect == True")
    for p, k in zip(zs.Patient, zs.metric):                                    # 엑셀 0값 의심(Nperp-A 17, Nperp-Pog 5) → 결측
        if p in d.index: d.loc[p, "x_"+k.replace("-", "")] = np.nan
    # 주의: m_* 중 Nperp-A·Nperp-Pog·PHI·PMCA·NAV는 엑셀에 맞춘 환산값이다. 예측력 비교에는 써도 되지만
    # 엑셀 − 자동 차이 분석에 쓰면 순환이 된다(osa_excel_diff_check.py는 최종표·default 값을 쓴다).
    global SKEL_XY
    xc = [c for c in f if c.startswith("c_") and c.split("_")[1] in bf.SKEL]; hy = [c for c in f if c.startswith("c_Hy")]; eb = [c for c in f if c.startswith("c_Eb")]
    SKEL_XY = (xc, hy+eb+aw)
    EFA_SETS.update({"skel": xc+hy, "skel_eb": xc+hy+eb, "skel_eb_air": xc+hy+eb+aw})
    if SOURCE == "final_v2": d = apply_final(d, GRADES)
    for st, cols in EFA_SETS.items():
        model, _ = bf.fit_efa(d.loc[d.restored.astype(bool), cols].dropna()); C = model["fa"].components_
        EFA_REF[st] = C[np.argsort(-(C**2).sum(1))]                         # varimax 출력 순서는 분산순이 아니므로 정렬
    return d


def expand(add):
    """모델 항목 → (일반 열, PLS 사용 여부, EFA 항목 목록)."""
    return [c for c in add if c != "PLS" and not c.startswith("EFA:")], "PLS" in add, [c for c in add if c.startswith("EFA:")]


def needed(base, add):
    cols, pls, efa = expand(add); need = base+cols
    if pls: need += SKEL_XY[0]+SKEL_XY[1]
    for t in efa: need += EFA_SETS[t.split(":")[1]]
    return need


_EFA_CACHE = {}


def efa_score(d, tr, te, tok):
    """EFA:<세트>:<k> — k는 전체 적합 요인을 적재량 제곱합 내림차순으로 정렬한 순위(data()에서 정렬)."""
    _, st, k = tok.split(":"); cols = EFA_SETS[st]; ref = EFA_REF[st][int(k)-1]
    key = (st, tuple(d.index[tr]))                                         # 환자 ID 기준(표본마다 행 위치가 달라서)
    if key not in _EFA_CACHE: _EFA_CACHE[key] = bf.fit_efa(d.iloc[tr][cols])
    model, s_tr = _EFA_CACHE[key]; C = model["fa"].components_
    cos = C@ref/(np.linalg.norm(C, axis=1)*np.linalg.norm(ref)+1e-12); j = int(np.argmax(np.abs(cos))); sg = np.sign(cos[j])
    return s_tr.iloc[:, j].to_numpy()*sg, bf.transform_efa(model, d.iloc[te][cols]).iloc[:, j].to_numpy()*sg


def design(d, tr, te, base, add):
    cols, pls, efa = expand(add); A, B = d.iloc[tr][base+cols].to_numpy(float), d.iloc[te][base+cols].to_numpy(float)
    if pls:
        X, Y = d[SKEL_XY[0]], d[SKEL_XY[1]]
        model, s_tr = bf.fit_pls(X.iloc[tr], Y.iloc[tr], perms=0)
        A, B = np.hstack([A, s_tr.to_numpy()]), np.hstack([B, bf.transform_pls(model, X.iloc[te]).to_numpy()])
    for t in efa:
        a, b = efa_score(d, tr, te, t); A, B = np.c_[A, a], np.c_[B, b]
    return A, B


def cv(d, y, base, add):
    """(교차검증 R² 배열, 교차검증 RMSE 배열) — 반복마다 하나."""
    yv = d[y].to_numpy(float); sst = ((yv-yv.mean())**2).sum(); r2, rm = [], []
    for rep in range(REPEATS):
        pred = np.zeros(len(d))
        for tr, te in KFold(FOLDS, shuffle=True, random_state=rep).split(d):
            A, B = design(d, tr, te, base, add); mu, sd = A.mean(0), A.std(0)+1e-12
            fit = sm.OLS(yv[tr], sm.add_constant((A-mu)/sd)).fit(); pred[te] = fit.predict(sm.add_constant((B-mu)/sd, has_constant="add"))
        sse = ((yv-pred)**2).sum(); r2.append(1-sse/sst); rm.append(np.sqrt(sse/len(yv)))
    return np.array(r2), np.array(rm)


def compare(d, y, base, add, ref_add=()):
    """기준(base+ref_add)과 비교 모델(base+ref_add+add)을 같은 환자에서. 반환 dict."""
    full = list(ref_add)+list(add)
    s = d.dropna(subset=[y]+needed(base, full))
    if len(s) < 25: return {"n": len(s)}
    (r0, m0), (r1, m1) = cv(s, y, base, list(ref_add)), cv(s, y, base, full); dr = r1-r0
    idx = np.arange(len(s)); yv = s[y].to_numpy(float)
    A, _ = design(s, idx, idx, base, full); fit = sm.OLS(yv, sm.add_constant(A)).fit()
    A0, _ = design(s, idx, idx, base, list(ref_add)); fit0 = sm.OLS(yv, sm.add_constant(A0)).fit()
    k = A.shape[1]-A0.shape[1]; coef = fit.params[-k:]; p = fit.pvalues[-k:]
    return {"n": len(s), "R2cv_ref": r0.mean(), "R2cv": r1.mean(), "dR2": dr.mean(), "dR2_lo": np.percentile(dr, 2.5), "dR2_hi": np.percentile(dr, 97.5),
            "share_better": (dr > 0).mean(), "RMSEcv_ref": m0.mean(), "RMSEcv": m1.mean(), "dRMSE": (m1-m0).mean(),
            "adjR2_ref": fit0.rsquared_adj, "adjR2": fit.rsquared_adj, "dAdjR2": fit.rsquared_adj-fit0.rsquared_adj,
            "F_p": float(fit.compare_f_test(fit0)[1]), "coef_std": ";".join(f"{c:+.3f}" for c in coef*np.r_[A[:, -k:].std(0)]),
            "p_coef": ";".join(f"{x:.4f}" for x in p)}


def samples(d):
    return {"main59": d[d.analysis_list.astype(bool)], "sens85": d[d.restored.astype(bool)]}


def fdr(t, by=("sample",)):
    t = pd.DataFrame(t)
    for _, g in t.groupby(list(by)):
        k = g.index[g.F_p.notna()]; t.loc[k, "q_BH"] = multipletests(t.loc[k, "F_p"], method="fdr_bh")[1]
    return t


def run(name, rows):
    t = pd.DataFrame(rows); DEST.mkdir(parents=True, exist_ok=True); t.round(4).to_csv(DEST/f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\n== {name}"); print(t.round(3).to_string(index=False)); return t


# ---------- Step 1: 기존 지표 기준선 ----------
def S1(d):
    rows = []
    for sn, s in samples(d).items():
        for h, y in H.items():
            for label, add in (("+ANB(자동)", ["m_ANB"]), ("+ANB·MPFH(자동)", ["m_ANB", "m_MPFH"]), ("+Roh골격(ANB·MPFH·TDI)", ROH_SKEL),
                               ("+Roh골격+기도(PMCA·PAL)", ROH_SKEL+ROH_AIR), ("+ANB+기도(PMCA·PAL)", ["m_ANB"]+ROH_AIR), ("+엑셀 Roh5(기존 M0)", ["ANB", "MPFH", "TDI", "PMCA", "PAL"])):
                rows.append({"part": "1-1 기준선", "sample": sn, "hyp": h, "model": label, **compare(s, y, B0, add)})
            rows.append({"part": "1-2 기도 가림", "sample": sn, "hyp": h, "model": "ANB 위에 +PMCA·PAL", **compare(s, y, B0, ROH_AIR, ref_add=["m_ANB"])})
    run("S1_baseline", rows)
    rows = []
    for k in METRICS:
        a, e = "m_"+k.replace("-", ""), "x_"+k.replace("-", "")
        for sn, s in samples(d).items():
            s = s.dropna(subset=[a, e]); r = np.corrcoef(s[a], s[e])[0, 1] if len(s) > 3 else np.nan
            for h, y in H.items():
                ex, au = compare(s, y, B0, [e]), compare(s, y, B0, [a])
                rows.append({"metric": k, "sample": sn, "hyp": h, "n": ex.get("n"), "r_auto_excel": r, "dR2_excel": ex.get("dR2"), "p_excel": ex.get("F_p"),
                             "dAdjR2_excel": ex.get("dAdjR2"), "dR2_auto": au.get("dR2"), "p_auto": au.get("F_p"), "dAdjR2_auto": au.get("dAdjR2")})
    return run("S1_excel_vs_auto_per_metric", rows)


# ---------- Step 2: 골격 → 기도 ----------
def airway_raw(d):
    """Step 2 결정(2026-11-16): 환산 전 원값. PMCA = 중심선 수직 단면, PAV = 정중 좌표계 아래 경계, Eb 검출 방법."""
    v2 = R_/"CBCT/patient_pipeline/v2"; ca = pd.read_csv(bf.EXT/"compare_all.csv", encoding="utf-8-sig").pivot(index="Patient", columns="metric", values="restored")
    pv = pd.concat([pd.read_csv(v2/"pav_review/v1/pav.csv", encoding="utf-8-sig")[["Patient", "PAV_midline"]],
                    pd.read_csv(bf.EXT/"new_alternatives.csv", encoding="utf-8-sig")[["Patient", "PAV_midline"]]]).drop_duplicates("Patient").set_index("Patient")
    from osa_eb_review import eb_methods
    d["a_PMCA"], d["a_PAL"] = ca.PMCA.reindex(d.index), ca.PAL.reindex(d.index)
    d["a_PAV"] = pv.PAV_midline.reindex(d.index)/1000                         # mm³ → mL
    d["Eb_method"] = eb_methods().reindex(d.index)
    return d


def size_axis(d):
    """S5 사전 등록 정의: roi_*_size 12개 표준화 첫 주성분(클수록 큰 머리) + 보조 z 평균."""
    size = [c for c in d if c.startswith("roi_") and c.endswith("_size")]; assert len(size) == 12, size
    Z = (d[size]-d[size].mean())/d[size].std(); ok = Z.notna().all(1)
    u = np.linalg.svd(Z[ok].to_numpy(), full_matrices=False)[2][0]; u = u if u.sum() > 0 else -u
    d["SIZE_PC1"] = (Z@u).where(ok); d["SIZE_meanz"] = Z.mean(1).where(ok)
    for g in ("MAND", "MAX", "CB"): d[f"SIZE_{g}"] = Z[[c for c in size if c.startswith(f"roi_{g}_")]].mean(1).where(ok)
    return size, u


def S2(d):
    d = airway_raw(d); size_axis(d); rows = []; s85 = d[d.restored.astype(bool)]
    for sn, s in (("85명", s85), ("Eb 직접 검출", s85[s85.Eb_method == "vallecula"])):
        for y in ("a_PMCA", "a_PAL", "a_PAV", "HD_mm"):
            for add in (["m_ANB"], ["Pog_Ba_ant"], ["Overjet_mm"], ["PLS"], ["SIZE_PC1"], ["Sup_Go"], ["S_Go"], ROH_SKEL):
                rows.append({"sample": sn, "target": y, "part": "B0 위", "add": "+".join(add), **compare(s, y, B0, add)})
            for add in (["Pog_Ba_ant"], ["PLS"], ["SIZE_PC1"]):
                rows.append({"sample": sn, "target": y, "part": "ANB 위", "add": "+".join(add), **compare(s, y, B0, add, ref_add=["m_ANB"])})
    return run("S2_airway_raw", fdr(rows, ("sample", "target", "part")).to_dict("records"))


# ---------- Step 3: 순수 골격 총효과 (주 가설 H1·H2) ----------
def S3(d):
    rows = []
    for sn, s in samples(d).items():
        for h, y in H.items():
            for add in PRIMARY+[["Pog_Ba_ant", "Overjet_mm", "PLS"]]:
                rows.append({"part": "3-1 B0 위", "sample": sn, "hyp": h, "add": "+".join(add), **compare(s, y, B0, add)})
                rows.append({"part": "3-2 기존 지표(ANB) 위", "sample": sn, "hyp": h, "add": "+".join(add), **compare(s, y, B0, add, ref_add=["m_ANB"])})
            rows.append({"part": "3-3 교호작용", "sample": sn, "hyp": h, "add": "ANBxBMI", **compare(s, y, B0, ["ANBxBMI"], ref_add=["m_ANB"])})
            rows.append({"part": "3-3 교호작용", "sample": sn, "hyp": h, "add": "PogBaxMPFH", **compare(s, y, B0, ["PogBaxMPFH"], ref_add=["Pog_Ba_ant", "m_MPFH"])})
        for y in ("logODI", "LSaO2"):
            for add in (["m_ANB"],)+tuple(PRIMARY):
                rows.append({"part": "3-4 보조 목표", "sample": sn, "hyp": y, "add": "+".join(add), **compare(s, y, B0, add)})
    return run("S3_primary_H1_H2", fdr(rows, ("sample", "part")).to_dict("records"))


# ---------- Step 4: 연동·기구학 탐색 ----------
CONFIRMED = ["m_ANB", "Pog_Ba_ant"]                                        # Step 1 확정 인자 (B0 위)
REFS = {"확정(ANB+Pog_Ba) 위": CONFIRMED, "확정+크기 위": CONFIRMED+["SIZE_PC1"]}


def S4(d):
    """Step 4 확정판(2026-11-16): 기준 = B0 + 최종 ANB + Pog_Ba (+ SIZE_PC1). 출력 이름에 _v3."""
    size_axis(d); rows = []
    for sn, s in samples(d).items():
        for h, y in H.items():
            for st in EFA_SETS:
                for k in (1, 2, 3):
                    tok = f"EFA:{st}:{k}"
                    for lab, ref in REFS.items():
                        rows.append({"part": "4-A EFA(fold 안), "+lab, "sample": sn, "hyp": h, "add": tok, **compare(s, y, B0, [tok], ref_add=ref)})
            for c in ("lever_Pog_mm", "lever_Hy_mm", "lever_arm_Pog_sup"):
                for lab, ref in REFS.items():
                    rows.append({"part": "4-B 지렛대, "+lab, "sample": sn, "hyp": h, "add": c, **compare(s, y, B0, [c], ref_add=ref)})
    run("S4AB_efa_lever_v3", fdr(rows, ("sample", "hyp", "part")).to_dict("records"))
    cat = pd.read_csv(bf.DEST/"feature_catalog.csv", encoding="utf-8-sig")
    skip = set(CONFIRMED) | {"SIZE_PC1"}
    feats = [c for c in cat.feature[cat.role.str.startswith("탐색")] if c in d and c not in skip and d[c].notna().sum() >= 50 and not c.startswith(("EFA_", "lever"))]
    rows = []
    for sn, s in samples(d).items():
        for h, y in H.items():
            for c in feats:
                for lab, ref in REFS.items():
                    rows.append({"part": "4-C "+lab, "sample": sn, "hyp": h, "feature": c, **compare(s, y, B0, [c], ref_add=ref)})
    t = fdr(rows, ("sample", "hyp", "part")).sort_values(["part", "hyp", "sample", "F_p"])
    return run("S4C_exploratory_fdr_v3", t.to_dict("records"))


# ---------- S5: 구조 크기 축 (사전 등록, 실험기록 "S5 사전 등록") ----------
def S5(d):
    size, u = size_axis(d); print("SIZE_PC1 적재량", dict(zip(size, u.round(2))))
    rows = []
    for sn, s in samples(d).items():
        for h, y in H.items():
            for add in ("SIZE_PC1", "SIZE_meanz", "SIZE_MAND", "SIZE_MAX", "SIZE_CB"):
                for ref, lab in (([], "B0 위"), (["m_ANB"], "ANB 위"), (["m_ANB", "Pog_Ba_ant"], "ANB+Pog_Ba 위")):
                    rows.append({"part": lab, "primary": add == "SIZE_PC1", "sample": sn, "hyp": h, "add": add, **compare(s, y, B0, [add], ref_add=ref)})
    return run("S5_size_axis", rows)


def S4E(d):
    """Step 4-A 수정판(2026-11-16): 요인을 분산순으로 정렬해 전부 검정 + 세트 전체 묶음 검정.
    주 판단 세트는 skel·skel_eb. skel_eb_air는 기도(매개) 정보가 들어 있어 참고용."""
    size_axis(d); rows = []
    for sn, s in samples(d).items():
        for h, y in H.items():
            for st in EFA_SETS:
                toks = [f"EFA:{st}:{k}" for k in range(1, len(EFA_REF[st])+1)]
                for lab, ref in REFS.items():
                    for t in toks:
                        rows.append({"part": "4-A 요인별, "+lab, "set": st, "sample": sn, "hyp": h, "add": t, **compare(s, y, B0, [t], ref_add=ref)})
                    rows.append({"part": "4-A 세트 전체, "+lab, "set": st, "sample": sn, "hyp": h, "add": f"EFA:{st}:1-{len(toks)}", **compare(s, y, B0, toks, ref_add=ref)})
    return run("S4A_efa_v4", fdr(rows, ("sample", "hyp", "part")).to_dict("records"))


SOFT_MAIN = ["tongue_back_ant", "tongue_mandible_ratio"]
SOFT_EXP = ["tongue_c_ant", "tongue_c_sup", "tongue_low_sup", "tongue_len", "tongue_height", "tongue_mL", "masseter_mL", "lat_pterygoid_mL",
            "med_pterygoid_mL", "chew_mL", "sinus_max_mL", "sinus_max_air_frac", "sinus_frontal_mL"]


def S6(d):
    """연조직 피처 (사전 등록: 실험기록 "S6 사전 등록"). 사전 가설 2개 + 탐색 13개, 기준 B0 / 확정 / 확정+크기."""
    size_axis(d); rows = []
    for sn, s in samples(d).items():
        for h, y in H.items():
            for c in SOFT_MAIN+SOFT_EXP:
                for lab, ref in (("B0 위", []),)+tuple(REFS.items()):
                    rows.append({"part": lab, "kind": "사전 가설" if c in SOFT_MAIN else "탐색", "sample": sn, "hyp": h, "feature": c,
                                 **compare(s, y, B0, [c], ref_add=ref)})
    return run("S6_soft_tissue", fdr(rows, ("sample", "hyp", "part", "kind")).to_dict("records"))


STEPS = {"S1": S1, "S2": S2, "S3": S3, "S4": S4, "S4E": S4E, "S5": S5, "S6": S6}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--source" in args: i = args.index("--source"); SOURCE = args[i+1]; del args[i:i+2]
    if "--grades" in args: i = args.index("--grades"); GRADES = args[i+1]; del args[i:i+2]
    if SOURCE != "current": DEST = R_/("analysis/osa_experiments_v2_final"+("" if GRADES == "ABCD" else "_"+GRADES))
    d = data(); todo = args or list(STEPS)
    out = {}
    for k in todo:
        t = STEPS[k](d); out[k] = len(t)
    f = DEST/"summary.json"; prev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    f.write_text(json.dumps({**prev, **{k: {"rows": v, "source": SOURCE, "grades": GRADES, "features_mtime": (bf.DEST/"features.csv").stat().st_mtime,
                                            "efa_reference_factors": {s: int(len(c)) for s, c in EFA_REF.items()}} for k, v in out.items()}},
                            ensure_ascii=False, indent=1), encoding="utf-8")
