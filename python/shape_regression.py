"""형태 벡터의 추가 설명력: PCA 요약과 M0 대비 교차검증 회귀. 결과변수 = 엑셀 AHI (민감도: log(AHI+1)).

M0 (Phase 1과 같음): AHI ~ 나이 + 남성 + BMI + 엑셀 ANB·MP-FH·TDI·PMCA·PAL. 성별은 정정값(psg_verified V_Sex), 나이는 검사일 기준(V_Age).
비교 블록 (결과를 보기 전에 정함, 블록마다 특징 2~4개):
  M0r: Roh 5지표를 복원값(엑셀 정렬 채택값)으로 바꿈
  M1 : + 하악 전역 SSM Procrustes PC1~3 (폴드 안 PCA)
  M2 : + 하악 국소 ROI(턱끝·하악각·하악지·과두) 모양 PC1 각 1개 (폴드 안 PCA)
  M3 : + 기도·연조직: 기도 최소 단면적(AW_min), 기도 길이(AW_length), 연구개 길이, MP-H
  M4 : + 치열궁: 상·하 제1대구치 폭, 상악 깊이
표본: 주 분석 = 분석 명단(cohort Include) ∩ Phase 1 주 표본(Class III·CBCT 불일치 제외) ∩ 결측 없음 / 민감도 = 복원 85명 ∩ 같은 기준.
평가: 반복 5겹 CV(50회), 폴드 밖 예측으로 R²_cv, 반복마다 ΔR²(블록 − M0)의 평균과 2.5–97.5 백분위. 참고로 표본 안 adj R²와 블록 F 검정.
PCA 요약: 85명 골격 랜드마크 좌표 PCA(머리 좌표계, 중심화), 구조별 SSM 설명분산, PC와 Roh 지표·AHI의 상관(탐색용).
출력: Result_py/analysis/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

from shape_ssm import gpa

ROOT = Path(__file__).resolve().parents[1]; R = ROOT/"Result_py"; SH = R/"shape"; DEST = R/"analysis"
EXT = R/"CBCT/patient_pipeline/v2/restore_ext/v1"
ROH = ["ANB", "MPFH", "TDI", "PMCA", "PAL"]; COV = ["Age", "Male", "BMI"]
REPEATS, FOLDS = 50, 5
ROI_M2 = ["symphysis", "gonial", "ramus_condyle"]
B3 = ["AW_min", "AW_length", "soft_palate_length", "MP_H"]; B4 = ["upper_width_M1", "lower_width_M1", "upper_depth"]


def table():
    m = pd.read_csv(R/"phase1/patient_master.csv", encoding="utf-8-sig").set_index("Patient")
    v = pd.read_csv(R/"psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
    d = pd.DataFrame(index=m.index)
    d["AHI"] = m.AHI; d["logAHI"] = np.log1p(m.AHI); d["Age"] = v.V_Age.reindex(d.index).fillna(m.Age)
    d["Male"] = (v.V_Sex.reindex(d.index) == "M").astype(float); d["BMI"] = m.BMI
    for k in ROH: d[k] = m[k]
    al = pd.read_csv(EXT/"alignment/aligned_metrics.csv", encoding="utf-8-sig")
    rm = {"ANB": "ANB", "MP-FH": "MPFH", "TDI": "TDI", "PMCA": "PMCA", "PAL": "PAL"}
    for k, c in rm.items(): d[c+"_r"] = al[al.metric == k].set_index("Patient").aligned.reindex(d.index)
    d = d.join(pd.read_csv(SH/"extra/airway_profile.csv", encoding="utf-8-sig").set_index("Patient")[["AW_min", "AW_length"]])
    d = d.join(pd.read_csv(SH/"fine/soft_tissue.csv", encoding="utf-8-sig").set_index("Patient")[["soft_palate_length"]])
    d = d.join(pd.read_csv(SH/"extra/relation.csv", encoding="utf-8-sig").set_index("Patient")[["MP_H"]])
    d = d.join(pd.read_csv(SH/"fine/dental_arch.csv", encoding="utf-8-sig").set_index("Patient")[B4])
    inc = pd.read_csv(R/"CBCT/patient_pipeline/v2/cohort.csv", dtype=str, encoding="utf-8-sig").set_index("Patient").Include.str.lower() == "true"
    d["analysis_list"] = inc.reindex(d.index).fillna(False); d["valid_main"] = m.Valid_Main
    d["restored"] = d.index.isin(pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig").Patient)
    return d


def shape_blocks(ids):
    """폴드 안 PCA용 원자료: 하악 전역(GPA 후 평탄화)과 국소 ROI별(GPA 후 평탄화). GPA는 결과변수를 쓰지 않는다."""
    z = np.load(SH/"MAND/points.npz"); pid = [f"P_{i}" for i in z["ids"]]; X = z["frame"]; lab = np.load(SH/"local/MAND_roi_labels.npy")
    g = gpa(X)[0].reshape(len(X), -1); out = {"global": pd.DataFrame(g, index=pid)}
    for r in ROI_M2: out[r] = pd.DataFrame(gpa(X[:, lab == r])[0].reshape(len(X), -1), index=pid)
    return {k: v.reindex(ids) for k, v in out.items()}


def pc_fit(tr, te, k):
    mu = tr.mean(0); vt = np.linalg.svd(tr-mu, full_matrices=False)[2][:k]
    return (tr-mu)@vt.T, (te-mu)@vt.T


def design(d, tr, te, model, S):
    base = (ROH if model != "M0r" else [c+"_r" for c in ROH])
    Xtr, Xte = d.iloc[tr][COV+base].to_numpy(float), d.iloc[te][COV+base].to_numpy(float)
    add_tr, add_te = [], []
    if model == "M1":
        a, b = pc_fit(S["global"].iloc[tr].to_numpy(), S["global"].iloc[te].to_numpy(), 3); add_tr.append(a); add_te.append(b)
    if model == "M2":
        for r in ROI_M2:
            a, b = pc_fit(S[r].iloc[tr].to_numpy(), S[r].iloc[te].to_numpy(), 1); add_tr.append(a); add_te.append(b)
    if model == "M3": add_tr.append(d.iloc[tr][B3].to_numpy(float)); add_te.append(d.iloc[te][B3].to_numpy(float))
    if model == "M4": add_tr.append(d.iloc[tr][B4].to_numpy(float)); add_te.append(d.iloc[te][B4].to_numpy(float))
    return np.hstack([Xtr]+add_tr), np.hstack([Xte]+add_te)


MODELS = ["M0", "M0r", "M1", "M2", "M3", "M4"]


def cv(d, y, S, models=MODELS):
    yv = d[y].to_numpy(float); sst = ((yv-yv.mean())**2).sum(); res = {m: [] for m in models}
    for rep in range(REPEATS):
        pred = {m: np.zeros(len(d)) for m in models}
        for tr, te in KFold(FOLDS, shuffle=True, random_state=rep).split(d):
            for m in models:
                A, B = design(d, tr, te, m, S); mu, sd = A.mean(0), A.std(0)+1e-12
                fit = sm.OLS(yv[tr], sm.add_constant((A-mu)/sd)).fit(); pred[m][te] = fit.predict(sm.add_constant((B-mu)/sd, has_constant="add"))
        for m in models: res[m].append(1-((yv-pred[m])**2).sum()/sst)
    return {m: np.array(v) for m, v in res.items()}


def insample(d, y, S, models=MODELS):
    out = {}; idx = np.arange(len(d)); yv = d[y].to_numpy(float); f0 = None
    for m in models:
        A, _ = design(d, idx, idx, m, S); fit = sm.OLS(yv, sm.add_constant(A)).fit()
        if m == "M0": f0 = fit
        p = float(fit.compare_f_test(f0)[1]) if m not in ("M0", "M0r") else np.nan
        out[m] = {"adjR2": fit.rsquared_adj, "k": A.shape[1], "F_p_vs_M0": p}
    return out


def pca_summary(d):
    L = pd.read_csv(EXT/"experiments/export/skeletal_landmarks_head.csv", encoding="utf-8-sig").set_index("Patient")
    from batch_cbct_pipeline import LMS
    cols = [f"{k}_{a}" for k in LMS for a in ("left", "ant", "sup") if f"{k}_{a}" in L and not (k == "N")]
    X = L[cols].dropna()
    q = pd.read_csv(R/"CBCT/patient_pipeline/v2/last_run.csv", encoding="utf-8-sig").drop_duplicates("Patient", keep="last").set_index("Patient").QC
    use = (q.reindex(X.index) == "qc_pass_pending_visual_review").to_numpy()      # 축은 랜드마크 거리 QC 통과 환자로만, 나머지는 투영 (P_35처럼 Po·Or 실패 환자가 PC1을 끌지 않게)
    mu = X[use].mean(); _, s, vt = np.linalg.svd(X[use]-mu, full_matrices=False); var = s**2/(s**2).sum()
    sc = pd.DataFrame((X-mu).to_numpy()@vt[:5].T, index=X.index, columns=[f"PC{i}" for i in range(1, 6)]); sc["in_pca_axes"] = use
    d = d.assign(**{c: d[c] for c in d})    # 상관은 축을 만든 환자로만 계산
    d = d.loc[X.index[use]]
    load = pd.DataFrame(vt[:3].T, index=cols, columns=["PC1", "PC2", "PC3"])
    top = {c: load[c].abs().sort_values(ascending=False).head(6).index.tolist() for c in load}
    j = sc.drop(columns="in_pca_axes").join(d[["AHI"]+[c+"_r" for c in ROH]], how="inner")
    corr = j.corr().loc[[f"PC{i}" for i in range(1, 6)], ["AHI"]+[c+"_r" for c in ROH]]
    ssm = {}
    for s_ in ("MAND", "CB", "MAX", "UAW"):
        f = SH/s_/"review/variance_procrustes.csv"
        if f.exists(): v = pd.read_csv(f, encoding="utf-8-sig"); ssm[s_] = v.iloc[:5, -1].round(3).tolist() if v.shape[1] else []
    return var[:10], sc, top, corr, ssm


def main():
    DEST.mkdir(parents=True, exist_ok=True); d = table()
    # 모델마다 그 블록에 필요한 변수가 있는 환자로, 같은 표본에서 M0와 1:1 비교
    extra = {"M0": [], "M0r": [c+"_r" for c in ROH], "M1": [], "M2": [], "M3": B3, "M4": B4}
    samples = {"주 분석: 분석 명단": d[d.analysis_list & d.valid_main], "민감도: 복원 85명": d[d.restored & d.valid_main]}
    rows, ins = [], []
    for name, s0 in samples.items():
        S0 = shape_blocks(s0.index.tolist())
        for m in MODELS:
            ok = s0[COV+ROH+extra[m]+["AHI"]].notna().all(1) & S0["global"].notna().all(1)
            s = s0[ok]; S = {k: v[ok.to_numpy()] for k, v in S0.items()}; pair = ["M0"] if m == "M0" else ["M0", m]
            for y in ("AHI", "logAHI"):
                res = cv(s, y, S, pair); dr = res[m]-res["M0"]
                rows.append({"sample": name, "y": y, "n": len(s), "model": m, "R2_cv": res[m].mean(), "R2_cv_lo": np.percentile(res[m], 2.5), "R2_cv_hi": np.percentile(res[m], 97.5),
                             "M0_R2_cv_same_n": res["M0"].mean(), "dR2_vs_M0": dr.mean(), "dR2_lo": np.percentile(dr, 2.5), "dR2_hi": np.percentile(dr, 97.5), "share_better": float((dr > 0).mean())})
                v = insample(s, y, S, pair)[m]; ins.append({"sample": name, "y": y, "n": len(s), "model": m, **v,
                                                           "M0_adjR2_same_n": insample(s, y, S, ["M0"])["M0"]["adjR2"]})
        print(name, "후보", len(s0), flush=True)
    r = pd.DataFrame(rows); r.round(4).to_csv(DEST/"regression_cv.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ins).round(4).to_csv(DEST/"regression_insample.csv", index=False, encoding="utf-8-sig")
    var, sc, top, corr, ssm = pca_summary(d)
    sc.round(3).to_csv(DEST/"pca_landmark_scores.csv", encoding="utf-8-sig"); corr.round(3).to_csv(DEST/"pca_landmark_corr.csv", encoding="utf-8-sig")
    json.dump({"landmark_pca_var": [round(float(x), 4) for x in var], "landmark_pca_top_loadings": top, "ssm_procrustes_cumvar_PC1_5": ssm},
              open(DEST/"pca_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(r[r.y == "AHI"].set_index(["sample", "model"])[["n", "R2_cv", "M0_R2_cv_same_n", "dR2_vs_M0", "dR2_lo", "dR2_hi", "share_better"]].round(3).to_string())
    print(pd.DataFrame(ins).query("y == \"AHI\"").set_index(["sample", "model"])[["n", "adjR2", "M0_adjR2_same_n", "F_p_vs_M0"]].round(3).to_string())
    print("landmark PCA var", np.round(var[:5], 3)); print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
