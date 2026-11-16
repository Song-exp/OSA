"""A 단계: 기존 인자(B0 + 최종 ANB + Pog_Ba) 위 추가 변수 발굴. 사전 등록은 실험기록 "A 단계 사전 등록".
A1 크기 교란 / A2 기도 모양·연구개 / A3 잔차 피처 / A4 elastic net 안정성 선택 / A5 교호작용 / A6 앙와위 초과.
실행: python osa_addition.py [A1 A2 ...]   출력 Result_py/analysis/osa_experiments_v2_final/A_addition/
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
import osa_experiments as X

X.SOURCE = "final_v2"
BASE = X.B0+X.CONFIRMED                                                    # 기존 인자
OUT = X.R_/"analysis/osa_experiments_v2_final/A_addition"; OUT.mkdir(parents=True, exist_ok=True)
cz = lambda s: s-s.mean()


def load():
    d = X.data(); X.size_axis(d); d = d[d.restored.astype(bool)].copy()
    st = pd.read_csv(X.R_/"shape/fine/soft_tissue.csv", encoding="utf-8-sig").set_index("Patient")
    for c in st:
        if c.startswith(("nasopharynx_m", "oropharynx_m", "hypopharynx_m", "soft_palate_")) and "volume" not in c: d[c] = st[c].reindex(d.index)
    d["oro_naso_min_ratio"] = d.oropharynx_min_area/d.nasopharynx_min_area
    d["AW_min_pos"] = pd.read_csv(X.R_/"shape/extra/airway_profile.csv", encoding="utf-8-sig").set_index("Patient").AW_min_pos.reindex(d.index)
    d["Age2"] = cz(d.Age)**2
    d["SIZE_sexz"] = d.groupby("Male").SIZE_PC1.transform(lambda s: (s-s.mean())/s.std())
    d["SUP_excess"] = d.logAHI_SUP-d.logAHI
    for a, b in (("m_ANB", "SIZE_PC1"), ("m_ANB", "Male"), ("Pog_Ba_ant", "Male"), ("SIZE_PC1", "BMI"), ("m_ANB", "tongue_c_ant"), ("SIZE_PC1", "Male"), ("SIZE_PC1", "Age")):
        d[f"{a}x{b}"] = cz(d[a])*cz(d[b])
    # A3 잔차: 기존 인자 + 크기 + 성별로 설명되는 부분 제거(AHI 미사용)
    R = ["m_ANB", "Pog_Ba_ant", "SIZE_PC1", "Male"]
    for c in ("tongue_c_ant", "tongue_back_ant", "tongue_mL", "lat_pterygoid_mL", "MMV_mL", "L1_MP_deg"):
        ok = d[[c]+R].notna().all(1); f = sm.OLS(d.loc[ok, c], sm.add_constant(d.loc[ok, R])).fit()
        d.loc[ok, "res_"+c] = f.resid
    return d


def samples(d):
    return {"main59": d[d.analysis_list.astype(bool)], "sens85": d}


def table(name, rows):
    t = X.fdr(rows, ("sample", "hyp", "part")); t.round(4).to_csv(OUT/f"{name}.csv", index=False, encoding="utf-8-sig")
    show = [c for c in ("part", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std") if c in t]
    print(f"\n== {name}"); print(t[show].round(3).to_string(index=False)); return t


def A1(d):
    rows = []
    for sn, s in samples(d).items():
        for h, y in X.H.items():
            for lab, add, ref in (("기본", ["SIZE_PC1"], X.CONFIRMED), ("+나이²", ["SIZE_PC1"], X.CONFIRMED+["Age2"]),
                                  ("성별 안 z", ["SIZE_sexz"], X.CONFIRMED), ("+나이²·성별 안 z", ["SIZE_sexz"], X.CONFIRMED+["Age2"]),
                                  ("성별×크기", ["SIZE_PC1xMale"], X.CONFIRMED+["SIZE_PC1"]), ("나이×크기", ["SIZE_PC1xAge"], X.CONFIRMED+["SIZE_PC1"])):
                rows.append({"part": lab, "sample": sn, "hyp": h, "add": add[0], **X.compare(s, y, X.B0, add, ref_add=ref)})
    return table("A1_size_confound", rows)


A2_FEATS = ["HD_mm", "AW_min_pos", "nasopharynx_min_area", "nasopharynx_mean_area", "oropharynx_min_area", "oropharynx_mean_area",
            "hypopharynx_min_area", "hypopharynx_mean_area", "oro_naso_min_ratio", "soft_palate_length", "soft_palate_thickness",
            "soft_palate_angle", "soft_palate_PNS_gap"]
A3_FEATS = ["res_tongue_c_ant", "res_tongue_back_ant", "res_tongue_mL", "res_lat_pterygoid_mL", "res_MMV_mL", "res_L1_MP_deg"]


def single(d, feats, name):
    rows = []
    for sn, s in samples(d).items():
        for h, y in X.H.items():
            for c in feats:
                for lab, ref in (("기존 인자 위", X.CONFIRMED), ("기존 인자+크기 위", X.CONFIRMED+["SIZE_PC1"])):
                    rows.append({"part": lab, "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
    return table(name, rows)


def pool(d):
    cat = pd.read_csv(X.bf.DEST/"feature_catalog.csv", encoding="utf-8-sig")
    f = [c for c in cat.feature[cat.role.str.startswith(("탐색", "2-B"))] if c in d and not c.startswith(("EFA_", "PLS_", "m_"))]
    f += A2_FEATS+A3_FEATS+["SIZE_PC1"]
    f = [c for c in dict.fromkeys(f) if c not in BASE and c not in ("tongue_MMV",) and d[c].notna().sum() >= 75 and d[c].std() > 0]
    return f


def resid(Y, B):
    """B(기존 인자, 상수 포함)로 잔차화. Y: (n,k)."""
    Bc = np.c_[np.ones(len(B)), B]; beta = np.linalg.lstsq(Bc, Y, rcond=None)[0]; return Y-Bc@beta, beta


def enet_1se(Z, y, seed=0):
    """ElasticNetCV 경로에서 1-SE 규칙(최소 MSE + 1 표준오차 이내 가장 큰 alpha)으로 고른 뒤 재적합. 표본이 작아 최소 MSE alpha가 0 근처로 튀는 것을 막음."""
    cv = ElasticNetCV(l1_ratio=.5, cv=5, random_state=seed, max_iter=20000).fit(Z, y)
    mse = cv.mse_path_.mean(1); se = cv.mse_path_.std(1)/np.sqrt(cv.mse_path_.shape[1]); i = np.argmin(mse)
    a = cv.alphas_[mse <= mse[i]+se[i]].max()
    return ElasticNet(alpha=a, l1_ratio=.5, max_iter=20000).fit(Z, y)


def zclip(tr, te):
    """학습 fold 평균·SD로 표준화하고 ±3으로 자름(이상치 환자 1명이 예측을 끌어올리는 것 방지)."""
    mu, sd = tr.mean(0), tr.std(0)+1e-12
    return np.clip((tr-mu)/sd, -3, 3), np.clip((te-mu)/sd, -3, 3)


def enet_select(Btr, Ftr, ytr, seed=0):
    ry, _ = resid(ytr[:, None], Btr); rf, _ = resid(Ftr, Btr)
    Z, _ = zclip(rf, rf[:1]); m = enet_1se(Z, ry.ravel(), seed)
    return np.flatnonzero(np.abs(m.coef_) > 1e-8)


def A4(d, reps=20, nboot=200):
    """안정성 선택 빈도는 무작위 기준율(평균 선택 수 ÷ 후보 수)과 비교해 읽는다."""
    feats = pool(d); rows, freq_rows = [], []
    for sn, s0 in samples(d).items():
        for h, y in X.H.items():
            s = s0.dropna(subset=[y]+BASE+feats); B, F, Y = s[BASE].to_numpy(float), s[feats].to_numpy(float), s[y].to_numpy(float)
            rng = np.random.default_rng(0); cnt = np.zeros(len(feats))
            for _ in range(nboot):
                i = rng.choice(len(s), len(s)//2, replace=False); cnt[enet_select(B[i], F[i], Y[i])] += 1
            for c, v in zip(feats, cnt/nboot): freq_rows.append({"sample": sn, "hyp": h, "feature": c, "freq": v, "n": len(s)})
            r0, r1, nsel = [], [], []
            sst = ((Y-Y.mean())**2).sum()
            for rep in range(reps):
                p0, p1 = np.zeros(len(s)), np.zeros(len(s))
                for tr, te in KFold(5, shuffle=True, random_state=rep).split(s):
                    # 기존 인자 OLS(벌점 없음) + 잔차에 elastic net(축소 계수 그대로 예측). 선택 변수 OLS 재적합은 과적합으로 폭주해 쓰지 않음
                    Bc = np.c_[np.ones(len(tr)), B[tr]]; bt = np.linalg.lstsq(Bc, Y[tr], rcond=None)[0]; base_te = np.c_[np.ones(len(te)), B[te]]@bt
                    ry = Y[tr]-Bc@bt; bf_ = np.linalg.lstsq(Bc, F[tr], rcond=None)[0]; rf = F[tr]-Bc@bf_; rft = F[te]-np.c_[np.ones(len(te)), B[te]]@bf_
                    Ztr, Zte = zclip(rf, rft); m = enet_1se(Ztr, ry, rep)
                    nsel.append(int((np.abs(m.coef_) > 1e-8).sum())); p0[te] = base_te; p1[te] = base_te+m.predict(Zte)
                r0.append(1-((Y-p0)**2).sum()/sst); r1.append(1-((Y-p1)**2).sum()/sst)
            dr = np.array(r1)-np.array(r0)
            rows.append({"sample": sn, "hyp": h, "n": len(s), "n_candidates": len(feats), "R2cv_base": np.mean(r0), "R2cv_ext": np.mean(r1),
                         "dR2": dr.mean(), "dR2_lo": np.percentile(dr, 2.5), "dR2_hi": np.percentile(dr, 97.5), "share_better": (dr > 0).mean(),
                         "mean_selected": np.mean(nsel)})
    t = pd.DataFrame(rows); t.round(4).to_csv(OUT/"A4_enet_extended.csv", index=False, encoding="utf-8-sig"); print("\n== A4 확장 모델"); print(t.round(3).to_string(index=False))
    fr = pd.DataFrame(freq_rows); fr.round(3).to_csv(OUT/"A4_stability.csv", index=False, encoding="utf-8-sig")
    top = fr.pivot_table(index="feature", columns=["sample", "hyp"], values="freq"); top = top[top.max(axis=1) >= .3].sort_values(("main59", "H1"), ascending=False)
    print("\n== A4 안정성 선택 빈도 (최대 ≥ 0.3)"); print(top.round(2).to_string()); return t


def A5(d):
    rows = []
    for sn, s in samples(d).items():
        for h, y in X.H.items():
            for a, b in (("m_ANB", "SIZE_PC1"), ("m_ANB", "Male"), ("Pog_Ba_ant", "Male"), ("SIZE_PC1", "BMI"), ("m_ANB", "tongue_c_ant")):
                ref = list(dict.fromkeys(X.CONFIRMED+[a, b]))
                ref = [c for c in ref if c not in X.B0]
                rows.append({"part": "교호작용", "sample": sn, "hyp": h, "add": f"{a}x{b}", **X.compare(s, y, X.B0, [f"{a}x{b}"], ref_add=ref)})
    return table("A5_interactions", rows)


def A6(d):
    rows = []; y = "SUP_excess"
    for sn, s in samples(d).items():
        for c in ("m_ANB", "Pog_Ba_ant", "SIZE_PC1"):
            rows.append({"part": "B0 위", "sample": sn, "hyp": "앙와위 초과", "add": c, **X.compare(s, y, X.B0, [c])})
        for c in A2_FEATS+["tongue_c_ant", "res_tongue_c_ant"]:
            rows.append({"part": "기존 인자 위", "sample": sn, "hyp": "앙와위 초과", "add": c, **X.compare(s, y, X.B0, [c], ref_add=X.CONFIRMED)})
    return table("A6_supine_excess", rows)


if __name__ == "__main__":
    d = load(); todo = sys.argv[1:] or ["A1", "A2", "A3", "A5", "A6", "A4"]
    for k in todo:
        {"A1": lambda: A1(d), "A2": lambda: single(d, A2_FEATS, "A2_airway_shape"), "A3": lambda: single(d, A3_FEATS, "A3_residual"),
         "A4": lambda: A4(d), "A5": lambda: A5(d), "A6": lambda: A6(d)}[k]()
