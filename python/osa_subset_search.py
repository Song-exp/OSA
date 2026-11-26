"""회귀 인자 조합 전수 탐색(조합 크기 3 이하). Y = log(AHI_r + 1). 같은 환자 집합에서 모든 조합을 OLS로 적합한다.

버전 A: Y ~ 나이 + 성별 + BMI + [조합] (보정 변수 고정) / 버전 B: Y ~ [조합]
후보(2026-11-26 사용자 결정): osa_feature_heatmap_v2 60개 피처를 |r| ≥ 0.7 묶음별 대표 2개로 축약(파생변수 우선, 묶음 안 평균 |r| 순)하고,
  엑셀·자동 같은 지표 쌍(PAV·PAL·NAV)은 파생변수만, 혀 부피는 도메인상 유지 → 51개. Nperp-A(엑셀 0값 의심 17명)는 공통 환자 확보를 위해 제외 → 50개.
순위: 반복 교차검증 RMSE(5겹 × 20회, 조합끼리 같은 분할) 주 기준, BIC 보조. 최종 = 1-SE 규칙(최소 RMSE + SE 이내에서 k 최소, 같으면 BIC 낮은 쪽).
주의: 최종 조합의 p값은 조합 선택 뒤 같은 자료로 계산한 값이라 실제보다 작게 나온다(탐색적). 중첩 교차검증·부트스트랩 안정성은 아직 하지 않았다.
출력: Result_py/analysis/subset_search/ (all_subsets_{A,B}.csv, top_by_k.csv, final_choice.json)
  --excel-fill: 자동 추출 결측을 같은 지표 엑셀 값(PAV_fh←PAV, NAV_ts←NAV, H_col←PAL)으로, 검증 나이 결측을 엑셀 나이로 채움 → subset_search/excel_fill/
"""
from itertools import combinations
import json
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

import osa_feature_heatmap_v2 as H

DEST = H.R_/"analysis/subset_search"; KMAX, SEED = 3, 2026
DROP = ["터키안_크기", "구개_크기", "하악각_크기", "안와저_크기", "하악지·과두_크기", "앞 치조부_크기",   # 0.7 묶음 대표 2개 축약
        "[기본] PAV", "[기본] PAL", "[기본] NAV",                                                    # 같은 지표 쌍은 파생변수만
        "[기본] Nperp-A"]                                                                         # 0값 의심 17명


EXCEL_FILL = {"PAV_fh": "[기본] PAV", "NAV_ts": "[기본] NAV", "H_col": "[기본] PAL"}    # --excel-fill: 자동 추출 결측을 같은 지표의 엑셀 값으로


def data(excel_fill=False):
    d = H.table(); d = d.join(H.skeletal()[0].reindex(d.index))
    if excel_fill:
        for a, b in EXCEL_FILL.items(): d[a] = d[a].fillna(d[b])
    o = pd.read_csv(H.R_/"phase1/psg_restored_own.csv", encoding="utf-8-sig", index_col=0)
    v = pd.read_csv(H.R_/"psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
    m = pd.read_csv(H.R_/"phase1/patient_master.csv", encoding="utf-8-sig").set_index("Patient")
    assert set(v.V_Sex.dropna()) <= {"M", "F"}, set(v.V_Sex.dropna())
    age = v.V_Age.fillna(m.Age) if excel_fill else v.V_Age                                     # 검증 나이 결측 → 엑셀 나이
    cov = pd.DataFrame({"Age": age, "Male": (v.V_Sex == "M").astype(float).where(v.V_Sex.notna()), "BMI": m.BMI}).reindex(d.index)
    y = np.log1p(o.AHI_r.reindex(d.index)).rename("logAHI")
    cands = [c for c in d.columns if c not in DROP]
    assert len(cands) == 50, len(cands)
    full = pd.concat([y, cov, d[cands]], axis=1).dropna()
    return full, cands


def ols_stats(X, y):
    n, p = X.shape; b, *_ = np.linalg.lstsq(X, y, rcond=None); rss = float(((y-X@b)**2).sum())
    tss = float(((y-y.mean())**2).sum()); r2 = 1-rss/tss
    ll = -n/2*(np.log(2*np.pi*rss/n)+1)
    return r2, 1-(1-r2)*(n-1)/(n-p), -2*ll+2*p, -2*ll+np.log(n)*p


def cv_rmse(X, y, folds):
    e = []
    for tr, te in folds:
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); e.append(np.sqrt(np.mean((y[te]-X[te]@b)**2)))
    e = np.array(e); return e.mean(), e.std(ddof=1)/np.sqrt(len(e))


def search(full, cands, fixed):
    y = full.logAHI.to_numpy(); Z = full[cands].to_numpy(); F = np.c_[np.ones(len(full)), full[fixed].to_numpy()]
    folds = list(RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED).split(Z))
    base = cv_rmse(F, y, folds); rows = []
    for k in range(1, KMAX+1):
        for combo in combinations(range(len(cands)), k):
            X = np.c_[F, Z[:, combo]]; r2, ar2, aic, bic = ols_stats(X, y); rm, se = cv_rmse(X, y, folds)
            mv = max(vif(X, F.shape[1]+i) for i in range(k)) if k > 1 else 1.0
            rows.append({"vars": " + ".join(cands[i] for i in combo), "k": k, "R2": r2, "adjR2": ar2, "AIC": aic, "BIC": bic, "cv_rmse": rm, "cv_se": se, "max_vif": mv})
    t = pd.DataFrame(rows); t["vif_flag"] = t.max_vif > 5
    return t.sort_values("cv_rmse").reset_index(drop=True), base


def final(t, full, fixed):
    ok = t[~t.vif_flag]; best = ok.iloc[0]; band = ok[ok.cv_rmse <= best.cv_rmse+best.cv_se]
    pick = band.sort_values(["k", "BIC"]).iloc[0]; cols = pick.vars.split(" + ")
    Xs = (full[fixed+cols]-full[fixed+cols].mean())/full[fixed+cols].std(); ys = (full.logAHI-full.logAHI.mean())/full.logAHI.std()
    f = sm.OLS(full.logAHI, sm.add_constant(full[fixed+cols])).fit(); fb = sm.OLS(ys, sm.add_constant(Xs)).fit()
    X = sm.add_constant(full[fixed+cols]).to_numpy()
    terms = {c: {"B": float(f.params[c]), "beta": float(fb.params[c]), "p": float(f.pvalues[c]), "VIF": float(vif(X, i+1))} for i, c in enumerate(fixed+cols)}
    return {"vars": cols, "k": int(pick.k), "cv_rmse": float(pick.cv_rmse), "cv_se": float(pick.cv_se), "BIC": float(pick.BIC), "adjR2": float(pick.adjR2),
            "best_rmse_model": best.vars, "best_rmse": float(best.cv_rmse), "n_within_1se": int(len(band)), "terms": terms}


def main():
    global DEST
    import sys
    fill = "--excel-fill" in sys.argv
    if fill: DEST = DEST/"excel_fill"
    DEST.mkdir(parents=True, exist_ok=True); t0 = time.time()
    full, cands = data(fill); print(f"공통 환자 {len(full)}명, 후보 {len(cands)}개, 엑셀 채움 {fill}", flush=True)
    out, tops = {"n": len(full), "candidates": cands, "seed": SEED, "kmax": KMAX, "cv": "RepeatedKFold 5x20", "time": time.strftime("%Y-%m-%d %H:%M")}, []
    for ver, fixed in (("A", ["Age", "Male", "BMI"]), ("B", [])):
        t, base = search(full, cands, fixed); t.round(5).to_csv(DEST/f"all_subsets_{ver}.csv", index=False, encoding="utf-8-sig")
        fc = final(t, full, fixed); fc["baseline_cv_rmse"] = float(base[0]); out[ver] = fc
        tops.append(t[~t.vif_flag].groupby("k").head(10).assign(version=ver))
        print(f"[{ver}] 기준선 RMSE {base[0]:.4f} | 최소 RMSE {fc['best_rmse']:.4f} ({fc['best_rmse_model']}) | 1-SE 선택 k={fc['k']}: {' + '.join(fc['vars'])} "
              f"RMSE {fc['cv_rmse']:.4f} adjR2 {fc['adjR2']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    pd.concat(tops).round(5).to_csv(DEST/"top_by_k.csv", index=False, encoding="utf-8-sig")
    (DEST/"final_choice.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
