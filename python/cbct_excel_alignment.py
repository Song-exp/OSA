"""엑셀 계측값과의 오차를 줄이는 방향의 복원값 정렬. 엑셀을 기준값으로 보되, 의심 엑셀 값은 학습에서 뺀다.

1) 엑셀 기준값 점검: 정확히 0인 값(입력 누락 의심), 지표별 오차 이상치(robust z > 3.5) → 학습 제외, 검토 목록
2) 지표별 모델 비교 (85명 반복 교차검증, 5겹 × 20회):
     M0 기본 정의 그대로 / M1 기본 정의의 이상치에 강한 직선 환산(Huber) / M2 계산 가능한 변형 전부로 엑셀을 예측(Ridge, 안쪽 CV로 강도 선택)
   채택 규칙: 더 복잡한 모델은 바깥 CV MAE가 5% 이상 줄고, 반복의 80% 이상에서 나을 때만. 아니면 단순한 쪽을 유지.
3) 채택 모델을 학습 제외가 아닌 환자 전체로 맞춰 85명 값을 만든다. 보고 오차는 교차검증(학습에 안 쓴 환자) 값이다.
입력: restore_ext/v1/compare_all.csv, experiments/stage2_values.csv·stage4_values.csv, 동결·신규 대안 CSV
출력: restore_ext/v1/alignment/ (excel_flags.csv, cv_results.csv, choices.json, aligned_metrics.csv, README.md)
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cbct_accuracy_experiments as X
from cbct_patient_pipeline import save

EXT = X.EXT; EXP = X.EXP; DEST = EXT/"alignment"
DEFAULT = {"ANB": "jung", "Nperp-A": "jung", "Nperp-Pog": "jung", "MP-FH": "jung",
           **{m: cols[0] for m, (_, cols, _) in X.ALT.items()}, "PAL": "PAL_mm"}
MARGIN, WIN, REPEATS, FOLDS, ZCUT = .05, .80, 20, 5, 3.5


def features():
    """{지표: DataFrame(Patient, excel, 후보 열...)} — 85명, 계산 가능한 변형 전부. 첫 열 = 기본 정의."""
    out = {}
    s2 = X.pd.read_csv(EXP/"stage2_values.csv", encoding="utf-8-sig")
    s4 = X.pd.read_csv(EXP/"stage4_values.csv", encoding="utf-8-sig")
    for m in X.SKEL:
        w = s2[s2.metric == m].pivot(index="Patient", columns="variant", values="value")
        w["tta"] = s4[s4.metric == m].set_index("Patient").tta
        w["excel"] = s2[s2.metric == m].drop_duplicates("Patient").set_index("Patient").excel
        out[m] = w[["jung"]+[c for c in w if c not in ("jung", "excel")]+["excel"]].reset_index()
    dev, hold = X.alt_table(False), X.alt_table(True)
    for m in X.ALT: out[m] = pd.concat([dev[m], hold[m]], ignore_index=True)
    ca = pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig"); p = ca[ca.metric == "PAL"]
    out["PAL"] = pd.DataFrame({"Patient": p.Patient, "PAL_mm": p.restored, "excel": p.excel})
    for m, t in out.items(): assert t.Patient.nunique() == len(t) == 85, (m, len(t))
    return out


def flags(F):
    rows = []
    for m, t in F.items():
        e = t[DEFAULT[m]]-t.excel; ok = e.notna()
        med = e[ok].median(); mad = 1.4826*(e[ok]-med).abs().median(); z = (e-med)/mad
        for _, r in t.assign(z=z).iterrows():
            if pd.isna(r.excel) or pd.isna(r[DEFAULT[m]]): continue
            why = "엑셀 값 0 (입력 누락 의심)" if r.excel == 0 and m in ("Nperp-A", "Nperp-Pog", "TDI") else ("오차 이상치" if abs(r.z) > ZCUT else "")
            if why: rows.append(dict(metric=m, Patient=r.Patient, excel=r.excel, restored=r[DEFAULT[m]], robust_z=r.z, reason=why))
    return pd.DataFrame(rows)


def models(cols):
    m = {"M0": None, "M1": ([cols[0]], make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, max_iter=2000)))}    # 부피 단위(수만)에서도 수렴하게 표준화
    if len(cols) > 1: m["M2"] = (cols, make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)))
    return m


def predict(spec, cols, tr, te):
    if spec is None: return te[cols[0]].to_numpy(float)
    use, est = spec; Xtr = tr[use].to_numpy(float); Xte = te[use].to_numpy(float)
    est.fit(Xtr, tr.excel.to_numpy(float)); return est.predict(Xte)


def complete(t, cols):
    """변형 일부가 빈 환자는 기본 정의 값으로 채운다 (M2 입력용). 기본 정의도 비면 제외."""
    t = t.dropna(subset=[cols[0]]).copy()
    for c in cols[1:]: t[c] = t[c].fillna(t[cols[0]])
    return t


def cv(t, cols):
    t = complete(t, cols).dropna(subset=["excel"]); specs = models(cols); res = {k: [] for k in specs}
    for rep in range(REPEATS):
        err = {k: np.full(len(t), np.nan) for k in specs}
        for tr, te in KFold(FOLDS, shuffle=True, random_state=rep).split(t):
            for k, s in specs.items():
                err[k][te] = np.abs(predict(s, cols, t.iloc[tr], t.iloc[te])-t.excel.iloc[te].to_numpy(float))
        for k in specs: res[k].append(float(np.mean(err[k])))
    return {k: np.array(v) for k, v in res.items()}, len(t)


def choose(res):
    """M0 → M1 → M2 순으로, 앞 모델보다 확실히 나을 때만 넘어간다."""
    cur = "M0"
    for nxt in ("M1", "M2"):
        if nxt not in res: continue
        gain = 1-res[nxt].mean()/res[cur].mean(); wins = float(np.mean(res[nxt] < res[cur]))
        if gain >= MARGIN and wins >= WIN: cur = nxt
    return cur


def main():
    DEST.mkdir(parents=True, exist_ok=True); F = features(); fl = flags(F)
    fl.round(3).to_csv(DEST/"excel_flags.csv", index=False, encoding="utf-8-sig")
    excl = set(zip(fl.metric, fl.Patient)); rows, choice, aligned = [], {}, []
    for m, t in F.items():
        cols = [c for c in t if c not in ("Patient", "excel")]
        train = t[[(m, p) not in excl for p in t.Patient]]
        res, n = cv(train, cols); pick = choose(res)
        for k, v in res.items(): rows.append(dict(metric=m, model=k, n_train=n, MAE_mean=v.mean(), MAE_sd=v.std(), win_vs_M0=float(np.mean(v < res["M0"]))))
        spec = models(cols)[pick]; full = complete(train, cols).dropna(subset=["excel"]); allp = complete(t, cols)
        pred = predict(spec, cols, full, allp)
        coef = None
        if spec is not None:
            est = spec[1]; coef = (dict(zip(spec[0], est[-1].coef_ / est[0].scale_)) if hasattr(est, "steps") else {"slope": float(est.coef_[0]), "intercept": float(est.intercept_)})
        choice[m] = {"model": pick, "inputs": cols if pick == "M2" else [cols[0]], "n_train": n, "n_excluded": int(sum(fl.metric == m)),
                     "MAE_cv": {k: round(float(v.mean()), 4) for k, v in res.items()}, "coef": coef}
        aligned.append(pd.DataFrame({"Patient": allp.Patient, "metric": m, "default": allp[cols[0]], "aligned": pred,
                                     "excel": allp.excel, "excel_flag": [(m, p) in excl for p in allp.Patient]}))
    pd.DataFrame(rows).round(4).to_csv(DEST/"cv_results.csv", index=False, encoding="utf-8-sig")
    save(DEST/"choices.json", {"rule": f"복잡한 모델은 CV MAE {MARGIN:.0%} 이상 감소 & 반복 {WIN:.0%} 이상 우세일 때만", "metrics": choice})
    a = pd.concat(aligned, ignore_index=True); a.round(4).to_csv(DEST/"aligned_metrics.csv", index=False, encoding="utf-8-sig")
    (DEST/"README.md").write_text(f"""# 엑셀 정렬 결과

- excel_flags.csv: 학습에서 뺀 엑셀 값 (0 입력 의심 / 오차 이상치 robust z>{ZCUT}). 원인 확인 전이며 엑셀 오류로 확정한 것이 아님.
- cv_results.csv: 지표별 M0(기본 정의)·M1(직선 환산, Huber)·M2(변형 전부 Ridge)의 반복 교차검증 MAE ({FOLDS}겹 × {REPEATS}회, 85명 중 학습 가능 환자).
- choices.json: 채택 모델과 계수. 규칙: 복잡한 모델은 CV MAE {MARGIN:.0%} 이상 감소 & 반복의 {WIN:.0%} 이상에서 우세할 때만.
- aligned_metrics.csv: 85명 × 지표의 기본 정의 값(default)과 정렬 값(aligned). aligned는 엑셀 정의 쪽으로 옮긴 값이며, 학습 환자에서는 표본 안 값이다. 성능은 cv_results를 본다.
""", encoding="utf-8")
    t = pd.DataFrame(rows).pivot(index="metric", columns="model", values="MAE_mean").round(3)
    t["채택"] = [choice[m]["model"] for m in t.index]; t["학습 n"] = [choice[m]["n_train"] for m in t.index]; t["제외"] = [choice[m]["n_excluded"] for m in t.index]
    print(t.to_string()); print("엑셀 점검 표시:", len(fl), "건")


if __name__ == "__main__":
    main()
