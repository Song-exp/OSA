"""S2 검토 1·2 (2026-11-16): 골격 → PAV 관련성이 크기(기하) 결합인지, Eb 직접 검출 45명에서 '더 강한' 것이 의미 있는 차이인지.

1. 크기 보정: PAV는 S–PNS 평면 ~ Eb 평면 사이 부피라 머리가 크면 기둥이 길어진다.
   - 목표 a_PAV_h = PAV / 기둥 높이(PNS(원점) → Eb 수직 거리, mm) = 평균 단면적(mm²)
   - 목표 log a_PAV 에서 SIZE_PC1을 기준 모델에 넣은 뒤(ref_add) PLS·Pog_Ba·ANB가 더 설명하는지
2. 45명 vs 나머지 40명: OLS 비교 모델 ΔAdj-R²의 층화 부트스트랩 95% 구간(집단별, 차이), 피처×집단 교호작용 p.
   부트스트랩은 표본 안 OLS(교차검증 아님). PLS는 부트스트랩 표본마다 다시 적합.
입력: osa_experiments.data()(--source final_v2, 등급 ABCD). 출력: Result_py/analysis/osa_experiments_v2_final/S2b_size_check/
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import build_features as bf
import osa_experiments as oe

B = 1000
FEATS = ["SIZE_PC1", "Pog_Ba_ant", "m_ANB", "PLS"]


def prep():
    oe.SOURCE = "final_v2"; d = oe.data(); d = oe.airway_raw(d); oe.size_axis(d)
    d = d[d.restored.astype(bool)].copy()
    d["H_col"] = -d.c_Eb_sup                                     # PNS 원점 → Eb 아래로 거리(mm)
    d["a_PAV_h"] = d.a_PAV*1000/d.H_col                          # mL→mm³ / mm = 평균 단면적 mm²
    d["log_PAV"] = np.log(d.a_PAV)
    d["Eb_direct"] = (d.Eb_method == "vallecula").astype(float)
    return d


def part1(d):
    rows = []
    for sn, s in (("85명", d), ("Eb 직접 검출", d[d.Eb_direct == 1])):
        for y, ref in (("a_PAV", []), ("H_col", []), ("a_PAV_h", []), ("log_PAV", ["SIZE_PC1"]), ("log_PAV", ["SIZE_PC1", "H_col"])):
            for add in (["SIZE_PC1"], ["Pog_Ba_ant"], ["m_ANB"], ["PLS"]):
                if add[0] in ref: continue
                rows.append({"sample": sn, "target": y, "ref": "B0+"+"+".join(ref) if ref else "B0", "add": add[0], **oe.compare(s, y, oe.B0, add, ref_add=ref)})
    t = oe.fdr(rows, ("sample", "target", "ref")); return t


def score(s, feat):
    """피처 값(PLS면 s 전체로 적합한 점수)."""
    if feat != "PLS": return s[feat].to_numpy(float)
    X, Y = s[oe.SKEL_XY[0]], s[oe.SKEL_XY[1]]; _, sc = bf.fit_pls(X, Y, perms=0); return sc.to_numpy()[:, 0]


def dadj(s, y, feat):
    yv = s[y].to_numpy(float); A0 = s[oe.B0].to_numpy(float)
    f0 = sm.OLS(yv, sm.add_constant(A0)).fit(); f1 = sm.OLS(yv, sm.add_constant(np.c_[A0, score(s, feat)])).fit()
    return f1.rsquared_adj-f0.rsquared_adj


def part2(d, y="a_PAV"):
    rng = np.random.default_rng(0); rows = []
    for feat in FEATS:
        s = d.dropna(subset=[y, "Eb_method"]+oe.needed(oe.B0, [feat])); g1, g0 = s[s.Eb_direct == 1], s[s.Eb_direct == 0]
        est = {"all": dadj(s, y, feat), "direct": dadj(g1, y, feat), "estimated": dadj(g0, y, feat)}
        bs = {k: [] for k in ("all", "direct", "estimated", "diff")}
        for _ in range(B):
            b1, b0 = g1.sample(len(g1), replace=True, random_state=rng.integers(1 << 31)), g0.sample(len(g0), replace=True, random_state=rng.integers(1 << 31))
            try:
                a, c, e = dadj(pd.concat([b1, b0]), y, feat), dadj(b1, y, feat), dadj(b0, y, feat)
            except Exception: continue
            bs["all"].append(a); bs["direct"].append(c); bs["estimated"].append(e); bs["diff"].append(c-e)
        sc = score(s, feat); z = (sc-sc.mean())/sc.std()
        X = np.c_[s[oe.B0].to_numpy(float), z, s.Eb_direct, z*s.Eb_direct]
        p_int = sm.OLS(s[y].to_numpy(float), sm.add_constant(X)).fit().pvalues[-1]
        r = {"target": y, "feature": feat, "n_direct": len(g1), "n_estimated": len(g0), "p_interaction": p_int, "n_boot": len(bs["diff"])}
        for k in bs:
            v = np.array(bs[k]); r[f"{k}"] = est.get(k, est["direct"]-est["estimated"]); r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows)


def balance(d):
    cols = ["Age", "Male", "BMI", "SIZE_PC1", "Pog_Ba_ant", "m_ANB", "a_PAV", "H_col", "a_PAV_h"]
    g = d.groupby(d.Eb_direct.map({1: "direct", 0: "estimated"}))[cols]
    return pd.concat([g.mean().T.add_suffix("_mean"), g.count().T.add_suffix("_n")], axis=1)


if __name__ == "__main__":
    out = bf.R_/"analysis/osa_experiments_v2_final/S2b_size_check"; out.mkdir(parents=True, exist_ok=True)
    d = prep()
    print("SIZE_PC1 ~ H_col r =", round(d[["SIZE_PC1", "H_col"]].corr().iloc[0, 1], 3), "| a_PAV ~ H_col r =", round(d[["a_PAV", "H_col"]].corr().iloc[0, 1], 3))
    bal = balance(d); bal.round(3).to_csv(out/"group_balance.csv", encoding="utf-8-sig"); print(bal.round(2).to_string())
    t1 = part1(d); t1.round(4).to_csv(out/"part1_size_adjusted.csv", index=False, encoding="utf-8-sig")
    print(t1[["sample", "target", "ref", "add", "n", "dR2", "dR2_lo", "dR2_hi", "dAdjR2", "F_p", "q_BH", "coef_std"]].round(3).to_string(index=False))
    t2 = pd.concat([part2(d, "a_PAV"), part2(d, "a_PAV_h")]); t2.round(4).to_csv(out/"part2_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(t2.round(3).to_string(index=False))
