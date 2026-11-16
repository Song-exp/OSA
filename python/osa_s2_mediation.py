"""S2c: 크기 축 → 기도 단면적 → AHI 매개분석 (설계는 실험기록 "S2c 매개분석 설계").
간접효과 a×b의 비모수 부트스트랩(2000회) 백분위 구간 + 상대성장 확인(크기 위에 단면적, 교차검증 ΔR²).
출력 Result_py/analysis/osa_experiments_v2_final/S2c_mediation/
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
import osa_experiments as oe
from osa_s2_size_check import prep

B = 2000
z = lambda v: (v-v.mean())/v.std()


def paths(s, y, m, cov):
    X = z(s.SIZE_PC1).to_numpy(); M = z(s[m]).to_numpy(); C = s[cov].to_numpy(float); Y = s[y].to_numpy(float)
    a = sm.OLS(M, sm.add_constant(np.c_[X, C])).fit().params[1]
    fb = sm.OLS(Y, sm.add_constant(np.c_[M, X, C])).fit(); b, cp = fb.params[1], fb.params[2]
    c = sm.OLS(Y, sm.add_constant(np.c_[X, C])).fit().params[1]
    return a, b, cp, c


def assert_paths():
    """총효과 = 직접 + 간접 (OLS 항등식)."""
    r = np.random.default_rng(1); n = 200; x = r.normal(size=n); m = .5*x+r.normal(size=n); y = .3*m+.2*x+r.normal(size=n)
    s = pd.DataFrame({"SIZE_PC1": x, "M": m, "Y": y, "k": r.normal(size=n)}); a, b, cp, c = paths(s, "Y", "M", ["k"])
    assert abs(c-(cp+a*b)) < 1e-8


if __name__ == "__main__":
    assert_paths()
    out = oe.R_/"analysis/osa_experiments_v2_final/S2c_mediation"; out.mkdir(parents=True, exist_ok=True)
    d = prep(); rng = np.random.default_rng(0); rows, rows2 = [], []
    for sn, s0 in (("main59", d[d.analysis_list.astype(bool)]), ("sens85", d)):
        for cov_name, cov in (("B0", oe.B0), ("B0+ANB", oe.B0+["m_ANB"])):
            for h, y in oe.H.items():
                for m in ("a_PAV_h", "a_PAV", "a_PMCA"):
                    s = s0.dropna(subset=[y, m, "SIZE_PC1"]+cov); a, b, cp, c = paths(s, y, m, cov); ind = []
                    for _ in range(B):
                        bs = s.sample(len(s), replace=True, random_state=rng.integers(1 << 31)); aa, bb, _, _ = paths(bs, y, m, cov); ind.append(aa*bb)
                    lo, hi = np.percentile(ind, [2.5, 97.5])
                    rows.append({"sample": sn, "cov": cov_name, "hyp": h, "mediator": m, "n": len(s), "a": a, "b": b, "indirect": a*b,
                                 "ind_lo": lo, "ind_hi": hi, "direct": cp, "total": c, "prop_mediated": a*b/c if c else np.nan})
                # 상대성장: 크기 위에 단면적
                for m in ("a_PAV_h", "a_PMCA"):
                    rows2.append({"sample": sn, "cov": cov_name, "hyp": h, "add": m, "ref": "+SIZE_PC1",
                                  **oe.compare(s0, y, oe.B0, [m], ref_add=cov[len(oe.B0):]+["SIZE_PC1"])})
    t = pd.DataFrame(rows); t.round(4).to_csv(out/"mediation.csv", index=False, encoding="utf-8-sig"); print(t.round(3).to_string(index=False))
    t2 = pd.DataFrame(rows2); t2.round(4).to_csv(out/"allometry.csv", index=False, encoding="utf-8-sig")
    print(t2[["sample", "cov", "hyp", "add", "n", "dR2", "dR2_lo", "dR2_hi", "F_p", "coef_std"]].round(3).to_string(index=False))
