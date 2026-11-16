"""G 단계: 수직 배치 대표 지표 VERT_Z (사전 등록: 실험기록 "G 단계 사전 등록").
VERT_Z = z(MAND_rPC4), z(LAFH_ratio), z(F1_PC4)의 평균(부호는 LAFH_ratio와 양의 상관이 되게). 85명 기준 표준화.
값을 Result_py/analysis/validation/frozen_vertical.csv에 동결(이미 있으면 거부). 발굴 표본 기술 검정 출력.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import osa_experiments as X
import osa_validate_holdout as VH
import osa_residual_shape as RS

OUT = X.R_/"analysis/validation"; FV = OUT/"frozen_vertical.csv"
COMP = ["MAND_rPC4", "LAFH_ratio", "F1_PC4"]


def build(d):
    sc, *_ = RS.residual_pcs(d, "MAND"); d = d.join(sc)
    d = d.join(pd.read_csv(X.R_/"analysis/shape_map/E_joint_axes/joint_axes.csv", encoding="utf-8-sig", index_col=0)[["F1_PC4"]])
    s = d[d.restored.astype(bool)]; Z = pd.DataFrame(index=d.index); info = []
    for c in COMP:
        sign = np.sign(spearmanr(s[c], s.LAFH_ratio, nan_policy="omit")[0]) if c != "LAFH_ratio" else 1.
        Z[c] = sign*(d[c]-s[c].mean())/s[c].std(); info.append({"component": c, "sign": sign, "mean": s[c].mean(), "sd": s[c].std()})
    d["VERT_Z"] = Z[COMP].mean(axis=1, skipna=False)
    return d, Z, pd.DataFrame(info)


def alpha(Z):
    k = Z.shape[1]; return k/(k-1)*(1-Z.var().sum()/Z.sum(1).var())


if __name__ == "__main__":
    d, f = VH.load_frozen(); d, Z, info = build(d); s = d[d.restored.astype(bool)]
    zc = Z.loc[s.index].dropna()
    print(info.round(3).to_string(index=False)); print(zc.corr(method="spearman").round(2).to_string()); print("크론바흐 α", round(alpha(zc), 3), "n", len(zc))
    print(s[["VERT_Z", "m_ANB", "Pog_Ba_ant", "SIZE_PC1", "Male", "BMI", "Age"]].corr(method="spearman").loc["VERT_Z"].round(2).to_string())
    if not FV.exists():
        d.loc[d.restored.astype(bool), ["VERT_Z"]+COMP].round(5).to_csv(FV, encoding="utf-8-sig"); print("동결:", FV)
    else: print("frozen_vertical.csv 이미 있음 — 동결값 유지")
    rows = []
    for sn, ss in (("main59", d[d.index.isin(f["discovery_ids"])]), ("sens85", s)):
        for h, y in X.H.items():
            for lab, ref in (("B0 위", []), ("확정 위", X.CONFIRMED), ("확정+크기 위", X.CONFIRMED+["SIZE_PC1"])):
                rows.append({"part": lab, "sample": sn, "hyp": h, **X.compare(ss, y, X.B0, ["VERT_Z"], ref_add=ref)})
    t = pd.DataFrame(rows); t.round(4).to_csv(X.R_/"analysis/shape_map/G_vertical_index.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200); print(t[["part", "sample", "hyp", "n", "dR2", "share_better", "F_p", "coef_std"]].round(3).to_string(index=False))
