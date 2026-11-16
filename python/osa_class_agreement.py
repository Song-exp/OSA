"""Roh 분류(시상·수직·횡·CSP)를 엑셀 값과 자동 값으로 각각 나눴을 때의 일치도.
기준(Roh 2025): Class I 0<ANB<4, II ANB>=4 / normo 25<=MP-FH<30, hyper >=30, hypo <25 / TSD TDI<-2.5, nonTSD TDI>-2
CSP2D = II+hyper+nonTSD, CSP3D = II+hyper+TSD, nonCSP = I+normo+nonTSD.
자동: 현재(jung FH, m_*) / 최종표(yao4 FH, 등급 A~D). 출력 Result_py/analysis/osa_experiments_v2/S1e_class_agreement/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
import osa_experiments as X


def classify(anb, mpfh, tdi):
    sag = np.select([anb >= 4, anb > 0], ["II", "I"], "III"); sag = np.where(np.isnan(anb), None, sag)
    ver = np.select([mpfh >= 30, mpfh >= 25], ["hyper", "normo"], "hypo"); ver = np.where(np.isnan(mpfh), None, ver)
    tr = np.select([tdi < -2.5, tdi > -2], ["TSD", "nonTSD"], "boundary"); tr = np.where(np.isnan(tdi), None, tr)
    ph = np.select([(sag == "II") & (ver == "hyper") & (tr == "TSD"), (sag == "II") & (ver == "hyper") & (tr == "nonTSD"),
                    (sag == "I") & (ver == "normo") & (tr == "nonTSD")], ["CSP3D", "CSP2D", "nonCSP"], "other")
    ph = np.where(pd.isna(sag) | pd.isna(ver) | pd.isna(tr), None, ph)
    return {"시상": sag, "수직": ver, "횡": tr, "CSP": ph}


assert list(classify(np.array([5., 2, -1]), np.array([31., 27, 20]), np.array([-3., -1, -2.2]))["CSP"]) == ["CSP3D", "nonCSP", "other"]

d = X.data(); fin = pd.read_csv(X.FINAL, encoding="utf-8-sig").set_index("Patient")
f = {k: fin[k].where(fin[k+"_grade"].isin(list("ABCD"))).reindex(d.index) for k in ("ANB", "MP-FH")}
OUT = X.R_/"analysis/osa_experiments_v2/S1e_class_agreement"; OUT.mkdir(parents=True, exist_ok=True)
SRC = {"엑셀": (d.x_ANB, d.x_MPFH, d.x_TDI), "자동(현재 jung)": (d.m_ANB, d.m_MPFH, d.m_TDI), "자동(최종 yao4)": (f["ANB"], f["MP-FH"], d.m_TDI)}
C = {k: {a: pd.Series(v, index=d.index) for a, v in classify(*(s.to_numpy(float) for s in vals)).items()} for k, vals in SRC.items()}

rows, tabs = [], []
for sn, s in X.samples(d).items():
    idx = s.index
    for auto in ("자동(현재 jung)", "자동(최종 yao4)"):
        for axis in ("시상", "수직", "횡", "CSP"):
            e, a = C["엑셀"][axis][idx], C[auto][axis][idx]; ok = e.notna() & a.notna(); e, a = e[ok], a[ok]
            rows.append({"sample": sn, "auto": auto, "axis": axis, "n": int(ok.sum()), "agree": (e == a).mean(), "kappa": cohen_kappa_score(e, a),
                         "excel_counts": e.value_counts().to_dict(), "auto_counts": a.value_counts().to_dict()})
            tabs.append(pd.crosstab(e.rename("엑셀"), a.rename("자동")).assign(sample=sn, auto=auto, axis=axis))
    # 경계 근처: 엑셀 MP-FH가 30° 또는 25°에서 자동–엑셀 차이 SD(약 3°) 안
    m = d.loc[idx, "x_MPFH"]
    for b in (30, 25):
        rows.append({"sample": sn, "auto": "-", "axis": f"엑셀 MP-FH {b}°±3° 안", "n": int(m.notna().sum()), "agree": ((m-b).abs() < 3).mean(),
                     "excel_counts": {"명": int(((m-b).abs() < 3).sum())}})
t = pd.DataFrame(rows); t.round(3).to_csv(OUT/"agreement.csv", index=False, encoding="utf-8-sig")
pd.concat(tabs).to_csv(OUT/"crosstabs.csv", encoding="utf-8-sig")
print(t.round(3).to_string(index=False))
for tb in tabs:
    if tb["sample"].iloc[0] == "main59" and tb["axis"].iloc[0] in ("수직", "CSP"): print("\n", tb.iloc[0]["auto"], tb.iloc[0]["axis"]); print(tb.drop(columns=["sample", "auto", "axis"]))
