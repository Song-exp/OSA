"""분석 집합 80 / 77 / 62명의 분포 비교 (중앙값 [IQR], 남성 비율, 중증도 구성)와, 빠진 환자 vs 남은 환자.
모형을 다시 맞추지 않고 분포만 봄. cohort_n.py 다음에 실행.
결과: Result_py/phase1/cohort_compare.csv, cohort_dropped.csv
"""
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Result_py/phase1"
VARS = ["Age", "BMI", "AHI", "ODI", "LSaO2", "ANB", "MPFH", "TDI", "PMCA", "PAL", "GapDays"]


def med_iqr(x):
    x = x.dropna()
    return f"{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]" if len(x) else ""


def main():
    m = pd.read_csv(R / "patient_master.csv", index_col="Patient", encoding="utf-8-sig")
    e = pd.read_csv(R / "cohort_eligibility.csv", index_col=0, encoding="utf-8-sig").astype(bool)
    m["GapDays"] = m.GapDays.abs()
    base = e.Valid_Main & e.CBCT_processed
    sets = {"기본 (주 분석 ∩ CBCT)": base, "∩ PSG 출처 확인": base & e.PSG_source_confirmed,
            "∩ 출처 확인 ∩ 간격 1년 이내": base & e.PSG_source_confirmed & e.Gap_le_1y}
    rows = {}
    for name, s in sets.items():
        d = m[s]
        col = {"N": str(len(d)), "남성 %": f"{100 * d.Male.mean():.0f}"}
        col.update({v: med_iqr(d[v]) for v in VARS})
        col.update({f"중증도 {k}": str(n) for k, n in d.Severity.value_counts().sort_index().items()})
        rows[name] = col
    t = pd.DataFrame(rows)
    t.to_csv(R / "cohort_compare.csv", encoding="utf-8-sig")
    print(t.to_string())

    # 62명 집합에서 빠진 18명 vs 남은 62명 (기본 80명 안에서). 탐색용 p값, 다중비교 보정 없음
    keep = sets["∩ 출처 확인 ∩ 간격 1년 이내"][base]
    d = m[base]
    out = []
    for v in VARS:
        a, b = d.loc[keep, v].dropna(), d.loc[~keep, v].dropna()
        out.append({"var": v, "남은 62": med_iqr(a), "빠진 18": med_iqr(b), "p (Mann-Whitney)": round(mannwhitneyu(a, b).pvalue, 3)})
    tab = pd.crosstab(~keep, d.Male)
    out.append({"var": "남성 %", "남은 62": f"{100 * d.Male[keep].mean():.0f}", "빠진 18": f"{100 * d.Male[~keep].mean():.0f}",
                "p (Mann-Whitney)": round(fisher_exact(tab)[1], 3)})
    o = pd.DataFrame(out)
    o.to_csv(R / "cohort_dropped.csv", index=False, encoding="utf-8-sig")
    print("\n빠진 18명 vs 남은 62명 (남성은 Fisher):\n" + o.to_string(index=False))


if __name__ == "__main__":
    main()
