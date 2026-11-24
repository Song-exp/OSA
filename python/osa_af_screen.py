"""Screen A–F measurements one at a time and build the aggregate-only Sites report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import osa_all83 as cohort
import osa_feature_guide as guide
import osa_newfactor as models
from osa_var_detail import detail

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/analysis/all83/af_screen"
SITE = ROOT / "sites/osa-af-association/dist"
OUTCOMES = {"logAHI": "전체 AHI", "logAHI_SUP": "앙와위 AHI"}
GROUPS = {key: guide.SRC[key][0] for key in "ABCDEF"}
# These are calculated from other measurements and kept for the later derived-variable pass.
DERIVED = {"SoftLoad_MMV", "tongue_MMVhull", "tongue_MMV"}


def variables(data):
    """Use the same measurement universe as the feature-definition report."""
    names = set(models.X11.values()) | set(models.M11.values()) | set(models.STEP2) | set(models.STEP3) | set(models.step4(data))
    names.discard("a_PAV_mm3")
    names.add("a_PAV")
    catalog = cohort.catalog()
    for name in sorted(names):
        group, _ = guide.where(name)
        if group not in GROUPS or name in DERIVED:
            continue
        description = (detail(name) or catalog.get(name) or (name, ""))[0]
        yield group, name, description


def screen(data):
    rows = []
    for group, name, description in variables(data):
        for outcome in OUTCOMES:
            row = {"group": group, "variable": name, "description": description, "outcome": outcome,
                   "n": 0, "status": "분석 열 없음", "b": None, "beta": None, "low": None,
                   "high": None, "p": None, "r2_base": None, "delta_r2": None}
            rows.append(row)
            if name not in data:
                continue
            sample = data[[outcome, *cohort.B0, name]].apply(pd.to_numeric, errors="coerce")
            sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
            row["n"] = len(sample)
            if len(sample) <= len(cohort.B0) + 2:
                row["status"] = "분석 인원 부족"
                continue
            y = sample[outcome]
            x = sm.add_constant(sample[[*cohort.B0, name]], has_constant="add")
            if y.nunique() < 2 or np.linalg.matrix_rank(x) < x.shape[1]:
                row["status"] = "변동 부족 또는 완전 공선성"
                continue
            base = sm.OLS(y, x.drop(columns=name)).fit()
            fit = sm.OLS(y, x).fit()
            scale = sample[name].std() / y.std()
            low, high = fit.conf_int().loc[name] * scale
            p = float(fit.pvalues[name])
            if not np.isfinite([p, low, high, fit.rsquared]).all():
                row["status"] = "계수 검정 불가"
                continue
            assert np.isclose(p, fit.compare_f_test(base)[1], atol=1e-10)
            row.update(status="계산 완료", b=float(fit.params[name]), beta=float(fit.params[name]*scale),
                       low=float(low), high=float(high), p=p, r2_base=float(base.rsquared),
                       delta_r2=max(0., float(fit.rsquared-base.rsquared)))
    result = pd.DataFrame(rows)
    for outcome in OUTCOMES:
        tested = result.outcome.eq(outcome) & result.p.notna()
        result.loc[tested, "q"] = multipletests(result.loc[tested, "p"], method="fdr_bh")[1]
    return result.sort_values(["group", "variable", "outcome"])


def main():
    data = cohort.data()
    assert len(data) == 83 and data.index.is_unique
    result = screen(data)
    OUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    result.to_csv(SITE / "results.csv", index=False, encoding="utf-8-sig")
    payload = {"cohort": len(data), "groups": GROUPS, "outcomes": OUTCOMES,
               "results": json.loads(result.to_json(orient="records", double_precision=15))}
    template = (ROOT / "sites/osa_af_report.html").read_text(encoding="utf-8")
    (SITE / "index.html").write_text(template.replace("/* REPORT_DATA */", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")
    for key in OUTCOMES:
        d = result[result.outcome == key]
        print(key, "variables", len(d), "tested", d.p.count(), "p<.05", d.p.lt(.05).sum(),
              "q<.05", d.q.lt(.05).sum(), "n tested", d.loc[d.p.notna(), "n"].agg(["min", "max"]).tolist())
        print(d.nsmallest(10, "p")[["group", "variable", "n", "beta", "p", "q", "delta_r2"]].to_string(index=False))
    print("By group:", result.drop_duplicates("variable").groupby("group").size().to_dict())
    print("Untested:", result[result.p.isna()][["variable", "outcome", "n", "status"]].to_dict("records"))


if __name__ == "__main__":
    main()
