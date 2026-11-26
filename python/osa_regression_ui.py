"""Run the OSA feature regression interface on the local computer."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser

import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm

from osa_feature_heatmap_v2 import BASE, R_, S2, STV, skeletal, table
from osa_regression_ct import atlas, feature_layers


from roi_centroid_pca import REGIONS

CLINICAL = {"Age": "나이(세)", "Male": "성별(남성 1, 여성 0)", "BMI": "체질량지수(kg/m²)"}
# Roh et al. (2025), Table 3, p < .05 for both AHI and supine AHI.
ROH_SIGNIFICANT = {"[기본] ANB", "[기본] TDI", "[기본] PMCA"}
BONES = {f"sk_{roi}": title for structure, title in
         (("MAX", "상악골"), ("MAND", "하악골"), ("CB", "두개저")) for roi in REGIONS[structure]}
OUTCOMES = {
    "logAHI": "log(1 + AHI)",
    "AHI": "AHI 원값",
    "logAHI_SUP": "log(1 + 앙와위 AHI)",
    "AHI_SUP": "앙와위 AHI 원값",
}
BASE_GROUPS = {
    "ANB": "악골 계측", "Nperp-A": "악골 계측", "Nperp-Pog": "악골 계측", "MP-FH": "악골 계측",
    "TDI": "치열·구개 계측", "PHI": "치열·구개 계측",
    "NCWR": "비강", "NAV": "비강", "PAV": "인두 기도", "PMCA": "인두 기도", "PAL": "인두 기도",
}


def predictor_group(key, domains, labels):
    if key in domains:
        return labels[domains[key]["d"]]
    if key.startswith("[기본] "):
        return BASE_GROUPS[key.removeprefix("[기본] ")]
    if key.startswith("soft_palate_"):
        return "연구개"
    if key.startswith("nasal_") or key == "NAV_ts":
        return "비강"
    if key.startswith("tongue_"):
        return "혀"
    return "인두 기도"


def load_data():
    base = table()
    bone, bone_descriptions, domains, labels = skeletal()
    verified = pd.read_csv(R_ / "psg_verified.csv", index_col="Patient")
    master = pd.read_csv(R_ / "phase1/patient_master.csv", index_col="Patient")
    verified_age = verified.V_Age.reindex(base.index)
    source_age = master.Age.reindex(base.index)
    age_restored = int((verified_age.isna() & source_age.notna()).sum())
    clinical = pd.DataFrame({
        "Age": verified_age.fillna(source_age),
        "Male": verified.V_Sex.map({"M": 1, "F": 0}),
        "BMI": master.BMI,
    }, index=base.index)
    features = base.join(bone).join(clinical)
    psg = pd.read_csv(R_ / "phase1/psg_restored_own.csv", encoding="utf-8-sig", index_col=0)
    outcomes = psg.loc[features.index, ["AHI_r", "AHI_SUP_r"]].rename(
        columns={"AHI_r": "AHI", "AHI_SUP_r": "AHI_SUP"}
    )
    outcomes = outcomes.where(outcomes >= 0)
    outcomes["logAHI"] = np.log1p(outcomes["AHI"])
    outcomes["logAHI_SUP"] = np.log1p(outcomes["AHI_SUP"])
    descriptions = {**{f"[기본] {key}": value for key, value in BASE.items()},
                    **CLINICAL, **S2, **STV, **bone_descriptions}
    if age_restored:
        descriptions["Age"] += f" · 원본 환자표에서 결측 {age_restored}명 보완"
    groups = {}
    for key in features:
        paper = key in CLINICAL or key.startswith("[기본] ")
        if key in CLINICAL:
            region = subgroup = stage = "환자 기본정보"
        elif key in domains:
            domain = domains[key]["d"]
            subgroup = labels[domain].removeprefix("골격·")
            if domain == "sk_layout":
                region = stage = "골격 전체 배치"
            else:
                region = BONES[domain]
                stage = "골격 부위별 크기" if domains[key]["c"] == "크기" else "골격 부위별 모양 PC"
        else:
            subgroup = predictor_group(key, domains, labels)
            region = {"비강": "기도", "인두 기도": "기도", "악골 계측": "악골·치열 계측",
                      "치열·구개 계측": "악골·치열 계측"}.get(subgroup, subgroup)
            if paper:
                stage = "논문 해부학적 계측"
            elif key in S2:
                stage = "인두·연구개 계측"
            else:
                stage = "기도·연조직 모양 PC" if "shapePC" in key else "기도·연조직 부피"
        display = {"Age": "나이 (Age)", "Male": "성별 (Male)"}.get(key, key.removeprefix("[기본] "))
        groups.setdefault(region, []).append({
            "id": key, "label": display + (" [논문]" if paper else " [파생]"),
            "subgroup": subgroup, "stage": stage, "description": descriptions[key],
            "n": int(features[key].notna().sum()), "layers": [] if key in CLINICAL else feature_layers(key, domains),
            "rohSignificant": key in ROH_SIGNIFICANT,
        })
    metadata = {
        "groups": [{"name": name, "features": values} for name, values in groups.items()],
        "outcomes": [{"id": key, "label": label} for key, label in OUTCOMES.items()],
        "n": len(features), "featureCount": len(features.columns),
    }
    return features, outcomes, metadata


def regress(features, outcomes, predictors, outcome):
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        raise ValueError("결과변수를 선택해 주세요.")
    if not isinstance(predictors, list) or not predictors:
        raise ValueError("피처를 하나 이상 선택해 주세요.")
    if any(not isinstance(key, str) or key not in features for key in predictors):
        raise ValueError("피처 목록에 없는 변수가 들어 있습니다.")
    if len(set(predictors)) != len(predictors):
        raise ValueError("같은 피처가 중복 선택됐습니다.")

    sample = features[predictors].join(outcomes[outcome]).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(sample)
    if n <= len(predictors) + 1:
        raise ValueError(f"완전한 자료는 {n}명입니다. 피처 {len(predictors)}개의 계수와 표준오차를 "
                         f"계산하려면 최소 {len(predictors) + 2}명이 필요합니다.")
    x = sample[predictors].to_numpy(dtype=float)
    y = sample[outcome].to_numpy(dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    constant = [name for name, value in zip(predictors, scales) if value == 0]
    if constant:
        raise ValueError("분석 표본에서 값이 일정한 피처가 있습니다. " + ", ".join(constant))
    if np.ptp(y) == 0:
        raise ValueError("분석 표본에서 결과변수 값이 일정합니다.")

    # Scale the design for a stable fit, then return coefficients in input units.
    design = sm.add_constant((x - means) / scales, has_constant="add")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("선택한 피처 사이에 선형 종속이 있어 개별 회귀계수를 계산할 수 없습니다. 일부 피처를 빼 주세요.")
    result = sm.OLS(y, design).fit()
    transform = np.diag(np.r_[1., 1 / scales])
    transform[0, 1:] = -means / scales
    coefficients = transform @ result.params
    errors = np.sqrt(np.maximum(0, np.diag(transform @ result.cov_params() @ transform.T)))
    critical = t.ppf(.975, result.df_resid)
    terms = []
    for name, coefficient, error in zip(["절편", *predictors], coefficients, errors):
        terms.append({
            "name": name, "b": float(coefficient), "se": float(error),
            "p": float(2 * t.sf(abs(coefficient / error), result.df_resid)) if error > 0 else None,
            "low": float(coefficient - critical * error), "high": float(coefficient + critical * error),
        })
    return {
        "outcome": OUTCOMES[outcome], "n": n, "missing": len(features) - n, "terms": terms,
        "r2": float(result.rsquared), "adjustedR2": float(result.rsquared_adj),
    }


def make_handler(features, outcomes, metadata, page):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status, data):
            self.send(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                      "application/json; charset=utf-8")

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/features":
                self.send_json(200, metadata)
            elif self.path == "/api/anatomy":
                try:
                    self.send_json(200, atlas())
                except (OSError, RuntimeError, ValueError) as error:
                    self.log_error("CBCT overlay generation: %s", error)
                    self.send_json(503, {"error": "CBCT 부위 이미지를 준비하지 못했습니다."})
            else:
                self.send_json(404, {"error": "페이지를 찾을 수 없습니다."})

        def do_POST(self):
            if self.path != "/api/regression":
                self.send_json(404, {"error": "페이지를 찾을 수 없습니다."})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32_768 or self.headers.get_content_type() != "application/json":
                    raise ValueError("회귀 요청 형식이 올바르지 않습니다.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("회귀 요청 형식이 올바르지 않습니다.")
                result = regress(features, outcomes, payload.get("predictors"), payload.get("outcome"))
            except (ValueError, UnicodeError, np.linalg.LinAlgError) as error:
                self.send_json(400, {"error": str(error)})
                return
            self.send_json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="Local port (0 chooses a free port)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    features, outcomes, metadata = load_data()
    page = Path(__file__).with_name("osa_regression_ui.html").read_bytes()
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(features, outcomes, metadata, page)) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"OSA regression: {url}", flush=True)
        if not args.no_browser:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
