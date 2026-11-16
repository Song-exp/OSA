"""형태 연구 Track B: 국소(Local) ROI 형태 벡터. 전역 SSM과 같은 대응점을 부위별로 나눠 부위마다 Procrustes + PCA.

ROI 정의: 템플릿 환자의 대응점을 ALI 랜드마크로 만든 기준점 중 가장 가까운 것에 배정 (엑셀·PSG 미사용, 한 번 고정).
  MAND: 턱끝(Pog·Me·Gn·B 평균) / 하악체(좌우 Gn–Go 중점) / 하악각(좌우 Go) / 하악지·과두(좌우 Go에서 위로 35 mm)
  MAX : 앞 치조(A) / 구개(ANS–PNS 중점) / 뒤쪽(PNS) / 안와저(좌우 Or)
  CB  : 앞 두개저(N–S 중점) / 터키안(S) / 사대(S–Ba 중점, Ba) / 측두골(좌우 Po)
  좌우 기준점은 한 ROI로 묶는다(양쪽을 함께 본 국소 형태).
ROI별 출력: PC1~3 점수(국소 모양, 크기 제거), 중심 크기(국소 크기), 머리 좌표계 중심 위치(상대 위치, relation).
주의: 여기 PCA는 탐색용 전체 표본 PCA다. 결과변수(AHI) 예측 평가 때는 교차검증 폴드 안에서 다시 한다(형태연구_계획 2장).
입력: Result_py/shape/<구조>/points.npz (shape_ssm.py), ALI 랜드마크. 출력: Result_py/shape/local/
사용: python shape_local_roi.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cbct_patient_pipeline import landmarks
from shape_ssm import gpa, pca

ROOT = Path(__file__).resolve().parents[1]
SHAPE = ROOT/"Result_py/shape"; OUT = SHAPE/"local"
NPC = 3
SUP = np.array([0, 0, 1.])


def anchors(struct, L):
    """{ROI 이름: [기준점 좌표...]} — L은 머리 좌표계(좌, 앞, 위) 랜드마크."""
    m = lambda *k: np.mean([L[x] for x in k], 0)
    if struct == "MAND":
        return {"symphysis": [m("Pog", "Me", "Gn", "B")], "body": [m("Gn", "RGo"), m("Gn", "LGo")],
                "gonial": [L["RGo"], L["LGo"]], "ramus_condyle": [L["RGo"]+35*SUP, L["LGo"]+35*SUP]}
    if struct == "MAX":
        return {"anterior_alveolar": [L["A"]], "palate": [m("ANS", "PNS")], "posterior": [L["PNS"]], "orbital_floor": [L["ROr"], L["LOr"]]}
    return {"anterior_base": [m("N", "S")], "sella": [L["S"]], "clivus": [m("S", "Ba"), L["Ba"]], "temporal": [L["RPo"], L["LPo"]]}


def roi_labels(struct, z):
    ids = list(z["ids"]); t = ids.index(int(z["template_id"])); tpl = z["frame"][t]
    P = landmarks(int(z["template_id"]))[0]; o, R = z["head_origin"][t], z["head_R"][t]
    L = {k: (v-o)@R.T for k, v in P.items()}
    gap = {k: float(np.min(np.linalg.norm(tpl-L[k], axis=1))) for k in {"MAND": ("Pog", "Me", "RGo"), "CB": ("S", "Ba"), "MAX": ("ANS", "PNS", "A")}[struct]}
    assert max(gap.values()) < 8, f"랜드마크와 대응점 좌표계가 어긋남: {gap}"    # 같은 좌표계면 랜드마크가 표면 가까이 있다
    A = anchors(struct, L); names = list(A)
    d = np.stack([np.min([np.linalg.norm(tpl-a, axis=1) for a in A[n]], 0) for n in names], 1)
    return np.array(names)[d.argmin(1)], names, gap


def main():
    OUT.mkdir(parents=True, exist_ok=True); wide, var, defs = None, [], {}
    for struct in [s for s in ("MAND", "CB", "MAX") if (SHAPE/s/"points.npz").exists()]:
        z = np.load(SHAPE/struct/"points.npz"); lab, names, gap = roi_labels(struct, z)
        ids = [f"P_{i}" for i in z["ids"]]; X = z["frame"]
        rv = SHAPE/struct/"review/qc_review.csv"
        use = np.ones(len(ids), bool)
        if rv.exists():
            q = pd.read_csv(rv, encoding="utf-8-sig").set_index("Patient")
            if "main_analysis" in q: use = q.main_analysis.reindex(ids).fillna(True).astype(bool).to_numpy()
        df = pd.DataFrame({"Patient": ids})
        for n in names:
            sel = lab == n; Y = X[:, sel]
            gp, size = gpa(Y); _, _, v, sc = pca(gp, use)
            for k in range(NPC): df[f"{struct}_{n}_PC{k+1}"] = sc[:, k]
            df[f"{struct}_{n}_size"] = size
            c = Y.mean(1); df[[f"{struct}_{n}_c{a}" for a in ("left", "ant", "sup")]] = c
            var += [{"struct": struct, "roi": n, "n_points": int(sel.sum()), "PC": k+1, "var_ratio": float(v[k]/v.sum())} for k in range(NPC)]
        defs[struct] = {"template_id": int(z["template_id"]), "points_per_roi": {n: int((lab == n).sum()) for n in names},
                        "landmark_to_surface_mm": gap, "pca_patients": int(use.sum()), "patients": len(ids)}
        np.save(OUT/f"{struct}_roi_labels.npy", lab)
        wide = df if wide is None else wide.merge(df, on="Patient", how="outer")
    wide.round(5).to_csv(OUT/"local_roi_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(var).round(4).to_csv(OUT/"local_roi_variance.csv", index=False, encoding="utf-8-sig")
    (OUT/"roi_definition.json").write_text(json.dumps(defs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(defs, ensure_ascii=False)); print(pd.DataFrame(var).pivot_table(index=["struct", "roi"], columns="PC", values="var_ratio").round(3).to_string())
    print("환자", wide.Patient.nunique(), "열", wide.shape[1]-1)


if __name__ == "__main__":
    main()
