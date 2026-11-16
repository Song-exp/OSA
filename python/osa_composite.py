"""F 단계: 얼굴 구조 종합 지표 (사전 등록: 실험기록 "F 단계 사전 등록").
F1 전형성 거리(E 단계 F2 공동 PC1~10 마할라노비스 D², Procrustes 거리) / F2 성장 축(Y축, 얼굴 축 근사) / F3 Björk 합 항등식 확인.
입력: Result_py/analysis/shape_map/E_joint_axes/joint_axes.csv (osa_joint_axes.py 먼저). 출력: Result_py/analysis/shape_map/F_composite/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import osa_experiments as X
import osa_validate_holdout as VH
import osa_joint_axes as JA
import build_features as bf
from shape_ssm import gpa

OUT = X.R_/"analysis/shape_map/F_composite"; OUT.mkdir(parents=True, exist_ok=True)


def sang(a, b):
    """정중 시상면(앞, 위) 투영 두 벡터 사이 각(도)."""
    a, b = a[1:], b[1:]; return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b), -1, 1))))


def growth(d):
    rel = pd.read_csv(X.R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
    cond = pd.read_csv(bf.EXT/"condyle_all.csv", encoding="utf-8-sig").set_index("Patient"); rows = {}
    for p in d.index:
        q, _ = bf.coords(int(p[2:]), rel.loc[p].to_dict() if p in rel.index else {}, cond.loc[p] if p in cond.index else None)
        gn = (q["Pog"]+q["Me"])/2; go = (q["RGo"]+q["LGo"])/2; r = {"Yaxis": sang(q["N"]-q["S"], gn-q["S"]), "FacialAxis_proxy": sang(q["N"]-q["Ba"], gn-q["PNS"])}
        if "RSup" in q and np.isfinite(q["RSup"]).all() and np.isfinite(q["LSup"]).all():
            co = (q["RSup"]+q["LSup"])/2
            r["Bjork_sum"] = sang(q["N"]-q["S"], co-q["S"])+sang(q["S"]-co, go-co)+sang(co-go, q["Me"]-go)
        rows[p] = r
    return d.join(pd.DataFrame(rows).T.astype(float))


def atypicality(d):
    A = pd.read_csv(X.R_/"analysis/shape_map/E_joint_axes/joint_axes.csv", encoding="utf-8-sig", index_col=0)
    Z = A[[f"F2_PC{i}" for i in range(1, 11)]]; d["Atyp_D2"] = (Z**2).sum(1).reindex(d.index)        # PC 점수는 SD 1로 표준화돼 있음
    ids, C, _, _ = JA.combined(("MAND", "MAX", "CB")); G, _ = gpa(C); m = G.mean(0)
    d["Atyp_Procrustes"] = pd.Series(np.sqrt(((G-m)**2).sum((1, 2))), index=ids).reindex(d.index)
    return d


if __name__ == "__main__":
    d, f = VH.load_frozen(); d = growth(d); d = atypicality(d)
    chk = (d.Bjork_sum-(d.SN_MP+360)).abs(); print("Björk 합 − (SN_MP + 360) 최대 차이(도):", round(float(chk.max()), 3), "| 상관", round(d.Bjork_sum.corr(d.SN_MP), 3))
    IDX = ["Atyp_D2", "Atyp_Procrustes", "Yaxis", "FacialAxis_proxy"]
    s85 = d[d.restored.astype(bool)]; print(s85[IDX+["m_ANB", "Pog_Ba_ant", "SIZE_PC1", "SN_MP", "Male"]].corr(method="spearman").loc[IDX].round(2).to_string())
    S = {"main59": d[d.index.isin(f["discovery_ids"])], "sens85": s85}; rows = []
    for sn, s in S.items():
        for h, y in X.H.items():
            for c in IDX:
                for lab, ref in (("B0 위", []), ("확정 위", X.CONFIRMED), ("확정+크기 위", X.CONFIRMED+["SIZE_PC1"])):
                    rows.append({"part": lab, "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
    t = X.fdr(rows, ("sample", "hyp", "part")); t.round(4).to_csv(OUT/"F_results.csv", index=False, encoding="utf-8-sig")
    d[IDX+["Bjork_sum"]].round(3).to_csv(OUT/"F_values.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[["part", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]].round(3).to_string(index=False))
