"""E 단계: AMASSS 구조 수치화 축 — 구조(MAND·MAX·CB) × 종류(크기·위치·모양) 블록별 PCA (사전 등록: 실험기록 "E 단계 사전 등록").
크기 = 부위 중심 크기 4개 → PC1~2 / 위치 = 부위 중심 앞뒤·위아래 8개(정중 좌표계) → PC1~3 / 모양 = 구조 GPA → PC1~3.
축을 만든 뒤 확정 인자 위·확정 인자 + SIZE_PC1(동결) 위에서 AHI 검정.
출력: Result_py/analysis/shape_map/E_face_axes/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import osa_experiments as X
import osa_residual_shape as RS
import osa_validate_holdout as VH
from shape_local_roi import roi_labels
from shape_ssm import gpa

OUT = X.R_/"analysis/shape_map/E_face_axes"; OUT.mkdir(parents=True, exist_ok=True)
NK = {"size": 2, "pos": 3, "shape": 3}
KR = {"symphysis": "턱끝", "body": "하악체", "gonial": "하악각", "ramus_condyle": "하악지·과두", "anterior_alveolar": "상악 앞치조", "palate": "구개",
      "posterior": "상악 후방", "orbital_floor": "안와저", "anterior_base": "앞 두개저", "sella": "터키안장", "clivus": "사대", "temporal": "측두"}


def pca_block(M, k):
    """M: DataFrame(완전 사례). 표준화 PCA → 점수(SD 1), 적재량, 설명 분산."""
    Z = (M-M.mean())/M.std(); u, s, vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    for i in range(k):                                                # 부호: 적재량 절대값 최대 항목이 양수가 되게
        if vt[i, np.argmax(np.abs(vt[i]))] < 0: vt[i] *= -1
    sc = Z.to_numpy()@vt[:k].T; sc /= sc.std(0, ddof=1)
    return sc, vt[:k], (s**2/(s**2).sum())[:k]


def structure_axes(struct):
    ids, S, use = RS.load(struct); z = np.load(X.R_/f"shape/{struct}/points.npz"); lab, names, _ = roi_labels(struct, z)
    ids = np.array(ids)[use]; S = S[use]; feats = {}
    for r in names:
        P = S[:, lab == r]; c = P.mean(1)
        feats[f"{r}_size"] = np.sqrt(((P-c[:, None])**2).sum((1, 2))); feats[f"{r}_ant"] = c[:, 1]; feats[f"{r}_sup"] = c[:, 2]
    F = pd.DataFrame(feats, index=ids); out, load = pd.DataFrame(index=ids), []
    for kind, cols in (("size", [f"{r}_size" for r in names]), ("pos", [f"{r}_{a}" for r in names for a in ("ant", "sup")])):
        sc, vt, var = pca_block(F[cols], NK[kind])
        for i in range(NK[kind]):
            nm = f"AX_{struct}_{kind}_PC{i+1}"; out[nm] = sc[:, i]
            top = np.argsort(-np.abs(vt[i]))[:3]
            load.append({"axis": nm, "var": var[i], "top": "; ".join(f"{KR[cols[j].rsplit('_', 1)[0]]} {cols[j].rsplit('_', 1)[1]} {vt[i, j]:+.2f}" for j in top),
                         **{c: vt[i, j] for j, c in enumerate(cols)}})
    G, csize = gpa(S); Y = G.reshape(len(G), -1); Y = Y-Y.mean(0); u, s, vt = np.linalg.svd(Y, full_matrices=False); var = s**2/(s**2).sum()
    for i in range(NK["shape"]):
        nm = f"AX_{struct}_shape_PC{i+1}"; sc = Y@vt[i]; out[nm] = sc/sc.std(ddof=1)
        disp = vt[i].reshape(-1, 3); mag = {r: float(np.linalg.norm(disp[lab == r], axis=1).mean()) for r in names}
        load.append({"axis": nm, "var": var[i], "top": "변형 큰 부위: "+", ".join(f"{KR[r]}" for r in sorted(mag, key=mag.get, reverse=True)[:2])})
    return out, F, pd.DataFrame(load)


if __name__ == "__main__":
    d, f = VH.load_frozen(); axes, raw, loads = [], [], []
    for st in ("MAND", "MAX", "CB"):
        a, F, L = structure_axes(st); axes.append(a); raw.append(F.add_prefix(st+"_")); loads.append(L)
    A = pd.concat(axes, axis=1); A.round(4).to_csv(OUT/"axes.csv", encoding="utf-8-sig"); pd.concat(raw, axis=1).round(3).to_csv(OUT/"raw_block_features.csv", encoding="utf-8-sig")
    L = pd.concat(loads); L.round(3).to_csv(OUT/"loadings.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); print(L[["axis", "var", "top"]].round(3).to_string(index=False))
    d = d.join(A); cols = list(A.columns)
    print(d.loc[d.restored.astype(bool), cols+["SIZE_PC1", "m_ANB", "Pog_Ba_ant", "Male"]].corr(method="spearman").loc[cols, ["SIZE_PC1", "m_ANB", "Pog_Ba_ant", "Male"]].round(2).to_string())
    S = {"main59": d[d.index.isin(f["discovery_ids"])], "sens85": d[d.restored.astype(bool)]}; rows = []
    for sn, s in S.items():
        for h, y in X.H.items():
            for lab, ref in (("확정 위", X.CONFIRMED), ("확정+크기 위", X.CONFIRMED+["SIZE_PC1"])):
                for c in cols: rows.append({"part": lab, "unit": "축", "sample": sn, "hyp": h, "add": c, **X.compare(s, y, X.B0, [c], ref_add=ref)})
                for st in ("MAND", "MAX", "CB"):
                    for kind in NK:
                        blk = [c for c in cols if c.startswith(f"AX_{st}_{kind}_")]
                        rows.append({"part": lab, "unit": "블록", "sample": sn, "hyp": h, "add": f"{st}_{kind}", **X.compare(s, y, X.B0, blk, ref_add=ref)})
    t = X.fdr(rows, ("sample", "hyp", "part", "unit")); t.round(4).to_csv(OUT/"E_results.csv", index=False, encoding="utf-8-sig")
    show = ["part", "unit", "sample", "hyp", "add", "n", "dR2", "share_better", "F_p", "q_BH", "coef_std"]
    print(t[t.F_p < .10][show].round(3).to_string(index=False)); print("q<0.05:", int((t.q_BH < .05).sum()))
