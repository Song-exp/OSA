"""(v2: 원격 원본 osa_feature_heatmap.py --groups 방식과 별개. 골격·연조직 파생변수 포함 판) 기본(Roh 11, 엑셀) · Step 2 잔여 · 연조직 파생변수 Spearman 상관 히트맵 자료(data.json). Y(AHI)는 보지 않는다.

2026-11-26 사용자 결정: 인두·비강·혀와 의미가 겹치는 기존 Step 2 피처는 연조직 크기 제거 PCA 파생변수(softtissue_profile_pca.py)로 대체한다.
  빠짐: a_PAV_h, AWshape_PC1~3, AW_min, AW_min_pos, HD_mm, AW_length(H_col과 같은 값), TS 인두 구간(비·구·하인두 최소·평균, 구인두÷비인두 비)
  남김(Step 2): 연구개 4개, H_col
  추가(연조직): 인두 기도 부피 PAV_fh·모양 PC1~2, 비강 부피 NAV_ts·모양 PC1~3, 혀 부피·모양 PC1~3 (개별 설명분산 10% 이상 PC)
대상: phase1/psg_restored_own.csv Include_own_c3 (83명). 상관은 두 값이 모두 있는 환자로 계산.
묶음: 거리 1 − |r|, 평균 연결 계층 군집, |r| 기준 0.6·0.7·0.8에서 자름(기존 아티팩트와 같은 규칙, 기존 data.json으로 재현 확인).
골격 파생변수 33개(skeletal): 12부위 모양 PC 17개 + 전체 12점 배치 PC1~4 + 부위 중심 크기 12개 (2026-11-26 추가, 히트맵은 색만 표시).
출력: Result_py/analysis/feature_heatmap_base_s2_st/ (data.json, figs/softtissue_*.png)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from run_ali import ROOT

R_ = ROOT/"Result_py"; V2 = R_/"CBCT/patient_pipeline/v2"; ST = R_/"analysis/softtissue_pca"
DEST = R_/"analysis/feature_heatmap_base_s2_st"
CUTS, STRONG = (0.6, 0.7, 0.8), 0.8
BASE = {"ANB": "ANB 각(°): N–A선과 N–B선 사이 각", "Nperp-A": "Nperp-A(mm): N 수직선에서 A점까지 앞뒤 거리", "Nperp-Pog": "Nperp-Pog(mm): N 수직선에서 Pog까지 앞뒤 거리",
        "MP-FH": "MP-FH(°): 하악평면과 FH 평면 사이 각", "TDI": "TDI: 위아래 치열 폭 차이 지수", "PHI": "PHI: 입천장 높이 지수", "NCWR": "NCWR: 비강 폭 ÷ 상악 기저폭",
        "NAV": "NAV(mm³): 비강 기도 부피", "PAV": "PAV(mm³): 인두 기도 부피", "PMCA": "PMCA(mm²): 인두 기도 최소 단면적", "PAL": "PAL(mm): PNS–Eb 인두 기도 길이"}
S2 = {"H_col": "인두 기둥 높이(mm): PNS에서 Eb까지의 수직 거리. 클수록 인두가 위아래로 김",
      "soft_palate_length": "연구개 길이(mm): 정중 단면에서 연구개의 장축 방향 길이. 길수록 연구개가 기도 쪽으로 늘어짐",
      "soft_palate_thickness": "연구개 최대 두께(mm): 장축에 수직인 방향의 최대 폭",
      "soft_palate_angle": "연구개 기울기(°): 연구개 장축이 앞 축(구개 방향)과 이루는 각. 클수록 아래로 처짐",
      "soft_palate_PNS_gap": "PNS와 연구개 사이 가장 가까운 거리(mm)"}
STV = {"PAV_fh": "인두 기도 부피(mm³, 자동 추출): Eb를 지나는 FH 평행면 위 −1200~−400 HU 공기. 인두 기도 크기",
       "airway_shapePC1": "인두 기도 모양 PC1(49.3%): 크기를 뺀 20구간 단면 프로파일에서 좁은 곳이 아래(혀뿌리 뒤)인지 위(연구개 뒤)인지",
       "airway_shapePC2": "인두 기도 모양 PC2(28.8%): 가운데가 잘록한 정도(모래시계 모양)",
       "NAV_ts": "비강 기도 부피(mm³, 자동 추출): TotalSegmentator 비강 라벨 안 −400 HU 미만 공기",
       "nasal_shapePC1": "비강 모양 PC1(38.3%): 크기를 뺀 PNS→ANS 20구간 관상 단면 프로파일의 앞쪽·뒤쪽 넓이 균형",
       "nasal_shapePC2": "비강 모양 PC2(21.5%): 비강 프로파일 둘째 축", "nasal_shapePC3": "비강 모양 PC3(16.1%): 비강 프로파일 셋째 축",
       "tongue_vol": "혀 부피(mm³): TotalSegmentator 혀 라벨(0.5 mL 미만 조각 제거)",
       "tongue_shapePC1": "혀 모양 PC1(64.6%): 크기를 뺀 혀뿌리→혀끝 20구간 관상 단면 프로파일에서 덩어리가 혀뿌리 쪽에 몰렸는지 앞쪽에 몰렸는지",
       "tongue_shapePC2": "혀 모양 PC2(18.4%): 혀 프로파일 둘째 축", "tongue_shapePC3": "혀 모양 PC3(10.1%): 혀 프로파일 셋째 축"}


def table():
    o = pd.read_csv(R_/"phase1/psg_restored_own.csv", encoding="utf-8-sig", index_col=0)
    ids = [p for p in o.index if o.at[p, "Include_own_c3"]]
    ca = pd.read_csv(V2/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig")
    x = ca.pivot(index="Patient", columns="metric", values="excel")
    zs = pd.read_csv(V2/"develop85/final_criteria_v2/final_values.csv", encoding="utf-8-sig").query("excel_zero_suspect == True")
    for p, k in zip(zs.Patient, zs.metric): x.loc[p, k] = np.nan                                   # 엑셀 0값 의심 → 결측
    d = pd.DataFrame(index=ids)
    for k in BASE: d[f"[기본] {k}"] = x[k].reindex(ids)
    f = pd.read_csv(R_/"features/v1/features.csv", encoding="utf-8-sig", index_col=0)
    st = pd.read_csv(R_/"shape/fine/soft_tissue.csv", encoding="utf-8-sig").set_index("Patient")
    d["H_col"] = -f.c_Eb_sup.reindex(ids)
    for k in list(S2)[1:]: d[k] = st[k].reindex(ids)
    r = ca.pivot(index="Patient", columns="metric", values="restored")
    d["PAV_fh"], d["NAV_ts"] = r.PAV.reindex(ids), r.NAV.reindex(ids)
    for key, n in (("airway", 2), ("nasal", 3), ("tongue", 3)):
        s = pd.read_csv(ST/f"{key}_scores.csv", encoding="utf-8-sig", index_col=0)
        for i in range(1, n+1): d[f"{key}_shapePC{i}"] = s[f"PC{i}"].reindex(ids)
    d["tongue_vol"] = pd.read_csv(ST/"tongue_profile.csv", encoding="utf-8-sig").set_index("Patient").TG_volume.reindex(ids)
    return d.dropna(how="all")                                                                      # CBCT 계측이 하나도 없는 환자 제외(기존 83명과 같게)


def skeletal():
    """골격 파생변수 33개: 12부위 모양 PC(개별 설명분산 10% 이상, roi_shape_pca_report와 같은 GPA+PCA 95%) 17개,
    전체 12점 배치 PC1~4(roi_centroid_pca/scores.csv), 부위 중심 크기 12개(local_roi_features *_size). 반환: (표, 설명, 도메인)"""
    from roi_centroid_pca import NAMES, REGIONS
    from roi_shape_pca_report import fit
    from shape_ssm import gpa
    S = R_/"shape"; cols, desc, dom = {}, {}, {}; i = 0
    size = pd.read_csv(S/"local/local_roi_features.csv").set_index("Patient")
    for structure, names in REGIONS.items():
        with np.load(S/structure/"points.npz") as z: ids, pts = z["ids"], z["frame"]
        labels = np.load(S/"local"/f"{structure}_roi_labels.npy")
        qc = pd.read_csv(S/structure/"review/qc_review.csv").set_index("Patient")
        use = qc.loc[[f"P_{k}" for k in ids], "confirmed_fail_procrustes"].astype("string").fillna("").str.strip().eq("").to_numpy()
        for name in names:
            nm = NAMES[i]; i += 1
            aligned, _ = gpa(pts[:, labels == name][use]); m = fit(aligned)
            sc = m.transform(aligned.reshape(len(aligned), -1)); pid = [f"P_{k}" for k in ids[use]]
            for k, r in enumerate(m.explained_variance_ratio_):
                if r < .10: continue
                c = f"{nm}_PC{k+1}"; cols[c] = pd.Series(sc[:, k], index=pid)
                desc[c] = f"{nm} 모양 PC{k+1}({r*100:.1f}%): 위치·회전·크기를 맞춘(GPA) 표면점의 부위 모양 축"
                dom[c] = {"d": f"sk_{name}", "c": f"모양 PC{k+1}", "p": 1, "m": desc[c]}
            c = f"{nm}_크기"; cols[c] = size[f"{structure}_{name}_size"]
            desc[c] = f"{nm} 중심 크기: 부위 표면점이 자기 중심에서 퍼진 정도(mm)"; dom[c] = {"d": f"sk_{name}", "c": "크기", "p": 1, "m": desc[c]}
    j = pd.read_csv(R_/"analysis/roi_centroid_pca/scores.csv", encoding="utf-8-sig").set_index("Patient")
    j = j[j.in_pca_axes.astype(str).str.lower() == "true"]
    for k in range(1, 5):
        c = f"12점배치_PC{k}"; cols[c] = j[f"PC{k}"]
        desc[c] = f"전체 12점 상대 배치 PC{k}: 12부위 중심점의 크기를 맞춘 배치 모양 축"; dom[c] = {"d": "sk_layout", "c": f"배치 PC{k}", "p": 1, "m": desc[c]}
    dim = {f"sk_{name}": f"골격·{NAMES[ix]}" for ix, name in enumerate(n for g in REGIONS.values() for n in g)} | {"sk_layout": "골격·전체 12점 배치"}
    return pd.DataFrame(cols), desc, dom, dim


def clusters(names, R, cut):
    dist = 1-np.abs(R); np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"), 1-cut, "distance"); out = []
    for k in sorted(set(lab)):
        idx = np.flatnonzero(lab == k)
        if len(idx) < 2: continue
        sub = np.abs(R[np.ix_(idx, idx)]); mem = [names[i] for i in idx]
        mean = {m: float(np.round((sub[j].sum()-1)/(len(idx)-1), 3)) for j, m in enumerate(mem)}
        out.append({"members": mem, "rep": max(mem, key=mean.get), "base": [m for m in mem if m.startswith("[기본]")], "size": len(idx),
                    "min_abs_r": float(sub[np.triu_indices(len(idx), 1)].min()), "mean_abs_r": mean})
    return out


def figs():
    import softtissue_profile_pca as T
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    (DEST/"figs").mkdir(parents=True, exist_ok=True); ex = T.example_masks()
    for key in T.REGIONS:
        fig, ax = plt.subplots(figsize=(4.8, 4.4)); T.ct_panel(ax, key, ex)
        ax.set_title(f"{T.REGIONS[key][0]} (P_2{', 좌우 ±15 mm 겹침' if key == 'nasal' else ''})", fontsize=11); fig.tight_layout()
        fig.savefig(DEST/f"figs/softtissue_{key}.png", dpi=80); plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    sk, sk_desc, sk_dom, sk_dim = skeletal()
    d = table(); d = d.join(sk.reindex(d.index)); C = d.corr(method="spearman").to_numpy()
    dist = 1-np.abs(C); np.fill_diagonal(dist, 0)
    order = [d.columns[i] for i in leaves_list(linkage(squareform(dist, checks=False), "average"))]
    d = d[order]; R = d.corr(method="spearman").to_numpy(); names = list(d.columns)
    n = {c: int(d[c].notna().sum()) for c in names}
    group = {c: "base" if c.startswith("[기본]") else "s2" if c in S2 else "sk" if c in sk_desc else "st" for c in names}
    pairs = sorted(([a, b, float(R[i, j]), int((d[a].notna() & d[b].notna()).sum())] for i, a in enumerate(names) for j, b in enumerate(names)
                    if i < j and abs(R[i, j]) >= STRONG), key=lambda p: -abs(p[2]))
    cbc = {str(c): clusters(names, R, c) for c in CUTS}
    data = {"order": names, "group": group, "groupLabel": {"base": "기본 (Roh 11개, 엑셀)", "s2": "Step 2 잔여 (연구개·기도 길이)", "st": "연조직 파생변수 (크기 제거 PCA)", "sk": "골격 파생변수 (부위 모양·크기·전체 배치)"},
            "desc": {**{f"[기본] {k}": v for k, v in BASE.items()}, **S2, **STV, **sk_desc}, "dom": sk_dom, "dim": sk_dim, "n": n, "r": np.round(R, 3).tolist(),
            "clusters": cbc[str(CUTS[0])], "clustersByCut": cbc, "pairs": pairs, "cut": CUTS[0], "strong": STRONG, "nPatients": len(d), "dropped": [], "missing": []}
    (DEST/"data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    figs()
    print(len(names), "피처 |", {g: sum(v == g for v in group.values()) for g in ("base", "s2", "st", "sk")}, "| 환자", len(d))
    for c in CUTS: print(f"|r|≥{c}: 묶음 {len(cbc[str(c)])}개", [cl["members"] for cl in cbc[str(c)]])


if __name__ == "__main__":
    main()
