"""기본 피처(Roh 11개, 엑셀) + 신규 피처(Step 2~4) 상관 히트맵: 회귀 전에 서로 대체할 수 있는 인자 찾기.
Y는 보지 않는다(선택 편향 없음). Spearman 상관(짝별 결측 제외), 거리 1 − |r|, 평균 연결 계층 군집.
군집 기준 |r| ≥ 0.6 (회귀_조합실험_지시서.md 8절), 강한 짝 |r| ≥ 0.8 별도 표.
입력(읽기만): compare_all.csv(엑셀 Roh 11개, NAV·PAV 보충), final_values.csv(0값 의심 → 결측), analysis/all83/patient_values_all.csv(83명 피처)
출력: Result_py/analysis/feature_heatmap/{heatmap.png, corr_spearman.csv, clusters.csv, strong_pairs.csv, data.json}
실행: python osa_feature_heatmap.py (전체) / python osa_feature_heatmap.py --groups base,s2 (기본 + Step 2만 → feature_heatmap_base_s2/)
"""
import json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from osa_var_detail import detail

ROOT = Path(__file__).resolve().parents[1]; R = ROOT/"Result_py"; P = R/"CBCT/patient_pipeline/v2"
OUT = R/"analysis/feature_heatmap"; OUT.mkdir(parents=True, exist_ok=True)
CUT, STRONG, MIN_N = .6, .8, 30

R11 = ["ANB", "Nperp-A", "Nperp-Pog", "MP-FH", "TDI", "PHI", "NCWR", "NAV", "PAV", "PMCA", "PAL"]
# osa_newfactor.STEP2 / STEP3 / step4 과 같은 목록 (자동 추출판 Roh 지표 m_*·a_PAV는 기본 피처와 중복이라 제외)
STEP2 = ["a_PAV_h", "AW_min", "HD_mm", "nasopharynx_min_area", "nasopharynx_mean_area", "oropharynx_min_area", "oropharynx_mean_area",
         "hypopharynx_min_area", "hypopharynx_mean_area", "oro_naso_min_ratio", "H_col", "AW_length", "AW_min_pos", "soft_palate_length",
         "soft_palate_thickness", "soft_palate_angle", "soft_palate_PNS_gap", "AWshape_PC1", "AWshape_PC2", "AWshape_PC3"]
STEP3 = ["Pog_Ba_ant", "SNB", "SNA", "Pog_PNS_ant", "PNS_B_ant", "Overjet_mm", "Co_Gn", "Go_Me", "body_len", "body_len_res", "McN_diff", "F1_PC1",
         "PP_MP", "SN_MP", "LAFH_ratio", "VERT_Z", "gonial", "Jarabak", "ANS_Me", "N_Me", "S_Go", "Co_Go", "ramus_body", "Yaxis", "FacialAxis_proxy", "F1_PC4", "overbite_mm",
         "upper_width_M1", "lower_width_M1", "InterGonial", "Upper_Taper", "upper_depth",
         "SIZE_PC1", "SIZE_MAND", "SIZE_MAX", "SIZE_CB", "S_N", "S_Ba", "NSBa", "ANS_PNS", "PNS_Ba_depth", "Co_A", "palate_cb", "mand_cb", "logCS_all", "AlloShape",
         "Atyp_D2", "Atyp_Procrustes", "MaxPost_Clivus_ant"]
STEP4 = ["tongue_mL", "tongue_c_ant", "tongue_c_sup", "tongue_low_sup", "tongue_len", "tongue_height", "tongue_back_ant", "tongue_mandible_ratio", "tongue_MMV",
         "tongue_MMVhull", "SoftLoad_MMV", "MMV_mL", "MMV_hull_mL", "masseter_mL", "lat_pterygoid_mL", "med_pterygoid_mL", "chew_mL", "sinus_max_mL", "sinus_max_air_frac",
         "sinus_frontal_mL", "SDp_mm", "MP_H", "Hy_Me", "Hy_Ba_dist", "Hy_sup", "Hy_ant", "L1_MP_deg", "Condyle_Axial_mean", "Condyle_Lat_mean",
         "lever_Pog_mm", "lever_Hy_mm", "lever_arm_Pog_sup"]
STEP4_PRE = ("EFA_", "MAND_rPC", "MAX_rPC", "CB_rPC", "res_", "F1_PC", "F2_PC", "F2MM_PC", "roi_")
BASE_DESC = {"ANB": "ANB 각(°): N–A선과 N–B선 사이 각", "Nperp-A": "Nperp-A(mm): N 수직선에서 A점까지 앞뒤 거리", "Nperp-Pog": "Nperp-Pog(mm): N 수직선에서 Pog까지 앞뒤 거리",
             "MP-FH": "MP-FH(°): 하악평면과 FH 평면 사이 각", "TDI": "TDI: 위아래 치열 폭 차이 지수", "PHI": "PHI: 입천장 높이 지수",
             "NCWR": "NCWR: 비강 폭 ÷ 상악 기저폭", "NAV": "NAV(mL): 비강 기도 부피", "PAV": "PAV(mm³): 인두 기도 부피",
             "PMCA": "PMCA(mm²): 인두 기도 최소 단면적", "PAL": "PAL(mm): PNS–Eb 인두 기도 길이"}
GROUP = {"base": "기본 (Roh 11개, 엑셀)", "s2": "신규 Step 2 기도·연조직", "s3": "신규 Step 3 골격", "s4": "신규 Step 4 혀·근육·형태"}


def load():
    c = pd.read_csv(P/"restore_ext/v1/compare_all.csv"); z = pd.read_csv(P/"develop85/final_criteria_v2/final_values.csv")
    x = c.pivot(index="Patient", columns="metric", values="excel")
    for _, r in z[z.excel_zero_suspect == True].iterrows(): x.loc[r.Patient, r.metric] = np.nan
    x = x[R11].add_prefix("[기본] ")
    v = pd.read_csv(R/"analysis/all83/patient_values_all.csv").set_index("Patient")
    for f, cols in (("shape/extra/airway_profile.csv", ["AW_min", "AW_length"]), ("shape/fine/dental_arch.csv", ["upper_width_M1", "lower_width_M1", "upper_depth"])):
        v = v.join(pd.read_csv(R/f, encoding="utf-8-sig").set_index("Patient")[cols])   # osa_all83.data가 실행 중에 붙이는 열
    s4 = STEP4+[c for c in v if c.startswith(STEP4_PRE)]+["F2_logCS", "F2MM_logCS"]
    grp = {f"[기본] {k}": "base" for k in R11}
    for g, names in (("s2", STEP2), ("s3", STEP3), ("s4", s4)):
        for n in names:
            if n in v and n not in grp: grp[n] = g
    d = x.join(v, how="inner")[list(grp)]
    missing = [n for n in STEP2+STEP3+STEP4 if n not in v]
    keep = [n for n in grp if d[n].notna().sum() >= MIN_N and d[n].std() > 0]
    dropped = [n for n in grp if n not in keep]
    return d[keep], {k: grp[k] for k in keep}, missing, dropped


def clusters(Z, names, order, r, grp, cut):
    lab = fcluster(Z, 1-cut, "distance"); cl = dict(zip(names, lab)); rows = []
    for k in sorted(set(lab)):
        m = [n for n in order if cl[n] == k]
        if len(m) < 2: continue
        sub = r.loc[m, m].abs(); mean = {n: (sub.loc[n].sum()-1)/(len(m)-1) for n in m}
        rep = max(m, key=mean.get); base = [n for n in m if grp[n] == "base"]
        rows.append({"members": m, "rep": rep, "base": base, "size": len(m), "min_abs_r": float(sub.to_numpy()[np.triu_indices(len(m), 1)].min()),
                     "mean_abs_r": {n: round(mean[n], 3) for n in m}})
    rows.sort(key=lambda c: (-len(c["base"]), -c["size"]))
    return rows


def main(groups=None, out=OUT, cuts=(.6, .7, .8)):
    d, grp, missing, dropped = load()
    if groups:
        keep = [n for n in d if grp[n] in groups]; d = d[keep]; grp = {n: grp[n] for n in keep}
    out.mkdir(parents=True, exist_ok=True); names = list(d)
    r = d.corr("spearman", min_periods=MIN_N); nn = d.notna().astype(int); npair = nn.T @ nn
    a = r.abs().fillna(0).to_numpy(); np.fill_diagonal(a, 1)
    Z = linkage(squareform(1-a, checks=False), "average"); order = [names[i] for i in leaves_list(Z)]
    byCut = {f"{c:.1f}": clusters(Z, names, order, r, grp, c) for c in cuts}; rows = byCut[f"{CUT:.1f}"]
    pairs = [(i, j, float(r.at[i, j]), int(npair.at[i, j])) for x, i in enumerate(names) for j in names[x+1:] if abs(r.at[i, j]) >= STRONG]
    pairs.sort(key=lambda p: -abs(p[2]))
    desc = {n: BASE_DESC[n[5:]] if grp[n] == "base" else (detail(n) or ("", ""))[0] for n in names}

    r.loc[order, order].round(3).to_csv(out/"corr_spearman.csv", encoding="utf-8-sig")
    pd.DataFrame([{"군집": i+1, "크기": c["size"], "대표 후보(군집 안 평균 |r| 최대)": c["rep"], "기본 피처": ", ".join(c["base"]),
                   "최소 |r|": round(c["min_abs_r"], 2), "구성": ", ".join(c["members"])} for i, c in enumerate(rows)]).to_csv(out/"clusters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs, columns=["a", "b", "spearman_r", "n"]).to_csv(out/"strong_pairs.csv", index=False, encoding="utf-8-sig")
    json.dump({"order": order, "group": grp, "groupLabel": GROUP, "desc": desc, "n": {n: int(d[n].notna().sum()) for n in names},
               "r": [[None if pd.isna(r.at[i, j]) else round(float(r.at[i, j]), 3) for j in order] for i in order],
               "clusters": rows, "clustersByCut": byCut, "pairs": pairs, "cut": CUT, "strong": STRONG, "nPatients": len(d), "dropped": dropped, "missing": missing},
              open(out/"data.json", "w", encoding="utf-8"), ensure_ascii=False)

    plt.rcParams["font.family"] = "Malgun Gothic"; m = len(order)
    fig, ax = plt.subplots(figsize=(max(.12*m+3, .32*m+3) if m <= 40 else .12*m+3, max(.12*m+2, .32*m+2) if m <= 40 else .12*m+2))
    ax.imshow(r.loc[order, order].to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    col = {"base": "#c0262d", "s2": "#1f6fb5", "s3": "#1a8a52", "s4": "#a1630b"}
    fs = 7 if m <= 40 else 4; ax.set_xticks(range(m), order, rotation=90, fontsize=fs); ax.set_yticks(range(m), order, fontsize=fs)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_color(col[grp[t.get_text()]]); t.set_fontweight("bold" if grp[t.get_text()] == "base" else "normal")
    if m <= 40:
        for i in range(m):
            for j in range(m):
                v = r.at[order[i], order[j]]
                if pd.notna(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white" if abs(v) > .6 else "black")
    fig.tight_layout(); fig.savefig(out/"heatmap.png", dpi=160); plt.close(fig)
    print(f"피처 {m}개 (기본 {sum(g == 'base' for g in grp.values())}), 환자 {len(d)}명, 군집(2개 이상) {len(rows)}개, |r|≥{STRONG} 짝 {len(pairs)}개")
    print("목록에 있으나 데이터에 없음:", missing); print("n<30 또는 상수라 제외:", dropped)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--groups", help="원천 고르기, 쉼표로 (base,s2,s3,s4). 없으면 전부")
    a = ap.parse_args()
    if a.groups:
        g = a.groups.split(","); main(set(g), OUT.parent/f"feature_heatmap_{'_'.join(g)}")
    else: main()
