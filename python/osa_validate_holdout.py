"""V 단계: 추가 환자 외부 검증 (사전 등록: 실험기록 "V 단계 사전 등록").

python osa_validate_holdout.py --freeze     # 발굴 ID·SIZE_PC1 정의 동결(한 번만. 이미 있으면 거부)
python osa_validate_holdout.py --dry-run    # PSG 확인 전 환자로 스크립트 시험(검증으로 해석 금지)
python osa_validate_holdout.py              # 정식 검증: 동결 발굴 ID 밖 + cohort Include = true
python osa_validate_holdout.py --describe   # SIZE_PC1 구성(적재량·관련 변수) 표
출력: Result_py/analysis/validation/
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
from datetime import date
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
import osa_experiments as X
import osa_addition as A
import osa_vertical as V

OUT = X.R_/"analysis/validation"; OUT.mkdir(parents=True, exist_ok=True); FROZEN = OUT/"frozen.json"
M0 = X.B0+X.CONFIRMED
MODELS = {"M0 확정": M0, "M1 = M0 + SIZE_PC1": M0+["SIZE_PC1"], "M1 + 구인두 최소 단면적": M0+["SIZE_PC1", "oropharynx_min_area"],
          "M1 + 아래 앞얼굴 비율": M0+["SIZE_PC1", "LAFH_ratio"], "M1 + ANB × 혀 위치": M0+["SIZE_PC1", "tongue_c_ant", "m_ANBxtongue_c_ant"],
          "M0 + VERT_Z": M0+["VERT_Z"], "M1 + VERT_Z": M0+["SIZE_PC1", "VERT_Z"]}      # G 단계 추가(수직 배치 대표 지표)
SIGN = {"SIZE_PC1": -1, "oropharynx_min_area": -1, "LAFH_ratio": 1, "m_ANBxtongue_c_ant": 1, "VERT_Z": 1}    # 발굴 계수 방향


def size_cols(d):
    return [c for c in d if c.startswith("roi_") and c.endswith("_size")]


def freeze():
    assert not FROZEN.exists(), f"{FROZEN} 이미 있음(덮어쓰기 금지)"
    d = A.load(); disc = d[d.analysis_list.astype(bool) & d.logAHI.notna()].index.tolist()
    sc = size_cols(d); Z = d[sc]; mu, sd = Z.mean(), Z.std(); z = ((Z-mu)/sd).dropna()
    u = np.linalg.svd(z.to_numpy(), full_matrices=False)[2][0]; u = u if u.sum() > 0 else -u
    FROZEN.write_text(json.dumps({"date": str(date.today()), "discovery_ids": disc, "size_cols": sc, "size_mean": mu.tolist(), "size_sd": sd.tolist(),
                                  "size_loading": u.tolist(), "note": "SIZE_PC1 = ((roi_size − mean)/sd) @ loading"}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("동결:", len(disc), "명, 크기 피처", len(sc))


def load_frozen():
    f = json.loads(FROZEN.read_text(encoding="utf-8")); d = V.vertical(A.load())
    z = (d[f["size_cols"]]-np.array(f["size_mean"]))/np.array(f["size_sd"]); d["SIZE_PC1"] = z@np.array(f["size_loading"])  # 고정 정의로 덮어씀
    fv = OUT/"frozen_vertical.csv"
    if fv.exists(): d = d.join(pd.read_csv(fv, encoding="utf-8-sig", index_col=0)[["VERT_Z"]])       # G 단계 동결값
    d["m_ANBxtongue_c_ant"] = (d.m_ANB-d.loc[f["discovery_ids"], "m_ANB"].mean())*(d.tongue_c_ant-d.loc[f["discovery_ids"], "tongue_c_ant"].mean())
    return d, f


def r2(y, p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()


def validate(dry, own=False, no_gap=False, class3=False):
    """own: 연구실 확인 없이 우리 규칙으로 복원한 PSG(psg_restored_own.csv)로 검증군을 정하고 결과 변수를 바꿈(실험기록 "V 단계 변경")."""
    d, f = load_frozen(); disc = d.loc[d.index.isin(f["discovery_ids"])]
    pool = d[~d.index.isin(f["discovery_ids"]) & d.restored.astype(bool)]
    if own:
        o = pd.read_csv(X.R_/"phase1/psg_restored_own.csv", encoding="utf-8-sig", index_col=0).reindex(pool.index)
        keep = o["Include_own_c3" if class3 else "Include_own"].fillna(False).astype(bool) & (~o.Gap_gt1y.fillna(False).astype(bool) if no_gap else True)
        pool = pool[keep].copy(); o = o.loc[pool.index]
        pool["logAHI"], pool["logAHI_SUP"] = np.log1p(o.AHI_r), np.log1p(o.AHI_SUP_r)
        val = pool
    else:
        pool = pool[pool.logAHI.notna()]; val = pool if dry else pool[pool.analysis_list.astype(bool)]
    terms = list(dict.fromkeys(c for v in MODELS.values() for c in v)); rng = np.random.default_rng(0); rows = []
    for h, y in X.H.items():
        D, Vv = disc.dropna(subset=terms+[y]), val.dropna(subset=terms+[y])
        if len(Vv) < 10: print(f"[{h}] 검증 환자 {len(Vv)}명 — 10명 미만이라 중단"); continue
        pred = {}
        for name, cols in MODELS.items():
            fit = sm.OLS(D[y], sm.add_constant(D[cols])).fit(); pred[name] = fit.predict(sm.add_constant(Vv[cols], has_constant="add")).to_numpy()
        yv = Vv[y].to_numpy(); idx = rng.integers(0, len(Vv), (2000, len(Vv)))
        for name, cols in MODELS.items():
            ref = "M0 확정" if name.startswith(("M1 =", "M0 +")) else ("M1 = M0 + SIZE_PC1" if name != "M0 확정" else None)
            r = {"hyp": h, "model": name, "n_disc": len(D), "n_val": len(Vv), "R2_val": r2(yv, pred[name])}
            if ref:
                dr = r2(yv, pred[name])-r2(yv, pred[ref]); boot = [r2(yv[i], pred[name][i])-r2(yv[i], pred[ref][i]) for i in idx]
                new = [c for c in cols if c not in MODELS[ref]][-1]
                fv = sm.OLS(Vv[y], sm.add_constant(Vv[cols])).fit(); b, p2 = fv.params[new], fv.pvalues[new]
                one = p2/2 if np.sign(b) == SIGN[new] else 1-p2/2
                r |= {"vs": ref, "dR2_val": dr, "dR2_lo": np.percentile(boot, 2.5), "dR2_hi": np.percentile(boot, 97.5), "term": new,
                      "beta_val": b, "p_one_sided": one, "same_sign": np.sign(b) == SIGN[new],
                      "pass": bool(dr > 0 and np.sign(b) == SIGN[new] and one < .05)}
            rows.append(r)
    t = pd.DataFrame(rows); tag = "dryrun" if dry else ("validation_own"+("_c3" if class3 else "")+("_nogap" if no_gap else "") if own else "validation")
    t.round(4).to_csv(OUT/f"{tag}_{date.today()}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); print(("[DRY-RUN: PSG 확인 전 환자, 검증 아님]\n" if dry else "")+t.round(3).to_string(index=False))


def describe():
    """SIZE_PC1 구성: 동결 적재량 + 관련 변수 상관(85명)."""
    d, f = load_frozen(); s = d[d.restored.astype(bool)]
    names = {"MAND_symphysis": "하악 턱끝", "MAND_body": "하악체", "MAND_gonial": "하악각", "MAND_ramus_condyle": "하악지·과두",
             "MAX_anterior_alveolar": "상악 앞치조", "MAX_palate": "구개", "MAX_posterior": "상악 후방", "MAX_orbital_floor": "안와저",
             "CB_anterior_base": "앞 두개저", "CB_sella": "터키안장", "CB_clivus": "사대", "CB_temporal": "측두 두개저"}
    load = pd.DataFrame({"feature": f["size_cols"], "부위": [names[c[4:-5]] for c in f["size_cols"]], "loading": f["size_loading"],
                         "rho_with_SIZE": [spearmanr(s[c], s.SIZE_PC1, nan_policy="omit")[0] for c in f["size_cols"]]})
    rel = ["Male", "Age", "BMI", "m_ANB", "Pog_Ba_ant", "Co_Gn", "S_Ba", "S_N", "N_Me", "S_Go", "Co_Go", "ANS_PNS", "MMV_mL", "InterGonial",
           "tongue_mL", "chew_mL", "masseter_mL", "sinus_max_mL", "oropharynx_min_area", "LAFH_ratio", "logAHI", "logAHI_SUP"]
    cor = pd.DataFrame([{"variable": c, "rho": spearmanr(s[c], s.SIZE_PC1, nan_policy="omit")[0], "n": int(s[c].notna().sum())} for c in rel if c in s])
    load.round(3).to_csv(OUT/"size_loadings.csv", index=False, encoding="utf-8-sig"); cor.round(3).to_csv(OUT/"size_correlates.csv", index=False, encoding="utf-8-sig")
    print(load.round(2).to_string(index=False)); print(cor.round(2).sort_values("rho", key=abs, ascending=False).to_string(index=False))
    ev = np.linalg.svd(((s[f["size_cols"]]-np.array(f["size_mean"]))/np.array(f["size_sd"])).dropna().to_numpy(), compute_uv=False)**2
    print("SIZE_PC1 설명 분산", round(ev[0]/ev.sum(), 3))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--freeze" in a: freeze()
    elif "--describe" in a: describe()
    else: validate("--dry-run" in a, own="--own" in a, no_gap="--no-gap" in a, class3="--class3" in a)
