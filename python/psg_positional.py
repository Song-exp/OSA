"""P 단계: 판독 파일의 자세 정보로 앙와위·비앙와위 AHI와 앙와위 수면시간 계산 (사전 등록: 실험기록 "P 단계 사전 등록").
Embla: EN-Stage / EN-Apnea Events의 Position 열. Nox: Data.ndb에서 psg_reconcile.nox_counts와 같은 층·같은 규칙 + position 구간.
원본은 읽기만 한다. 결과에 날짜·식별정보를 쓰지 않는다.
출력: Result_py/analysis/positional/positional.csv (환자별 지표), 이후 분석은 __main__.
"""
import sqlite3, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from osa.psg_events import _read_text, _stage_level
from psg_reconcile import EVENT

ROOT = Path(__file__).resolve().parents[1]; PSG = ROOT/"Data/PSG"; OUT = ROOT/"Result_py/analysis/positional"; OUT.mkdir(parents=True, exist_ok=True)
AHI_EV = {"OA", "CA", "MA", "H"}
POS = {"SUPINE": "sup", "LEFT": "non", "RIGHT": "non", "PRONE": "non"}       # UPRIGHT·UNKNOWN은 제외


def pos_code(x):
    x = str(x).upper().replace("POSITION-", "").strip(); return POS.get(x, "other")


def table(f):
    lines = _read_text(f, "cp949").splitlines()
    for k, line in enumerate(lines):
        low = [c.strip().lower() for c in line.split("\t")]
        if "event" in low and any(c.startswith("time") for c in low):
            if "position" not in low: return None
            e, p = low.index("event"), low.index("position")
            rows = [l.split("\t") for l in lines[k+1:] if l.strip()]
            return pd.DataFrame({"event": [r[e].strip().upper() for r in rows if len(r) > max(e, p)],
                                 "pos": [pos_code(r[p]) for r in rows if len(r) > max(e, p)]})
    return None


def embla(folder):
    st, ap = table(folder/"EN-Stage Events.txt"), table(folder/"EN-Apnea Events.txt")
    if st is None or ap is None: return None
    lv = st.event.map(_stage_level); sl = st[lv.notna() & (lv != _stage_level("SLEEP-S0"))]   # 각성 제외 수면 epoch(30초)
    mins = sl.pos.value_counts()*0.5
    ev = ap[ap.event.map(EVENT).isin(AHI_EV)]
    return {"device": "E", "TST_r": len(sl)*0.5, "sup_min": mins.get("sup", 0.), "non_min": mins.get("non", 0.),
            "ev_all": len(ev), "ev_sup": int((ev.pos == "sup").sum()), "ev_non": int((ev.pos == "non").sum())}


def nox(folder):
    db = sqlite3.connect(f"file:{(folder/'Data.ndb').as_posix()}?mode=ro", uri=True)
    try:
        tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}; layers = {}
        for pre in ("scoring", "temporary_scoring"):
            if f"{pre}_marker" not in tabs: continue
            m = pd.read_sql(f"select CAST(m.starts_at AS TEXT) a, CAST(m.ends_at AS TEXT) b, m.type, m.is_deleted d, k.type kt "
                            f"from {pre}_marker m join {pre}_key k on m.key_id = k.id", db)
            m = m[m.d == 0]
            if len(m): layers[pre] = m.assign(a=m.a.astype("int64"), b=m.b.astype("int64"), type=m.type.astype(str))
    finally:
        db.close()
    if not layers: return None
    m = layers[max(layers, key=lambda k: (layers[k].kt == "Manual").sum())]                # psg_reconcile.nox_counts와 같은 층
    man = m[m.kt == "Manual"]
    st = m[m.type.str.startswith("sleep-")].sort_values("kt").drop_duplicates("a", keep="last")      # Manual 우선
    a0, a1 = man[man.type == "period-analysisstart"].a.min(), man[man.type == "period-analysisstop"].a.max()
    if not (pd.notna(a0) and pd.notna(a1)): a0, a1 = st.a.min(), st.b.max()
    sl = st[(st.type != "sleep-wake") & (st.a >= a0) & (st.a < a1)]
    ps = m[m.type.str.startswith("position-")].sort_values("a").drop_duplicates("a")
    if not len(ps): return None
    pa, pb, pc = ps.a.values, ps.b.values, ps.type.map(pos_code).values
    def at(t):
        k = np.flatnonzero((pa <= t) & (t < pb)); return pc[k[-1]] if len(k) else "other"
    epos = np.array([at((a+b)//2) for a, b in zip(sl.a.values, sl.b.values)]); dur = (sl.b.values-sl.a.values)/1e7/60
    A, B = sl.a.values, sl.b.values
    ev = man[man.type.str.upper().map(EVENT).isin(AHI_EV) & (man.a >= a0) & (man.a <= a1)]
    ev = ev[[((A <= a) & (a < B)).any() for a in ev.a.values]]
    evp = np.array([at(a) for a in ev.a.values])
    return {"device": "N", "TST_r": dur.sum(), "sup_min": dur[epos == "sup"].sum(), "non_min": dur[epos == "non"].sum(),
            "ev_all": len(ev), "ev_sup": int((evp == "sup").sum()), "ev_non": int((evp == "non").sum())}


def build():
    rows = []
    for d in sorted(PSG.glob("P_*_PSG_*"), key=lambda p: int(p.name.split("_")[1])):
        p = "_".join(d.name.split("_")[:2])
        try: r = embla(d) if d.name.endswith("_E") else nox(d) if d.name.endswith("_N") else None
        except Exception as ex: r = {"error": type(ex).__name__}
        if r: rows.append({"Patient": p, **r})
    t = pd.DataFrame(rows).set_index("Patient")
    t["AHI_r"] = t.ev_all/(t.TST_r/60); t["AHI_sup_r"] = t.ev_sup/(t.sup_min/60); t["AHI_non_r"] = t.ev_non/(t.non_min/60)
    t["sup_frac"] = t.sup_min/t.TST_r
    ok = (t.sup_min >= 30) & (t.non_min >= 30)
    t["POS_ratio"] = np.where(ok, np.log((t.AHI_sup_r+1)/(t.AHI_non_r+1)), np.nan)
    t["POSA"] = np.where(ok, (t.AHI_sup_r >= 2*t.AHI_non_r).astype(float), np.nan)
    t.round(4).to_csv(OUT/"positional.csv", encoding="utf-8-sig"); return t


if __name__ == "__main__":
    t = build(); print(t.device.value_counts().to_dict(), "| 계산", len(t), "| 판정 가능(각 자세 30분 이상)", int(t.POS_ratio.notna().sum()))
    print(t[["TST_r", "sup_frac", "AHI_r", "AHI_sup_r", "AHI_non_r", "POS_ratio", "POSA"]].describe().round(2).to_string())


def analyze():
    """사전 가설 P1(ANB → POS_ratio), P2(ANB → 앙와위 대 비앙와위 AHI). P_75는 기사 판독 사건이 거의 없어(원신호 EDA) 제외."""
    import statsmodels.api as sm
    import osa_experiments as X, osa_validate_holdout as VH
    d, f = VH.load_frozen(); t = pd.read_csv(OUT/"positional.csv", encoding="utf-8-sig", index_col=0)
    v = pd.read_csv(ROOT/"Result_py/psg_verified.csv", encoding="utf-8-sig").set_index("Patient")
    o = pd.read_csv(ROOT/"Result_py/phase1/psg_restored_own.csv", encoding="utf-8-sig", index_col=0)
    t = t[(v.SameExam.reindex(t.index) == True) & (t.index != "P_75")]
    t = t.assign(logSUP=np.log1p(t.AHI_sup_r), logNON=np.log1p(t.AHI_non_r)); d = d.join(t[["POS_ratio", "POSA", "logSUP", "logNON", "sup_frac", "sup_min", "non_min"]])
    S = {"발굴": d[d.index.isin(f["discovery_ids"])], "최종 80명": d[d.restored.astype(bool) & o.Include_own.reindex(d.index).fillna(False).astype(bool)]}
    rows = []
    for sn, s in S.items():
        s = s[s.POS_ratio.notna()]
        for y in ("POS_ratio", "logSUP", "logNON"):
            for c, adj in (("m_ANB", []), ("m_ANB", ["sup_frac"]), ("Pog_Ba_ant", []), ("VERT_Z", ["m_ANB", "Pog_Ba_ant"])):
                cols = X.B0+adj+[c]; u = s.dropna(subset=[y]+cols); Z = (u[cols]-u[cols].mean())/u[cols].std()
                fit = sm.OLS(u[y], sm.add_constant(Z)).fit(); cv = X.compare(u, y, X.B0, [c], ref_add=adj)
                rows.append({"sample": sn, "y": y, "x": c, "adj": "+".join(adj) or "-", "n": len(u), "beta_std": fit.params[c], "p": fit.pvalues[c],
                             "dR2cv": cv.get("dR2"), "share_better": cv.get("share_better")})
        u = s.dropna(subset=["POSA", "m_ANB"]+X.B0)
        lg = sm.Logit(u.POSA, sm.add_constant((u[X.B0+["m_ANB"]]-u[X.B0+["m_ANB"]].mean())/u[X.B0+["m_ANB"]].std())).fit(disp=0)
        rows.append({"sample": sn, "y": "POSA(로지스틱)", "x": "m_ANB", "adj": "-", "n": len(u), "beta_std": lg.params["m_ANB"], "p": lg.pvalues["m_ANB"],
                     "dR2cv": np.nan, "share_better": np.nan, "POSA_rate": u.POSA.mean()})
    r = pd.DataFrame(rows); r.round(4).to_csv(OUT/"positional_tests.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220); print(r.round(3).to_string(index=False))
    return r
