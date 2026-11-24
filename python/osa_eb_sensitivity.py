"""Eb 검출 자체 선택 수치의 민감도: 정중 공기 투영 두께(±W), 후두개 두께 하한·상한(GAP), 최소 연속 줄 수.
아래 한계 = 설골 몸통 아래 끝(확정 규칙), 설골 검출(뼈 400, ±3 mm)은 고정. 한 번에 하나씩 바꾸고 나머지는 기본값.
판정은 엑셀 PAL(연구실 수작업 PNS–Eb)과의 오차 — 회귀 결과는 쓰지 않는다. 기존 결과 파일은 건드리지 않는다.
출력: Result_py/analysis/all83/eb_sensitivity/{variants.csv, summary.csv}
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import map_coordinates
import cbct_eb_detect as E
from cbct_patient_pipeline import CB, OUT as PIPE, landmarks
from check_landmarks import read_nrrd
import osa_experiments as X

OUT = X.R_/"analysis/all83/eb_sensitivity"
BASE = dict(W=6.0, gap=(1.0, 10.0), rows=3)
VARIANTS = ([dict(BASE, W=w) for w in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0)]
            + [dict(BASE, gap=(g, 10.0)) for g in (0.4, 0.8, 1.2, 1.6, 2.0)]
            + [dict(BASE, gap=(1.0, g)) for g in (4.0, 6.0, 15.0)]
            + [dict(BASE, rows=n) for n in (1, 2, 4, 5, 6, 8)])


COMBO = [dict(W=w, gap=(0.8, 10.0), rows=2) for w in (2.0, 4.0, 6.0)] + [BASE]    # 하한 0.8 mm·2줄을 묶은 상태에서 두께 비교


def key(v): return f"W{v['W']:g}_gap{v['gap'][0]:g}-{v['gap'][1]:g}_rows{v['rows']}"


def volume(pid):
    P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); o, R, _ = E.midline_frame(P)
    ant, sup, Ls = np.arange(*E.ANT, E.STEP), np.arange(*E.SUP, E.STEP), np.arange(-15.2, 15.2+1e-9, E.STEP)    # 0과 ±6이 격자 위(cbct_eb_detect와 같은 칸)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-3000).reshape(len(Ls), len(ant), len(sup)); del v
    vol[vol < -2500] = np.nan
    return vol, Ls, ant, sup


def main(variants=VARIANTS, tag=""):
    OUT.mkdir(parents=True, exist_ok=True); VARIANTS = variants
    ids = sorted(int(p.stem.split("_")[1]) for p in (CB/"nrrd").glob("P_*.nrrd")); rows = []
    for pid in ids:
        try: vol, Ls, ant, sup = volume(pid)
        except Exception as e: print("skip", pid, type(e).__name__); continue
        mx = np.nan_to_num(np.nanmax(vol[np.abs(Ls) <= 3], 0), nan=-3000); hy = E.hyoid(mx, ant, sup)
        if not hy: continue
        air = {w: np.nanmin(vol[np.abs(Ls) <= w+1e-9], 0) < E.AIR for w in {v["W"] for v in VARIANTS}}
        for v in VARIANTS:
            E.GAP = v["gap"]; r = E.vallecula(air[v["W"]], ant, sup, hy)
            ok = r is not None and r[2] >= v["rows"]
            rows.append(dict(Patient=f"P_{pid}", variant=key(v), W=v["W"], gap_lo=v["gap"][0], gap_hi=v["gap"][1], rows_min=v["rows"],
                             hy_ant=hy[1], hy_sup=hy[2], Eb_ant=r[0] if ok else np.nan, Eb_sup=r[1] if ok else np.nan, n_rows=r[2] if r else 0))
        E.GAP = BASE["gap"]; print(pid, end=" ", flush=True)
    d = pd.DataFrame(rows); d["PAL"] = np.hypot(d.Eb_ant, d.Eb_sup); d.round(2).to_csv(OUT/f"variants{tag}.csv", index=False, encoding="utf-8-sig")
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'PAL'").set_index("Patient").excel
    base = d[d.variant == key(BASE)].set_index("Patient"); common = base.PAL.dropna().index.intersection(xl.dropna().index)
    out = []
    for k, g in d.groupby("variant", sort=False):
        g = g.set_index("Patient"); e = (g.PAL-xl).dropna(); ec = (g.PAL.reindex(common)-xl.reindex(common)).dropna()
        mv = np.hypot(g.Eb_ant-base.Eb_ant, g.Eb_sup-base.Eb_sup).reindex(common)
        out.append(dict(variant=k, W=g.W.iloc[0], gap_lo=g.gap_lo.iloc[0], gap_hi=g.gap_hi.iloc[0], rows_min=g.rows_min.iloc[0],
                        direct=int(g.PAL.notna().sum()), common_n=len(ec), common_MAE=ec.abs().mean(), common_r=np.corrcoef(g.PAL.reindex(ec.index), xl.reindex(ec.index))[0, 1] if len(ec) > 2 else np.nan,
                        all_n=len(e), all_MAE=e.abs().mean(), over3=int((e.abs() > 3).sum()), moved_gt1mm=int((mv > 1).sum()), lost_vs_base=int(mv.isna().sum())))
    s = pd.DataFrame(out).round(3); s.to_csv(OUT/f"summary{tag}.csv", index=False, encoding="utf-8-sig"); print("\n", s.to_string(index=False))



# ---------- 앞뒤 탐색 범위·탐색 창 뒤쪽 끝 (확정: 아래 한계 = 설골 몸통 아래 끝, ±6 mm, 하한 0.8 mm, 2줄) ----------
AP_BASE = dict(c_lo=-18.0, c_hi=2.0, run_back=25.0, win=-35.0)
AP_VARIANTS = ([dict(AP_BASE, c_lo=c) for c in (-10.0, -14.0, -18.0, -22.0, -26.0, -30.0)]
               + [dict(AP_BASE, c_hi=c) for c in (-4.0, -2.0, 0.0, 5.0, 8.0)]
               + [dict(AP_BASE, run_back=r) for r in (15.0, 20.0, 30.0, 35.0, 99.0)]
               + [dict(AP_BASE, win=w) for w in (-30.0, -40.0, -45.0, -50.0)])


def ap_key(v): return f"clo{v['c_lo']:g}_chi{v['c_hi']:g}_back{v['run_back']:g}_win{v['win']:g}"


def ap_main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.stem.split("_")[1]) for p in (CB/"nrrd").glob("P_*.nrrd")); rows = []
    for pid in ids:
        P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); o, R, _ = E.midline_frame(P)
        ant, sup, Ls = np.arange(-55.0, E.ANT[1], E.STEP), np.arange(*E.SUP, E.STEP), np.arange(-6, 6+1e-9, E.STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
        vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-3000).reshape(len(Ls), len(ant), len(sup)); del v
        vol[vol < -2500] = np.nan
        mip = np.nanmin(vol, 0); mx = np.nan_to_num(np.nanmax(vol[np.abs(Ls) <= 3], 0), nan=-3000)
        k0 = np.flatnonzero(ant >= E.ANT[0]-1e-9)[0]
        hy = E.hyoid(mx[k0:], ant[k0:], sup)                      # 설골은 기존 창 그대로
        if not hy: continue
        for vv in AP_VARIANTS:
            k = np.flatnonzero(ant >= vv["win"]-1e-9)[0]
            r = E.vallecula(mip[k:] < E.AIR, ant[k:], sup, hy, vv["c_lo"], vv["c_hi"], vv["run_back"])
            ok = r is not None and r[2] >= E.MIN_ROWS
            rows.append(dict(Patient=f"P_{pid}", variant=ap_key(vv), **vv, hy_ant=hy[1], hy_sup=hy[2],
                             Eb_ant=r[0] if ok else np.nan, Eb_sup=r[1] if ok else np.nan, n_rows=r[2] if r else 0))
        print(pid, end=" ", flush=True)
    d = pd.DataFrame(rows); d.round(2).to_csv(OUT/"variants_ap.csv", index=False, encoding="utf-8-sig")
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'PAL'").set_index("Patient").excel.dropna()
    base = d[d.variant == ap_key(AP_BASE)].set_index("Patient"); out = []
    for k, g in d.groupby("variant", sort=False):
        g = g.set_index("Patient"); dd = g.Eb_ant.notna(); off = ((g.Eb_ant-g.hy_ant)[dd].median(), (g.Eb_sup-g.hy_sup)[dd].median())
        ea, es = g.Eb_ant.where(dd, g.hy_ant+off[0]), g.Eb_sup.where(dd, g.hy_sup+off[1]); pal = np.hypot(ea, es)
        idx = pal.index.intersection(xl.index); e = pal[idx]-xl[idx]
        ch = ((g.Eb_ant-base.Eb_ant).abs() > .01) | (dd != base.Eb_ant.notna())
        out.append(dict(variant=k, **{c: g[c].iloc[0] for c in ("c_lo", "c_hi", "run_back", "win")}, direct=int(dd.sum()), n=len(idx),
                        MAE=e.abs().mean(), r=np.corrcoef(pal[idx], xl[idx])[0, 1], over3=int((e.abs() > 3).sum()), changed=int(ch.sum())))
    s = pd.DataFrame(out).round(3); s.to_csv(OUT/"summary_ap.csv", index=False, encoding="utf-8-sig"); print("\n", s.drop(columns="variant").to_string(index=False))



# ---------- 설골: 정중 두께(±T)·탐색 창 앞뒤 경계. Eb는 확정 규칙으로 따라 다시 찾는다 ----------
HY_BASE = dict(T=3.0, lo=-33.0, hi=12.0)
HY_VARIANTS = ([dict(HY_BASE, T=t) for t in (0.0, 1.2, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)]
               + [dict(HY_BASE, lo=l) for l in (-25.0, -29.0, -37.0, -41.0)]
               + [dict(HY_BASE, hi=h) for h in (4.0, 8.0, 16.0, 20.0)])


def hy_key(v): return f"T{v['T']:g}_lo{v['lo']:g}_hi{v['hi']:g}"


def hy_main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.stem.split("_")[1]) for p in (CB/"nrrd").glob("P_*.nrrd")); rows = []; keep = E.HYOID_ANT
    for pid in ids:
        P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); o, R, _ = E.midline_frame(P)
        ant, sup, Ls = np.arange(*E.ANT, E.STEP), np.arange(*E.SUP, E.STEP), np.arange(-12, 12+1e-9, E.STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
        vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-3000).reshape(len(Ls), len(ant), len(sup)); del v
        vol[vol < -2500] = np.nan
        air = np.nanmin(vol[np.abs(Ls) <= 6+1e-9], 0) < E.AIR
        mxs = {t: np.nan_to_num(np.nanmax(vol[np.abs(Ls) <= t+1e-9], 0), nan=-3000) for t in {vv["T"] for vv in HY_VARIANTS}}
        for vv in HY_VARIANTS:
            E.HYOID_ANT = (vv["lo"], vv["hi"]); hy = E.hyoid(mxs[vv["T"]], ant, sup); E.HYOID_ANT = keep
            r = E.vallecula(air, ant, sup, hy) if hy else None; ok = r is not None and r[2] >= E.MIN_ROWS
            rows.append(dict(Patient=f"P_{pid}", variant=hy_key(vv), **vv, found=bool(hy),
                             hy_ant=hy[1] if hy else np.nan, hy_sup=hy[2] if hy else np.nan, H_ant=hy[3] if hy else np.nan, H_sup=hy[4] if hy else np.nan,
                             area=hy[0] if hy else np.nan, Eb_ant=r[0] if ok else np.nan, Eb_sup=r[1] if ok else np.nan))
        print(pid, end=" ", flush=True)
    d = pd.DataFrame(rows); d.round(2).to_csv(OUT/"variants_hy.csv", index=False, encoding="utf-8-sig")
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'PAL'").set_index("Patient").excel.dropna()
    base = d[d.variant == hy_key(HY_BASE)].set_index("Patient"); out = []
    for k, g in d.groupby("variant", sort=False):
        g = g.set_index("Patient"); dd = g.Eb_ant.notna(); off = ((g.Eb_ant-g.hy_ant)[dd].median(), (g.Eb_sup-g.hy_sup)[dd].median())
        ea, es = g.Eb_ant.where(dd, g.hy_ant+off[0]), g.Eb_sup.where(dd, g.hy_sup+off[1]); pal = np.hypot(ea, es)
        idx = pal.dropna().index.intersection(xl.index); e = pal[idx]-xl[idx]; hm = np.hypot(g.H_ant-base.H_ant, g.H_sup-base.H_sup)
        out.append(dict(variant=k, **{c: g[c].iloc[0] for c in ("T", "lo", "hi")}, hyoid=int(g.found.sum()), H_moved_gt1=int((hm > 1).sum()),
                        H_lost=int((base.found & ~g.found).sum()), H_new=int((~base.found & g.found).sum()), direct=int(dd.sum()), n=len(idx),
                        MAE=e.abs().mean(), r=np.corrcoef(pal[idx], xl[idx])[0, 1], over3=int((e.abs() > 3).sum())))
    s = pd.DataFrame(out).round(3); s.to_csv(OUT/"summary_hy.csv", index=False, encoding="utf-8-sig"); print("\n", s.drop(columns="variant").to_string(index=False))


if __name__ == "__main__":
    import sys
    hy_main() if "--hy" in sys.argv else ap_main() if "--ap" in sys.argv else main(COMBO, "_combo") if "--combo" in sys.argv else main()
