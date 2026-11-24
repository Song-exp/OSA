"""비골 하단(Rh)·NAV 자체 선택 수치의 민감도: 공기 기준(Roh −400~−500 범위 안 선택), Rh 뼈 밝기 300, 정중 ±1 mm, 물러남 3 mm, N 아래 8 mm부터.
복원 NAV(NAV_ts = 경계 평면 안 + TotalSegmentator 비강 라벨 안 공기)를 하나씩 바꿔 다시 잰다. 판정은 엑셀 NAV(연구실 수작업) 오차.
cbct_nav.measure와 같은 영상 준비·경계 계산. 기존 결과 파일은 건드리지 않는다.
출력: Result_py/analysis/all83/nav_sensitivity/{variants.csv, summary.csv}
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import map_coordinates
import cbct_nav as nv
from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import CB, OUT as PIPE, landmarks
from check_landmarks import read_nrrd
import osa_experiments as X

OUT = X.R_/"analysis/all83/nav_sensitivity"
BASE = dict(air=-400.0, bone=300.0, w=1.0, jump=3.0, skip=8.0)
VARIANTS = [BASE] + ([dict(BASE, air=a) for a in (-300.0, -450.0, -500.0, -600.0)]
                     + [dict(BASE, bone=b) for b in (200.0, 400.0, 500.0)]
                     + [dict(BASE, w=w) for w in (0.4, 2.0, 3.0)]
                     + [dict(BASE, jump=j) for j in (1.5, 2.0, 4.0, 6.0)]
                     + [dict(BASE, skip=s) for s in (4.0, 6.0, 10.0, 12.0)])


def key(v): return f"air{v['air']:g}_bone{v['bone']:g}_w{v['w']:g}_j{v['jump']:g}_s{v['skip']:g}"


def main():
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    ids = sorted(int(p.stem.split("_")[1]) for p in (CB/"nrrd").glob("P_*.nrrd"))
    for pid in ids:
        P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
        o, R, _ = midline_frame(P); pts = {k: R@(P[k]-o) for k in ("N", "Ba", "S", "ANS")}
        Ls = np.arange(-nv.HALF_L, nv.HALF_L+1e-9, nv.STEP); ant = np.arange(-40., pts["N"][1]+25, nv.STEP); sup = np.arange(0., pts["N"][2]+10, nv.STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
        vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup)); del v
        ts = nv.ts_nasal(pid, o+G@R); del G
        if ts is None: print("ts?", pid, end=" "); continue
        ts = ts.reshape(vol.shape)
        rws = np.flatnonzero((sup <= pts["N"][2]) & (sup >= 5.))[::-1]; a_lim = ant <= pts["N"][1]+20
        A, Z = np.meshgrid(ant, sup, indexing="ij"); N, Ba, S, ANS = ((pts[k][1], pts[k][2]) for k in ("N", "Ba", "S", "ANS"))
        fixed = nv.side_of_line(A, Z, N, Ba, (ANS[0]-10, 5.)) & nv.side_of_line(A, Z, S, (0, 0), (ANS[0]/2, 10.)) & (Z >= 0)
        airs = {}; mids = {}
        for vv in VARIANTS:
            mk = (vv["bone"], vv["w"])
            if mk not in mids: mids[mk] = vol[np.abs(Ls) <= vv["w"]+1e-9].max(0) > vv["bone"]
            mid = mids[mk]
            front = np.array([ant[(mid[:, k] & a_lim)].max() if (mid[:, k] & a_lim).any() else np.nan for k in rws])
            nv.JUMP, nv.SKIP_BELOW_N = vv["jump"], vv["skip"]; i = nv.rhinion(front, sup[rws]); nv.JUMP, nv.SKIP_BELOW_N = BASE["jump"], BASE["skip"]
            if i is None: rows.append(dict(Patient=f"P_{pid}", variant=key(vv), **vv)); continue
            rh = (front[i], sup[rws[i]])
            box = fixed & nv.side_of_line(A, Z, rh, ANS, (ANS[0]/2, 10.)) & (nv.side_of_line(A, Z, N, rh, (ANS[0]/2, 10.)) | (Z < rh[1]))
            if vv["air"] not in airs: airs[vv["air"]] = (vol > -1200) & (vol < vv["air"]) & ts
            rows.append(dict(Patient=f"P_{pid}", variant=key(vv), **vv, Rh_ant=rh[0], Rh_sup=rh[1], NAV_ts=float((airs[vv["air"]] & box[None]).sum())*nv.STEP**3))
        del vol, ts, airs, mids; print(pid, end=" ", flush=True)
    d = pd.DataFrame(rows); d.round(2).to_csv(OUT/"variants.csv", index=False, encoding="utf-8-sig")
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'NAV'").set_index("Patient").excel.dropna()
    base = d[d.variant == key(BASE)].set_index("Patient"); out = []
    for k, g in d.groupby("variant", sort=False):
        g = g.set_index("Patient"); idx = g.NAV_ts.dropna().index.intersection(xl.index); e = g.NAV_ts[idx]-xl[idx]
        mv = np.hypot(g.Rh_ant-base.Rh_ant, g.Rh_sup-base.Rh_sup)
        out.append(dict(variant=k, **{c: g[c].iloc[0] for c in ("air", "bone", "w", "jump", "skip")}, rh_found=int(g.Rh_ant.notna().sum()), rh_moved_gt1=int((mv > 1).sum()),
                        n=len(idx), r=np.corrcoef(g.NAV_ts[idx], xl[idx])[0, 1], MAE_mL=e.abs().mean()/1000, bias_mL=e.mean()/1000, rel_MAE=(e.abs()/xl[idx]).median()))
    s = pd.DataFrame(out).round(3); s.to_csv(OUT/"summary.csv", index=False, encoding="utf-8-sig"); print("\n", s.drop(columns="variant").to_string(index=False))


if __name__ == "__main__":
    main()
