"""설골 뼈 밝기 기준 비교: 450(기존, 근거 기록 없음) 대 400(Dong 2019, 복셀 0.40 mm 최적값), 참고 226(Mimics CT 기본)·300.
cbct_eb_detect.analyse와 같은 영상 준비(정중 좌표계, 정중 ±3 mm 최대 투영, ±6 mm 최소 투영)를 환자마다 한 번 하고, 기준만 바꿔 hyoid·vallecula를 다시 찾는다.
기존 결과 파일은 건드리지 않는다. 출력: Result_py/analysis/all83/bone_threshold/compare.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import map_coordinates
import cbct_eb_detect as E
from cbct_patient_pipeline import CB, landmarks
from check_landmarks import read_nrrd
import osa_experiments as X

OUT = X.R_/"analysis/all83/bone_threshold"; TH = (450, 400, 300, 226)


def prep(pid):
    P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
    o, R, _ = E.midline_frame(P)
    ant, sup, Ls = np.arange(*E.ANT, E.STEP), np.arange(*E.SUP, E.STEP), np.arange(-6, 6+1e-9, E.STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-3000).reshape(len(Ls), len(ant), len(sup))
    vol[vol < -2500] = np.nan
    mip, mx = np.nanmin(vol, 0), np.nanmax(vol[np.abs(Ls) <= 3], 0)
    return np.nan_to_num(mx, nan=-3000), mip < E.AIR, ant, sup


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.stem.split("_")[1]) for p in (CB/"nrrd").glob("P_*.nrrd"))
    rows = []
    for pid in ids:
        try: mx, air, ant, sup = prep(pid)
        except Exception as e: print("skip", pid, type(e).__name__); continue
        r = {"Patient": f"P_{pid}"}
        for th in TH:
            E.BONE = th; hy = E.hyoid(mx, ant, sup)
            if hy:
                r |= {f"hy_ant_{th}": hy[1], f"hy_sup_{th}": hy[2], f"hy_area_{th}": hy[0]}
                val = E.vallecula(air, ant, sup, hy)
                if val and val[2] >= E.MIN_ROWS: r |= {f"Eb_ant_{th}": val[0], f"Eb_sup_{th}": val[1]}
        rows.append(r); print(pid, end=" ", flush=True)
    E.BONE = 450
    d = pd.DataFrame(rows).set_index("Patient"); d.round(2).to_csv(OUT/"compare.csv", encoding="utf-8-sig")
    print()
    for th in TH[1:]:
        hy = np.hypot(d[f"hy_ant_{th}"]-d.hy_ant_450, d[f"hy_sup_{th}"]-d.hy_sup_450)
        eb = np.hypot(d.get(f"Eb_ant_{th}")-d.Eb_ant_450, d.get(f"Eb_sup_{th}")-d.Eb_sup_450)
        print(f"{th} vs 450 | 설골 찾음 {d[f'hy_ant_{th}'].notna().sum()} (450: {d.hy_ant_450.notna().sum()}) | 설골 이동 중앙 {hy.median():.2f} mm, 최대 {hy.max():.1f} ({hy.idxmax() if hy.notna().any() else '-'}), 1 mm 초과 {int((hy > 1).sum())}명"
              f" | Eb(직접) 찾음 {d[f'Eb_ant_{th}'].notna().sum()} (450: {d.Eb_ant_450.notna().sum()}) | Eb 이동 중앙 {eb.median():.2f}, 1 mm 초과 {int((eb > 1).sum())}명")
    return d


if __name__ == "__main__":
    main()
