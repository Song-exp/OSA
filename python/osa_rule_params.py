"""밝기 규칙(설골·Eb)의 자체 선택 수치가 실제로 무엇을 거르는지 보는 분포 점검. 기존 결과는 건드리지 않는다.
환자마다 cbct_eb_detect와 같은 투영 영상(osa_bone_threshold_check.prep)을 만들고 현재 기준(뼈 400)으로
- 설골: 탐색 창 안 뼈 덩어리 전부의 넓이·앞뒤 길이·위아래 높이·가장자리 접촉, 고른 덩어리, 앞쪽 1/3 꼭대기(문헌형 점 후보)
- Eb: 탐색 높이 안에서 공기가 두 갈래 이상인 줄마다 간격(후두개 두께)·앞 갈래 중심·채택 여부
출력: Result_py/analysis/all83/rule_params/{hyoid_blobs.csv, eb_rows.csv, per_patient.csv}
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import label
import cbct_eb_detect as E
from osa_bone_threshold_check import prep
import osa_experiments as X
from cbct_patient_pipeline import CB

OUT = X.R_/"analysis/all83/rule_params"


def hyoid_blobs(mx, ant, sup):
    win = (ant > E.HYOID_ANT[0]) & (ant < E.HYOID_ANT[1]); lo, hi = np.flatnonzero(win)[[0, -1]]
    lab, n = label((mx > E.BONE) & win[:, None]); rows = []
    for i in range(1, n+1):
        xs, zs = np.nonzero(lab == i)
        a, s = ant[xs], sup[zs]; front = a >= np.percentile(a, 67); back = a <= np.percentile(a, 33)
        rows.append(dict(area=xs.size*E.STEP**2, ap_len=(xs.max()-xs.min())*E.STEP, si_len=(zs.max()-zs.min())*E.STEP,
                         edge=bool(xs.min() <= lo or xs.max() >= hi or zs.min() == 0 or zs.max() == len(sup)-1),
                         back_ant=float(a[back].mean()), back_sup=float(s[back].max()),
                         front_ant=float(a[front][np.argmax(s[front])]), front_sup=float(s[front].max()),
                         ant_mean=float(a.mean()), sup_mean=float(s.mean())))
    return rows


def eb_rows(air, ant, sup, hy):
    lab, n = label(air)
    if not n: return []
    main = np.bincount(lab.ravel())[1:].argmax()+1; out = []
    for k, z in enumerate(sup):
        if not hy[2]-20 <= z <= hy[2]+40: continue                  # 기본 범위(−3~+22)보다 넓게 기록
        rs = [r for r in E.runs(lab[:, k] == main) if ant[r[1]] >= hy[1]-25]
        if len(rs) < 2: continue
        back, front = rs[-2], rs[-1]; c = ant[(front[0]+front[1])//2]; gap = (front[0]-back[1]-1)*E.STEP
        ok = E.GAP[0] <= gap <= E.GAP[1] and hy[1]-18 <= c <= hy[1]+2 and hy[2]-3 <= z <= hy[2]+22
        out.append(dict(dz=float(z-hy[2]), dc=float(c-hy[1]), gap=float(gap), front_w=(front[1]-front[0]+1)*E.STEP, accepted=ok))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.stem.split("_")[1]) for p in (CB/"nrrd").glob("P_*.nrrd"))
    B, R, P = [], [], []
    for pid in ids:
        try: mx, air, ant, sup = prep(pid)
        except Exception as e: print("skip", pid, type(e).__name__); continue
        p = f"P_{pid}"; bl = hyoid_blobs(mx, ant, sup); hy = E.hyoid(mx, ant, sup)
        for b in bl: b |= {"Patient": p, "chosen": bool(hy and abs(b["area"]-hy[0]) < 1e-6)}
        B += bl; rec = {"Patient": p, "n_blobs": len(bl), "hyoid": bool(hy)}
        if hy:
            ch = [b for b in bl if b["chosen"]][0]
            rec |= {"hy_area": ch["area"], "hy_ap": ch["ap_len"], "hy_si": ch["si_len"], "hy_ant": hy[1], "hy_sup": hy[2],
                    "front_ant": ch["front_ant"], "front_sup": ch["front_sup"],
                    "back_vs_front_mm": float(np.hypot(ch["front_ant"]-hy[1], ch["front_sup"]-hy[2]))}
            rows = eb_rows(air, ant, sup, hy)
            for r in rows: r["Patient"] = p
            R += rows; acc = [r for r in rows if r["accepted"]]
            rec |= {"eb_rows_accepted": len(acc), "eb_direct": len(acc) >= E.MIN_ROWS,
                    "eb_dz": acc[0]["dz"] if acc else np.nan, "eb_dc": acc[0]["dc"] if acc else np.nan}
        P.append(rec); print(pid, end=" ", flush=True)
    pd.DataFrame(B).round(2).to_csv(OUT/"hyoid_blobs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(R).round(2).to_csv(OUT/"eb_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(P).round(2).to_csv(OUT/"per_patient.csv", index=False, encoding="utf-8-sig"); print("\nsaved", OUT)


if __name__ == "__main__":
    main()
