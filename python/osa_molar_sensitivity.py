"""제1대구치 저항중심(→ TDI) 자체 선택 수치의 민감도. 정의(저항중심 = 뿌리 분지부 가운데, Koo 2017)는 선행연구를 따르고,
"갈라졌다"는 판단 수치(덩어리 넓이 3 mm², 1.5 mm 지속), 최소 치관 길이 3 mm, 좌우 앞뒤 차 상한 8 mm, ALI 옆 치아 판정 하한 4 mm를 하나씩 바꾼다.
TDI = (위 좌우 저항중심 좌우 거리) − (아래 좌우 저항중심 좌우 거리) — cbct_restore_extend와 같은 정의(MxAW_rc − MnAW_rc).
판정은 엑셀 TDI(연구실 수작업) 오차. 환자마다 치아 분할을 한 번 읽는다. 기존 결과 파일은 건드리지 않는다.
출력: Result_py/analysis/all83/molar_sensitivity/{variants.csv, summary.csv}
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, SimpleITK as sitk
import cbct_molar_section as M
from cbct_evidence_figs import teeth_points
from cbct_fh_repair import repair
from cbct_patient_pipeline import CB, OUT as PIPE, landmarks
from osa.ceph import head_frame
from run_ali import ascii_path
import osa_experiments as X

OUT = X.R_/"analysis/all83/molar_sensitivity"
BASE = dict(area=3.0, persist=1.5, crown=3.0, da=8.0, nb=4.0)
VARIANTS = [BASE] + ([dict(BASE, area=a) for a in (1.0, 2.0, 5.0, 8.0)] + [dict(BASE, persist=p) for p in (0.5, 1.0, 2.5, 4.0)]
                     + [dict(BASE, crown=c) for c in (1.0, 2.0, 5.0)] + [dict(BASE, da=d) for d in (5.0, 6.0, 10.0, 15.0)]
                     + [dict(BASE, nb=n) for n in (2.0, 3.0, 6.0)])


def key(v): return "_".join(f"{k}{v[k]:g}" for k in ("area", "persist", "crown", "da", "nb"))


def teeth(pid):
    im = sitk.ReadImage(str(ascii_path(CB/f"seg_teeth/{M.MODEL}/P_{pid}.nii.gz"))); seg = sitk.GetArrayFromImage(im)
    org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
    P = repair(landmarks(pid)[0])[0]; o, R = head_frame(P); pts = teeth_points(pid); out = {}
    for t, l in M.TEETH.items():
        vox = np.argwhere(seg == l); lps = org+(vox[:, ::-1]*sp)@D.T; vol = len(vox)*sp.prod()/1000
        dR = float(np.linalg.norm(lps-pts[t+"R"], axis=1).min()) if t+"R" in pts and len(vox) else np.nan
        out[t] = ((lps-o)@R.T, vol, dR, sp.max())
    return out


def centre(t, q, vol, dR, res, v):
    if not M.MIN_VOL_ML <= vol <= M.MAX_VOL_ML: return None
    f = M.furcation(q, t[0] == "U", res, min_area=v["area"], persist=v["persist"])
    if f and f[1] < v["crown"]: f = None
    if f: c = f[0]
    else:
        if not dR <= 2.: return None
        c = M.level_center(q, t[0] == "U", M.FIXED_MM[t[0]])
        if c is None: return None
    if v["nb"] < dR <= M.ALI_NEIGHBOR[1]: return None                       # 옆 치아 라벨 의심
    return c, bool(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    ids = sorted(int(p.name.split(".")[0].split("_")[1]) for p in (CB/f"seg_teeth/{M.MODEL}").glob("P_*.nii.gz"))
    for pid in ids:
        try: T = teeth(pid)
        except Exception as e: print("skip", pid, type(e).__name__); continue
        for v in VARIANTS:
            C = {t: centre(t, *T[t], v) for t in T}; r = dict(Patient=f"P_{pid}", variant=key(v), **v)
            for arch, a, b in (("Mx", "UL6", "UR6"), ("Mn", "LL6", "LR6")):
                ca, cb = C[a], C[b]
                ok = ca is not None and cb is not None and M.WIDTH[0] <= ca[0][0]-cb[0][0] <= M.WIDTH[1] and abs(ca[0][2]-cb[0][2]) <= M.MAX_DZ and abs(ca[0][1]-cb[0][1]) <= v["da"]
                r[f"{arch}AW"] = ca[0][0]-cb[0][0] if ok else np.nan; r[f"{arch}_furc"] = int(ok and ca[1])+int(ok and cb[1])
            r["TDI"] = r["MxAW"]-r["MnAW"]; rows.append(r)
        print(pid, end=" ", flush=True)
    d = pd.DataFrame(rows); d.round(3).to_csv(OUT/"variants.csv", index=False, encoding="utf-8-sig")
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'TDI'").set_index("Patient").excel.dropna()
    base = d[d.variant == key(BASE)].set_index("Patient"); out = []
    for k, g in d.groupby("variant", sort=False):
        g = g.set_index("Patient"); idx = g.TDI.dropna().index.intersection(xl.index); e = g.TDI[idx]-xl[idx]
        out.append(dict(variant=k, **{c: g[c].iloc[0] for c in BASE}, TDI_n=int(g.TDI.notna().sum()), n_xl=len(idx), r=np.corrcoef(g.TDI[idx], xl[idx])[0, 1],
                        MAE=e.abs().mean(), bias=e.mean(), furc_teeth=int(g.Mx_furc.sum()+g.Mn_furc.sum()), changed=int(((g.TDI-base.TDI).abs() > .3).sum())))
    s = pd.DataFrame(out).round(3); s.to_csv(OUT/"summary.csv", index=False, encoding="utf-8-sig"); print("\n", s.drop(columns="variant").to_string(index=False))


if __name__ == "__main__":
    main()
