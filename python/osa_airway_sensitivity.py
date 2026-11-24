"""인두 기도 자체 선택 수치의 민감도: 기도 덩어리 고르기 정중 폭(±8 mm), PMCA 아래 여유(Eb 위 5 mm), 중심선 매끄럽게(5 mm).
Eb는 확정 규칙의 값(shape/extra/relation.csv, 85명 같은 규칙). 하나씩 바꾸고 나머지는 기본값. 판정은 엑셀 PMCA(연구실 수작업)와의 오차.
cbct_pmca.measure와 같은 영상 준비·단면 계산을 쓴다. 기존 결과 파일은 건드리지 않는다.
출력: Result_py/analysis/all83/airway_sensitivity/{variants.csv, summary.csv}
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.ndimage import label, map_coordinates
import cbct_pmca as pm
from cbct_eb_detect import midline_frame
from cbct_patient_pipeline import CB, OUT as PIPE, landmarks
from check_landmarks import read_nrrd
import osa_experiments as X

OUT = X.R_/"analysis/all83/airway_sensitivity"
BASE = dict(width=8.0, bottom=5.0, smooth=5.0)
VARIANTS = ([dict(BASE, width=w) for w in (2.0, 4.0, 8.0, 12.0, 20.0)]
            + [dict(BASE, bottom=b) for b in (0.0, 1.0, 3.0, 5.0, 8.0, 12.0)]
            + [dict(BASE, smooth=s) for s in (0.4, 2.0, 5.0, 8.0, 12.0)])


def key(v): return f"w{v['width']:g}_b{v['bottom']:g}_s{v['smooth']:g}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rel = pd.read_csv(X.R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient"); rows = []
    for p, eb in rel.Eb_sup.dropna().items():
        pid = int(p[2:]); P = landmarks(pid)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32)
        o, R, _ = midline_frame(P); s = R@(P["S"]-o)
        Ls = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP); ant = np.arange(pm.ANT[0], pm.ANT[1]+1e-9, pm.STEP); sup = np.arange(eb, pm.TOP+1e-9, pm.STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
        vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup)); del v
        A, Z = np.meshgrid(ant, sup, indexing="ij")
        behind = (s[1]*Z-s[2]*A)*np.sign(s[1]*(-30.0)-s[2]*(-30.0)) > 0      # cbct_pmca.airway와 같은 S–PNS 뒤쪽 조건
        lab, n = label((vol > pm.AIR[0]) & (vol < pm.AIR[1]) & behind[None]); del vol
        cache = {}
        for vv in VARIANTS:
            ck = (vv["width"], vv["smooth"])
            if ck not in cache:
                counts = np.bincount(lab[np.abs(Ls) < vv["width"]].ravel(), minlength=n+1); counts[0] = 0
                m = lab == counts.argmax() if n and counts.max() else None
                if m is None: cache[ck] = None
                else:
                    keep = pm.SMOOTH_MM; pm.SMOOTH_MM = vv["smooth"]; _, pe, _ = pm.sections(m, Ls, ant, sup); pm.SMOOTH_MM = keep
                    cache[ck] = pe
            pe = cache[ck]
            A_, z_ = pm.pick(pe, sup, eb+vv["bottom"], 0.0) if pe is not None else (np.nan, np.nan)
            rows.append(dict(Patient=p, variant=key(vv), **vv, PMCA=A_, z=z_, z_above_bottom=z_-(eb+vv["bottom"]) if pe is not None else np.nan))
        print(pid, end=" ", flush=True)
    d = pd.DataFrame(rows); d.round(2).to_csv(OUT/"variants.csv", index=False, encoding="utf-8-sig")
    xl = pd.read_csv(PIPE/"restore_ext/v1/compare_all.csv", encoding="utf-8-sig").query("metric == 'PMCA'").set_index("Patient").excel.dropna()
    base = d[d.variant == key(BASE)].set_index("Patient"); out = []
    for k, g in d.groupby("variant", sort=False):
        g = g.set_index("Patient"); idx = g.PMCA.dropna().index.intersection(xl.index); e = g.PMCA[idx]-xl[idx]
        i2 = idx.drop("P_44", errors="ignore"); e2 = g.PMCA[i2]-xl[i2]
        out.append(dict(variant=k, **{c: g[c].iloc[0] for c in ("width", "bottom", "smooth")}, n=len(idx), MAE=e.abs().mean(), r=np.corrcoef(g.PMCA[idx], xl[idx])[0, 1],
                        r_no44=np.corrcoef(g.PMCA[i2], xl[i2])[0, 1], MAE_no44=e2.abs().mean(), at_bottom=int((g.z_above_bottom < 1).sum()),
                        changed=int(((g.PMCA-base.PMCA).abs() > 1).sum()), P44=g.PMCA.get("P_44", np.nan)))
    s = pd.DataFrame(out).round(3); s.to_csv(OUT/"summary.csv", index=False, encoding="utf-8-sig"); print("\n", s.drop(columns="variant").to_string(index=False))


if __name__ == "__main__":
    main()
