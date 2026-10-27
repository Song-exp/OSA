"""A·B·Pog 후보 ①~④b 비교 (파일럿 12명, docs/파일럿_자동계측_1차.md 7장).

① ALI 그대로 / ② FH 최후방점(osa.refine) / ③ msp 현 기준 / ④a sil 현 기준 / ④b drr 현 기준
결과: Result_py/pilot/cand_measures.csv, cand_errors.csv, cand_shift.csv, cand_P_n.png (검수 그림)
파일럿 결과는 방법 선택용이며 최종 성능이 아닙니다.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from osa.ceph import measures
from osa.profile_points import candidates
from osa.refine import refine
from pilot_diagnose import points
from pilot_measure import OUT, PILOT, ROOT, SEG, excel
from run_ali import ascii_path

NAMES = {"ali": "① ALI", "fhmin": "② FH최후방", "msp": "③ 정중면 현", "sil": "④a 마스크투영 현", "drr": "④b DRR 현"}
MEAS = {"ANB": "ANB", "Nperp_A": "Nperp_A", "Nperp_Pog": "Nperp_Pog"}
COL = {"ali": "white", "fhmin": "gray", "msp": "orange", "sil": "lime", "drr": "magenta"}


def load(p):
    rd = lambda f: sitk.ReadImage(str(ascii_path(f)))
    ct = rd(ROOT / f"Result_py/CBCT/nrrd/P_{p}.nrrd")
    dirs = np.array(ct.GetDirection()).reshape(3, 3).T * np.array(ct.GetSpacing())[:, None]
    arr = lambda s: sitk.GetArrayFromImage(rd(SEG / f"P_{p}_seg_{s}.nrrd"))
    return sitk.GetArrayFromImage(ct), arr("MAND"), arr("MAX"), dirs, np.array(ct.GetOrigin())


def figure(p, sets, aux):
    ant, sup, o, R = aux["ant"], aux["sup"], aux["o"], aux["R"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    ext = [ant[0], ant[-1], sup[0], sup[-1]]
    ax[0].imshow(aux["msp_img"].T, cmap="gray", vmin=-500, vmax=2500, origin="lower", extent=ext)
    ax[1].imshow(aux["drr"].T, cmap="gray", origin="lower", extent=ext)
    for a, title in zip(ax, ("정중 단면 (CT)", "모의 측면 X-ray (DRR)")):
        for k, pr in (("msp", "orange"), ("sil", "lime"), ("drr", "magenta")):
            for j in ("MAX", "MAND"):
                a.plot(aux["prof"][k][j], sup, color=pr, lw=0.6, alpha=0.8)
        for k, P in sets.items():
            for lm, mk in (("A", "o"), ("B", "s"), ("Pog", "^")):
                hp = R @ (P[lm] - o)
                a.plot(hp[1], hp[2], mk, mfc="none", mec=COL[k], ms=8, mew=1.5)
        a.set_title(f"P_{p} {title}", fontfamily="Malgun Gothic", fontsize=10)
        a.set_xlim(ant[0], ant[-1])
    ax[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc="none", mec=COL[k], label=NAMES[k]) for k in sets],
                 loc="lower left", fontsize=8, prop={"family": "Malgun Gothic", "size": 8})
    fig.text(0.5, 0.01, "○=A  □=B  △=Pog · 선: 윤곽(주황 msp, 초록 sil, 자홍 drr) · 오른쪽 = 앞",
             ha="center", fontfamily="Malgun Gothic", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / f"cand_P_{p}.png", dpi=80)
    plt.close(fig)


def main():
    xl = excel()
    rows, shifts = [], []
    for p in PILOT:
        P = points(p)
        ct, mand, maxm, dirs, org = load(p)
        sets = {"ali": P, "fhmin": refine(P, mand, maxm, dirs, org)[0]}
        cand, aux = candidates(P, ct, mand, maxm, dirs, org)
        sets.update(cand)
        from osa.profile_points import sample
        aux["msp_img"] = sample(ct.astype(np.float32), dirs, org, aux["o"], aux["R"], np.array([0.0]),
                                aux["ant"], aux["sup"], order=1)[0]
        for k, Q in sets.items():
            m = measures(Q)
            rows.append({"Patient": f"P_{p}", "Cand": k, **{v: m[v] for v in MEAS},
                         **{f"Excel_{v}": xl.loc[p, e] for v, e in MEAS.items()}})
            for lm in ("A", "B", "Pog"):
                d = aux["R"] @ (Q[lm] - P[lm])
                shifts.append({"Patient": f"P_{p}", "Cand": k, "Landmark": lm, "d_ant": d[1], "d_sup": d[2],
                               "dist": np.linalg.norm(d)})
        figure(p, sets, aux)
        print(f"P_{p} 완료", flush=True)
    d = pd.DataFrame(rows)
    d.round(2).to_csv(OUT / "cand_measures.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(shifts)
    s.round(2).to_csv(OUT / "cand_shift.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(0)
    err = []
    for k, g in d.groupby("Cand", sort=False):
        for v in MEAS:
            e = (g[v] - g[f"Excel_{v}"]).to_numpy()
            bs = [np.abs(rng.choice(e, e.size)).mean() for _ in range(5000)]
            err.append({"Cand": NAMES[k], "Variable": v, "MeanDiff": e.mean(), "MAE": np.abs(e).mean(),
                        "CI_low": np.percentile(bs, 2.5), "CI_high": np.percentile(bs, 97.5), "MaxAbs": np.abs(e).max(),
                        "r": np.corrcoef(g[v], g[f"Excel_{v}"])[0, 1]})
        agree = ((g.ANB >= 4) == (g.Excel_ANB >= 4)).mean()
        err.append({"Cand": NAMES[k], "Variable": "ANB≥4 분류 일치", "MAE": agree})
    e = pd.DataFrame(err)
    e.round(2).to_csv(OUT / "cand_errors.csv", index=False, encoding="utf-8-sig")
    print("\n후보별 오차 (자동 − 엑셀, 파일럿 12명; 방법 선택용)\n" + e.round(2).to_string(index=False))
    print("\nALI 대비 평균 이동 (mm; + = 앞/위)\n" +
          s[s.Cand != "ali"].groupby(["Cand", "Landmark"])[["d_ant", "d_sup", "dist"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
