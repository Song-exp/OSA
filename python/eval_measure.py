"""평가 세트 71명 자동 계측과 엑셀 대조 — 파일럿에서 고정한 방법으로 한 번만 실행 (계획 7장).

고정 방법: ALI 좌표 그대로, Jung FH(RPo·LPo·ROr), ANB는 MSP 2D 투영, MP-FH는 Go–Me 직선.
결과를 보고 방법을 바꾸지 않습니다. 이 결과가 최종 성능 수치입니다.
랜드마크 QC(사후 추가, 방법은 그대로): 랜드마크 간 거리 13개의 로버스트 z(중앙값·MAD, 83명 기준)가 |z|>4이면 표시.
  주 결과는 71명 전체, QC 표시 환자를 뺀 결과는 보조로만 보고합니다.
결과: Result_py/eval/eval_measures.csv, eval_errors.csv, eval_qc.csv, bland_altman.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from check_landmarks import read_points
from osa.ceph import measures
from pilot_measure import excel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Result_py/eval"
PAIRS = [("ANB", "ANB"), ("Nperp_A", "Nperp_A"), ("Nperp_Pog", "Nperp_Pog"), ("MPFH_GoMe", "MPFH")]


QC_PAIRS = [("RPo", "LPo"), ("ROr", "LOr"), ("RPo", "ROr"), ("LPo", "LOr"), ("N", "Ba"), ("Me", "RGo"), ("Me", "LGo"),
            ("RGo", "LGo"), ("N", "Me"), ("A", "B"), ("B", "Pog"), ("Pog", "Me"), ("N", "RPo")]


def landmark_qc(ids):
    """환자별 랜드마크 간 거리의 로버스트 z가 4를 넘는 항목."""
    d = {}
    for p in ids:
        P = {k: r[["x", "y", "z"]].to_numpy(float) for k, r in read_points(p).iterrows()}
        d[f"P_{p}"] = {f"{a}-{b}": np.linalg.norm(P[a] - P[b]) for a, b in QC_PAIRS}
    d = pd.DataFrame(d).T
    z = (d - d.median()) / (1.4826 * (d - d.median()).abs().median())
    return z.abs().gt(4).apply(lambda r: ", ".join(r.index[r]), axis=1).rename("qc_flag")


def errors(d):
    out = []
    for a, b in PAIRS:
        g = d[[a, f"Excel_{b}"]].dropna()
        e = (g[a] - g[f"Excel_{b}"]).to_numpy()
        lo, hi = boot_ci(e)
        loa = e.mean() + np.array([-1.96, 1.96]) * e.std(ddof=1)
        out.append({"Variable": a, "N": len(e), "MeanDiff": e.mean(), "SD": e.std(ddof=1), "MAE": np.abs(e).mean(),
                    "MAE_CI_lo": lo, "MAE_CI_hi": hi, "LoA_lo": loa[0], "LoA_hi": loa[1],
                    "MaxAbs": np.abs(e).max(), "r": np.corrcoef(g[a], g[f"Excel_{b}"])[0, 1]})
    return pd.DataFrame(out).round(2)


def boot_ci(e, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [np.abs(rng.choice(e, len(e))).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT / "Result_py/cbct_split.csv")
    ids = s[s.Set == "eval"].Patient.str[2:].astype(int).tolist()
    xl = excel()
    rows = []
    for p in ids:
        P = {k: r[["x", "y", "z"]].to_numpy(float) for k, r in read_points(p).iterrows()}
        m = measures(P, "jung")
        rows.append({"Patient": f"P_{p}", **{a: m[a] for a, _ in PAIRS},
                     **{f"Excel_{b}": xl.loc[p, b] for _, b in PAIRS}})
    d = pd.DataFrame(rows)
    qc = landmark_qc(s[s.Set.isin(["pilot", "eval"])].Patient.str[2:].astype(int))
    d = d.join(qc, on="Patient")
    qc.to_csv(OUT / "eval_qc.csv", encoding="utf-8-sig")
    d.round(2).to_csv(OUT / "eval_measures.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i, (a, b) in enumerate(PAIRS):
        g = d[[a, f"Excel_{b}"]].dropna()
        e = (g[a] - g[f"Excel_{b}"]).to_numpy()
        loa = e.mean() + np.array([-1.96, 1.96]) * e.std(ddof=1)
        mean = (g[a] + g[f"Excel_{b}"]) / 2
        ax[i].scatter(mean, e, s=12)
        for y, ls in ((e.mean(), "-"), (loa[0], "--"), (loa[1], "--")):
            ax[i].axhline(y, color="0.4", ls=ls, lw=1)
        ax[i].set_title(f"{a} (n={len(e)})")
        ax[i].set_xlabel("(자동 + 엑셀) / 2")
    ax[0].set_ylabel("자동 - 엑셀")
    fig.tight_layout()
    fig.savefig(OUT / "bland_altman.png", dpi=90)
    ok = d[d.qc_flag == ""]
    e = pd.concat([errors(d).assign(Set="all"), errors(ok).assign(Set="qc_pass")])
    e.to_csv(OUT / "eval_errors.csv", index=False, encoding="utf-8-sig")
    print("자동 − 엑셀 (평가 세트, 고정 방법)\n" + e.to_string(index=False))
    print("\nQC 표시 환자:", d.loc[d.qc_flag != "", ["Patient", "qc_flag"]].to_dict("records"))

    agree = lambda x, y, cut: ((x >= cut) == (y >= cut))[x.notna() & y.notna()].mean()
    for name, g in (("all", d), ("qc_pass", ok)):
        print(f"분류 일치 ({name}, n={len(g)}): ANB≥4 {agree(g.ANB, g.Excel_ANB, 4):.0%}"
              f" | MP-FH≥30 {agree(g.MPFH_GoMe, g.Excel_MPFH, 30):.0%}")
    d["absANB"] = (d.ANB - d.Excel_ANB).abs()
    print("ANB 오차 큰 환자 (>3°):", d.loc[d.absANB > 3, ["Patient", "ANB", "Excel_ANB"]].round(1).to_dict("records"))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
