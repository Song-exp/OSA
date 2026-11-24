"""전처리 보고서 6절(밝기 규칙) 그림 2장. 예시 환자 1명(후두개곡 직접 검출 환자).
rule_common.png: (가) 원본 정중 단면 (나) ① 밝기로 칸 표시(공기 −1200~−400, 뼈 400 이상) (다) ② 덩어리 고르기(인두 공기·설골만 남김)
rule_hyoid.png: 정중 ±3 mm 뼈 최대 투영에서 설골 덩어리, 뒤쪽 1/3, 설골점, 하악평면(Go–Me), Eb
계산은 cbct_eb_detect·cbct_pmca 함수를 그대로 쓴다(그림용 재현). 출력: Result_py/analysis/all83/preproc/tools/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, map_coordinates
import cbct_eb_detect as E, cbct_pmca as pm
from cbct_patient_pipeline import CB, OUT as PIPE, landmarks
from check_landmarks import read_nrrd
import osa_experiments as X

OUT = X.R_/"analysis/all83/preproc/tools"; PID = 5
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
AIRC, BONEC = "#2f8cff", "#ff8a1f"


def sample(v, dirs, org, o, R, Ls, ant, sup):
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    return map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-3000).reshape(len(Ls), len(ant), len(sup))


def mask_img(ax, m, ext, col, a):
    ax.imshow(np.ma.masked_where(~m.T, m.T), origin="lower", cmap=ListedColormap([col]), alpha=a, extent=ext, interpolation="nearest")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = landmarks(PID)[0]; v, dirs, org = read_nrrd(CB/f"nrrd/P_{PID}.nrrd"); v = v.astype(np.float32); o, R, _ = E.midline_frame(P)
    q = {k: R@(P[k]-o) for k in ("S", "PNS", "ANS", "RGo", "LGo", "Me")}
    eb = pd.read_csv(PIPE/"eb_review/v2/eb_candidates.csv", encoding="utf-8-sig").set_index("Patient").loc[f"P_{PID}"]
    # 넓은 정중 단면 (L = 0)
    ant, sup = np.arange(-70, 45, E.STEP), np.arange(-100, 35, E.STEP); ext = [ant[0], ant[-1], sup[0], sup[-1]]
    sl = sample(v, dirs, org, o, R, np.array([0.]), ant, sup)[0]
    air, bone = (sl > pm.AIR[0]) & (sl < pm.AIR[1]), sl > E.BONE
    # 인두 기도 덩어리: cbct_pmca.airway (3D, 위 경계 TOP, 아래 Eb, S–PNS 뒤쪽) → 정중 단면
    Ls = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP); a2 = np.arange(pm.ANT[0], pm.ANT[1]+1e-9, pm.STEP); s2 = np.arange(eb.Eb_sup, pm.TOP+1e-9, pm.STEP)
    aw = pm.airway(sample(v, dirs, org, o, R, Ls, a2, s2), Ls, a2, s2, q["S"][1], q["S"][2])
    k0 = int(np.argmin(np.abs(Ls))); keep_air = np.zeros_like(air)
    ia, iz = np.searchsorted(ant, a2[0]), np.searchsorted(sup, s2[0])
    keep_air[ia:ia+len(a2), iz:iz+len(s2)] = aw[k0][:min(len(a2), len(ant)-ia), :min(len(s2), len(sup)-iz)]
    # 설골 덩어리: cbct_eb_detect.hyoid와 같은 규칙(정중 ±3 mm 최대 투영)
    Lh = np.arange(-3, 3+1e-9, E.STEP); ah, sh = np.arange(*E.ANT, E.STEP), np.arange(*E.SUP, E.STEP)
    mx = sample(v, dirs, org, o, R, Lh, ah, sh).max(0)    # 영상 밖 = −3000 (cbct_eb_detect의 NaN 처리와 같은 결과)
    win = (ah > E.HYOID_ANT[0]) & (ah < E.HYOID_ANT[1]); lo, hi = np.flatnonzero(win)[[0, -1]]
    lab, n = label((mx > E.BONE) & win[:, None]); best, blob = None, None
    for i in range(1, n+1):
        xs, zs = np.nonzero(lab == i); area = xs.size*E.STEP**2
        if xs.min() <= lo or xs.max() >= hi or zs.min() == 0 or zs.max() == len(sh)-1: continue
        if 8 <= area <= 150 and (xs.max()-xs.min())*E.STEP < 18 and (best is None or area > best): best, blob = area, lab == i
    hy = E.hyoid(mx, ah, sh)
    keep_bone = np.zeros_like(bone)
    ib, jb = np.searchsorted(ant, ah[0]), np.searchsorted(sup, sh[0])
    keep_bone[ib:ib+len(ah), jb:jb+len(sh)] = blob
    # 그림 1: 공통 과정
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    for a, t in zip(ax, ("(가) 원본 정중 단면", "① 밝기로 칸 표시: 공기(파랑) -1200~-400, 뼈(주황) 400 이상", "② 덩어리 고르기: 인두 공기·설골만 남김")):
        a.imshow(sl.T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=ext); a.set_title(t, fontsize=10)
        a.set_xlabel("앞 + (mm, 원점 PNS)", fontsize=8); a.set_ylabel("위 + (mm)", fontsize=8); a.tick_params(labelsize=7)
    mask_img(ax[1], air, ext, AIRC, .75); mask_img(ax[1], bone, ext, BONEC, .75)
    mask_img(ax[2], air & ~keep_air, ext, AIRC, .18); mask_img(ax[2], bone & ~keep_bone, ext, BONEC, .15)
    mask_img(ax[2], keep_air, ext, AIRC, .9); mask_img(ax[2], keep_bone, ext, BONEC, .95)
    ax[2].annotate("인두 기도 덩어리\n(아래 끝 = Eb 높이에서 자름)", (-12, -20), xytext=(15, -5), color="white", fontsize=9, arrowprops=dict(arrowstyle="->", color="white"), bbox=dict(fc="black", alpha=.6, ec="none"))
    if hy: ax[2].annotate("설골 덩어리", (hy[1], hy[2]), xytext=(hy[1]+18, hy[2]-14), color="white", fontsize=9, arrowprops=dict(arrowstyle="->", color="white"), bbox=dict(fc="black", alpha=.6, ec="none"))
    ax[2].annotate("입안·부비동 공기 등\n(따로 떨어진 덩어리 → 버림)", (25, -15), xytext=(5, 15), color="white", fontsize=8, bbox=dict(fc="black", alpha=.6, ec="none"))
    fig.suptitle(f"밝기 규칙 공통 과정 (예시 환자 P_{PID}, 정중 단면 · 실제 계산은 3D 영상 전체)", fontsize=11)
    fig.savefig(OUT/"rule_common.png", dpi=85); plt.close(fig)
    # 그림 2: 설골
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained"); e2 = [ah[0], ah[-1], sh[0], sh[-1]]
    ax.imshow(mx.T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=e2)
    mask_img(ax, blob, e2, BONEC, .7)
    if hy:
        xs = ah[np.nonzero(blob)[0]]; ax.axvline(np.percentile(xs, 33), color=BONEC, ls="--", lw=1.2, label="설골 뒤쪽 1/3 경계")
        ax.plot(hy[3], hy[4], "o", color="#ffd400", ms=9, mew=2, mfc="none", label="설골점 H (몸통의 가장 앞쪽·위쪽 점)")
        ax.plot(hy[1], hy[2], "+", color="#ffd400", ms=14, mew=2.5, label="Eb 탐색 기준점 (뒤쪽 1/3의 꼭대기)")
        ax.axhline(hy[5], color="#ff6b6b", ls=":", lw=1.5, label="Eb 탐색 아래 한계 (설골 몸통 아래 끝)")
    go = (q["RGo"]+q["LGo"])/2; me = q["Me"]; xx = np.array(e2[:2]); t = (xx-go[1])/(me[1]-go[1])
    ax.plot(xx, go[2]+t*(me[2]-go[2]), color="#5ee36b", lw=2, label="하악평면 (Go–Me, ALI-CBCT 점)")
    if np.isfinite(eb.Eb_ant): ax.plot(eb.Eb_ant, eb.Eb_sup, "x", color=AIRC, ms=12, mew=3, label="Eb (후두개곡 최하점)")
    ax.set_xlim(e2[:2]); ax.set_ylim(e2[2:]); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("설골 찾기 (정중 ±3 mm 뼈 최대 투영, 400 이상 덩어리)", fontsize=9)
    ax.set_xlabel("앞 + (mm, 원점 PNS)", fontsize=8); ax.set_ylabel("위 + (mm)", fontsize=8)
    fig.savefig(OUT/"rule_hyoid.png", dpi=85); plt.close(fig)
    print("saved", hy, eb.method)


if __name__ == "__main__":
    main()
