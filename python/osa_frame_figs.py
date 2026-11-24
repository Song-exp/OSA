"""머리 좌표계 설명 그림 3장 (전처리 보고서 2절 ①용).
(가) 좌표계 세우기: 예시 환자의 시상·축 단면에 FH, 원점 N, 축, 정중면
(나) 문제: 귓구멍 점이 잘못 찍힌 환자(P_37, 오른쪽 Po)의 관상 단면에서 기운 FH
(다) 해결 장치: 같은 환자의 대칭 보정 Po와 3점 FH, 4점 최소제곱 평면
출력: Result_py/analysis/all83/preproc/tools/frame_{setup,error,fix}.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from cbct_patient_pipeline import CB, landmarks
from cbct_fh_repair import repair
from check_landmarks import read_nrrd
from osa.ceph import head_frame, fh_normal
import osa_experiments as X

OUT = X.R_/"analysis/all83/preproc/tools"
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
RED, CYAN, YEL, GRN = "#ff4d4d", "#2fd3ff", "#ffd400", "#5ee36b"


class View:
    def __init__(self, pid, P_frame):
        self.raw, self.dirs, self.org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); self.o, self.R = head_frame(P_frame)

    def loc(self, p): return self.R@(np.asarray(p, float)-self.o)                       # 머리 좌표 (좌, 앞, 위)

    def img(self, plane, a, b, c):
        A, B = np.meshgrid(a, b, indexing="ij"); z = np.full_like(A, c)
        Q = {"sag": (z, A, B), "cor": (A, z, B), "ax": (A, B, z)}[plane]; Q = np.stack(Q, -1).reshape(-1, 3)
        w = self.o+Q@self.R; idx = np.linalg.solve(self.dirs.T, (w-self.org).T)[::-1]
        return map_coordinates(self.raw, idx, order=1, cval=-1000).reshape(A.shape)

    def show(self, ax, plane, a, b, c, xl, yl):
        ax.imshow(self.img(plane, a, b, c).T, origin="lower", cmap="gray", vmin=-700, vmax=1800, extent=[a[0], a[-1], b[0], b[-1]])
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8); ax.tick_params(labelsize=7)


def mark(ax, x, y, name, col, m="o"):
    ax.plot(x, y, m, color=col, ms=6, mew=2); ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color="white", bbox=dict(fc="black", alpha=.6, ec="none", pad=1))


def plane_line(ax, V, pts, nrm, a0, xs, col, lab, ls="-"):
    """평면(점 pts 평균, 법선 nrm, 영상 좌표)이 관상 단면(앞 = a0)과 만나는 선."""
    p0, n = V.loc(np.mean(pts, 0)), V.R@nrm
    zs = (n@p0-n[1]*a0-n[0]*xs)/n[2]; ax.plot(xs, zs, ls, color=col, lw=2, label=lab)


def setup(pid=4):
    P = landmarks(pid)[0]; V = View(pid, P); q = {k: V.loc(v) for k, v in P.items()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    a, s = np.arange(-110, 30, .5), np.arange(-120, 40, .5); V.show(ax[0], "sag", a, s, 0., "앞 + (mm)", "위 + (mm)")
    fz = np.mean([q[k][2] for k in ("RPo", "LPo", "ROr")]); ax[0].axhline(fz, color=YEL, lw=2, label="FH (RPo·LPo·ROr)")
    for k, col in (("RPo", RED), ("ROr", YEL), ("Ba", GRN)): mark(ax[0], q[k][1], q[k][2], k, col)
    mark(ax[0], 0, 0, "N (원점)", CYAN)
    ax[0].annotate("", (18, 0), (0, 0), arrowprops=dict(arrowstyle="->", color=CYAN, lw=2)); ax[0].annotate("", (0, 18), (0, 0), arrowprops=dict(arrowstyle="->", color=CYAN, lw=2))
    ax[0].text(20, -3, "앞", color=CYAN, fontsize=9); ax[0].text(2, 20, "위", color=CYAN, fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_title("옆에서 본 단면: FH와 원점 N (좌우 점은 정중면에 투영)", fontsize=9)
    L = np.arange(-75, 75, .5); a2 = np.arange(-110, 30, .5); V.show(ax[1], "ax", L, a2, fz, "왼쪽 + (mm)", "앞 + (mm)")
    ax[1].axvline(0, color=CYAN, lw=2, label="정중면 (N·Ba를 지나 FH에 수직)")
    for k, col in (("RPo", RED), ("LPo", RED), ("ROr", YEL), ("LOr", YEL), ("N", CYAN), ("Ba", GRN)): mark(ax[1], q[k][0], q[k][1], k, col)
    ax[1].legend(fontsize=7, loc="lower left"); ax[1].set_title("위에서 본 단면 (FH 높이)", fontsize=9)
    fig.suptitle(f"머리 좌표계 세우기 (예시 환자 P_{pid})", fontsize=11); fig.savefig(OUT/"frame_setup.png", dpi=90); plt.close(fig)


def error_fix(pid=37):
    P = landmarks(pid)[0]; Q, log = repair(P); V = View(pid, Q)                       # 화면은 보정한 좌표계로 세움
    raw = {k: V.loc(P[k]) for k in ("RPo", "LPo", "ROr", "LOr")}; fix = {k: V.loc(Q[k]) for k in ("RPo", "LPo", "ROr", "LOr")}
    a0 = float(np.mean([fix["RPo"][1], fix["LPo"][1]])); L = np.arange(-85, 85, .5); S = np.arange(-90, 40, .5)
    def base(title):
        fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained"); V.show(ax, "cor", L, S, a0, "왼쪽 + (mm)", "위 + (mm)"); ax.set_title(title, fontsize=9); return fig, ax
    fig, ax = base(f"P_{pid}: 오른쪽 귓구멍 점(RPo)이 잘못 찍혀 FH가 기욺 (귓구멍 높이 관상 단면)")
    plane_line(ax, V, [P["RPo"], P["LPo"], P["ROr"]], fh_normal(P, "jung"), a0, L, RED, "잘못 찍힌 점으로 만든 FH")
    mark(ax, raw["RPo"][0], raw["RPo"][2], "RPo (잘못 찍힘)", RED, "x"); mark(ax, raw["LPo"][0], raw["LPo"][2], "LPo", GRN)
    ax.legend(fontsize=7, loc="lower left"); ax.set_ylim(S[0], S[-1]); fig.savefig(OUT/"frame_error.png", dpi=90); plt.close(fig)
    fig, ax = base(f"P_{pid}: 대칭 보정과 4점 수평면")
    plane_line(ax, V, [P["RPo"], P["LPo"], P["ROr"]], fh_normal(P, "jung"), a0, L, RED, "보정 전 3점 FH", "--")
    plane_line(ax, V, [Q["RPo"], Q["LPo"], Q["ROr"]], fh_normal(Q, "jung"), a0, L, YEL, "대칭 보정 후 3점 FH")
    plane_line(ax, V, [Q[k] for k in ("RPo", "LPo", "ROr", "LOr")], fh_normal(Q, "yao4"), a0, L, CYAN, "4점 최소제곱 평면", ":")
    mark(ax, raw["RPo"][0], raw["RPo"][2], "원래 RPo", RED, "x"); mark(ax, fix["RPo"][0], fix["RPo"][2], "대칭 대체 RPo", YEL); mark(ax, fix["LPo"][0], fix["LPo"][2], "LPo", GRN)
    ax.legend(fontsize=7, loc="lower left"); ax.set_ylim(S[0], S[-1]); fig.savefig(OUT/"frame_fix.png", dpi=90); plt.close(fig)
    return log


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); setup(); print(error_fix())
