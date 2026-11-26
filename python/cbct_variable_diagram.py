"""CBCT 계측 변수 7개(ANB, Nperp-A, Nperp-Pog, MP-FH, TDI, PHI, NCWR)가 두개골 어디를 재는지 보여 주는 모식도.

실제 환자 영상이 아니라 설명용 그림입니다. 점 위치는 예시이며 실제 평균 위치가 아닙니다.
정의는 docs/CBCT_측정_계획.md 3장(랜드마크·변수 정의)을 따릅니다.
PHI·NCWR의 기준점·높이와 TDI 측정점은 그 문서에서 C등급(정의 불명확)이라 그림에 표시해 둡니다.
결과 숫자: 엑셀 CBCT 변수, P_39(CBCT 본인 영상 없음) 제외 86명.

결과: Result_py/cbct_variable_diagram/cbct_variable_diagram.png
사용: python cbct_variable_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Ellipse, Polygon, Rectangle
from scipy.interpolate import splev, splprep

ROOT = Path(__file__).resolve().parents[1]
EXCEL = ROOT / "Patient_Info_Data_87명_261019.xlsx"
OUT = ROOT / "Result_py" / "cbct_variable_diagram"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9})

INK, MUTED, BONE, BONE_EDGE, SOFT = "#1f2328", "#6b7280", "#f3efe6", "#8a8172", "#e9eef5"
C1, C2, C3 = "#2a78d6", "#e8590c", "#2f9e44"      # 변수 묶음별 색 (글자에는 쓰지 않고 선·점에만)
# 엑셀 열 번호 (0부터) ↔ 머리글 3행 이름
COLS = {22: ("ANB", "ANB"), 23: ("NA", "Nperp_A"), 24: ("NPog", "Nperp_Pog"), 25: ("MPFH", "MP-FH"),
        26: ("NCWR", "NCWR(%)"), 27: ("PHI", "PHI(%)"), 28: ("TDI", "TDI")}
EXCLUDE = [39]                                     # CBCT 본인 영상 없음


def read_excel() -> pd.DataFrame:
    x = pd.read_excel(EXCEL, header=None)
    for c, (_, name) in COLS.items():
        assert str(x.iloc[2, c]).strip() == name, f"엑셀 {c}열 머리글이 예상과 다름: {x.iloc[2, c]!r}"
    r = x.iloc[3:]
    r = r[r[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    d = pd.DataFrame({k: pd.to_numeric(r[c]).values for c, (k, _) in COLS.items()}, index=r[0].astype(int).values)
    return d.drop(EXCLUDE)


def smooth(pts, closed=True, n=400):
    p = np.array(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    tck, _ = splprep(p.T, s=0, per=closed, k=3)
    return np.array(splev(np.linspace(0, 1, n), tck)).T


def dot(ax, xy, label, dx=2, dy=2, ha="left", color=INK):
    ax.plot(*xy, "o", ms=6.5, color=color, mec="white", mew=1.2, zorder=6)
    ax.annotate(label, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9,
                color=INK, fontweight="bold", zorder=7)


def dim(ax, a, b, color, label=None, at=None, ha="center", va="center", fs=9):
    """두 점 사이 양방향 화살표. 글자는 at(데이터 좌표)에 둠."""
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="<->", color=color, lw=1.8, shrinkA=0, shrinkB=0), zorder=5)
    if label:
        ax.text(*at, label, ha=ha, va=va, fontsize=fs, color=INK, zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))


def sagittal(ax):
    # 얼굴이 오른쪽. FH를 수평(y=45)으로 두어 Nperp가 수직선(x=88)이 되게 함
    N, Or, Po = (88, 60), (76, 45), (8, 45)
    A, B, Pog, Me, Go = (92, 0), (84.5, -30), (84, -45), (77, -54), (18, -22)
    skull = [N, (91, 50), (93, 38), (93, 22), (98, 10), (93, 4), A, (93, -9), (70, -10), (50, -6), (42, 4),
             (30, 18), (12, 22), (-5, 26), (-18, 50), (-16, 82), (4, 106), (40, 117), (70, 107), (85, 88), (90, 72)]
    mand = [(89, -15), B, Pog, (83, -51), Me, (50, -38), (32, -30), Go, (14, 0), (14, 20), (17, 30), (23, 30),
            (30, 22), (38, 28), (43, 10), (48, -8), (65, -13)]
    for pts in (skull, mand):
        ax.add_patch(Polygon(smooth(pts), closed=True, fc=BONE, ec=BONE_EDGE, lw=1.4, zorder=1))
    ax.add_patch(Ellipse((76, 56), 16, 22, fc="white", ec=BONE_EDGE, lw=1.2, zorder=2))      # 눈확
    ax.add_patch(Ellipse((8, 41), 6, 8, fc="white", ec=BONE_EDGE, lw=1.2, zorder=2))         # 바깥귀길
    ax.plot([93, 95], [-9, -15], color=BONE_EDGE, lw=3, solid_capstyle="round", zorder=2)      # 위 앞니
    ax.plot([89, 92], [-15, -11], color=BONE_EDGE, lw=3, solid_capstyle="round", zorder=2)     # 아래 앞니

    # 기준선: FH, Nperp
    ax.plot([-10, 106], [45, 45], color=MUTED, lw=1.1, ls="--", zorder=3)
    ax.text(24, 43.5, "FH 평면 (Po–Or)", color=MUTED, fontsize=8.5, va="top")
    ax.plot([88, 88], [72, -62], color=MUTED, lw=1.1, ls="--", zorder=3)
    ax.text(89.5, -62, "Nperp (N을 지나 FH에 수직)", color=MUTED, fontsize=8.5, va="bottom")

    # ANB: N–A선과 N–B선 사이 각
    for P in (A, B):
        v = np.subtract(P, N)
        e = np.add(P, v * 0.08)
        ax.plot([N[0], e[0]], [N[1], e[1]], color=C1, lw=1.8, zorder=4)
    ang = lambda P: np.degrees(np.arctan2(P[1] - N[1], P[0] - N[0]))
    ax.add_patch(Arc(N, 100, 100, theta1=ang(B), theta2=ang(A), color=C1, lw=2.5, zorder=5))
    ax.annotate("ANB", (88.5, 10.5), xytext=(104, 22), fontsize=10, color=INK, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=C1, lw=1), zorder=7)

    # Nperp-A, Nperp-Pog: Nperp에서 수평 거리 (앞쪽 +)
    dim(ax, (88, A[1]), A, C2, "Nperp-A (+)", at=(97, 0), ha="left")
    dim(ax, (88, Pog[1]), Pog, C2, "Nperp-Pog (-)", at=(92, -45), ha="left")

    # MP-FH: 하악평면(Go–Me)과 FH 사이 각
    t = np.array([Me[0] - Go[0], Me[1] - Go[1]], float)
    p0, p1 = np.array(Go) - t * 0.2, np.array(Me) + t * 0.12
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C3, lw=1.8, zorder=4)
    # 하악평면과 FH의 교점은 그림 밖이라, Go를 지나는 FH 평행선으로 같은 각을 표시
    ax.plot([Go[0] - 4, Go[0] + 40], [Go[1], Go[1]], color=C3, lw=1, ls=":", zorder=4)
    ax.text(Go[0] + 41, Go[1], "FH와 평행", color=MUTED, fontsize=8, va="center")
    mp = np.degrees(np.arctan2(t[1], t[0]))
    ax.add_patch(Arc(Go, 44, 44, theta1=mp, theta2=0, color=C3, lw=2.5, zorder=5))
    ax.text(Go[0] + 23, Go[1] - 9, "MP-FH", fontsize=10, color=INK, fontweight="bold", zorder=7)
    ax.text(44, -45, "하악평면 (Go–Me)", color=MUTED, fontsize=8.5, rotation=mp, rotation_mode="anchor", zorder=7)

    for P, lab, dx, dy, ha in [(N, "N", -8, 4, "right"), (Or, "Or", -4, -13, "right"), (Po, "Po", -6, 5, "right"),
                               (A, "A", 6, 6, "left"), (B, "B", -7, -2, "right"), (Pog, "Pog", -7, -6, "right"),
                               (Me, "Me", -4, -13, "right"), (Go, "Go", -7, -4, "right")]:
        dot(ax, P, lab, dx, dy, ha)
    ax.set_xlim(-28, 128)
    ax.set_ylim(-68, 122)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("① 옆에서 본 모습 (정중 시상면) · 앞뒤·위아래 위치", loc="left", fontsize=11, color=INK,
                 fontweight="bold")
    ax.text(-28, 116, "얼굴 앞쪽 →", color=MUTED, fontsize=8.5)


def coronal(ax):
    def mirror(pts):
        return pts + [(-x, y) for x, y in reversed(pts) if x != 0]     # 가운데 점은 한 번만
    # 위턱: 바깥 윤곽 → 어금니 자리 → 입천장 (좌우 대칭)
    maxilla = mirror([(0, 62), (18, 62), (40, 52), (46, 38), (40, 22), (34, 14), (40, 4), (41, -3), (37, -5),
                      (26, -5), (16, 5), (0, 9)])
    ax.add_patch(Polygon(smooth(maxilla), closed=True, fc=BONE, ec=BONE_EDGE, lw=1.4, zorder=1))
    nasal = mirror([(0, 58), (8, 57), (15, 46), (18, 35), (16, 24), (10, 18), (0, 17)])
    ax.add_patch(Polygon(smooth(nasal), closed=True, fc="white", ec=BONE_EDGE, lw=1.2, zorder=2))   # 코안
    ax.plot([0, 0], [17, 57], color=BONE_EDGE, lw=2, zorder=3)                                     # 비중격
    mand = [(41, -25), (43, -35), (41, -50), (33, -62), (24, -58), (23, -40), (25, -25)]
    for s in (1, -1):
        ax.add_patch(Polygon(smooth([(s * x, y) for x, y in mand]), closed=True, fc=BONE, ec=BONE_EDGE, lw=1.4,
                             zorder=1))
        ax.add_patch(Ellipse((s * 30, 37), 14, 22, fc="white", ec=BONE_EDGE, lw=1, zorder=2))   # 위턱굴
        ax.add_patch(Rectangle((s * 33 - 7, -15), 14, 10, fc="white", ec=BONE_EDGE, lw=1.2, zorder=3))   # 위 어금니
        ax.add_patch(Rectangle((s * 33 - 7, -25), 14, 10, fc="white", ec=BONE_EDGE, lw=1.2, zorder=3))   # 아래 어금니
    ax.text(-33, -10, "위 어금니", ha="center", va="center", fontsize=7, color=MUTED, zorder=4)
    ax.text(-33, -20, "아래 어금니", ha="center", va="center", fontsize=7, color=MUTED, zorder=4)
    ax.text(0, 40, "코안", ha="center", fontsize=8, color=MUTED, zorder=4,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    # TDI = MxAW - MnAW
    dim(ax, (-41, 0), (41, 0), C1, "MxAW", at=(-30, 3), va="bottom")
    dim(ax, (-43, -36), (43, -36), C1, "MnAW", at=(0, -39), va="top")
    # PHI = PH / MxOW
    dim(ax, (-26, -10), (26, -10), C2, "MxOW", at=(-14, -10))
    dim(ax, (0, -10), (0, 9), C2, "PH", at=(2, 5), ha="left")
    # NCWR = NCW / MxBW
    dim(ax, (-18, 33), (18, 33), C3, "NCW", at=(8, 30), va="top", fs=8.5)
    dim(ax, (-34, 14), (34, 14), C3, "MxBW", at=(-20, 14), fs=8.5)

    ax.text(0, 76, "TDI = MxAW - MnAW", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(0, 69, "PHI = PH ÷ MxOW × 100        NCWR = NCW ÷ MxBW × 100", ha="center", fontsize=9.5,
            color=INK)
    ax.text(-58, -68, "첫째 큰어금니 높이의 단면을 앞에서 본 모습입니다. 화살표 위치는 예시입니다.\n"
                      "TDI 측정점, PHI 기준점(MxOW), NCWR 높이(NCW·MxBW)는 논문 정의가 불명확해\n"
                      "아직 확정되지 않았습니다 (CBCT_측정_계획.md에서 C등급).",
            fontsize=8, color=MUTED, va="top")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-84, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("② 앞에서 본 단면 (첫째 큰어금니, 관상면) · 좌우 폭", loc="left", fontsize=11, color=INK,
                 fontweight="bold")


def results_text(d: pd.DataFrame) -> tuple[str, str]:
    a, m, t = d.ANB, d.MPFH, d.TDI
    ms = lambda s: f"{s.mean():.1f}±{s.std():.1f}"
    left = "\n".join([
        f"ANB  {ms(a)}°  ·  N–A선과 N–B선 사이 각, 위·아래턱 앞뒤 관계",
        f"       Class III(≤0°) {(a <= 0).sum()}명 · I(0~4°) {((a > 0) & (a < 4)).sum()}명 · II(≥4°) {(a >= 4).sum()}명"
        f" · 가장 작은 값 P_{a.idxmin()} ({a.min():.1f}°)",
        f"Nperp-A  {ms(d.NA)} mm  ·  위턱(A점)이 Nperp보다 앞(+)/뒤(-)",
        f"Nperp-Pog  {ms(d.NPog)} mm  ·  턱끝(Pog)이 Nperp보다 앞(+)/뒤(-)",
        f"MP-FH  {ms(m)}°  ·  하악평면이 FH에 대해 기운 각",
        f"       hypo(<25°) {(m < 25).sum()}명 · normo(25~30°) {((m >= 25) & (m < 30)).sum()}명"
        f" · hyper(≥30°) {(m >= 30).sum()}명",
    ])
    right = "\n".join([
        f"TDI  {ms(t)} mm  ·  위턱 폭 - 아래턱 폭 (음수 = 위턱이 좁음)",
        f"       TSD(<-2.5) {(t < -2.5).sum()}명 · 경계(-2.5~-2) {((t >= -2.5) & (t <= -2)).sum()}명"
        f" · non-TSD(>-2) {(t > -2).sum()}명",
        f"PHI  {ms(d.PHI)}%  ·  입천장 높이 ÷ 위 어금니 사이 폭 (기준치 없음)",
        f"NCWR  {ms(d.NCWR)}%  ·  코안 최대 폭 ÷ 위턱 기저 폭 (기준치 없음)",
    ])
    return left, right


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = read_excel()
    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 0.2], left=0.02, right=0.98,
                          top=0.92, bottom=0.02, wspace=0.05, hspace=0.04)
    sagittal(fig.add_subplot(gs[0, 0]))
    coronal(fig.add_subplot(gs[0, 1]))
    left, right = results_text(d)
    for k, txt in enumerate((left, right)):
        ax = fig.add_subplot(gs[1, k])
        ax.axis("off")
        ax.text(0.01, 0.98, txt, va="top", fontsize=9.5, color=INK, linespacing=1.6, transform=ax.transAxes)
    fig.suptitle(f"CBCT 계측 변수는 두개골의 어디를 재는가 · 모식도 (실제 환자 영상 아님, 점 위치는 예시)\n"
                 f"아래 숫자 = 엑셀 CBCT 변수, {len(d)}명 (P_39 제외: CBCT 본인 영상 없음), 평균±표준편차",
                 fontsize=12.5, color=INK)
    fig.savefig(OUT / "cbct_variable_diagram.png", dpi=110)
    plt.close(fig)
    print("saved", OUT / "cbct_variable_diagram.png")


if __name__ == "__main__":
    main()
