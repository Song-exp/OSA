"""83명 회귀표용 변수 위치 그림: 예시 환자 1명의 CBCT에 변수마다 기준점·선·각·분할 영역을 그린다.
좌표계: 정중 좌표계(cbct_eb_detect.midline_frame, PNS 원점, 축 좌·앞·위). 계산식으로 만든 2차 변수(주성분·비율·교호작용 등)는 그림 없음.
Roh 지표(엑셀·자동 공통 정의)는 기존 용어집 그림(cbct_glossary_examples)을 그대로 쓴다.
출력: Result_py/analysis/all83/figs/<변수>.png, 반환: {변수: 그림 경로}
"""
import shutil, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import map_coordinates
import SimpleITK as sitk

import osa_experiments as X, build_features as bf
from cbct_patient_pipeline import CB, OUT as PIPE, landmarks
from cbct_eb_detect import midline_frame
from check_landmarks import read_nrrd
from shape_local_roi import roi_labels
from cbct_extra import MODEL as TEETH_MODEL
from run_ali import ascii_path

FIG = X.R_/"analysis/all83/figs"; GLOSS = PIPE/"report/glossary_images"
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
COL = "#ff5a36"; COL2 = "#2fd3ff"


class Case:
    def __init__(self, pid):
        self.pid = pid; P = landmarks(pid)[0]; self.o, self.R, _ = midline_frame(P)
        rel = pd.read_csv(X.R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
        cond = pd.read_csv(bf.EXT/"condyle_all.csv", encoding="utf-8-sig").set_index("Patient")
        q, _ = bf.coords(pid, rel.loc[f"P_{pid}"].to_dict(), cond.loc[f"P_{pid}"] if f"P_{pid}" in cond.index else None)
        q["Go"] = (q["RGo"]+q["LGo"])/2; q["Co"] = (q["RSup"]+q["LSup"])/2; q["Gn"] = (q["Pog"]+q["Me"])/2
        self.q = q; self.raw, self.dirs, self.org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd")
        pts = np.array([v for v in q.values() if np.isfinite(v).all()])
        self.ant = np.arange(pts[:, 1].min()-25, pts[:, 1].max()+20, .5); self.sup = np.arange(pts[:, 2].min()-25, pts[:, 2].max()+20, .5)
        self.sag = self.slice_sag(0.)

    def world(self, Q): return self.o+Q@self.R                                   # 정중 좌표 → 영상(LPS) 좌표

    def ct(self, Q, shape):
        idx = np.linalg.solve(self.dirs.T, (self.world(Q)-self.org).T)[::-1]
        return map_coordinates(self.raw, idx, order=1, cval=-1000, output=np.float32).reshape(shape)

    def grid(self, a, b, plane, c):
        A, B = np.meshgrid(a, b, indexing="ij"); z = np.full_like(A, c)
        Q = {"sag": (z, A, B), "cor": (A, z, B), "ax": (A, B, z)}[plane]
        return np.stack(Q, -1).reshape(-1, 3), A.shape

    def slice_sag(self, L):
        Q, sh = self.grid(self.ant, self.sup, "sag", L); return np.mean([self.ct(Q+[dl, 0, 0], sh) for dl in (-1, 0, 1)], axis=0)

    def label(self, f, Q, shape):
        im = sitk.ReadImage(str(ascii_path(f))); a = sitk.GetArrayFromImage(im)
        org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
        idx = (np.linalg.solve(D, (self.world(Q)-org).T).T/sp)[:, ::-1].T
        return map_coordinates(a, idx, order=0, cval=0).reshape(shape)

    def label_pts(self, f, labs):
        im = sitk.ReadImage(str(ascii_path(f))); a = sitk.GetArrayFromImage(im)
        org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
        k = np.argwhere(np.isin(a, labs)); w = org+(k[:, ::-1]*sp)@D.T
        return (w-self.o)@self.R.T


def base(c, img=None, ext=None, xl="앞 + (mm)", yl="위 + (mm)"):
    fig, ax = plt.subplots(figsize=(4.4, 4.4), layout="constrained")
    img = c.sag if img is None else img; ext = [c.ant[0], c.ant[-1], c.sup[0], c.sup[-1]] if ext is None else ext
    ax.imshow(img.T, origin="lower", cmap="gray", vmin=-700, vmax=1800, extent=ext); ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6)
    return fig, ax


def pt(ax, c, k, color=COL):
    p = c.q[k] if isinstance(k, str) else k; name = k if isinstance(k, str) else ""
    ax.plot(p[1], p[2], "o", ms=4, color=color)
    if name: ax.annotate(name, (p[1], p[2]), xytext=(4, 4), textcoords="offset points", fontsize=7, color="white", bbox=dict(fc="black", alpha=.6, ec="none", pad=1))


def seg(ax, c, a, b, color=COL, ls="-"):
    p, r = c.q[a] if isinstance(a, str) else a, c.q[b] if isinstance(b, str) else b
    ax.plot([p[1], r[1]], [p[2], r[2]], ls, color=color, lw=1.6)


def save(fig, ax, name, title):
    ax.set_title(title, fontsize=8); f = FIG/f"{name}.png"; fig.savefig(f, dpi=80); plt.close(fig); return f


# ------------------------------------------------------------ 기준점 기하
def dist(c, a, b, title):
    def f(name):
        fig, ax = base(c); seg(ax, c, a, b); pt(ax, c, a); pt(ax, c, b); return save(fig, ax, name, title)
    return f


def angle(c, a, v, b, title, extra=()):
    def f(name):
        fig, ax = base(c); seg(ax, c, v, a); seg(ax, c, v, b)
        for k in (a, v, b)+tuple(extra): pt(ax, c, k)
        return save(fig, ax, name, title)
    return f


def lines(c, pairs, title, points=()):
    def f(name):
        fig, ax = base(c)
        for i, (a, b) in enumerate(pairs): seg(ax, c, a, b, color=[COL, COL2][i % 2])
        for k in dict.fromkeys([k for p in pairs for k in p]+list(points)): pt(ax, c, k)
        return save(fig, ax, name, title)
    return f


def antdiff(c, a, b, title):
    def f(name):
        fig, ax = base(c); p, r = c.q[a], c.q[b]; y = min(p[2], r[2])-12
        for s in (p, r): ax.plot([s[1], s[1]], [s[2], y], "--", color=COL2, lw=1)
        ax.annotate("", (p[1], y), (r[1], y), arrowprops=dict(arrowstyle="<->", color=COL, lw=1.6)); pt(ax, c, a); pt(ax, c, b)
        return save(fig, ax, name, title)
    return f


def vert(c, a, b, title):
    def f(name):
        fig, ax = base(c); p, r = c.q[a], c.q[b]; x = p[1]+12
        for v in (p, r): ax.plot([v[1], x], [v[2], v[2]], "--", color=COL2, lw=1)
        ax.annotate("", (x, p[2]), (x, r[2]), arrowprops=dict(arrowstyle="<->", color=COL, lw=1.6)); pt(ax, c, a); pt(ax, c, b)
        return save(fig, ax, name, title)
    return f


def position(c, k, title):
    def f(name):
        fig, ax = base(c); p = c.q[k]; ax.axhline(0, color=COL2, lw=.8, ls="--"); ax.axvline(0, color=COL2, lw=.8, ls="--")
        pt(ax, c, "PNS", COL2); pt(ax, c, k); return save(fig, ax, name, title)
    return f


def perp(c, a, b, k, title):
    def f(name):
        fig, ax = base(c); p, r, h = c.q[a][1:], c.q[b][1:], c.q[k][1:]; u = (r-p)/np.linalg.norm(r-p); foot = p+u*((h-p)@u)
        seg(ax, c, a, b); ax.plot([h[0], foot[0]], [h[1], foot[1]], color=COL2, lw=1.6); pt(ax, c, a); pt(ax, c, b); pt(ax, c, k)
        return save(fig, ax, name, title)
    return f


# ------------------------------------------------------------ 분할 영역
def sag_label(c, f, labs, title, marks=(), plane=None, L=0.):
    def g(name):
        img = c.sag if L == 0 else c.slice_sag(L); Q, sh = c.grid(c.ant, c.sup, "sag", L); m = np.isin(c.label(f, Q, sh), labs)
        fig, ax = base(c, img); ax.imshow(np.ma.masked_where(~m, m).T, origin="lower", cmap=ListedColormap([COL]), alpha=.55, extent=[c.ant[0], c.ant[-1], c.sup[0], c.sup[-1]])
        if plane: seg(ax, c, *plane, color=COL2, ls="--")
        for k in marks: pt(ax, c, k, COL2) if isinstance(k, str) else ax.plot(k[1], k[2], "x", color=COL2, ms=7, mew=2)
        return save(fig, ax, name, title)
    return g


def cor_label(c, f, labs, title):
    def g(name):
        P = c.label_pts(f, labs)
        if not len(P): return None
        a0 = float(np.median(P[:, 1])); Ls = np.arange(-75, 75, .5); sup = c.sup
        Q, sh = c.grid(Ls, sup, "cor", a0); img = c.ct(Q, sh); m = np.isin(c.label(f, Q, sh), labs)
        fig, ax = base(c, img, [Ls[0], Ls[-1], sup[0], sup[-1]], xl="왼쪽 + (mm)")
        ax.imshow(np.ma.masked_where(~m, m).T, origin="lower", cmap=ListedColormap([COL]), alpha=.55, extent=[Ls[0], Ls[-1], sup[0], sup[-1]])
        return save(fig, ax, name, f"{title} (PNS 기준 앞 {a0:.0f} mm 관상면)")
    return g


def teeth_ax(c, hi, title):
    def g(name):
        P = c.label_pts(CB/f"seg_teeth/{TEETH_MODEL}/P_{c.pid}.nii.gz", list(range(1, 33)))
        im = sitk.ReadImage(str(ascii_path(CB/f"seg_teeth/{TEETH_MODEL}/P_{c.pid}.nii.gz"))); a = sitk.GetArrayFromImage(im)
        org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
        k = np.argwhere((a >= 1) & (a <= 32)); lab = a[tuple(k.T)]; w = org+(k[:, ::-1]*sp)@D.T; P = (w-c.o)@c.R.T
        fig, ax = plt.subplots(figsize=(4.4, 4.4), layout="constrained"); ax.set_facecolor("black")
        sel = np.random.default_rng(0).choice(len(P), min(len(P), 40000), replace=False)
        ax.scatter(P[sel, 0], P[sel, 1], s=.3, c=np.where(np.isin(lab[sel], hi), COL, "#888888"))
        for t in hi:
            m = lab == t
            if m.any(): ax.annotate(str(t), P[m][:, :2].mean(0), color="white", fontsize=7)
        ax.set_aspect("equal"); ax.set_xlabel("왼쪽 + (mm)", fontsize=7); ax.set_ylabel("앞 + (mm)", fontsize=7); ax.tick_params(labelsize=6)
        return save(fig, ax, name, title+" (치아 분할, 위에서 본 투영)")
    return g


def incisors(c, title, mp=False):
    def g(name):
        f = CB/f"seg_teeth/{TEETH_MODEL}/P_{c.pid}.nii.gz"; P = c.label_pts(f, [8, 9, 24, 25]); L = float(np.median(P[:, 0]))
        img = c.slice_sag(L); Q, sh = c.grid(c.ant, c.sup, "sag", L); m = c.label(f, Q, sh) > 0
        fig, ax = base(c, img); ax.imshow(np.ma.masked_where(~m, m).T, origin="lower", cmap=ListedColormap([COL]), alpha=.5, extent=[c.ant[0], c.ant[-1], c.sup[0], c.sup[-1]])
        if mp: seg(ax, c, "Go", "Me", color=COL2); pt(ax, c, "Go", COL2); pt(ax, c, "Me", COL2)
        ax.set_xlim(c.q["B"][1]-25, c.q["A"][1]+20); ax.set_ylim(c.q["Me"][2]-5, c.q["ANS"][2]+10)
        return save(fig, ax, name, title)
    return g


def roi(c, struct, region, title, center=False):
    def g(name):
        z = np.load(X.R_/f"shape/{struct}/points.npz"); ids = list(z["ids"])
        if c.pid not in ids: return None
        i = ids.index(c.pid); lab = roi_labels(struct, z)[0]
        w = z["head_origin"][i]+z["frame"][i]@z["head_R"][i]; Q = (w-c.o)@c.R.T                    # 머리 좌표계 → 영상 → 정중 좌표계
        fig, ax = base(c); m = lab == region
        ax.scatter(Q[~m, 1], Q[~m, 2], s=1, c="#9ab", alpha=.35); ax.scatter(Q[m, 1], Q[m, 2], s=2, c=COL)
        if center: ax.plot(Q[m, 1].mean(), Q[m, 2].mean(), "x", color=COL2, ms=9, mew=2)
        ax.set_xlim(c.ant[0], c.ant[-1]); ax.set_ylim(c.sup[0], c.sup[-1])
        return save(fig, ax, name, title+"\n(좌우 양쪽 표면 대응점을 정중면에 겹쳐 투영)")
    return g


def gloss(key):
    def g(name):
        src = GLOSS/f"{key}.png"
        if not src.exists(): return None
        dst = FIG/f"{name}.png"; shutil.copy(src, dst); return dst
    return g


# ------------------------------------------------------------ 변수 → 그림
def specs(c):
    HG, HM, CF = (CB/f"seg_totalseg/{t}/P_{c.pid}.nii.gz" for t in ("head_glands_cavities", "head_muscles", "craniofacial_structures"))
    q = c.q; s = {}
    s.update({"m_ANB": angle(c, "A", "N", "B", "ANB: N–A선과 N–B선 사이 각"), "SNA": angle(c, "S", "N", "A", "SNA: N에서 S·A 방향 사이 각"),
              "SNB": angle(c, "S", "N", "B", "SNB: N에서 S·B 방향 사이 각"), "NSBa": angle(c, "N", "S", "Ba", "두개저 각 N–S–Ba"),
              "gonial": angle(c, "Co", "Go", "Me", "하악각 과두–Go–Me"), "Yaxis": angle(c, "N", "S", "Gn", "Y축: S–N과 S–Gn 사이 각"),
              "FacialAxis_proxy": lines(c, [("Ba", "N"), ("PNS", "Gn")], "얼굴 축 근사: Ba–N선과 PNS–Gn선 사이 각"),
              "PP_MP": lines(c, [("ANS", "PNS"), ("Go", "Me")], "구개평면(ANS–PNS)과 하악평면(Go–Me) 사이 각"),
              "SN_MP": lines(c, [("S", "N"), ("Go", "Me")], "S–N선과 하악평면(Go–Me) 사이 각"),
              "Pog_Ba_ant": antdiff(c, "Pog", "Ba", "턱끝(Pog)과 대후두공 앞점(Ba)의 앞뒤 거리"),
              "Pog_PNS_ant": antdiff(c, "Pog", "PNS", "PNS 기준 Pog 앞뒤 위치"), "PNS_Ba_depth": antdiff(c, "PNS", "Ba", "PNS 기준 Ba 앞뒤 깊이"),
              "PNS_B_ant": antdiff(c, "B", "PNS", "PNS 기준 B점 앞뒤 위치"),
              "lever_arm_Pog_sup": lines(c, [("Co", "Pog")], "과두 축과 Pog의 높이 차", ()), "lever_Pog_mm": lines(c, [("Co", "Pog")], "과두 축 회전(5°) 시 Pog가 뒤로 밀리는 거리"),
              "lever_Hy_mm": lines(c, [("Co", "Hy")], "과두 축 회전(5°) 시 설골이 뒤로 밀리는 거리"),
              "MP_H": perp(c, "Go", "Me", "Hy", "설골에서 하악평면(Go–Me)까지 수직 거리"),
              "Hy_sup": position(c, "Hy", "설골 위아래 위치 (PNS 원점)"), "Hy_ant": position(c, "Hy", "설골 앞뒤 위치 (PNS 원점)"),
              "H_col": vert(c, "PNS", "Eb", "기도 기둥 높이: PNS에서 Eb 높이까지 수직 거리"),
              "MaxPost_Clivus_ant": roi(c, "MAX", "posterior", "상악 후방 부위 중심의 앞뒤 위치 (사대 부위와의 차)", True)})
    for k, (a, b) in {"S_N": ("S", "N"), "S_Ba": ("S", "Ba"), "ANS_PNS": ("ANS", "PNS"), "Co_Gn": ("Co", "Gn"), "Co_A": ("Co", "A"), "Go_Me": ("Go", "Me"),
                      "Co_Go": ("Co", "Go"), "N_Me": ("N", "Me"), "ANS_Me": ("ANS", "Me"), "S_Go": ("S", "Go"), "body_len": ("Go", "Pog"),
                      "Hy_Me": ("Hy", "Me"), "Hy_Ba_dist": ("Hy", "Ba"), "InterGonial": ("RGo", "LGo")}.items():
        s[k] = dist(c, a, b, f"{a}–{b} 거리" + (" (좌우 하악각점, 정중면에 투영)" if k == "InterGonial" else ""))
    for k, key in {"x_ANB": "ANB", "x_NperpA": "Nperp", "m_NperpA": "Nperp", "x_NperpPog": "Nperp", "m_NperpPog": "Nperp", "x_MPFH": "MP", "m_MPFH": "MP",
                   "x_TDI": "TDI", "m_TDI": "TDI", "x_PHI": "PHI", "m_PHI": "PHI", "x_NCWR": "NCWR", "m_NCWR": "NCWR", "x_NAV": "NAV", "m_NAV": "NAV",
                   "x_PAL": "PAL", "a_PAL": "PAL", "x_PAV": "PAV", "a_PAV": "PAV", "x_PMCA": "PMCA", "a_PMCA": "PMCA", "HD_mm": "PMCA", "AW_min_pos": "PAV",
                   "Condyle_Axial_mean": "condyle", "Condyle_Lat_mean": "condyle"}.items():
        s[k] = gloss(key)
    for k, labs, t in (("nasopharynx", [11], "비인두"), ("oropharynx", [12], "구인두"), ("hypopharynx", [13], "하인두")):
        for v in ("min_area", "mean_area"): s[f"{k}_{v}"] = sag_label(c, HG, labs, f"{t} 영역 (TotalSegmentator, 1 mm 가로 단면의 {'최소' if v == 'min_area' else '평균'} 넓이)")
    for v, t in (("soft_palate_length", "연구개 길이"), ("soft_palate_thickness", "연구개 최대 두께"), ("soft_palate_angle", "연구개 기울기"), ("soft_palate_PNS_gap", "PNS와 연구개 시작점 사이 간격")):
        s[v] = sag_label(c, HG, [18], f"{t} (빨강: 연구개 분할)", marks=("PNS",))
    for v, t in (("tongue_mL", "혀 부피"), ("tongue_c_ant", "혀 무게중심 앞뒤 위치"), ("tongue_c_sup", "혀 무게중심 위아래 위치"), ("tongue_low_sup", "혀 최저점 높이"),
                 ("tongue_len", "혀 앞뒤 길이"), ("tongue_height", "혀 높이"), ("tongue_back_ant", "혀뿌리(뒤 경계) 앞뒤 위치")):
        s[v] = sag_label(c, HM, [9], f"{t} (빨강: 혀 분할, 정중 단면)", marks=("PNS",))
    s["MMV_mL"] = sag_label(c, HM, [9], "하악·상악 볼록 껍질 안 공간(구개평면 아래) 부피. 파란 점선: 구개평면", plane=("ANS", "PNS"))
    s["MMV_hull_mL"] = s["MMV_mL"]
    for v, labs, t in (("masseter_mL", [1, 2], "교근"), ("lat_pterygoid_mL", [5, 6], "외측익돌근"), ("med_pterygoid_mL", [7, 8], "내측익돌근"), ("chew_mL", [1, 2, 5, 6, 7, 8], "저작근 전체")):
        s[v] = cor_label(c, HM, labs, f"{t} 좌우 합 부피")
    s["sinus_max_mL"] = cor_label(c, CF, [5], "상악동 좌우 합 부피"); s["sinus_max_air_frac"] = cor_label(c, CF, [5], "상악동 안 공기 비율")
    s["sinus_frontal_mL"] = cor_label(c, CF, [6], "전두동 부피"); s["SDp_mm"] = cor_label(c, HG, [14, 15], "비중격 최대 편위 (빨강: 좌우 비강)")
    s["Overjet_mm"] = incisors(c, "위·아래 중절치 절단 끝의 앞뒤 거리 (빨강: 중절치)"); s["overbite_mm"] = incisors(c, "위·아래 중절치 절단 끝의 높이 차")
    s["L1_MP_deg"] = incisors(c, "아래 중절치 장축과 하악평면(Go–Me) 사이 각", mp=True)
    s["Upper_Taper"] = teeth_ax(c, [6, 11, 3, 14], "위 송곳니(6·11) 폭 ÷ 위 제1대구치(3·14) 폭")
    for st, regs in {"MAND": ("symphysis", "body", "gonial", "ramus_condyle"), "MAX": ("anterior_alveolar", "palate", "posterior", "orbital_floor"),
                     "CB": ("anterior_base", "sella", "clivus", "temporal")}.items():
        for r in regs:
            s[f"roi_{st}_{r}_size"] = roi(c, st, r, f"{st} {r} 부위 (빨강) 크기")
            for ax_, t in (("cleft", "좌우"), ("cant", "앞뒤"), ("csup", "위아래")): s[f"roi_{st}_{r}_{ax_}"] = roi(c, st, r, f"{st} {r} 부위 중심(×)의 {t} 위치", True)
    return s


def pick(d):
    """예시 환자: Class III 아님, ANB 등급 A, 귓구멍 점 문제 없음, 분할·대응점 모두 있음, ANB가 중앙값에 가까움."""
    bad = {"P_16", "P_20", "P_23", "P_35", "P_37"}
    ok = d[(d.grade_ANB == "A") & ~d.ClassIII.astype(bool) & d.Overjet_mm.notna() & d.c_Eb_sup.notna() & d.c_Hy_sup.notna() & d.tongue_mL.notna() & d.sinus_frontal_mL.notna()]
    ids = {st: set(f"P_{i}" for i in np.load(X.R_/f"shape/{st}/points.npz")["ids"]) for st in ("MAND", "MAX", "CB")}
    ok = ok[[p not in bad and all(p in v for v in ids.values()) and (CB/f"seg_totalseg/craniofacial_structures/{p}.nii.gz").exists() for p in ok.index]]
    return int((ok.m_ANB-d.m_ANB.median()).abs().idxmin()[2:])


def build(d, names):
    FIG.mkdir(parents=True, exist_ok=True); pid = pick(d); c = Case(pid); sp = specs(c); out = {}
    for n in names:
        if n in sp:
            try:
                f = sp[n](n)
                if f is not None: out[n] = f
            except Exception as e: print("그림 실패", n, type(e).__name__, e)
    return pid, out
