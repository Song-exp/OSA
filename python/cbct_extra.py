"""엑셀 추가 변수: 비중격 편위, 비강저 경사, 교합평면, 턱끝 편위. 전 환자 동일 규칙, 엑셀로 임계값을 맞추지 않는다.

좌표계: osa.ceph.head_frame (FH = RPo·LPo·ROr, MSP = N–Ba를 지나 FH에 수직, 원점 N), Po는 cbct_fh_repair로 이상값 대체.
비중격 편위 (원 논문 그림 4A): TotalSegmentator 우(14)·좌(15) 비강 라벨의 안쪽 끝 가운데를 비중격 위치로 보고(틈 0~8 mm인 곳만),
            3 mm 중앙값으로 다듬은 뒤 MSP에서 가장 먼 거리 SDp. > 2 mm면 1.
비강저 경사 (그림 4B): 상악 제1대구치 저항중심 앞뒤 위치의 관상면에서 좌우 비강 라벨 최하점을 이은 선과 MSP 각의 90°에서 벗어난 정도. > 2°면 1.
교합평면 (PPT 13): ALI UR6MB·UL6MB와 앞니 겹침 중간점(UNIVERSALLAB 상악 중절치 8·9 최하점 평균과 하악 중절치 24·25 최상점 평균의 중점).
            FH와의 각: (주) MSP에 투영한 각 / (대안) 3차원 평면 사이 각.
턱끝 편위: Me의 MSP 거리. 기준이 문서에 없어 2 mm(인수인계 메모의 예시)와 4 mm를 함께 기록.
출력: Result_py/CBCT/patient_pipeline/v2/extra_review/v1/extra.csv
"""
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

from cbct_evidence_figs import excel_col, teeth_points
from cbct_fh_repair import repair
from cbct_molar_section import MODEL, PILOT
from cbct_patient_pipeline import CB, OUT, landmarks
from osa.ceph import head_frame
from run_ali import ascii_path

DEST = OUT/"extra_review/v1"
XL = {"OP": 33, "Chin": 34, "Septal": 36, "NFcant": 37}
STEP = .4; GAP = (0., 8.); SMOOTH = 7; SEPTAL_MM = 2.; CANT_DEG = 2.
INC = {"U": (8, 9), "L": (24, 25)}


class Labels:
    def __init__(self, f):
        im = sitk.ReadImage(str(ascii_path(f))); self.a = sitk.GetArrayFromImage(im)
        self.org, self.sp, self.D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)

    def at(self, pts):
        k = np.rint(((pts-self.org)@np.linalg.inv(self.D).T)/self.sp).astype(int)[:, ::-1]; ok = ((k >= 0) & (k < self.a.shape)).all(1)
        out = np.zeros(len(k), self.a.dtype); out[ok] = self.a[tuple(k[ok].T)]
        return out

    def points(self, labels):
        v = np.argwhere(np.isin(self.a, labels))
        return self.org+(v[:, ::-1]*self.sp)@self.D.T


def septum(lab, Ls):
    """lab: (L, ant, sup) 라벨. 비중격 중심 L 좌표 2D 배열 (없으면 nan)."""
    r = np.where(lab == 14, Ls[:, None, None], -np.inf).max(0); l = np.where(lab == 15, Ls[:, None, None], np.inf).min(0)
    c = (r+l)/2; ok = np.isfinite(r) & np.isfinite(l) & (l-r > GAP[0]) & (l-r < GAP[1])
    return np.where(ok, c, np.nan)


def nan_median(a, size):
    filled = np.where(np.isfinite(a), a, 0.); n = ndi.uniform_filter(np.isfinite(a).astype(float), size)
    med = ndi.generic_filter(np.where(np.isfinite(a), a, np.nan), np.nanmedian, size=size, mode="constant", cval=np.nan)
    return np.where(np.isfinite(a) & (n > .5), med, np.nan) if filled.size else a


def msp_angle(n_plane, R):
    """평면과 FH의 MSP 투영 각 (도). n_plane: LPS 법선."""
    d = np.cross(n_plane, R[0]); d = R@d
    return float(abs(np.degrees(np.arctan2(d[2], d[1]))) if abs(d[1]) > 1e-9 else 90.)


def measure(pid, molars):
    r = {"Patient": f"P_{pid}"}; P = repair(landmarks(pid)[0])[0]; o, R = head_frame(P); h = lambda x: R@(np.asarray(x, float)-o)
    r["Chin_dev_mm"] = abs(h(P["Me"])[0])
    ts_f = CB/f"seg_totalseg/head_glands_cavities/P_{pid}.nii.gz"
    if ts_f.exists():
        ts = Labels(ts_f); pns, ans, n = h(P["PNS"]), h(P["ANS"]), h(P["N"])
        Ls = np.arange(-20, 20+1e-9, STEP); ant = np.arange(pns[1], ans[1], STEP); sup = np.arange(min(pns[2], ans[2])-8, n[2], STEP)
        G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3); lab = ts.at(o+G@R).reshape(len(Ls), len(ant), len(sup))
        c = nan_median(septum(lab, Ls), SMOOTH)
        if np.isfinite(c).any():
            i = np.nanargmax(np.abs(c)); a, z = np.unravel_index(i, c.shape)
            r.update(SDp_mm=float(abs(c[a, z])), SDp_ant=ant[a], SDp_sup=sup[z], septum_cells=int(np.isfinite(c).sum()))
        u = molars[(molars.Patient == f"P_{pid}") & molars.tooth.isin(["UR6", "UL6"]) & molars.status.eq("ok")]
        A0 = u.A.mean() if len(u) else (pns[1]+ans[1])/2; r["NF_section"] = "molar" if len(u) else "mid_PNS_ANS"
        Ls2, Zs = np.arange(-30, 30, .25), np.arange(min(pns[2], ans[2])-15, n[2], .25); LL, ZZ = np.meshgrid(Ls2, Zs, indexing="ij")
        sec = ts.at(o+np.stack([LL, 0*LL+A0, ZZ], -1).reshape(-1, 3)@R).reshape(LL.shape); low = {}
        for k, v in (("R", 14), ("L", 15)):
            m = sec == v
            if m.any():
                j = np.flatnonzero(m.any(0)).min(); low[k] = (Ls2[m[:, j]].mean(), Zs[j])
        if len(low) == 2:
            r["NF_cant_deg"] = float(abs(np.degrees(np.arctan2(low["L"][1]-low["R"][1], low["L"][0]-low["R"][0]))))
            r.update(NfR_L=low["R"][0], NfR_sup=low["R"][1], NfL_L=low["L"][0], NfL_sup=low["L"][1], NF_A=A0)
    q = teeth_points(pid); tf = CB/f"seg_teeth/{MODEL}/P_{pid}.nii.gz"
    if all(k in q for k in ("UR6MB", "UL6MB")) and tf.exists():
        t = Labels(tf); up, lo = t.points(list(INC["U"])), t.points(list(INC["L"]))
        if len(up) > 50 and len(lo) > 50:
            hu, hl = (up-o)@R.T, (lo-o)@R.T
            edge_u = np.mean([up[(t.at(up) == k)][np.argmin(hu[t.at(up) == k][:, 2])] for k in INC["U"] if (t.at(up) == k).any()], 0)
            edge_l = np.mean([lo[(t.at(lo) == k)][np.argmax(hl[t.at(lo) == k][:, 2])] for k in INC["L"] if (t.at(lo) == k).any()], 0)
            inc = (edge_u+edge_l)/2; nrm = np.cross(q["UL6MB"]-q["UR6MB"], inc-q["UR6MB"]); nrm /= np.linalg.norm(nrm)
            r["OP_FH_msp"] = msp_angle(nrm, R); r["OP_FH_3d"] = float(np.degrees(np.arccos(min(1., abs(nrm@R[2])))))
            r["overbite_mm"] = float(h(edge_u)[2]-h(edge_l)[2])*-1
    return r


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int); po = (a == b).mean(); pe = a.mean()*b.mean()+(1-a.mean())*(1-b.mean())
    return (po-pe)/(1-pe) if pe < 1 else np.nan


def main():
    ids = [int(a) for a in sys.argv[1:]] or [int(p[2:]) for p, i in pd.read_csv(OUT/"cohort.csv", dtype=str, encoding="utf-8-sig")[["Patient", "Include"]].values if i.lower() == "true"]
    DEST.mkdir(parents=True, exist_ok=True); molars = pd.read_csv(OUT/"tdi_review/v1/molars.csv", encoding="utf-8-sig"); rows = []
    for pid in ids:
        print(pid, end=" ", flush=True); rows.append(measure(pid, molars))
    d = pd.DataFrame(rows); d["Set"] = ["pilot" if int(p[2:]) in PILOT else "eval" for p in d.Patient]
    for k, c in XL.items(): d[k+"_xl"] = [excel_col(c).get(int(p[2:]), np.nan) for p in d.Patient]
    d["Septal_auto"] = (d.SDp_mm > SEPTAL_MM).where(d.SDp_mm.notna()); d["NFcant_auto"] = (d.NF_cant_deg > CANT_DEG).where(d.NF_cant_deg.notna())
    d["Chin_auto2"] = (d.Chin_dev_mm > 2).astype(float); d["Chin_auto4"] = (d.Chin_dev_mm > 4).astype(float)
    d.to_csv(DEST/"extra.csv", index=False, encoding="utf-8-sig"); print()
    for s in ("pilot", "eval"):
        g = d[d.Set == s]
        for c in ("OP_FH_msp", "OP_FH_3d"):
            x = g.dropna(subset=[c, "OP_xl"]); e = x[c]-x.OP_xl
            print(s, c, "n", len(x), "bias", round(e.mean(), 2), "MAE", round(e.abs().mean(), 2), "r", round(x[c].corr(x.OP_xl), 2))
        for c, xl, cont in (("Septal_auto", "Septal_xl", "SDp_mm"), ("NFcant_auto", "NFcant_xl", "NF_cant_deg"), ("Chin_auto2", "Chin_xl", "Chin_dev_mm"), ("Chin_auto4", "Chin_xl", "Chin_dev_mm")):
            x = g.dropna(subset=[c, xl])
            print(s, c, "n", len(x), "agree", round((x[c] == x[xl]).mean()*100, 1), "kappa", round(kappa(x[c], x[xl]), 2), "xl 1:", int(x[xl].sum()), "auto 1:", int(x[c].sum()),
                  f"{cont} median xl0/xl1", round(x[x[xl] == 0][cont].median(), 2), round(x[x[xl] == 1][cont].median(), 2))


if __name__ == "__main__":
    main()
