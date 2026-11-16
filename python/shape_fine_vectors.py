"""골격보다 세밀한 단위의 벡터: 연조직(연구개·경구개·인두 세 구간), 치아 단위·치열궁. 85명, 환자 한 명씩.

좌표계: Po를 쓰지 않는 정중 좌표계(cbct_eb_detect.midline_frame, 원점 PNS, 축 좌·앞·위 mm; 앞 축 = ANS–PNS 방향).
연조직: TotalSegmentator head_glands_cavities (CT 학습 모델 → CBCT 정확도 미검증, qc 그림으로 확인).
  연구개(18): 부피, 정중 ±2 mm 단면의 주축 길이·최대 두께·구개평면(앞 축)과의 각, 끝점 위치, PNS까지 거리(부착 확인용)
  경구개(19): 부피, 정중 앞뒤 길이, 정중 두께 중앙값
  비인두·구인두·하인두(11·12·13): 부피, 높이, 축상 단면적의 최소·평균과 최소 위치
치아: UNIVERSALLAB 치아 분할 1~32번(미국식). 치아별 존재, 부피, 중심(정중 좌표), 장축 단위벡터(치근 쪽 방향).
  라벨 오류 표시(label_ok): 부피 범위 밖, 상하악 높이 뒤바뀜, 제1대구치는 ALI 저항중심과 10 mm 초과. 치열궁 계산에서 뺀다.
  치열궁: 견치·제1소구치·제1대구치 좌우 폭, 중절치–제1대구치 깊이, 존재 치아를 이은 둘레, 결손 수.
출력: Result_py/shape/fine/{soft_tissue.csv, teeth_long.csv, dental_arch.csv, qc_softtissue_*.png}
사용: python shape_fine_vectors.py [환자번호...]
"""
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from cbct_accuracy_experiments import wait_ram
from cbct_eb_detect import midline_frame
from cbct_evidence_figs import teeth_points
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd
from run_ali import ascii_path

DEST = Path(__file__).resolve().parents[1]/"Result_py/shape/fine"
SOFT = {18: "soft_palate", 19: "hard_palate", 11: "nasopharynx", 12: "oropharynx", 13: "hypopharynx"}
UPPER, LOWER = list(range(1, 17)), list(range(32, 16, -1))     # 오른쪽 → 왼쪽 순서
TYPE = {1: "M3", 2: "M2", 3: "M1", 4: "P2", 5: "P1", 6: "C", 7: "I2", 8: "I1"}    # 사분면 안 순서 (1=제3대구치 … 8=중절치)
MIN_ML = .05
ALI_R = {3: "UR6R", 14: "UL6R", 19: "LL6R", 30: "LR6R"}; ALI_MAX_MM = 10.    # 제1대구치: ALI 저항중심과 10 mm 넘게 다르면 라벨 오류 (검증 중앙값 2.1 mm)
VOL_OK = (.1, 1.6)    # 영구치 부피 범위(mL). 밖이면 결손·부분 분할 또는 다른 치아와 합쳐진 분할로 본다 (TDI 규칙 0.3~2 mL보다 넓게: 절치 포함)


def tooth_type(t):
    i = t if t <= 8 else 17-t if t <= 16 else t-16 if t <= 24 else 33-t
    return TYPE[i]


def label_points(f, keep):
    """라벨 영상 → {라벨: LPS 점 (N×3)}, 복셀 부피(mm³)."""
    im = sitk.ReadImage(str(ascii_path(f))); a = sitk.GetArrayFromImage(im)
    org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
    out = {}
    for l in keep:
        k = np.argwhere(a == l)
        if len(k): out[l] = org+(k[:, ::-1]*sp)@D.T
    return out, float(np.prod(sp)), (a, org, sp, D)


def soft_tissue(pts, vv, o, R):
    r = {}
    for l, name in SOFT.items():
        if l not in pts: r[f"{name}_volume"] = 0.0; continue
        q = (pts[l]-o)@R.T; r[f"{name}_volume"] = len(q)*vv
        if name == "soft_palate":
            m = q[np.abs(q[:, 0]) <= 2][:, 1:]
            r["soft_palate_PNS_gap"] = float(np.min(np.linalg.norm(q, axis=1)))
            if len(m) >= 20:
                c = m.mean(0); ax = np.linalg.eigh(np.cov((m-c).T))[1][:, -1]
                if ax[1] > 0: ax = -ax                                   # 뒤(아래)쪽을 향하게
                t, n = (m-c)@ax, (m-c)@np.array([-ax[1], ax[0]])
                bins = np.arange(t.min(), t.max()+2, 2.); thick = [np.ptp(n[(t >= b) & (t < b+2)]) for b in bins[:-1] if ((t >= b) & (t < b+2)).sum() > 3]
                tip = m[np.argmax(t)]
                r |= {"soft_palate_length": float(np.ptp(t)), "soft_palate_thickness": float(max(thick)) if thick else np.nan,
                      "soft_palate_angle": float(np.degrees(np.arctan2(-ax[1], -ax[0]))), "soft_palate_tip_ant": tip[0], "soft_palate_tip_sup": tip[1]}
        elif name == "hard_palate":
            m = q[np.abs(q[:, 0]) <= 1]
            if len(m) >= 10:
                b = np.round(m[:, 1]); th = [np.ptp(m[b == x, 2]) for x in np.unique(b) if (b == x).sum() > 2]
                r |= {"hard_palate_length": float(np.ptp(m[:, 1])), "hard_palate_thickness": float(np.median(th)) if th else np.nan}
        else:
            z = np.round(q[:, 2]); zs, cnt = np.unique(z, return_counts=True); area = cnt*vv          # 1 mm 두께 축상 단면적(mm²) 근사
            inner = (zs > zs.min()+2) & (zs < zs.max()-2)                                             # 라벨 위아래 끝 2 mm는 단면이 잘려 뺀다
            if inner.sum() >= 3: zs, area = zs[inner], area[inner]
            r |= {f"{name}_height": float(np.ptp(q[:, 2])), f"{name}_min_area": float(area.min()), f"{name}_mean_area": float(area.mean()),
                  f"{name}_min_sup": float(zs[np.argmin(area)])}
    return r


def teeth(pid, o, R):
    f = CB/f"seg_teeth/UNIVERSALLAB_MODEL/P_{pid}.nii.gz"
    pts, vv, _ = label_points(f, range(1, 33)); rows = []; cen = {}
    for t in range(1, 33):
        q = (pts[t]-o)@R.T if t in pts else None; ml = len(q)*vv/1000 if q is not None else 0.0
        r = {"Patient": f"P_{pid}", "tooth": t, "type": tooth_type(t), "arch": "upper" if t <= 16 else "lower", "present": ml >= MIN_ML, "volume_mL": ml}
        if ml >= MIN_ML:
            c = q.mean(0); ax = np.linalg.eigh(np.cov((q-c).T))[1][:, -1]
            if (ax[2] < 0) == (t <= 16): ax = -ax                       # 치근 쪽: 상악은 위, 하악은 아래
            cen[t] = c; r |= {"c_left": c[0], "c_ant": c[1], "c_sup": c[2], "axis_left": ax[0], "axis_ant": ax[1], "axis_sup": ax[2]}
        rows.append(r)
    # 라벨 오류 표시: 부피가 범위 밖이거나, 상악 치아가 하악 치아 중심 높이 중앙값보다 아래(또는 반대)면 치열궁 계산에서 뺀다
    up = [c[2] for t, c in cen.items() if t <= 16]; lo = [c[2] for t, c in cen.items() if t > 16]
    ali = {k: R@(v-o) for k, v in teeth_points(pid).items()}
    for r in rows:
        t = r["tooth"]
        if t not in cen: r["label_ok"] = False; continue
        bad_vol = not (VOL_OK[0] <= r["volume_mL"] <= VOL_OK[1])
        bad_side = (t <= 16 and lo and cen[t][2] < np.median(lo)) or (t > 16 and up and cen[t][2] > np.median(up))
        bad_ali = t in ALI_R and ALI_R[t] in ali and np.linalg.norm(cen[t]-ali[ALI_R[t]]) > ALI_MAX_MM    # 별도 모델(ALI)과 크게 다르면 다른 치아에 붙은 라벨
        r["label_ok"] = not (bad_vol or bad_side or bad_ali); r["ali_gap_mm"] = float(np.linalg.norm(cen[t]-ali[ALI_R[t]])) if t in ALI_R and ALI_R[t] in ali else np.nan
        if not r["label_ok"]: cen.pop(t)
    a = {"Patient": f"P_{pid}", "teeth_label_flags": int(sum(not r["label_ok"] and r["present"] for r in rows))}
    for arch, order in (("upper", UPPER), ("lower", LOWER)):
        pair = lambda k: [t for t in order if tooth_type(t) == k]
        for k in ("C", "P1", "M1"):
            ts = pair(k); a[f"{arch}_width_{k}"] = float(np.linalg.norm(cen[ts[0]]-cen[ts[1]])) if all(t in cen for t in ts) else np.nan
        i1, m1 = pair("I1"), pair("M1")
        a[f"{arch}_depth"] = float(np.mean([cen[t][1] for t in i1])-np.mean([cen[t][1] for t in m1])) if all(t in cen for t in i1+m1) else np.nan
        seq = [cen[t] for t in order if t in cen and tooth_type(t) != "M3"]
        a[f"{arch}_perimeter"] = float(sum(np.linalg.norm(np.diff(seq, axis=0), axis=1))) if len(seq) > 1 else np.nan
        a[f"{arch}_missing"] = int(sum(t not in cen for t in order if tooth_type(t) != "M3"))
        a[f"{arch}_M3_present"] = int(sum(t in cen for t in order if tooth_type(t) == "M3"))
    return rows, a


def qc_panel(ax, p, o, R, ts):
    v, dirs, org = read_nrrd(CB/f"nrrd/{p}.nrrd"); v = v.astype(np.float32)
    A, S = np.arange(-60, 40, .5), np.arange(-90, 30, .5); G = np.stack(np.meshgrid(A, S, indexing="ij"), -1).reshape(-1, 2)
    Q = o+np.c_[np.zeros(len(G)), G]@R
    ct = map_coordinates(v, np.linalg.solve(dirs.T, (Q-org).T)[::-1], order=1, cval=-1000).reshape(len(A), len(S)); del v
    a, torg, sp, D = ts; k = np.rint(((Q-torg)@np.linalg.inv(D).T)/sp).astype(int)[:, ::-1]; ok = ((k >= 0) & (k < a.shape)).all(1)
    lab = np.zeros(len(Q), np.uint8); lab[ok] = a[tuple(k[ok].T)]; lab = lab.reshape(len(A), len(S))
    ax.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000, vmax=1500, extent=[A[0], A[-1], S[0], S[-1]])
    for l, c in ((18, "r"), (19, "y"), (11, "c"), (12, "lime"), (13, "b")):
        if (lab == l).any(): ax.contour(A, S, (lab == l).T.astype(float), [.5], colors=c, linewidths=1)
    ax.set_title(p, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ids = [int(a) for a in sys.argv[1:]] or sorted(int(p.parent.name[2:]) for p in OUT.glob("P_*/result.json"))
    soft, tl, arch = [], [], []; n = len(ids); cols = 10; rows_ = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(2.6*cols, 3.1*rows_), squeeze=False)
    for a in axes.flat: a.axis("off")
    for i, pid in enumerate(ids):
        wait_ram(4, f"P_{pid}"); p = f"P_{pid}"; print(p, end=" ", flush=True)
        P = landmarks(pid)[0]; o, R, _ = midline_frame(P)
        pts, vv, ts = label_points(CB/f"seg_totalseg/head_glands_cavities/{p}.nii.gz", SOFT)
        soft.append({"Patient": p, **soft_tissue(pts, vv, o, R)}); del pts
        axes.flat[i].axis("on"); qc_panel(axes.flat[i], p, o, R, ts); del ts
        t, a = teeth(pid, o, R); tl += t; arch.append(a); gc.collect()
    fig.suptitle("정중 시상 단면: 빨강 연구개, 노랑 경구개, 청록 비인두, 연두 구인두, 파랑 하인두 (TotalSegmentator)", fontsize=14)
    fig.tight_layout(); fig.savefig(DEST/"qc_softtissue_midsagittal.png", dpi=55); plt.close(fig)
    for name, rows in (("soft_tissue", soft), ("teeth_long", tl), ("dental_arch", arch)):
        f = DEST/f"{name}.csv"; d = pd.DataFrame(rows)
        if sys.argv[1:] and f.exists():
            old = pd.read_csv(f, encoding="utf-8-sig"); d = pd.concat([old[~old.Patient.isin(d.Patient)], d], ignore_index=True)
        d.round(4).to_csv(f, index=False, encoding="utf-8-sig")
    s = pd.DataFrame(soft); t = pd.DataFrame(tl)
    print("\n연구개 있음", int((s.soft_palate_volume > 0).sum()), "/", len(s), "| PNS 간격 중앙값", round(s.soft_palate_PNS_gap.median(), 1) if "soft_palate_PNS_gap" in s else "-",
          "| 인두 라벨 있음", {k: int((s[f"{k}_volume"] > 0).sum()) for k in ("nasopharynx", "oropharynx", "hypopharynx")})
    print("치아 수(제3대구치 제외) 중앙값", int(t[t.present & (t.type != "M3")].groupby("Patient").size().median()), "| 28개 미만 환자",
          int((t[t.present & (t.type != "M3")].groupby("Patient").size() < 28).sum()))


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    main()
