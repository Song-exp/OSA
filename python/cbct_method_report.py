"""측정 기준 보고서 (결과 보고서와 별도): 어떤 도구가 무엇을 어떤 기준으로 재는지 그림으로.

1) 도구: ALI-CBCT(랜드마크), AMASSS(구조 분할), 그 밖의 분할(TotalSegmentator, 치아 분할, ALI 치아)
2) 좌표계: 머리 좌표계(FH·MSP, 원점 N) / 정중 좌표계(Po 미사용, 원점 PNS)
3) ALI 랜드마크 16개 + 치아 점, 4) AMASSS 4구조, 5) 11개 지표의 정의·입력·규칙·한계, 6) 품질 기준
새 그림 3장(예시 P_5)은 여기서 만들고, 랜드마크·지표별 그림은 기존 용어 그림(report/glossary_images)을 그대로 쓴다.
출력: Result_py/report_85/CBCT_측정기준_보고서_2026-11-15.html (+ method_figs/)
"""
from html import escape
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
import numpy as np
from scipy.ndimage import map_coordinates

from batch_cbct_pipeline import LMS
from cbct_eb_detect import midline_frame
from cbct_evidence_figs import teeth_points
from cbct_fh_repair import DEST as FH_REF, repair
from cbct_patient_pipeline import CB, OUT, landmarks
from cbct_report_glossary import TERMS
from check_landmarks import read_nrrd
from osa.ceph import head_frame
from shape_visual_review import lab_sample

ROOT = Path(__file__).resolve().parents[1]; DEST = ROOT/"Result_py/report_85"; FIG = DEST/"method_figs"; GL = OUT/"report/glossary_images"
HTML = DEST/"CBCT_측정기준_보고서_2026-11-15.html"; P = "P_5"
COL = {"MAND": "#2a78d6", "MAX": "#eb6834", "CB": "#1baf7a", "UAW": "#eda100"}    # 참조 팔레트 1~4번
SIDE = {"R": "#00c8ff", "L": "#ffd400"}   # 회색 CT 위에서 잘 보이는 밝은 색
MID = "#ff2d2d"
# 겹치는 이름표 위치 (옆, 앞 투영; 점 기준 픽셀)
LBL = ({"Me": (-26, -6), "Gn": (7, -6), "Pog": (7, 2), "RPo": (-30, 8), "LPo": (-30, -14), "ROr": (6, 6), "LOr": (6, -12)},
       {"Me": (-26, -8), "Gn": (7, -8), "Pog": (7, 4), "S": (-18, -14), "N": (6, 6), "Ba": (-26, 6), "PNS": (7, -2), "ANS": (7, -4)})
OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]
PROJECTION_NOTE = "각 행의 왼쪽은 두개골 전체에서 확인 단면의 위치(하늘색 선)를 보여 줍니다. 가운데는 해당 위치의 원본 단면, 오른쪽은 완전히 같은 단면에 실제 자동 분할 경계를 표시한 영상입니다. 색은 그 단면과 교차하는 영역에만 표시했으며, 전체 깊이의 분할을 겹치지 않았습니다. 단면은 해당 구조의 면적이 가장 큰 위치를 선택했습니다."


def rel(p): return quote(os.path.relpath(p, DEST).replace("\\", "/"))


def grid(v, dirs, org, o, R, L, A, S):
    G = np.stack(np.meshgrid(L, A, S, indexing="ij"), -1).reshape(-1, 3); Q = o+G@R
    return map_coordinates(v, np.linalg.solve(dirs.T, (Q-org).T)[::-1], order=1, cval=-1000).reshape(len(L), len(A), len(S)), Q


def fig_ali(v, dirs, org):
    ref = json.loads((FH_REF/"reference.json").read_text(encoding="utf-8"))["points"]
    Pt = repair(landmarks(5)[0], ref)[0]; o, R = head_frame(Pt); H = {k: R@(x-o) for k, x in Pt.items()}
    T = {k: R@(x-o) for k, x in teeth_points(5).items()}
    fig, ax = plt.subplots(1, 3, figsize=(20, 7.2))
    A, S = np.arange(-115, 25, 1.), np.arange(-130, 30, 1.); vol, _ = grid(v, dirs, org, o, R, np.arange(-70, 71, 1.), A, S)
    ax[0].imshow(vol.max(0).T, origin="lower", cmap="gray", vmin=0, vmax=2500, extent=[A[0], A[-1], S[0], S[-1]])
    Lr = np.arange(-75, 76, 1.); vol2, _ = grid(v, dirs, org, o, R, Lr, np.arange(-60, 25, 1.), S)
    ax[1].imshow(vol2.max(1).T, origin="lower", cmap="gray", vmin=0, vmax=2500, extent=[Lr[0], Lr[-1], S[0], S[-1]])
    for k in LMS:
        c = SIDE[k[0]] if k[0] in "RL" and k[1:] in ("Po", "Or", "Go") else MID
        for i, (a, (x, y)) in enumerate(((ax[0], (H[k][1], H[k][2])), (ax[1], (H[k][0], H[k][2])))):
            a.plot(x, y, "o", ms=9, mfc=c, mec="black", mew=1.5)
            a.annotate(k, (x, y), color=c, fontsize=11, fontweight="bold", xytext=LBL[i].get(k, (5, 4)), textcoords="offset points", path_effects=OUTLINE)
    ax[0].set_title("ALI 두개 랜드마크 16개 — 옆 투영 (뼈 최대투영)", fontsize=11); ax[1].set_title("앞 투영: 하늘색 = 오른쪽(R), 노랑 = 왼쪽(L), 빨강 = 정중", fontsize=11)
    zt = np.mean([q[2] for q in T.values()]); a_ = [q[1] for q in T.values()]; Lz, Az = np.arange(-45, 46, .5), np.arange(min(a_)-15, max(a_)+20, .5)    # 원점 N 기준 치아 앞뒤 범위
    vol3, _ = grid(v, dirs, org, o, R, Lz, Az, np.arange(zt-12, zt+12, 1.))
    ax[2].imshow(vol3.max(2).T, origin="lower", cmap="gray", vmin=500, vmax=3000, extent=[Lz[0], Lz[-1], Az[0], Az[-1]])
    for k, q in T.items():
        c = "#00e676" if k.startswith("U") else "#ff9100"
        ax[2].plot(q[0], q[1], "o", ms=8, mfc=c, mec="black", mew=1.5); ax[2].annotate(k, (q[0], q[1]), color=c, fontsize=9, fontweight="bold", xytext=(4, 3), textcoords="offset points", path_effects=OUTLINE)
    ax[2].set_title("ALI 치아 점 18개 — 위에서 본 투영 (초록 위턱, 주황 아래턱)", fontsize=11)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{P}: ALI-CBCT 자동 랜드마크 (머리 좌표계, Po·Or 이상값 대체 적용)", fontsize=13)
    fig.tight_layout(); f = FIG/"ali_overview.png"; fig.savefig(f, dpi=90); plt.close(fig); return f


def fig_extra(v, dirs, org):
    """ALI가 찍지 않는 점(Eb·Rh·대구치 저항중심·과두 최상점)을 머리뼈 투영 위에 한꺼번에 표시."""
    import pandas as pd
    ref = json.loads((FH_REF/"reference.json").read_text(encoding="utf-8"))["points"]
    Pt = repair(landmarks(5)[0], ref)[0]; o, R = head_frame(Pt); om, Rm, _ = midline_frame(Pt)
    hd = lambda w: R@(w-o); mid = lambda a, z: hd(om+np.array([0., a, z])@Rm)
    rd = lambda f: pd.read_csv(OUT/f).set_index("Patient").loc[P]
    e, nv, c = rd("eb_review/v2/eb_candidates.csv"), rd("nav_review/v1/nav.csv"), rd("condyle_review/v1/condyle.csv")
    mo = pd.read_csv(OUT/"tdi_review/v1/molars.csv"); mo = mo[(mo.Patient == P) & mo.status.eq("ok")].set_index("tooth")
    X = {"Eb": (mid(e.Eb_ant, e.Eb_sup), "#ff4fd8"), "Rh": (mid(nv.Rh_ant, nv.Rh_sup), "#ffd400"),
         "Sup R": (np.array([c.Sup_R_L, c.Sup_R_ant, c.Sup_R_sup]), "#00c8ff"), "Sup L": (np.array([c.Sup_L_L, c.Sup_L_ant, c.Sup_L_sup]), "#00c8ff")}
    for t, r in mo.iterrows(): X[t] = (np.array([r.L, r.A, r.Z]), "#00e676" if t.startswith("U") else "#ff9100")
    tp = teeth_points(5); X.update({k: (hd(tp[k]), "#c39bff") for k in ("UR6MB", "UL6MB") if k in tp})   # ALI 치아 모델 점
    off = ({"UL6": (6, 8), "UR6": (6, -12), "LL6": (6, 6), "LR6": (6, -12), "Sup L": (-40, 6), "Sup R": (6, -12), "UR6MB": (8, -16), "UL6MB": (8, 10)},
           {"UR6MB": (-52, -6), "UL6MB": (8, -14), "UR6": (-34, 4), "UL6": (8, 6)})
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.6)); S = np.arange(-140, 30, 1.)
    A = np.arange(-115, 25, 1.); vol, _ = grid(v, dirs, org, o, R, np.arange(-70, 71, 1.), A, S)
    ax[0].imshow(vol.max(0).T, origin="lower", cmap="gray", vmin=0, vmax=2500, extent=[A[0], A[-1], S[0], S[-1]])
    Lr = np.arange(-75, 76, 1.); vol2, _ = grid(v, dirs, org, o, R, Lr, np.arange(-60, 25, 1.), S)
    ax[1].imshow(vol2.max(1).T, origin="lower", cmap="gray", vmin=0, vmax=2500, extent=[Lr[0], Lr[-1], S[0], S[-1]])
    for k in LMS:   # 위치 참고용 ALI 점 (작은 흰 점)
        ax[0].plot(hd(Pt[k])[1], hd(Pt[k])[2], "o", ms=4, mfc="white", mec="black", mew=.6, alpha=.8)
        ax[1].plot(hd(Pt[k])[0], hd(Pt[k])[2], "o", ms=4, mfc="white", mec="black", mew=.6, alpha=.8)
    for k, (q, col) in X.items():
        for i, (a, (x, y)) in enumerate(((ax[0], (q[1], q[2])), (ax[1], (q[0], q[2])))):
            a.plot(x, y, "D" if k in ("Eb", "Rh") else "o", ms=10, mfc=col, mec="black", mew=1.5)
            a.annotate(k, (x, y), color=col, fontsize=11, fontweight="bold", xytext=off[i].get(k, (6, 4)), textcoords="offset points", path_effects=OUTLINE)
    ax[0].set_title("옆 투영", fontsize=11); ax[1].set_title("앞 투영", fontsize=11)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{P}: 따로 구하는 점 — 분홍 Eb(후두개곡) · 노랑 Rh(비골 하단) · 하늘 Sup(과두 최상점) · 초록/주황 위/아래 제1대구치 저항중심 · 보라 MB(ALI 치아 모델) · 작은 흰 점 = ALI 기준점", fontsize=12)
    fig.tight_layout(); f = FIG/"extra_overview.png"; fig.savefig(f, dpi=90); plt.close(fig); return f


def skull_projection(v, dirs, org, groups, title, filename):
    """두개골 위치 안내 + 동일 원본 단면 전후 대조. 서로 다른 깊이는 겹치지 않는다."""
    from matplotlib.patches import Patch

    ref = json.loads((FH_REF/"reference.json").read_text(encoding="utf-8"))["points"]
    o, R = head_frame(repair(landmarks(5)[0], ref)[0])
    L, A, S = np.arange(-75, 76, 1.), np.arange(-115, 26, 1.), np.arange(-140, 31, 1.)
    ct, Q = grid(v, dirs, org, o, R, L, A, S)
    shape = ct.shape
    backgrounds = (ct.max(0).T, ct.max(1).T)
    extents = ([A[0]-.5, A[-1]+.5, S[0]-.5, S[-1]+.5],
               [L[0]-.5, L[-1]+.5, S[0]-.5, S[-1]+.5])
    fig, axes = plt.subplots(len(groups), 3, figsize=(16, 21), squeeze=False)
    planes = (1, 1, 1, 0) if filename == 'amasss_overview.png' else (1, 0, 0, 1)
    for row, ((name, path, labels), axis) in enumerate(zip(groups, planes)):
        sampled = lab_sample(path, Q).reshape(shape)
        values = [value for value, _, _ in labels]
        union = sampled > 0 if None in values else np.isin(sampled, values)
        index = int(union.sum(axis=tuple(i for i in range(3) if i != axis)).argmax())
        assert union.any(), name
        plain = np.take(ct, index, axis=axis).T
        section = np.take(sampled, index, axis=axis).T
        locator, raw, overlay = axes[row]
        locator_axis = 1-axis
        locator.imshow(backgrounds[locator_axis], origin='lower', cmap='gray', vmin=0, vmax=2500,
                       extent=extents[locator_axis], interpolation='nearest')
        position = (L if axis == 0 else A)[index]
        locator.axvline(position, color='#00c8ff', linewidth=1.6)
        plane_name = '시상' if axis == 0 else '관상'
        locator.set_title(f'{name}\n① 전체 두개골: 하늘색 = 확인 단면', fontsize=12)
        X = A if axis == 0 else L
        extent = extents[axis]
        for ax in (raw, overlay):
            ax.imshow(plain, origin='lower', cmap='gray', vmin=-1000, vmax=2000,
                      extent=extent, interpolation='nearest')
        raw.set_title(f'② 원본 {plane_name} 단면\n색 없이 경계 확인', fontsize=12)
        overlay.set_title(f'③ 동일 {plane_name} 단면 + 분할\n이 단면에 있는 영역만 표시', fontsize=12)
        handles = []
        for value, label, color in labels:
            mask = section > 0 if value is None else section == value
            if not mask.any():
                continue
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = matplotlib.colors.to_rgba(color, .22)
            overlay.imshow(rgba, origin='lower', extent=extent, interpolation='nearest')
            overlay.contour(X, S, mask.astype(float), [.5], colors=[color], linewidths=1)
            handles.append(Patch(facecolor=color, label=label))
        if handles and len(handles) <= 4:
            overlay.legend(handles=handles, loc='lower left', fontsize=10, framealpha=.95)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
        raw.text(.03, .03, f'{plane_name}면 위치 {position:+.0f} mm\n머리 좌표계, 구조 최대 단면',
                 transform=raw.transAxes, color='white', fontsize=9,
                 bbox=dict(facecolor='black', alpha=.7, edgecolor='none'))
        del sampled
    fig.suptitle(f'{P}: {title} — 두개골 전체 위치 안내와 실제 단면 대조', fontsize=17)
    fig.text(.5, .012, '왼쪽은 위치 안내용 뼈 투영 · 가운데/오른쪽은 동일 좌표·동일 밝기의 실제 단면 · 자동 분할의 오류도 그대로 표시', ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .035, 1, .965), h_pad=3, w_pad=2)
    path = FIG/filename
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def fig_amasss(v, dirs, org):
    names = {"MAND": "하악", "MAX": "상악", "CB": "두개저", "UAW": "상기도"}
    groups = [(names[key], CB/f"seg/{P}_seg_{key}.nrrd", [(None, names[key], color)])
              for key, color in COL.items()]
    return skull_projection(v, dirs, org, groups, "AMASSS 구조 분할", "amasss_overview.png")


def fig_other(v, dirs, org):
    ts = CB/f"seg_totalseg/head_glands_cavities/{P}.nii.gz"
    cm = plt.get_cmap("tab20")
    groups = [
        ("비강", ts, [(14, "오른쪽 비강", "#00c8ff"), (15, "왼쪽 비강", "#ffd400")]),
        ("인두", ts, [(11, "비인두", "#1baf7a"), (12, "구인두", "#2a78d6"), (13, "하인두", "#b28aff")]),
        ("구개", ts, [(18, "연구개", "#ff5959"), (19, "경구개", "#eda100")]),
        ("치아 (색으로 개별 치아 구분)", CB/f"seg_teeth/UNIVERSALLAB_MODEL/{P}.nii.gz",
         [(i, str(i), cm((i-1) % 20)) for i in range(1, 33)]),
    ]
    return skull_projection(v, dirs, org, groups, "연조직·치아 분할", "other_tools.png")


TOOLS = [
    dict(name="ALI-CBCT", what="CBCT 영상에서 해부학적 기준점(랜드마크)을 자동으로 찍는 프로그램입니다. 공개된 3D Slicer 치과 도구 모음(SlicerAutomatedDentalTools)의 사전학습 모델을 Slicer 없이 실행했습니다.",
         role="머리뼈 기준점 16개를 찍습니다. 이 점들로 좌표계를 세우고 골격 지표(ANB, MP-FH 등)를 계산하며, 기도·비강을 자르는 경계면도 이 점들에서 만듭니다.",
         weak="같은 영상을 4번 돌려도 좌표가 완전히 같았습니다. 초기 12명 점검에서 대부분의 점이 뼈 표면 0.4~0.8 mm 안에 있었습니다. 다만 귓구멍 점(Po)을 5명(P_16·20·23·35·37)에서 엉뚱한 곳에 찍었고, 턱끝 앞점(Pog)은 일부 환자에서 뼈 앞 연조직에, 턱끝 아래점(Me)은 늘 뼈 경계보다 약 1.5 mm 아래에 찍었습니다."),
    dict(name="ALI-CBCT 치아 모델", what="같은 프로그램의 치아용 모델로, 제1대구치의 교두·뿌리 쪽 점 18개를 찍습니다.",
         role="교합평면 계산, 그리고 치아 분할 결과가 맞는지 대조하는 데 씁니다.",
         weak="치아 분할 중심과 보통 2 mm 안팎으로 맞지만 일부 환자는 엉뚱한 곳에 찍습니다(예: P_55). 근심구개 교두점(MP)은 54명 중 15명에서 찾지 못했고 찾아도 벗어난 예가 있어(예시 P_5), 계산에는 쓰지 않습니다."),
    dict(name="AMASSS", what="영상에서 하악·상악 같은 구조를 색칠해 떼어 내는(분할) 프로그램입니다. 같은 도구 모음에 있으며 nnU-Net v2로 만든 모델입니다.",
         role="하악·상악·두개저·상기도 4구조를 칠합니다. 형태 분석의 표면, 턱끝 윤곽 보정, 과두 각도, 입천장 경계에 씁니다.",
         weak="초기 12명 점검에서 뼈 경계가 실제 영상 경계와 0.1~0.4 mm 안에서 맞았습니다(전체 환자 검증은 아님). 상기도는 기도 전체를 칠한 것이라, 위아래 경계를 정해 재는 논문의 기도 부피(PAV·NAV)를 대신하지 못합니다."),
    dict(name="TotalSegmentator", what="일반 CT용으로 공개된 분할 프로그램(2.18.0)의 머리·목 공간 모델입니다. CT로 학습했습니다.",
         role="좌우 비강, 비인두·구인두·하인두, 연구개·경구개를 칠합니다. 비강 부피·폭(NAV·NCWR)과 연조직·인두 정보에 씁니다.",
         weak="CBCT는 밝기 값이 장비·촬영마다 달라 CT로 배운 모델이 연조직끼리의 경계를 헷갈릴 수 있습니다. 공기와 맞닿은 경계는 CBCT 전용 분할(AMASSS)과 Dice 0.84로 잘 맞았지만, 연구개 앞면처럼 조직끼리 맞닿은 경계는 확인할 방법이 없습니다."),
    dict(name="치아 분할 (UNIVERSALLAB)", what="치아 32개를 하나씩 따로 칠하는 공개 nnU-Net 모델입니다.",
         role="제1대구치 위치(TDI·PHI 계산)와 치아·치열궁 정보에 씁니다.",
         weak="가끔 라벨이 다른 치아나 턱뼈에 붙습니다. 그래서 지표 계산(TDI·PHI)에는 부피 범위, ALI 치아 점과의 거리, 좌우 대칭으로 제1대구치 번호를 확인한 치아만 씁니다(3절).")]

USE = {"N": "머리 좌표계의 원점, ANB, Nperp, 비강 부피의 위 경계", "S": "정중 좌표계, 기도 부피·최소 단면적의 위 경계",
       "Ba": "정중면(MSP), 정중 좌표계, 비강 부피의 위 경계", "RPo": "수평면(FH) → ANB·MP-FH·Nperp", "LPo": "수평면(FH) → ANB·MP-FH·Nperp",
       "ROr": "수평면(FH) → ANB·MP-FH·Nperp", "LOr": "4점 수평면 → MP-FH·Nperp", "ANS": "정중 좌표계의 앞 축, 비강 부피의 아래 경계, 비강 프로파일",
       "PNS": "정중 좌표계의 원점, PAL, 기도·비강 경계", "A": "ANB, Nperp-A", "B": "ANB", "Pog": "Nperp-Pog", "Me": "하악평면 → MP-FH, 턱끝 편위",
       "Gn": "하악 부위별 형태(턱끝 부위) 구분", "RGo": "하악평면 → MP-FH, 하악 부위별 형태 구분", "LGo": "하악평면 → MP-FH, 하악 부위별 형태 구분"}
FAIL = {"RPo": "5명에서 엉뚱한 위치 → 대칭 보정(2절)", "LPo": "5명에서 엉뚱한 위치 → 대칭 보정(2절)", "Pog": "일부 환자에서 뼈 앞 연조직에 찍힘 → 윤곽 보정(6절 Nperp-Pog)",
        "Me": "늘 뼈 경계보다 약 1.5 mm 아래 (보정해도 MP-FH가 나아지지 않아 그대로 사용)"}
KOR = {"N": "코뿌리점", "S": "터키안장 중심점", "Ba": "대후두공 앞점", "RPo": "오른쪽 귓구멍 점", "LPo": "왼쪽 귓구멍 점", "ROr": "오른쪽 눈밑점", "LOr": "왼쪽 눈밑점",
       "ANS": "앞코가시(입천장 앞끝)", "PNS": "뒤코가시(입천장 뼈 뒤끝)", "A": "상악 A점", "B": "하악 B점", "Pog": "턱끝 앞점", "Me": "턱끝 아래점", "Gn": "턱끝점(Pog와 Me 사이)",
       "RGo": "오른쪽 하악각점", "LGo": "왼쪽 하악각점"}

# ALI가 찍지 않아 따로 구하는 점: (그림 키, 제목, 쓰이는 곳, 구하는 방법)
EXTRA = [
    (["Eb"], "1. 후두개곡 최하점 (Eb)",
     "혀뿌리와 후두개(음식이 기도로 넘어가지 않게 덮는 연골) 사이에 오목하게 파인 공간(후두개곡)의 가장 아랫점입니다.",
     "인두 기도 길이(PAL), 인두 기도 부피(PAV), 최소 단면적(PMCA)의 아래쪽 끝 경계.",
     [("설골 위치 찾기", "뒤코가시(PNS)를 원점으로 하는 정중 좌표계에서 정중선 좌우 3 mm 두께의 뼈 영상을 겹쳐 봅니다. 밝기 450 이상인 가장 큰 뼈 덩어리(설골)를 찾고, 그 뒤쪽 1/3의 꼭대기를 기준점으로 삼습니다."),
      ("공기 따라가기", "설골 부근에서 정중선 좌우 6 mm 안의 공기(밝기 −400 미만)를 위에서 아래로 한 줄씩 훑어 내려갑니다."),
      ("후두개곡 찾기", "공기가 1~10 mm 두께의 후두개를 사이에 두고 앞뒤 두 갈래로 갈라지는 곳을 찾습니다. 앞쪽 갈래(후두개곡)가 끝나는 가장 아랫점을 Eb로 정합니다."),
      ("못 찾을 때", "후두개곡 공기가 보이지 않으면(혀뿌리와 후두개가 맞닿은 경우 등), 직접 찾은 환자들의 '설골 → Eb' 거리 중앙값을 설골 위치에 더해 추정합니다. 설골이 보이지 않거나 Eb가 촬영 범위 밖이면 재지 않습니다(4명).")]),
    (["Rh"], "2. 비골 하단점 (Rh)",
     "콧등을 이루는 단단한 코뼈(비골)가 끝나고 아래쪽 연골로 바뀌는 코뼈 맨 아래 끝점입니다.",
     "코 안 공기 부피(NAV)의 앞쪽 경계면(Rh–ANS 면).",
     [("정중 코뼈만 남기기", "정중 좌표계에서 정중선 좌우 1 mm 안의 단단한 뼈(밝기 300 이상)만 남깁니다."),
      ("콧등 선 그리기", "코뿌리점(N) 높이부터 0.4 mm씩 아래로 내려가며, 높이마다 뼈가 가장 앞으로 나온 위치를 기록합니다."),
      ("코뼈 끝 찾기", "코뼈가 끝나면 연골은 밝기가 300보다 낮아 뼈 경계가 갑자기 뒤로 들어갑니다. N 아래 8 mm부터 보면서, 뼈 앞끝이 한 줄 만에 3 mm 넘게 뒤로 물러나는 곳의 바로 윗점을 Rh로 정합니다.")]),
    (["UR6", "LR6"], "3. 제1대구치 저항중심 (UR6·UL6·LR6·LL6)",
     "치아에 힘을 줬을 때 치아가 돌지 않고 평행하게 움직이는 중심점입니다. 뿌리가 여럿인 어금니는 뿌리가 갈라지기 시작하는 곳(분지부)을 저항중심으로 봅니다(Koo 2017).",
     "위아래 치열 폭 차이(TDI)와 입천장 높이 지수(PHI)에서 좌우 폭을 재는 기준점.",
     [("제1대구치 꺼내기", "치아 분할 결과에서 제1대구치 4개(미국식 번호 위 3·14번, 아래 19·30번)를 꺼냅니다. 부피가 정상 범위(0.3~2 mL)를 벗어나면 잘못 칠한 것으로 보고 뺍니다."),
      ("단면 내려가기", "머리 좌표계의 수평 단면으로, 치관이 가장 넓은 단면부터 뿌리 끝 방향(위 어금니는 위쪽, 아래 어금니는 아래쪽)으로 0.5 mm씩 이동합니다."),
      ("분지부 찾기", "하나이던 단면이 넓이 3 mm² 이상 덩어리 2개 이상으로 갈라진 상태가 1.5 mm(3장) 넘게 이어지는 첫 높이를 찾고, 갈라지기 바로 위 단면의 중심을 저항중심으로 정합니다."),
      ("못 찾을 때", "뿌리가 붙어 있어 분지부가 안 보이면, 다른 환자들의 '최대 단면 → 분지부' 거리 중앙값(위 7.5 mm, 아래 6.0 mm)만큼 내려간 단면의 중심을 씁니다."),
      ("치아 번호 확인", "분할 모델이 옆 치아(예: 제2대구치)에 제1대구치 번호를 붙이는 경우를 거릅니다. ① ALI 치아 모델이 찍은 같은 치아 점이 분할 치아에서 4~15 mm 떨어져 있으면 옆 치아로 보고 뺍니다(15 mm를 넘으면 ALI 점 쪽 오류라 판단에 쓰지 않음). ② 좌우 제1대구치의 앞뒤 위치가 8 mm 넘게 어긋나면 그 턱의 폭을 재지 않습니다(보통 차이는 위 1.0 mm, 아래 1.6 mm).")]),
    (["UR6MB"], "4. 제1대구치 근심협측 교두점 (MB)",
     "위 첫째 큰어금니의 씹는 면 봉우리(교두) 중 가장 앞쪽이면서 뺨 쪽에 있는 끝점입니다.",
     "엑셀 추가 변수인 교합평면(위아래 치아가 맞물리는 평면) 각도.",
     [("모델 좌표 그대로 쓰기", "영상 규칙을 거치지 않고 ALI-CBCT 치아 모델(머리뼈 기준점 모델과는 다른 모델)이 찍은 좌표를 그대로 씁니다. 안쪽 교두점(MP)은 찾지 못하는 경우가 많아 쓰지 않습니다."),
      ("교합평면 만들기", "좌우 위 제1대구치 MB 두 점과 앞니 겹침 중간점(위 중절치 아래끝과 아래 중절치 위끝의 가운데, 치아 분할로 구함), 모두 세 점으로 교합평면을 정합니다.")]),
    (["Sup"], "5. 과두 최상점 (Sup)",
     "아래턱뼈 양쪽 끝이 귀 앞 턱관절과 맞물리는 둥근 머리(하악과두)의 가장 높은 꼭대기 점입니다.",
     "턱관절 기울기를 나타내는 과두 측면각과 축상각.",
     [("찾을 범위 정하기", "AMASSS 하악 분할에서, 같은 쪽 하악각점(Go)을 기준으로 앞뒤 −10~+15 mm, 좌우 ±15 mm 상자만 봅니다. 하악 앞쪽의 뾰족한 근돌기가 과두로 잘못 잡히지 않도록 뺍니다."),
      ("꼭대기 찾기", "그 범위에서 머리 좌표계로 가장 높은 하악 뼈 점을 Sup으로 정합니다."),
      ("과두 목과 잇기", "Sup 아래로 내려가며 과두 목이 가장 잘록한 곳의 중점(Mid)을 찾고, Sup–Mid 직선으로 과두의 기울기(측면각)를 잽니다.")]),
]


# 지표 카드: 이름, 우리말 뜻, 재는 순서, 왜 이렇게, 한계, 이번에 바뀐 점, 논문 근거 등급
METRICS = [
    ("ANB", "위턱과 아래턱의 앞뒤 관계를 나타내는 각도(°)", ["코뿌리점(N), 상악 A점, 하악 B점을 ALI로 찍습니다.", "N→A, N→B 두 선을 긋습니다.", "두 선을 정중면에 그림자처럼 투영해 사이 각도를 잽니다."],
     "3차원 그대로 재면 머리가 좌우로 기운 정도가 섞이므로, 측면 사진처럼 정중면에 투영합니다.", "A·B는 '가장 오목한 곳'이라는 정의라 점이 조금만 움직여도 값이 바뀝니다.", "바꾸지 않았습니다. 기존 3점 수평면 방식이 가장 정확했습니다.", "A"),
    ("Nperp-A", "상악이 코뿌리 수직선보다 얼마나 앞에 있는가(mm, 앞쪽 +)", ["코뿌리점(N)을 지나 수평면(FH)에 수직인 선을 긋습니다.", "그 선에서 A점까지 앞뒤 거리를 잽니다."],
     "수평면에 수직인 선을 기준으로 하면 고개를 숙이거나 든 자세의 영향이 빠집니다.", "수평면이 귓구멍 점·눈밑점에 달려 있어, 이 점이 틀린 환자는 값이 크게 틀립니다.",
     "수평면을 4점(좌우 귓구멍·좌우 눈밑점)으로 잡아 점 하나의 오류 영향을 줄였고, 엑셀 측정 방식에 맞춘 직선 환산을 적용했습니다.", "A"),
    ("Nperp-Pog", "턱끝이 코뿌리 수직선보다 얼마나 앞에 있는가(mm, 앞쪽 +)", ["Nperp-A와 같은 수직선을 긋습니다.", "턱끝 앞점(Pog)까지 앞뒤 거리를 잽니다."],
     "Nperp-A와 같은 이유입니다.", "ALI가 Pog를 뼈 앞 연조직에 찍는 환자가 있습니다.",
     "4점 수평면에 더해, 턱끝 앞점을 하악 분할의 앞쪽 윤곽에서 다시 잡도록 했고(뼈 위로 옮김), 엑셀 측정 방식에 맞춘 직선 환산을 적용했습니다.", "A"),
    ("MP-FH", "아래턱 아랫면이 수평면에 대해 기운 각도(°)", ["좌우 하악각점(Go)의 가운데와 턱끝 아래점(Me)을 이어 하악평면을 만듭니다.", "하악평면과 수평면(FH)의 각도를 정중면에 투영해 잽니다."],
     "얼굴이 수직으로 긴지(하악평면이 가파른지) 보는 대표 지표입니다.", "Me가 늘 약 1.5 mm 아래 찍히고, 수평면이 귓구멍 점에 민감합니다.", "수평면을 4점으로 잡았습니다. 엑셀 방식 환산은 효과가 작아 적용하지 않았습니다.", "B"),
    ("TDI", "위턱 치열 폭과 아래턱 치열 폭의 차이(mm)", ["치아 분할에서 좌우 상악 제1대구치와 좌우 하악 제1대구치를 찾습니다.", "치아마다 저항중심(뿌리가 갈라지는 높이 부근)의 위치를 구합니다.", "상악 좌우 거리 − 하악 좌우 거리를 계산합니다."],
     "논문의 '치조골 폭' 점은 자동으로 찍을 수 없어, 대구치 저항중심 사이 폭으로 대신했습니다.", "치아 라벨이 틀리거나 대구치가 없으면 결측입니다(번호 확인 규칙은 3절).", "분할 모델이 옆 치아에 제1대구치 번호를 붙인 경우를 거르도록, 모든 대구치에 ALI 치아 점 거리와 좌우 대칭 확인을 넣었습니다.", "단면 B / 점 C"),
    ("PHI", "입천장이 폭에 비해 얼마나 높은가(%)", ["상악 제1대구치를 지나는 관상 단면을 잡습니다.", "입천장 천장에서 교합 높이까지를 구개 높이로 잽니다.", "좌우 제1대구치 저항중심 사이를 폭으로 잽니다.", "높이 ÷ 폭 × 100을 계산합니다."],
     "입천장이 좁고 깊은지 보는 지표입니다.", "원 측정에서 높이를 어디서 쟀는지 확인되지 않았습니다.", "엑셀 측정 방식에 맞춘 직선 환산을 적용했습니다.", "C"),
    ("NCWR", "코 안 폭이 위턱 폭에 비해 얼마나 넓은가(%)", ["상악 제1대구치 단면에서 비강 라벨의 가장 넓은 좌우 폭을 잽니다.", "같은 단면에서 위턱 기저 폭을 잽니다.", "비강 폭 ÷ 위턱 폭 × 100을 계산합니다."],
     "코 안이 좁은지 보는 지표입니다.", "분모(위턱 기저 폭)의 원 정의가 확인되지 않았고, 비강 폭에는 비갑개 같은 조직도 들어갑니다.", "바꾸지 않았습니다.", "C"),
    ("PAL", "입천장 뒤끝에서 후두개곡까지의 기도 길이(mm)", ["입천장 뼈 뒤끝(PNS)을 ALI로 찍습니다.", "후두개곡 최하점(Eb)을 영상에서 찾습니다: 정중 부근 공기에서 후두개 앞쪽 홈의 가장 아랫점입니다.", "홈이 보이지 않으면 설골 위치로 추정합니다.", "PNS–Eb 직선 거리를 잽니다."],
     "Eb는 ALI가 찍지 않는 점이라 영상 규칙으로 찾았습니다.", "후두개곡이 촬영 범위 밖이면 잴 수 없습니다(4명). 이 환자들은 기도가 긴 편이라 결측이 한쪽으로 몰립니다.", "바꾸지 않았습니다.", "A"),
    ("PAV", "인두 기도 부피(mm³)", ["공기로 볼 밝기 범위(−1200~−400)를 정합니다.", "위 경계 S–PNS 면, 앞 경계 PNS를 지나는 관상면, 아래 경계 Eb를 지나는 수평면으로 자릅니다.", "그 안에서 정중선과 이어진 공기 덩어리의 부피를 잽니다."],
     "논문(Roh)의 경계 정의를 그대로 따랐습니다.", "논문의 위 경계점(Se)이 무엇인지 불분명해 터키안장 중심(S)으로 대신했습니다. 엑셀 값이 있는 환자가 25명뿐입니다. CBCT 밝기 값은 보정되지 않아 공기 경계가 장비에 따라 달라질 수 있습니다.", "바꾸지 않았습니다.", "A"),
    ("NAV", "코 안 공기 부피(mm³)", ["위 N–Ba 면 아래, 아래 입천장(ANS–PNS) 면 위, 뒤 S–PNS 면 앞, 앞 비골 하단점(Rh)–ANS 면 뒤로 자릅니다.", "그 안의 공기 중 비강 라벨 안에 있는 것만 남깁니다.", "부피를 잽니다."],
     "비강 라벨로 거르는 이유는 옆의 부비동 공기가 섞이지 않게 하기 위해서입니다. Rh는 ALI가 찍지 않아 영상 규칙(코뼈 아래 끝)으로 찾았습니다.", "엑셀 값보다 일정하게 작게 나옵니다. 경계나 부비동 처리 방식이 다른 것으로 보입니다.", "엑셀 측정 방식에 맞춘 직선 환산을 적용했습니다.", "A (부비동 제거는 B)"),
    ("PMCA", "인두 기도에서 가장 좁은 단면의 넓이(mm²)", ["PAV와 같은 기도 공기를 씁니다.", "기도 중심선을 따라 중심선에 수직인 단면을 자릅니다.", "Eb 위 5 mm부터 PNS 높이 사이에서 가장 좁은 단면의 넓이를 잽니다."],
     "아래 경계 바로 위는 단면이 잘려 작게 나오기 쉬워 5 mm 여유를 뒀습니다.", "엑셀 값보다 일정하게 크게 나옵니다. 원 측정의 단면 방향이나 범위가 달랐던 것으로 보입니다.", "엑셀 측정 방식에 맞춘 직선 환산을 적용했습니다.", "B")]


CSS = """:root{--bg:#fcfcfb;--card:#fff;--fg:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6;--sum:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#222220;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--sum:#1d2633}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 "Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1150px;margin:0 auto;padding:24px 16px 80px}h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.25rem;margin:40px 0 10px;padding-top:8px;border-top:2px solid var(--accent)}
h3{font-size:1.05rem;margin:4px 0}.sub,.note{color:var(--muted)}.note{font-size:.88rem}a{color:var(--accent)}nav{display:flex;flex-wrap:wrap;gap:4px 14px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:.88rem;background:var(--card)}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--bg)}
img{max-width:100%;border:1px solid var(--line);background:#fff}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px}.card img{width:100%}.card p{margin:4px 0;font-size:.88rem}
.tool{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin:10px 0}.tool p{margin:4px 0}
.mcard{display:grid;grid-template-columns:minmax(260px,400px) 1fr;gap:16px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px;margin:14px 0}
@media (max-width:700px){.mcard{grid-template-columns:1fr}}dt{font-weight:700;font-size:.85rem;color:var(--muted);margin-top:8px}dd{margin:0}ol.steps{margin:2px 0 0;padding-left:20px}
.sum{background:var(--sum);border-radius:6px;padding:8px 12px;margin:12px 0}"""


def gimg(key):
    m = json.loads((GL/"manifest.json").read_text(encoding="utf-8")); e = m.get(key)
    if not e: return ""
    src = rel(GL/Path(e["src"]).name)
    return f'<a href="{src}" target="_blank"><img src="{src}" alt="{escape(key)}"></a><p class="note">{escape(re.sub(r"\s*(자동 계측 예시이며 |표시한 자동 높이·점은 )?전문가 확인 전입니다\.?", "", e["caption"]).strip())}</p>'


def main():
    from report_terms import check
    from report_editor import apply
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; FIG.mkdir(parents=True, exist_ok=True)
    v, dirs, org = read_nrrd(CB/f"nrrd/{P}.nrrd"); v = v.astype(np.float32)
    f_ali, f_am, f_ot, f_ex = fig_ali(v, dirs, org), fig_amasss(v, dirs, org), fig_other(v, dirs, org), fig_extra(v, dirs, org); del v
    tools = "".join(f'<div class="tool"><h3>{escape(t["name"])}</h3><p><b>무엇을 하나:</b> {escape(t["what"])}</p><p><b>이 연구에서 맡은 일:</b> {escape(t["role"])}</p>'
                    f'<p><b>확인된 정확도와 약점:</b> {escape(t["weak"])}</p></div>' for t in TOOLS)
    lm = "".join(f'<div class="card"><h3>{KOR[k]} ({k})</h3>{gimg(k)}<p>{escape(TERMS.get(k, ""))}</p><p><b>쓰이는 곳:</b> {escape(USE[k])}</p>'
                 + (f'<p><b>알려진 문제:</b> {escape(FAIL[k])}</p>' if k in FAIL else "") + "</div>" for k in LMS)
    extra = "".join(f'<div class="mcard"><div>{"".join(gimg(k) for k in ks)}</div><div><h3>{escape(t)}</h3><dl><dt>해부학적 의미</dt><dd>{escape(mean)}</dd>'
                    f'<dt>쓰이는 곳</dt><dd>{escape(use)}</dd><dt>구하는 방법</dt><dd><ol class="steps">{"".join(f"<li><b>{escape(a)}:</b> {escape(b)}</li>" for a, b in how)}</ol></dd></dl></div></div>'
                    for ks, t, mean, use, how in EXTRA)
    planes = "".join(f'<div class="card"><h3>{escape(t)}</h3>{gimg(k)}<p>{escape(TERMS[k])}</p></div>' for k, t in
                     (("FH", "수평면 (FH)"), ("MSP", "정중면 (MSP)"), ("MP", "하악평면 (MP)"), ("Nperp", "코뿌리 수직선 (Nperp)")))
    mc = "".join(f"""<div class="mcard"><div>{gimg(n)}</div><div><h3>{n} — {escape(kor)}</h3><dl>
<dt>재는 순서</dt><dd><ol class="steps">{"".join(f"<li>{escape(s)}</li>" for s in steps)}</ol></dd><dt>왜 이렇게 재나</dt><dd>{escape(why)}</dd>
<dt>한계</dt><dd>{escape(lim)}</dd><dt>이번 작업에서 바뀐 점</dt><dd>{escape(ch)}</dd><dt>논문 정의 근거</dt><dd>{escape(g)}</dd></dl></div></div>"""
                 for n, kor, steps, why, lim, ch, g in METRICS)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CBCT 측정 기준</title><style>{CSS}</style></head><body><main>
<h1>CBCT 측정 기준 보고서</h1>
<p class="sub">2026-11-15 · 어떤 프로그램이 무엇을 어떤 기준으로 재는지 · 측정 결과는 별도의 종합 보고서에 있습니다</p>
<div class="sum"><b>이 보고서를 읽는 법</b><ul>
<li>CBCT 측정은 두 단계입니다. 먼저 프로그램이 <b>기준점을 찍고</b>(ALI-CBCT) <b>구조를 색칠합니다</b>(AMASSS 등). 그다음 그 점과 영역으로 <b>각도·거리·부피를 계산</b>합니다.</li>
<li>1절은 프로그램, 2절은 계산의 바탕이 되는 좌표계, 3~5절은 프로그램이 찍고 칠한 결과, 6절은 지표 11개를 재는 방법, 7절은 품질 점검 기준입니다.</li>
<li>그림의 점·윤곽은 모두 프로그램이 자동으로 낸 결과입니다. 기존 용어 그림 속 영어 축 이름은 Anterior = 앞, Superior = 위, Left = 왼쪽입니다.</li></ul></div>
<nav><a href="#tools">1. 사용한 프로그램</a><a href="#frames">2. 좌표계</a><a href="#ali">3. 기준점</a><a href="#amasss">4. 구조 분할</a><a href="#other">5. 연조직·치아 분할</a><a href="#metrics">6. 지표 11개</a><a href="#qc">7. 품질 점검</a></nav>
<h2 id="tools">1. 사용한 프로그램</h2>
<p>사람이 영상마다 점을 찍는 대신, 공개된 사전학습 프로그램 다섯 개로 자동 측정했습니다. 다만 프로그램이 찍어 주지 않는 점(후두개곡, 비골 하단, 대구치 저항중심 등)은 영상 규칙으로 따로 찾았고, 모든 규칙은 환자 모두에게 똑같이 적용했습니다.</p>
{tools}
<div class="sum">한 줄 요약: 기준점은 ALI-CBCT, 뼈·기도 영역은 AMASSS, 비강·인두·연구개는 TotalSegmentator, 치아는 치아 분할 모델이 맡고, 각 프로그램의 약점은 2·6·7절의 보정과 점검으로 보완합니다.</div>
<h2 id="frames">2. 좌표계: 왜 두 가지인가</h2>
<p>각도와 길이를 재려면 먼저 "어디가 앞이고 어디가 위인지"를 정하는 기준 좌표계가 필요합니다. 이 연구는 좌표계를 두 가지로 나눴습니다. <b>ALI-CBCT가 양쪽 귓구멍 점(Po)을 가끔 잘못 찍기 때문에</b>, 이 오류의 영향을 받지 않는 좌표계가 따로 필요했습니다.</p>
<h3>① 머리 좌표계 — 골격 지표용 (뼈의 각도·길이)</h3>
<p>ANB, MP-FH, Nperp처럼 교정학에서 쓰는 전통적인 골격 지표를 잴 때 쓰는 표준 좌표계입니다.</p>
<ul><li><b>수평면(FH):</b> 양쪽 귓구멍 점과 오른쪽 눈밑점(RPo·LPo·ROr) 세 점으로 만든 면입니다.</li>
<li><b>정중면(MSP):</b> 코뿌리점(N)과 대후두공 앞점(Ba)을 지나면서 수평면에 수직으로 세운 면입니다. 머리를 좌우로 나눕니다.</li>
<li><b>원점과 방향:</b> 코뿌리점을 (0, 0, 0)으로 두고 왼쪽·앞·위를 + 방향으로 잡습니다(mm). 각도는 정중면에 그림자처럼 투영해 잽니다.</li></ul>
<p><b>문제:</b> 귓구멍 점이나 눈밑점이 틀리면 수평면 전체가 기우뚱해지고, 그 위에서 재는 지표가 모두 틀어집니다. 실제로 ALI-CBCT는 5명(P_16·20·23·35·37)에서 귓구멍 점을 엉뚱한 곳에 찍었습니다.</p>
<p><b>해결 장치 두 가지</b></p>
<ul><li><b>대칭 보정:</b> 한쪽 귓구멍 점이나 눈밑점이 다른 환자들의 같은 점 위치 분포에서 크게 벗어나면(중앙값에서 5 MAD 넘게), 정상인 반대쪽 점을 정중면에 거울처럼 비춰 그 자리로 대신합니다. 양쪽이 모두 벗어나면 고치지 않고 표시만 합니다.</li>
<li><b>4점 수평면:</b> MP-FH와 Nperp는 정확도 비교 실험에서, 세 점 대신 양쪽 귓구멍·양쪽 눈밑점 네 점을 모두 써서 가장 잘 맞는 평면(최소제곱 평면)을 만드는 쪽이 더 정확했습니다. 점 하나가 틀려도 나머지 세 점이 받쳐 줍니다. Nperp-Pog는 여기에 턱끝 윤곽 보정을 더했고, ANB는 기존 3점 방식이 가장 정확해 그대로 둡니다.</li></ul>
<h3>② 정중 좌표계 — 기도·형태·연조직용</h3>
<p>기도 길이·부피(PAL, PAV 등), 코·연조직, 형태 분석에 쓰는 좌표계입니다.</p>
<ul><li><b>왜 따로:</b> 기도와 코는 얼굴 한가운데 있는데, 오류가 잦은 귓구멍 점으로 기준을 잡으면 멀쩡한 기도 측정값까지 함께 틀어집니다.</li>
<li><b>정중면:</b> 귓구멍 점을 아예 쓰지 않고, 얼굴 한가운데 놓인 뼈 점 다섯 개(N·S·Ba·ANS·PNS)를 가장 잘 지나는 평면으로 잡습니다.</li>
<li><b>원점과 앞 방향:</b> 입천장 뼈 뒤끝(PNS)을 원점으로, 입천장 앞끝(ANS)과 뒤끝(PNS)을 잇는 선을 앞뒤 축으로 둡니다.</li></ul>
<div class="grid">{planes}</div>
<div class="sum">한 줄 요약: 골격 지표는 학계 기준에 맞춰 귓구멍 점 기반의 머리 좌표계를 쓰되 튀는 점을 보정하고, 기도·연조직은 귓구멍 점 오류에 흔들리지 않도록 얼굴 한가운데 다섯 점만으로 만든 정중 좌표계를 씁니다.</div>
<h2 id="ali">3. 기준점 (ALI-CBCT)</h2>
<p>ALI-CBCT가 찍는 머리뼈 기준점 16개입니다. 위 그림은 한 환자에서 16개 점과 제1대구치 점 18개가 어디에 있는지 한눈에 보여 주고, 아래 카드는 점마다 실제 단면 위치, 정의, 쓰이는 곳, 알려진 문제를 정리했습니다.</p>
<a href="{rel(f_ali)}" target="_blank"><img src="{rel(f_ali)}" alt="기준점 전체"></a>
<div class="grid" style="margin-top:12px">{lm}</div>
<h3 style="margin-top:20px">ALI-CBCT가 찍지 않아 따로 구하는 점</h3>
<p>ALI-CBCT 머리뼈 모델이 찍지 않는 점 다섯 가지를 영상 밝기 규칙, 분할 모양 분석, 치아 전용 모델로 찾습니다. 모든 환자에게 같은 규칙을 씁니다. 아래 그림은 한 환자에서 이 점들이 머리뼈 어디에 있는지 한눈에 보여 주고, 카드는 점마다 실제 단면 위치를 보여 줍니다.</p>
<a href="{rel(f_ex)}" target="_blank"><img src="{rel(f_ex)}" alt="따로 구하는 점 전체"></a>
{extra}
<div class="sum">한 줄 요약: 기준점 대부분은 뼈 위에 정확히 찍히지만, 귓구멍 점(5명)·턱끝 앞점(일부)·턱끝 아래점(늘 1.5 mm 아래)은 알려진 문제가 있어 보정하거나 한계로 둡니다.</div>
<h2 id="amasss">4. 구조 분할 (AMASSS)</h2>
<p>분할은 영상에서 특정 구조에 해당하는 부분을 색칠해 떼어 내는 작업입니다. AMASSS는 하악·상악·두개저·상기도 4구조를 원본 영상과 같은 칸(0.39 mm)에 칠합니다.</p>
<a href="{rel(f_am)}" target="_blank"><img src="{rel(f_am)}" alt="구조 분할"></a>
<p class="note">{PROJECTION_NOTE}</p>
<table style="margin-top:10px"><tr><th>구조</th><th>쓰인 곳</th></tr>
<tr><td>하악</td><td>하악 전체·부위별 형태 분석의 표면, 턱끝 앞점 윤곽 보정, 과두 각도</td></tr>
<tr><td>상악</td><td>상악 형태 분석, 턱끝 윤곽 보정, 입천장 높이(PHI)의 경계</td></tr>
<tr><td>두개저</td><td>두개저 형태 분석. 표면 맞추기 품질에 문제가 있어 해석에 주의</td></tr>
<tr><td>상기도</td><td>기도 형태 분석, 인두 라벨 점검(5절). 기도 전체를 칠한 것이라 논문 경계로 자르는 PAV·NAV를 대신하지 않음</td></tr></table>
<div class="sum">한 줄 요약: AMASSS는 뼈와 기도를 칠해 형태 분석과 점 보정의 바탕을 만들지만, 기도 부피 지표는 논문의 경계로 따로 잽니다.</div>
<h2 id="other">5. 연조직·치아 분할</h2>
<p>연조직과 치아는 AMASSS가 칠하지 않아 다른 공개 프로그램 두 개를 더 썼습니다.</p>
<a href="{rel(f_ot)}" target="_blank"><img src="{rel(f_ot)}" alt="연조직·치아 분할"></a>
<p class="note">{PROJECTION_NOTE}</p>
<ul><li><b>TotalSegmentator:</b> 일반 CT로 학습한 프로그램이라 CBCT에서는 주의가 필요합니다. CBCT는 밝기 값이 장비마다 달라 연조직끼리의 경계를 헷갈릴 수 있기 때문입니다. 그래서 공기와 맞닿은 경계(비강·인두 안쪽, 연구개 뒷면)만 믿고 씁니다. 인두 라벨은 CBCT 전용 분할(AMASSS 상기도)과 Dice 0.84로 잘 맞았습니다.</li>
<li><b>치아 분할:</b> 치아 32개를 미국식 번호(1~32)로 칠합니다. 가끔 다른 치아나 턱뼈에 라벨이 붙어서, 쓰기 전에 거릅니다. 지표용 제1대구치는 3절의 번호 확인 규칙을 씁니다.</li></ul>
<div class="sum">한 줄 요약: 연조직은 CT용 프로그램이라 공기 경계만 믿고, 치아는 라벨 오류를 세 가지 점검으로 걸러서 씁니다.</div>
<h2 id="metrics">6. 지표 11개를 재는 방법</h2>
<p>각 지표는 Roh 논문의 정의를 기준으로 했습니다. "논문 정의 근거"는 A = 논문 정의 그대로, B = 선행연구의 정의를 가져옴, C = 논문 정의가 불분명해 여러 후보 중 고름입니다.
"엑셀 측정 방식에 맞춘 직선 환산"은 측정을 고친 것이 아니라, 우리 값과 엑셀 값 사이의 일정한 차이를 직선으로 맞춰 엑셀 측정자의 방식으로 옮겨 읽은 것입니다.</p>
{mc}
<div class="sum">한 줄 요약: 골격 지표는 기준점과 좌표계로, 치아 지표는 치아 분할로, 기도 지표는 공기 밝기와 논문 경계면으로 잽니다. 논문 정의가 불분명한 지표(TDI·PHI·NCWR)와 엑셀과 일정한 차이가 나는 지표(NAV·PMCA)는 해석에 주의가 필요합니다.</div>
<h2 id="qc">7. 품질 점검 기준</h2>
<table><tr><th>무엇을 확인하나</th><th>기준</th><th>걸리면</th></tr>
<tr><td>기준점이 엉뚱한 곳에 찍혔는지</td><td>기준점 사이 거리가 다른 환자들과 비교해 유난히 다른지 (거리 분포는 초기 환자로 미리 고정)</td><td>"다시 확인 필요" 표시</td></tr>
<tr><td>귓구멍 점·눈밑점이 튀는지</td><td>다른 환자들의 같은 점 위치에서 중앙값 기준 5 MAD를 넘는지</td><td>반대쪽 점의 대칭 위치로 대신 (2절)</td></tr>
<tr><td>재려는 구조가 촬영 범위 안인지</td><td>기준점·경계가 영상 안에 있는지</td><td>범위 밖이면 재지 않음. P_78은 촬영 범위가 102 mm라 골격 지표 불가, P_1·20·50·68은 후두개곡이 영상 밖이라 PAL·PAV·PMCA 불가</td></tr>
<tr><td>공기를 어떻게 고르나</td><td>밝기 −1200~−400을 공기로 봄 (Roh 논문 기준)</td><td>CBCT 밝기는 보정되지 않아 경계가 장비마다 조금 달라질 수 있음</td></tr>
<tr><td>지표용 제1대구치 번호가 맞는지</td><td>부피 0.3~2 mL, ALI 치아 점과 4~15 mm 떨어지지 않을 것, 좌우 앞뒤 차이 8 mm 이내</td><td>그 치아(또는 그 턱의 폭)를 계산에서 뺌</td></tr>
<tr><td>형태 정보용 치아 라벨이 맞는지</td><td>치아 하나 부피 0.1~1.6 mL, 위턱 치아가 아래턱 치아보다 위, 제1대구치는 ALI 점과 10 mm 이내</td><td>그 치아를 치열궁 정보에서 뺌</td></tr></table>
<div class="sum">한 줄 요약: 자동 점검은 "명백히 이상한 값"을 거르는 장치이고, 걸린 환자는 영상을 직접 열어 확인합니다.</div>
<p class="note">생성: python/cbct_method_report.py · 측정 결과는 종합 보고서(CBCT_종합보고서_2026-11-15)를 보세요.</p>
</main></body></html>"""
    check(html, "측정 기준 보고서"); HTML.write_text(apply(html), encoding="utf-8"); print("saved", HTML)


def refresh_segmentation_figures():
    """기존 보고서의 다른 내용을 보존하고 4·5절 그림과 설명만 갱신."""
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    v, dirs, org = read_nrrd(CB/f'nrrd/{P}.nrrd')
    v = v.astype(np.float32)
    for draw in (fig_amasss, fig_other):
        print(draw.__name__, flush=True)
        draw(v, dirs, org)
    text = HTML.read_text(encoding='utf-8')
    pattern = r'(<a [^>]*><img [^>]*src="method_figs/(?:amasss_overview|other_tools).png"[^>]*></a>)(?:\s*<p class="note">[^<]*</p>)?'
    text, count = re.subn(pattern, lambda m: m[1]+'\n<p class="note">'+PROJECTION_NOTE+'</p>', text)
    assert count == 2, count
    HTML.write_text(text, encoding='utf-8')
    print('Updated segmentation figures and captions', flush=True)


if __name__ == "__main__":
    import sys
    if '--segmentation-only' in sys.argv:
        refresh_segmentation_figures()
    else:
        main()
