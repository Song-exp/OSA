"""OSA 분석용 피처 파이프라인: 기존 결과에서 실측 거리·연동 피처·교호작용을 한 번에 다시 만든다.

사용
  python build_features.py              피처만 다시 만든다(영상 계산은 입력이 바뀐 환자만)
  python build_features.py --upstream   상위 결과(상대 위치·과두·복원 지표·엑셀 환산)를 먼저 다시 만든 뒤 피처
  python build_features.py --selftest   계산식 점검(HD·지렛대)
원본이 바뀌면: ALI·AMASSS·TotalSegmentator를 다시 돌린 뒤(batch_cbct_pipeline.py, run_totalseg.py) --upstream으로 실행한다.
  상위 결과가 원본(랜드마크·분할·영상)보다 오래되면 실행할 때 경고한다.
  주의: cbct_restore_extend.py는 동결 54명의 Roh 지표를 동결 CSV에서 읽는다. 동결 환자의 원본이 바뀌면 해당 검토 스크립트를 따로 돌려야 한다.

좌표: 정중 좌표계(cbct_eb_detect.midline_frame, Po 미사용, 원점 PNS, 축 = 좌·앞·위, mm 실측). Procrustes 크기 제거 없음.
  설골(Hy)·후두개곡(Eb)은 정중 탐색이라 좌우 = 0. 과두 최상점(Sup)은 머리 좌표계 결과를 정중 좌표계로 옮긴다.
피처
 1그룹 실측 거리: Pog_Ba_ant(턱끝–Ba 앞뒤), PNS_B_ant(PNS–B 앞뒤), Hy_Ba_dist(설골–Ba 거리), 부위별 위치·크기(local ROI)
 2그룹 연동 피처
  - EFA(VARIMAX): 골격+설골 좌표(Eb 포함·제외 두 가지). 구성상 상수·잡음인 열(PNS 전부, ANS 좌우·위아래, 정중점 좌우)은 뺀다.
    요인 수 = 상관행렬 고유값 ≥ 1. 열별 표준화. 점수 = 회전 요인 점수.
  - 2-Block PLS: X = 골격 좌표, Y = 설골·Eb·기도 단면적 프로파일 20구간. 열별 표준화 후 교차공분산 SVD, 골격 점수 t1·t2 = X·u.
    첫 특이값의 순열 검정 p(999회).
    ※ 여기 점수는 전체 표본 적합(기술용). 예측 성능 평가는 fit_efa/fit_pls를 교차검증 fold 안에서 다시 적합해 써야 한다.
  - MMV(상·하악 둘레 공간): AMASSS 하악·상악 마스크 중 구개 평면(정중 좌표계 위아래 = 0) 아래 부분의 볼록 껍질 부피에서 뼈를 뺀 값(1 mm 격자).
    PAV_MMV = PAV ÷ MMV, tongue_MMV = 혀 ÷ MMV(혀 분할이 있을 때).
  - HD(수력학적 직경): PMCA와 같은 단면(기도 중심선 수직, Eb+5 mm~PNS 중 최소) 4A/P. 둘레는 Crofton 둘레(복셀 계단 보정).
  - 지렛대: 좌우 Sup 축으로 하악을 θ만큼 벌렸을 때(턱끝이 내려가는 방향) Pog·Hy의 뒤로 밀림(mm). Ba는 고정점이라 Pog_Ba_ant 감소량과 같다.
    작은 θ에서 값 ≈ θ × (Sup과 그 점의 높이 차)라 θ·α는 크기만 바꾼다(lever_arm_*_sup로 대조). Hy_eff = α × 강체 회전 값.
 3그룹 교호작용(중심화 후 곱): ANB×BMI, Pog_Ba_ant×MPFH. PAL_sqrtPMCA = PAL ÷ √PMCA는 기도(매개) 변수라 기도 단계 결과 변수로만 쓴다.
정의서: docs/파생변수_정의서.md. 새 파생변수는 반드시 여기에 행을 추가한다(없으면 빌드 때 경고, summary.json의 undocumented).
출력: Result_py/features/v1/{features.csv, feature_catalog.csv(정의·출처·단위·환자 수·분석 역할), efa_loadings_*.csv, pls_weights.csv, summary.json, cache/P_n.json}
1그룹 추가(2026-11-15 검토): Sup_Go·S_Go(2-A 전용), InterGonial, Overjet_mm(절단 끝), overbite_mm, L1_MP_deg, Upper_Taper, MaxPost_Clivus_ant, 과두 각도 평균, SDp_mm
"""
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull
from skimage.measure import perimeter_crofton
from sklearn.decomposition import FactorAnalysis

import cbct_pmca as pm
from cbct_eb_detect import midline_frame
from cbct_fh_repair import repair
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd
from osa.ceph import head_frame
from run_ali import ROOT, ascii_path
from run_teeth_seg import free_gb

VERSION = "v1"; CACHE_VERSION = "v1.2"    # 캐시 계산식이 바뀌면 CACHE_VERSION만 올린다
R_ = ROOT/"Result_py"; EXT = OUT/"restore_ext/v1"; DEST = R_/"features"/VERSION; CACHE = DEST/"cache"
THETA_DEG, ALPHA_HY = 5., .6
MMV_TOP, MMV_GRID = 0., 1.    # 구개 평면 아래, 1 mm 격자
SKEL = ["Ba", "S", "N", "ANS", "PNS", "A", "RGo", "LGo", "Me", "Pog", "B"]
MIDLINE = {"N", "S", "Ba", "ANS", "PNS", "A", "B", "Pog", "Me", "Hy", "Eb"}
AX = ("L", "ant", "sup")
UPSTREAM = [    # (설명, 명령, 출력)
    ("기도·비강 프로파일, 상대 위치(설골·Eb)", ["shape_extra_vectors.py"], [R_/"shape/extra/relation.csv", R_/"shape/extra/airway_profile.csv"]),
    ("과두 최상점", ["cbct_extra_condyle_extend.py"], [EXT/"condyle_all.csv"]),
    ("Roh 11지표 복원", ["cbct_restore_extend.py"], [EXT/"compare_all.csv"]),
    ("정확도 실험 최종값", ["cbct_accuracy_experiments.py", "final"], [EXT/"experiments/final_comparison.csv"]),
    ("엑셀 방식 환산", ["cbct_excel_alignment.py"], [EXT/"alignment/aligned_metrics.csv"]),
]


# ---------- 입력 ----------
def raw_files(pid):
    """상위 결과의 원본(랜드마크·영상·뼈 분할)."""
    return sorted((CB/"landmarks").glob(f"P_{pid}_*.mrk.json")) + [CB/f"nrrd/P_{pid}.nrrd", CB/f"seg/P_{pid}_seg_MAND.nrrd", CB/f"seg/P_{pid}_seg_MAX.nrrd"]


def cache_files(pid):
    """환자별 영상 피처 캐시의 입력(원본 + 혀 분할 + 치아 분할)."""
    return raw_files(pid) + [CB/f"seg_totalseg/head_muscles/P_{pid}.nii.gz", CB/f"seg_teeth/UNIVERSALLAB_MODEL/P_{pid}.nii.gz"]


def signature(pid, extra):
    items = [(f.name, f.stat().st_size, f.stat().st_mtime_ns) if f.exists() else (f.name, None) for f in cache_files(pid)]
    return hashlib.sha1(json.dumps([CACHE_VERSION, MMV_TOP, MMV_GRID, items, extra], default=str).encode()).hexdigest()


def stale_upstream(pids):
    newest = max((f.stat().st_mtime for p in pids for f in raw_files(p) if f.exists()), default=0)
    return [(name, o.name) for name, _, outs in UPSTREAM for o in outs if not o.exists() or o.stat().st_mtime < newest]


def run_upstream():
    for name, cmd, _ in UPSTREAM:
        print(f"[상위] {name}: {' '.join(cmd)}", flush=True)
        subprocess.run([sys.executable, "-X", "utf8", *cmd], cwd=ROOT/"python", check=True)


def coords(pid, rel, cond):
    """정중 좌표계 3D 좌표 dict."""
    P = landmarks(pid)[0]; o, R, _ = midline_frame(P)
    q = {k: R@(P[k]-o) for k in SKEL}
    for k in ("Hy", "Eb"):
        a, s = rel.get(f"{k}_ant", np.nan), rel.get(f"{k}_sup", np.nan)
        q[k] = np.array([0., a, s]) if np.isfinite(a) and np.isfinite(s) else np.full(3, np.nan)
    if cond is not None:
        oh, Rh = head_frame(repair(P)[0])    # 과두 표는 FH 보정 머리 좌표계(cbct_condyle.py와 같음)
        for sd in "RL":
            h = np.array([cond[f"Sup_{sd}_L"], cond[f"Sup_{sd}_ant"], cond[f"Sup_{sd}_sup"]], float)
            q[f"{sd}Sup"] = R@(oh+Rh.T@h-o) if np.isfinite(h).all() else np.full(3, np.nan)
    return q, (P, o, R)


# ---------- 영상 피처 (환자별 캐시) ----------
def hd_at_pmca(pid, eb_sup, frame):
    P, o, R = frame; v, dirs, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd"); v = v.astype(np.float32); s = R@(P["S"]-o)
    Ls = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP); ant = np.arange(pm.ANT[0], pm.ANT[1]+1e-9, pm.STEP); sup = np.arange(eb_sup, pm.TOP+1e-9, pm.STEP)
    G = np.stack(np.meshgrid(Ls, ant, sup, indexing="ij"), -1).reshape(-1, 3)
    vol = map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=0).reshape(len(Ls), len(ant), len(sup)); del v
    m = pm.airway(vol, Ls, ant, sup, s[1], s[2])
    if m is None: return {}
    _, perp, cen = pm.sections(m, Ls, ant, sup)
    A, z = pm.pick(perp, sup, eb_sup+pm.BOTTOM_CENTER, 0.)
    if not np.isfinite(A): return {}
    k = int(np.argmin(np.abs(sup-z))); pts = np.c_[cen, sup]; tan = np.gradient(pts, axis=0); t = tan[k]/np.linalg.norm(tan[k])
    a = np.cross(t, [1., 0., 0.]); a /= np.linalg.norm(a); b = np.cross(t, a); g = np.arange(-pm.HALF_L, pm.HALF_L+1e-9, pm.STEP)
    U, V = np.meshgrid(g, g, indexing="ij"); qq = pts[k]+U[..., None]*a+V[..., None]*b
    idx = np.stack([(qq[..., 0]-Ls[0])/pm.STEP, (qq[..., 1]-ant[0])/pm.STEP, (qq[..., 2]-sup[0])/pm.STEP])
    sec = map_coordinates(m.astype(np.float32), idx, order=1, cval=0) > .5
    return hd_of(sec, pm.STEP) | {"PMCA_section_mm2": float(A), "PMCA_z": float(z)}


def hd_of(sec, step):
    area = float(sec.sum())*step**2; per = float(perimeter_crofton(sec, directions=4))*step
    return {"HD_mm": 4*area/per if per > 0 else np.nan, "HD_area_mm2": area, "HD_perimeter_mm": per}


def mmv(pid, frame, tongue_ml):
    P, o, R = frame; parts, bone = [], []
    for s in ("MAND", "MAX"):
        f = CB/f"seg/P_{pid}_seg_{s}.nrrd"
        if not f.exists(): return {}
        m, dirs, org = read_nrrd(f); ijk = np.argwhere(m > 0)[:, ::-1].astype(float)
        parts.append(((org+ijk@dirs)-o)@R.T); bone.append((m > 0, dirs, org))
    q = np.vstack(parts); q = q[q[:, 2] <= MMV_TOP]
    if len(q) < 100: return {}
    q = q[::max(1, len(q)//200000)]; hull = ConvexHull(q)
    lo, hi = q.min(0), q.max(0); axes = [np.arange(lo[i], hi[i]+MMV_GRID, MMV_GRID) for i in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3); inside = np.ones(len(G), bool)
    for c in range(0, len(G), 500000):
        inside[c:c+500000] = (G[c:c+500000]@hull.equations[:, :3].T+hull.equations[:, 3] <= 1e-9).all(1)
    Gi = G[inside]; w = o+Gi@R; is_bone = np.zeros(len(Gi), bool)
    for msk, dirs, org in bone:
        is_bone |= map_coordinates(msk.astype(np.uint8), np.linalg.solve(dirs.T, (w-org).T)[::-1], order=0, cval=0) > 0
    cell = MMV_GRID**3/1000; out = {"MMV_hull_mL": float(inside.sum())*cell, "MMV_mL": float((~is_bone).sum())*cell}
    if np.isfinite(tongue_ml): out["tongue_MMV"] = tongue_ml/out["MMV_mL"]
    return out


def tongue_ml(pid):
    f = CB/f"seg_totalseg/head_muscles/P_{pid}.nii.gz"
    if not f.exists(): return np.nan
    import SimpleITK as sitk
    from shape_tongue_sinus import clean
    im = sitk.ReadImage(str(ascii_path(f))); v = float(np.prod(im.GetSpacing()))/1000
    return float(clean(sitk.GetArrayFromImage(im) == 9, v).sum())*v


def overjet(pid, frame):
    """위 중절치(8·9) 절단 끝과 아래 중절치(24·25) 절단 끝의 앞뒤 거리(mm, 위가 앞이면 +). 절단 끝 정의는 cbct_extra.py overbite와 같다."""
    from cbct_extra import INC, MODEL as TM, Labels
    P, o, R = frame; tf = CB/f"seg_teeth/{TM}/P_{pid}.nii.gz"
    if not tf.exists(): return {}
    t = Labels(tf); up, lo = t.points(list(INC["U"])), t.points(list(INC["L"]))
    if len(up) <= 50 or len(lo) <= 50: return {}
    hu, hl = (up-o)@R.T, (lo-o)@R.T; lu, ll = t.at(up), t.at(lo)
    eu = [hu[lu == k][np.argmin(hu[lu == k][:, 2])] for k in INC["U"] if (lu == k).any()]
    el = [hl[ll == k][np.argmax(hl[ll == k][:, 2])] for k in INC["L"] if (ll == k).any()]
    if not eu or not el: return {}
    return {"Overjet_mm": float(np.mean(eu, 0)[1]-np.mean(el, 0)[1])}


def voxel_features(pid, eb_sup, frame):
    f = CACHE/f"P_{pid}.json"; sig = signature(pid, eb_sup)
    if f.exists():
        c = json.loads(f.read_text(encoding="utf-8"))
        if c.get("sig") == sig: return c["values"]
    while free_gb() < 6: print(f"  P_{pid} 메모리 대기 ({free_gb():.1f} GB)", flush=True); time.sleep(20)
    t = tongue_ml(pid); vals = {"tongue_mL": t}
    vals |= hd_at_pmca(pid, eb_sup, frame) if np.isfinite(eb_sup) else {}
    vals |= mmv(pid, frame, t)
    vals |= overjet(pid, frame)
    CACHE.mkdir(parents=True, exist_ok=True); tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sig": sig, "values": vals}, default=float), encoding="utf-8"); tmp.replace(f)
    return vals


# ---------- 2그룹: 연동 피처 ----------
def flat(q, keys, drop_construct=True):
    out = {}
    for k in keys:
        for i, ax in enumerate(AX):
            if drop_construct and (k == "PNS" or (k == "ANS" and ax != "ant") or (k in MIDLINE and ax == "L")): continue
            out[f"{k}_{ax}"] = q[k][i]
    return out


def zscore(X):
    mu, sd = X.mean(0), X.std(0, ddof=1); sd[sd == 0] = 1.; return (X-mu)/sd, (mu, sd)


def fit_efa(X):
    """X: DataFrame(완전 사례). 반환 (모델 dict, 점수 DataFrame). 교차검증에서는 fold 학습 자료로 호출하고 transform_efa로 투영."""
    Z, norm = zscore(X.to_numpy(float)); ev = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
    m = int(max(1, min((ev >= 1).sum(), X.shape[1]//2)))
    fa = FactorAnalysis(n_components=m, rotation="varimax", random_state=0).fit(Z)
    model = {"fa": fa, "norm": norm, "cols": list(X.columns), "eig": ev, "m": m}
    return model, transform_efa(model, X)


def transform_efa(model, X):
    Z = (X[model["cols"]].to_numpy(float)-model["norm"][0])/model["norm"][1]
    return pd.DataFrame(model["fa"].transform(Z), index=X.index, columns=[f"F{i+1}" for i in range(model["m"])])


def fit_pls(X, Y, k=2, perms=999, seed=0):
    Zx, nx = zscore(X.to_numpy(float)); Zy, _ = zscore(Y.to_numpy(float)); n = len(Zx)
    U, S, Vt = np.linalg.svd(Zx.T@Zy/(n-1), full_matrices=False)
    rng = np.random.default_rng(seed); null = [np.linalg.svd(Zx.T@Zy[rng.permutation(n)]/(n-1), compute_uv=False)[0] for _ in range(perms)]
    model = {"u": U[:, :k], "v": Vt[:k].T, "s": S, "norm": nx, "xcols": list(X.columns), "ycols": list(Y.columns),
             "p_sv1": float((1+sum(x >= S[0] for x in null))/(perms+1)), "cov_frac": (S**2/np.sum(S**2))[:k]}
    return model, transform_pls(model, X)


def transform_pls(model, X):
    Z = (X[model["xcols"]].to_numpy(float)-model["norm"][0])/model["norm"][1]
    return pd.DataFrame(Z@model["u"], index=X.index, columns=[f"PLS_t{i+1}" for i in range(model["u"].shape[1])])


def l1_mp(t, q, go):
    """아래 중절치(24·25) 장축(치관 쪽이 위)과 Me→Go 방향의 정중면 각도. 라벨이 정상인 치아만."""
    if t.empty: return {}
    ax = t[["axis_ant", "axis_sup"]].to_numpy(float); ax *= np.sign(ax[:, 1:2]+1e-12); a = ax.mean(0)
    mp = (go-q["Me"])[1:]; c = a@mp/np.linalg.norm(a)/np.linalg.norm(mp)
    return {"L1_MP_deg": float(np.degrees(np.arccos(np.clip(c, -1, 1))))}


def rodrigues(p, o, k, th):
    r = p-o; return o+r*np.cos(th)+np.cross(k, r)*np.sin(th)+k*np.dot(k, r)*(1-np.cos(th))


def lever(q, theta=np.radians(THETA_DEG)):
    """좌우 Sup 축 강체 회전에서 Pog·Hy가 뒤로 밀린 거리(mm, +면 뒤로). 회전 방향은 Pog가 내려가는 쪽."""
    if not (np.isfinite(q["RSup"]).all() and np.isfinite(q["LSup"]).all()): return {}
    o = (q["RSup"]+q["LSup"])/2; k = (q["LSup"]-q["RSup"]); k /= np.linalg.norm(k)
    th = theta if rodrigues(q["Pog"], o, k, theta)[2] < q["Pog"][2] else -theta
    out = {}
    for name in ("Pog", "Hy"):
        if np.isfinite(q[name]).all():
            out[f"lever_{name}_mm"] = float(q[name][1]-rodrigues(q[name], o, k, th)[1]); out[f"lever_arm_{name}_sup"] = float(o[2]-q[name][2])
    if "lever_Hy_mm" in out: out["lever_Hy_eff_mm"] = ALPHA_HY*out["lever_Hy_mm"]
    return out


# 피처 목록: (열 이름 정규식, 그룹, 정의, 출처, 단위, 분석 역할). 위에서부터 처음 맞는 것을 쓴다.
ROLE_MAIN, ROLE_2A, ROLE_MED, ROLE_BASE, ROLE_EXP, ROLE_IN = "2-B 사전 가설 후보", "2-A 전용(골격→기도)", "기도·매개 변수(2-B 제외)", "기존 골격 지표(기준선 비교용)", "탐색(FDR 보고)", "입력·보조값(직접 회귀 투입 안 함)"
CATALOG = [
    (r"^c_", "입력", "정중 좌표계 3D 좌표(PNS 원점, mm). 구성상 상수·잡음 열 제외", "ALI 랜드마크, 설골·Eb 규칙", "mm", ROLE_IN),
    (r"^Pog_Ba_ant$", "1그룹 실측", "턱끝 앞점(Pog)과 대후두공 앞점(Ba)의 앞뒤 거리", "ALI", "mm", ROLE_MAIN),
    (r"^PNS_B_ant$", "1그룹 실측", "PNS 기준 하악 B점의 앞뒤 위치", "ALI", "mm", ROLE_EXP),
    (r"^Hy_Ba_dist$", "1그룹 실측", "설골–Ba 정중면 거리", "설골 규칙, ALI", "mm", ROLE_EXP),
    (r"^Sup_Go$", "1그룹 실측", "좌우 과두 최상점 중점–좌우 하악각점 중점 3D 거리(뒤얼굴 수직 높이)", "AMASSS 하악, ALI", "mm", ROLE_2A),
    (r"^S_Go$", "1그룹 실측", "터키안장(S)–하악각점 중점 정중면 거리", "ALI", "mm", ROLE_2A),
    (r"^InterGonial$", "1그룹 실측", "좌우 하악각점 사이 3D 거리(하악 폭)", "ALI", "mm", ROLE_EXP),
    (r"^Overjet_mm$", "1그룹 실측", "위 중절치 절단 끝과 아래 중절치 절단 끝의 앞뒤 거리(위가 앞이면 +)", "치아 분할", "mm", ROLE_MAIN),
    (r"^overbite_mm$", "1그룹 실측", "위·아래 중절치 절단 끝 높이 차(겹침이 깊으면 +)", "치아 분할(cbct_extra)", "mm", ROLE_EXP),
    (r"^L1_MP_deg$", "1그룹 실측", "아래 중절치 장축과 하악평면(Me→Go) 정중면 각도. 클수록 앞으로 기욺(IMPA 방식)", "치아 분할 장축, ALI", "°", ROLE_EXP),
    (r"^Upper_Taper$", "1그룹 실측", "위 송곳니 폭 ÷ 위 제1대구치 폭", "치아 분할(치열궁)", "비", ROLE_EXP),
    (r"^MaxPost_Clivus_ant$", "1그룹 실측", "상악 후방 부위 중심 − 두개저 사대 부위 중심(앞뒤)", "부위별 형태(SSM)", "mm", ROLE_EXP),
    (r"^Condyle_(Axial|Lat)_mean$", "1그룹 실측", "과두 축상각·측면각 좌우 평균", "AMASSS 하악", "°", ROLE_EXP),
    (r"^SDp_mm$", "1그룹 실측", "비중격 최대 편위", "TotalSegmentator 비강", "mm", ROLE_EXP),
    (r"^roi_", "1그룹 실측", "부위 12곳의 크기(_size)·중심 위치(_cleft/_cant/_csup)", "부위별 형태(SSM)", "mm", ROLE_EXP),
    (r"^EFA_", "2그룹 연동", "골격+설골(±Eb) 좌표 VARIMAX 요인 점수(전체 표본 적합, 기술용)", "좌표", "점수", ROLE_EXP),
    (r"^PLS_t", "2그룹 연동", "골격 좌표 ↔ 설골·Eb·기도 프로파일 2-Block PLS 골격 점수(전체 표본 적합, 기술용)", "좌표, 기도 프로파일", "점수", ROLE_MAIN),
    (r"^MMV", "2그룹 연동", "구개 평면 아래 하악·상악 볼록 껍질 부피(MMV_mL는 뼈 제외)", "AMASSS 하악·상악", "mL", ROLE_EXP),
    (r"^tongue_MMV$", "2그룹 연동", "혀 부피 ÷ MMV", "TotalSegmentator 혀, AMASSS", "비", ROLE_MAIN),
    (r"^tongue_mL$", "2그룹 연동", "혀 부피(0.5 mL 미만 조각 제거)", "TotalSegmentator head_muscles", "mL", ROLE_EXP),
    (r"^(tongue_back_ant|tongue_mandible_ratio)$", "4그룹 연조직", "혀뿌리(혀 뒤쪽 경계) 정중 좌표계 앞뒤 위치 / 혀 부피 ÷ AMASSS 하악 부피", "TotalSegmentator 혀, AMASSS", "mm, 비", ROLE_MAIN),
    (r"^tongue_(c_ant|c_sup|low_sup|len|height)$", "4그룹 연조직", "혀 무게중심 앞뒤·위아래, 최저점 높이, 앞뒤 길이, 높이(정중 좌표계)", "TotalSegmentator 혀", "mm", ROLE_EXP),
    (r"^(masseter|lat_pterygoid|med_pterygoid|chew)_mL$", "4그룹 연조직", "저작근 좌우 합 부피(교근·외측익돌근·내측익돌근, chew = 셋의 합). 측두근 제외(잘림)", "TotalSegmentator head_muscles", "mL", ROLE_EXP),
    (r"^sinus_(max_mL|max_air_frac|frontal_mL)$", "4그룹 연조직", "상악동 좌우 합 부피, 상악동 공기 비율, 전두동 부피(잘린 3명 결측)", "TotalSegmentator craniofacial_structures", "mL, 비", ROLE_EXP),
    (r"^PAV_MMV$", "2그룹 연동", "인두 기도 부피 ÷ MMV", "밝기 규칙, AMASSS", "비", ROLE_MED),
    (r"^HD_mm$", "2그룹 연동", "PMCA 단면의 수력학적 직경 4A/P(Crofton 둘레)", "밝기 규칙", "mm", ROLE_MED),
    (r"^(HD_area_mm2|HD_perimeter_mm|PMCA_section_mm2|PMCA_z)$", "2그룹 연동", "HD 계산 보조값(단면 넓이·둘레·높이)", "밝기 규칙", "mm/mm²", ROLE_IN),
    (r"^lever_arm_", "2그룹 연동", "과두 축과 그 점의 높이 차(지렛대 값의 대조용)", "AMASSS, ALI", "mm", ROLE_EXP),
    (r"^lever_", "2그룹 연동", "좌우 과두 축으로 하악을 THETA° 벌렸을 때 뒤로 밀린 거리(Hy_eff = ALPHA×강체 값)", "AMASSS, ALI", "mm", ROLE_EXP),
    (r"^(ANBxBMI|PogBaxMPFH)$", "3그룹 교호작용", "두 변수를 각각 중심화한 뒤 곱", "복원 지표, 엑셀 BMI", "곱", ROLE_EXP),
    (r"^PAL_sqrtPMCA$", "3그룹 교호작용", "PAL ÷ √PMCA(기도가 가늘고 긴 정도)", "복원 지표", "비", ROLE_MED),
    (r"^m_(PAL|PAV|PMCA|NAV)$", "Roh 지표", "자동 측정값(엑셀 방식 환산 적용분 포함)", "복원 파이프라인", "mm·mm²·mm³", ROLE_MED),
    (r"^m_", "Roh 지표", "자동 측정값(엑셀 방식 환산 적용분 포함)", "복원 파이프라인", "°·mm·%", ROLE_BASE),
]


DOC = ROOT/"docs/파생변수_정의서.md"


def undocumented(cols):
    """정의서 표 첫 칸(변수명)의 백틱 이름·패턴(*)에 맞지 않는 열 목록. 정의서가 없으면 전부."""
    import re
    if not DOC.exists(): return list(cols)
    pats = []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            for name in re.findall(r"`([^`]+)`", line.split("|")[1]):
                pats.append(re.compile("^"+re.escape(name).replace(r"\*", ".*")+"$"))
    return [c for c in cols if not any(x.match(c) for x in pats)]


def catalog(d):
    import re
    rows = []
    for c in d.columns:
        hit = next((x for x in CATALOG if re.search(x[0], c)), None)
        g, de, so, un, ro = hit[1:] if hit else ("미분류", "", "", "", "")
        de = de.replace("THETA", f"{THETA_DEG:g}").replace("ALPHA", f"{ALPHA_HY}")
        rows.append({"feature": c, "group": g, "definition": de, "source": so, "unit": un, "role": ro, "n": int(d[c].notna().sum())})
    return pd.DataFrame(rows)


# ---------- 조립 ----------
def build():
    rel = pd.read_csv(R_/"shape/extra/relation.csv", encoding="utf-8-sig").set_index("Patient")
    prof = pd.read_csv(R_/"shape/extra/airway_profile.csv", encoding="utf-8-sig").set_index("Patient")
    cond = pd.read_csv(EXT/"condyle_all.csv", encoding="utf-8-sig").set_index("Patient")
    al = pd.read_csv(EXT/"alignment/aligned_metrics.csv", encoding="utf-8-sig").pivot(index="Patient", columns="metric", values="aligned")
    roi = pd.read_csv(R_/"shape/local/local_roi_features.csv", encoding="utf-8-sig").set_index("Patient")
    master = pd.read_csv(R_/"phase1/patient_master.csv", encoding="utf-8-sig").set_index("Patient")
    ex = pd.read_csv(EXT/"extra_all.csv", encoding="utf-8-sig").set_index("Patient")
    arch = pd.read_csv(R_/"shape/fine/dental_arch.csv", encoding="utf-8-sig").set_index("Patient")
    tl = pd.read_csv(R_/"shape/fine/teeth_long.csv", encoding="utf-8-sig"); tl = tl[tl.label_ok.astype(str).str.lower().isin(["true", "1"]) & tl.tooth.isin([24, 25])]
    pids = sorted((int(p[2:]) for p in rel.index), key=int)
    warn = stale_upstream(pids)
    for w in warn: print(f"[경고] 상위 결과가 원본보다 오래됨: {w[0]} ({w[1]}) → --upstream으로 다시 만드세요", flush=True)
    rows, C = [], {}
    for pid in pids:
        p = f"P_{pid}"; r0 = rel.loc[p].to_dict()
        q, frame = coords(pid, r0, cond.loc[p] if p in cond.index else None); C[p] = q
        r = {"Patient": p, **{f"c_{k}": v for k, v in flat(q, SKEL+["Hy", "Eb"]).items()}}
        r |= {"Pog_Ba_ant": q["Pog"][1]-q["Ba"][1], "PNS_B_ant": q["B"][1], "Hy_Ba_dist": float(np.linalg.norm((q["Hy"]-q["Ba"])[1:]))}
        go = (q["RGo"]+q["LGo"])/2; r |= {"S_Go": float(np.linalg.norm((q["S"]-go)[1:])), "InterGonial": float(np.linalg.norm(q["RGo"]-q["LGo"]))}
        if "RSup" in q and np.isfinite(q["RSup"]).all() and np.isfinite(q["LSup"]).all(): r["Sup_Go"] = float(np.linalg.norm((q["RSup"]+q["LSup"])/2-go))
        r |= l1_mp(tl[tl.Patient == p], q, go)
        r |= lever(q)
        r |= voxel_features(pid, float(r0.get("Eb_sup", np.nan)), frame)
        rows.append(r); print(pid, end=" ", flush=True)
    print()
    d = pd.DataFrame(rows).set_index("Patient")
    d = d.join(roi[[c for c in roi if c.endswith(("_size", "_cleft", "_cant", "_csup"))]].add_prefix("roi_"))
    d["MaxPost_Clivus_ant"] = roi.MAX_posterior_cant-roi.CB_clivus_cant
    d["Condyle_Axial_mean"] = cond[["Axial_R", "Axial_L"]].mean(axis=1); d["Condyle_Lat_mean"] = cond[["Lat_R", "Lat_L"]].mean(axis=1)
    d["overbite_mm"] = ex.overbite_mm; d["SDp_mm"] = ex.SDp_mm; d["Upper_Taper"] = arch.upper_width_C/arch.upper_width_M1
    d = d.join(al.rename(columns=lambda c: c.replace("-", "")).add_prefix("m_"))
    d["PAV_MMV"] = d.get("m_PAV")/d.get("MMV_mL")/1000 if "MMV_mL" in d else np.nan    # PAV mm³ → mL
    d["PAL_sqrtPMCA"] = d["m_PAL"]/np.sqrt(d["m_PMCA"])
    bmi = master.BMI.reindex(d.index); cz = lambda s: s-s.mean()
    d["ANBxBMI"] = cz(d["m_ANB"])*cz(bmi); d["PogBaxMPFH"] = cz(d["Pog_Ba_ant"])*cz(d["m_MPFH"])
    # 연조직 편입(2026-11-16): 혀 형태·저작근(측두근 제외, 75% 잘림)·부비동. 원천 shape_tongue_sinus.py
    tsm = pd.read_csv(R_/"shape/fine/tongue_sinus_muscles.csv", encoding="utf-8-sig").set_index("Patient").reindex(d.index)
    for c in ("tongue_c_ant", "tongue_c_sup", "tongue_back_ant", "tongue_low_sup", "tongue_len", "tongue_height", "tongue_mandible_ratio"): d[c] = tsm[c]
    for m in ("masseter", "lat_pterygoid", "med_pterygoid"): d[f"{m}_mL"] = tsm[f"{m}_R_mL"]+tsm[f"{m}_L_mL"]
    d["chew_mL"] = d[["masseter_mL", "lat_pterygoid_mL", "med_pterygoid_mL"]].sum(axis=1, min_count=3)
    d["sinus_max_mL"] = tsm.sinus_max_R_mL+tsm.sinus_max_L_mL
    d["sinus_max_air_frac"] = (tsm.sinus_max_R_air_mL+tsm.sinus_max_L_air_mL)/d.sinus_max_mL
    d["sinus_frontal_mL"] = tsm.sinus_frontal_mL.where(tsm.sinus_frontal_cut != 1)

    summ = {"n_patients": len(d), "stale_upstream": warn, "theta_deg": THETA_DEG, "alpha_hy": ALPHA_HY}
    keys_all = SKEL+["Hy"]
    for name, keys in (("without_Eb", keys_all), ("with_Eb", keys_all+["Eb"])):
        X = pd.DataFrame({p: flat(C[p], keys) for p in d.index}).T.astype(float).dropna()
        model, sc = fit_efa(X); d = d.join(sc.add_prefix(f"EFA_{name}_"))
        ld = pd.DataFrame(model["fa"].components_.T, index=model["cols"], columns=sc.columns).round(3)
        ld.to_csv(DEST/f"efa_loadings_{name}.csv", encoding="utf-8-sig")
        summ[f"efa_{name}"] = {"n": len(X), "n_vars": X.shape[1], "n_factors": model["m"], "eigen_top8": model["eig"][:8].round(2).tolist()}
    X = pd.DataFrame({p: flat(C[p], SKEL) for p in d.index}).T.astype(float)
    Y = pd.DataFrame({p: {**flat(C[p], ["Hy", "Eb"]), **prof.loc[p, [f"AW_p{i:02d}" for i in range(20)]].to_dict()} for p in d.index}).T.astype(float)
    ok = X.notna().all(1) & Y.notna().all(1); model, sc = fit_pls(X[ok], Y[ok]); d = d.join(sc)
    pd.concat([pd.DataFrame(model["u"], index=model["xcols"], columns=["u1", "u2"]), pd.DataFrame(model["v"], index=model["ycols"], columns=["v1", "v2"])]).round(4).to_csv(DEST/"pls_weights.csv", encoding="utf-8-sig")
    summ["pls"] = {"n": int(ok.sum()), "p_sv1": model["p_sv1"], "cov_frac_t1_t2": model["cov_frac"].round(3).tolist()}
    summ["missing"] = {c: int(d[c].isna().sum()) for c in d if d[c].isna().any()}
    tmp = DEST/"features.tmp.csv"; d.round(4).to_csv(tmp, encoding="utf-8-sig"); tmp.replace(DEST/"features.csv")
    cat = catalog(d); cat.to_csv(DEST/"feature_catalog.csv", index=False, encoding="utf-8-sig")
    summ["unclassified"] = cat.feature[cat.group == "미분류"].tolist(); summ["role_counts"] = cat.role.value_counts().to_dict()
    summ["undocumented"] = undocumented(d.columns)
    for c in summ["undocumented"]: print(f"[경고] 정의서에 없는 파생변수: {c} → docs/파생변수_정의서.md에 기록하세요", flush=True)
    (DEST/"summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float)); print("saved", DEST/"features.csv", d.shape)


def selftest():
    g = np.arange(-20, 20, .4); X, Y = np.meshgrid(g, g); h = hd_of(X**2+Y**2 <= 100, .4)["HD_mm"]
    assert abs(h-20) < 1.0, h    # 반지름 10 mm 원의 HD = 지름 20 mm
    h2 = hd_of((np.abs(X) <= 15) & (np.abs(Y) <= 2.5), .4)["HD_mm"]; assert h2 < 10, h2    # 같은 넓이 근처라도 납작하면 HD가 작다
    q = {"RSup": np.array([-50., 0, 0]), "LSup": np.array([50., 0, 0]), "Pog": np.array([0., 80, -60]), "Hy": np.full(3, np.nan)}
    lv = lever(q); exp = 60*np.sin(np.radians(THETA_DEG))+80*(1-np.cos(np.radians(THETA_DEG)))
    assert abs(lv["lever_Pog_mm"]-exp) < 1e-6 and lv["lever_arm_Pog_sup"] == 60, lv
    print("selftest ok: HD 원", round(h, 2), "납작", round(h2, 2), "지렛대", round(lv["lever_Pog_mm"], 3))


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest(); sys.exit()
    DEST.mkdir(parents=True, exist_ok=True)
    if "--upstream" in sys.argv: run_upstream()
    build()
