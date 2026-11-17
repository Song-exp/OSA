"""Step 2~4 주성분·요인 축 적재량 표 그림: 행 = 구성 변수, 열 = 축, 칸 = 적재량(숫자 표시).
기도 프로파일 PC(AWshape), 구조 크기 축, PLS 가중치, 요인분석 적재량은 구성 변수 그대로.
공동 축(F1·F2·F2MM)·잔여 형상(rPC)은 구성 변수가 표면 대응점 좌표 수천 개라, 부위 × 방향으로 묶어
'축 +1 SD당 그 부위 점들의 평균 이동량(mm, GPA 축은 크기 제거 단위)'을 칸에 넣는다. 좌우는 양쪽이 상쇄되지 않게 절댓값 평균.
출력: Result_py/analysis/all83/figs/pc/*.png, 반환 {단계: [(제목, 그림 경로, 설명)]}
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import osa_experiments as X, osa_validate_holdout as VH, osa_joint_axes as JA, osa_residual_shape as RS
from shape_local_roi import roi_labels
from shape_ssm import gpa
from osa_var_detail import STRUCT, REGION

R = X.R_; PC = R/"analysis/all83/figs/pc"
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False


def heat(M, title, name, vmax=None, fmt="{:.2f}"):
    """M: DataFrame(행 = 구성 변수, 열 = 축) → 숫자가 들어간 히트맵."""
    v = np.nanmax(np.abs(M.to_numpy())) if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(max(4, .75*M.shape[1]+3.5), max(2.5, .3*M.shape[0]+1.2)))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, fontsize=8); ax.set_yticks(range(M.shape[0]), M.index, fontsize=7.5); ax.xaxis.tick_top()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            x = M.iat[i, j]
            if np.isfinite(x): ax.text(j, i, fmt.format(x), ha="center", va="center", fontsize=6.5, color="white" if abs(x) > .6*v else "black")
    fig.colorbar(im, ax=ax, shrink=.6); ax.set_title(title, fontsize=9, pad=22)
    f = PC/f"{name}.png"; fig.savefig(f, dpi=100, bbox_inches="tight"); plt.close(fig); return f


def awshape(full):
    aw = [f"AW_p{i:02d}" for i in range(20)]; P = full[aw].dropna(); Z = (P-P.mean())/P.std()
    _, s, vt = np.linalg.svd(Z.to_numpy(), full_matrices=False); var = s**2/(s**2).sum()
    for i in range(3):
        if vt[i, np.argmax(np.abs(vt[i]))] < 0: vt[i] *= -1                       # osa_all83.data()와 같은 부호 규칙
    M = pd.DataFrame(vt[:3].T, index=[f"AW_p{i:02d} (위치 {i/19:.2f})" for i in range(20)], columns=[f"PC{k+1} ({var[k]:.0%})" for k in range(3)])
    return [("기도 단면 프로파일 PC 적재량", heat(M, "AWshape 적재량: 행 = Eb(0)→PNS(1) 20구간 표준화 단면적, 열 = PC (분산 비율)", "AWshape", 1),
             "같은 부호·비슷한 크기면 그 구간들이 함께 커지고 작아진다. PC1은 전 구간이 양수라 '전체 굵기', PC2는 아래(−)·위(+) 부호가 갈려 '아래 대 위 굵기 대비'다.")]


def size_axis():
    t = pd.read_csv(R/"analysis/validation/size_loadings.csv", encoding="utf-8-sig")
    M = pd.DataFrame({"PC1 (69%)": t.loading.to_numpy(), "SIZE_PC1과 상관": t.rho_with_SIZE.to_numpy()}, index=[f"{f} ({b})" for f, b in zip(t.feature, t["부위"])])
    return [("구조 크기 축 적재량", heat(M, "구조 크기 축(SIZE_PC1): 행 = 12부위 크기, 열 = 적재량·축 점수와의 상관", "SIZE_PC1", 1),
             "12부위 적재량이 모두 양수(0.16~0.32)라 모든 부위가 함께 커지는 '전체 크기' 축이다.")]


def pls():
    f = R/"features/v1/pls_weights.csv"
    if not f.exists(): return []
    t = pd.read_csv(f, encoding="utf-8-sig", index_col=0)
    return [("골격–기도 PLS 가중치", heat(t, "PLS 골격 가중치: 행 = 골격 기준점 좌표, 열 = PLS 성분", "PLS_t1", 1), "절댓값이 큰 좌표가 그 PLS 점수를 주로 정한다(PLS_t1 = 첫째 열).")]


def efa():
    out = []
    for nm, lab in (("without_Eb", "Eb 제외"), ("with_Eb", "Eb 포함")):
        f = R/f"features/v1/efa_loadings_{nm}.csv"
        if f.exists():
            t = pd.read_csv(f, encoding="utf-8-sig", index_col=0); t.columns = [c.split("_")[-1] for c in t.columns]
            out.append((f"요인분석 적재량 ({lab})", heat(t, f"골격 연동 요인분석 적재량({lab}): 행 = 기준점 좌표(좌·앞·위), 열 = 요인", f"EFA_{nm}", 1),
                        "칸 = 요인 적재량. 절댓값이 큰 좌표들이 그 요인 점수와 함께 움직인다."))
    return out


def region_table(vt, sd, lab, st_of, k):
    rows = {}
    for st in dict.fromkeys(st_of):
        for g in dict.fromkeys(lab[st_of == st]):
            m = (st_of == st) & (lab == g)
            for j, dn in ((1, "앞뒤"), (2, "위아래"), (0, "좌우|")):
                rg = REGION.get(g, g); nm = rg if rg.startswith(STRUCT.get(st, st)) else f"{STRUCT.get(st, st)} {rg}"
                rows[f"{nm} · {dn.rstrip('|')}"] = [((vt[i].reshape(-1, 3)[m, j]*sd[i]) if j else np.abs(vt[i].reshape(-1, 3)[m, j]*sd[i])).mean() for i in range(k)]
    return pd.DataFrame(rows, index=[f"PC{i+1}" for i in range(k)]).T


def joint(names):
    out = []
    for name, structs, mode in (("F1", ("MAND", "MAX", "CB"), "form"), ("F2", ("MAND", "MAX", "CB"), "gpa"), ("F2MM", ("MAND", "MAX"), "gpa")):
        ks = [int(n.split("_PC")[1]) for n in names if n.startswith(name+"_PC")]
        if not ks: continue
        ids, C, lab, st_of = JA.combined(structs)
        if mode == "gpa": G, _ = gpa(C); Y = G.reshape(len(G), -1)
        else: Y = C.reshape(len(C), -1)
        K = max(ks); _, vt, var, sd = JA.pca(Y, K); M = region_table(vt, sd, lab, st_of, K); M.columns = [f"{c} ({v:.0%})" for c, v in zip(M.columns, var)]
        unit = "mm" if mode == "form" else "크기 제거 단위"
        out.append((f"공동 축 {name} 적재량 (부위 × 방향)", heat(M, f"{name} 공동 축: 행 = 부위 × 방향, 열 = PC (분산 비율), 칸 = 축 +1 SD당 평균 이동({unit})", f"{name}_load"),
                    "앞뒤·위아래는 +면 앞·위로 이동, 좌우는 이동 크기(절댓값). 큰 칸이 그 축이 나타내는 배치 변화다."))
    return out


def residual(full, names):
    out = []
    for st in ("MAND", "MAX", "CB"):
        ks = [int(n.split("_rPC")[1]) for n in names if n.startswith(st+"_rPC")]
        if not ks: continue
        sc, var, vt, sd, mean = RS.residual_pcs(full, st); lab = roi_labels(st, np.load(R/f"shape/{st}/points.npz"))[0]
        K = max(ks); M = region_table(vt, sd, lab, np.array([st]*len(lab)), K); M.columns = [f"{c} ({v:.0%})" for c, v in zip(M.columns, var[:K])]
        out.append((f"{STRUCT[st]} 잔여 형상 적재량 (부위 × 방향)", heat(M, f"{st} 잔여 형상 PC: 행 = 부위 × 방향, 열 = PC (잔여 분산 비율), 칸 = +1 SD당 평균 이동(mm)", f"{st}_rPC_load"),
                    "ANB·턱끝–대후두공 거리·크기·성별을 뺀 나머지 모양. 큰 칸이 그 성분이 나타내는 부위 변화다."))
    return out


def build(step_vars):
    """step_vars: {단계: [변수...]} → {단계: [(제목, 경로, 설명)]}"""
    PC.mkdir(parents=True, exist_ok=True)
    for f in PC.glob("*.png"): f.unlink()                                          # 이전 판(±2 SD 모양 그림) 정리
    full, _ = VH.load_frozen(); out = {}
    for s, vs in step_vars.items():
        items = []
        if any(v.startswith("AWshape") for v in vs): items += awshape(full)
        if "SIZE_PC1" in vs: items += size_axis()
        if "PLS_t1" in vs: items += pls()
        items += joint([v for v in vs if v.startswith(("F1_PC", "F2_PC", "F2MM_PC"))])
        items += residual(full, [v for v in vs if "_rPC" in v])
        if any(v.startswith("EFA_") for v in vs): items += efa()
        out[s] = items
    return out
