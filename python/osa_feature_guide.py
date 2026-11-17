"""피처 가이드(docs/피처_가이드.md) 생성: 원천별로 도구 원리 → 피처 표(뜻 · 만든 과정 · 코드 위치 · 쓰인 단계).
변수 설명은 osa_var_detail.detail → osa_all83.catalog 순으로 가져와 보고서와 어긋나지 않게 한다. 환자 값은 넣지 않는다.
실행: python osa_feature_guide.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import osa_newfactor as N, osa_all83 as A
from osa_var_detail import detail

DOC = Path(__file__).resolve().parents[1]/"docs/피처_가이드.md"

# 원천 → (제목, 원리 설명)
SRC = {
 "A": ("A. 엑셀 (연구실 수작업 계측)", "연구실이 Invivo·Mimics로 손으로 잰 Roh 2025의 CBCT 지표 11개. 변수 이름 앞에 `x_`. Roh 논문 정의를 따르며, 0으로 기록된 의심값은 결측 처리했다(`osa_experiments.data`)."),
 "B": ("B. CBCT 기준점 (ALI-CBCT)", "공개 모델 ALI-CBCT가 CBCT에서 해부학 기준점(S, N, A, B, Pog, Me, ANS, PNS, Ba, 좌우 Go·Po·Or 등)을 자동으로 찍는다(`run_ali.py`). "
       "점 좌표를 정중 좌표계로 옮긴 뒤 거리·각도·앞뒤 위치를 계산한다. Roh 지표의 자동 계측값(`m_`)도 이 점들로 잰다(`cbct_restore_extend.py`, 최종 등급 `cbct_final_criteria85.py`)."),
 "C": ("C. 밝기 규칙 (기도·설골·Eb)", "CT 밝기(HU)로 직접 찾는다. 공기(−1200~−400 HU)이면서 정중선과 이어진 덩어리를 인두 기도로 보고(`cbct_pmca.airway`), 기도 중심선을 따라 수직 단면 넓이를 잰다(`cbct_pmca.sections`). "
       "설골은 하악 아래 뼈 밝기 덩어리, Eb(후두개곡 최하점)는 설골 위 공기 끝으로 검출한다(`cbct_eb_detect.hyoid`, `vallecula`)."),
 "D": ("D. AMASSS 분할 (하악·상악·두개저)", "공개 모델 AMASSS가 하악(MAND)·상악(MAX)·두개저(CB)를 분할한다(`run_amasss.py`). 분할 표면에서 환자 간 대응점 2048개를 만들고(통계 형상 모델, `shape_ssm.py`), "
       "대응점을 가장 가까운 기준점 앵커로 부위 12곳에 나눠 부위 크기·중심을 잰다(`shape_local_roi.py`). 과두 최상점·과두 각도(`cbct_condyle.py`), 구강 공간 부피(`build_features.mmv`)도 여기서 나온다."),
 "E": ("E. 치아 분할", "공개 nnU-Net 치아 분할(UNIVERSALLAB, 범용 치아 번호, `run_teeth_seg.py`)로 치아별 중심점·절단 끝을 구해 치열궁 폭·깊이, 수평·수직 피개, 앞니 각도를 잰다(`shape_fine_vectors.teeth`, `build_features.overjet`, `cbct_extra.py`)."),
 "F": ("F. TotalSegmentator 분할 (연조직·인두 구간·부비동)", "TotalSegmentator의 머리 과제(head_glands_cavities, head_muscles, craniofacial_structures)로 인두 3구간·연구개·혀·저작근·부비동·비강을 분할한다(`run_totalseg.py`). "
       "라벨을 정중 좌표계로 옮겨 단면적·길이·부피·위치를 잰다(`shape_fine_vectors.soft_tissue`, `shape_tongue_sinus.measure`, `cbct_nav.py`)."),
 "G": ("G. 통계적 2차 파생 (PCA·요인분석·잔차·묶음 점수)", "위 피처나 대응점 좌표를 통계 모델로 요약한 변수. 주성분분석(PCA), 탐색적 요인분석(EFA), 회귀 잔차, z점수 평균 등. 대부분 85명 기준으로 한 번 적합해 고정했다."),
 "H": ("H. 분석용 변수 (결측 채움·환산)", "회귀에 넣으려고 만든 판. 엑셀·복원값의 결측을 서로 채운 값, 단위 환산 등. 피처 자체의 뜻은 원래 변수와 같다."),
}
CODE = [  # (조건, 원천, 코드 위치)
 (lambda v: v.startswith("x_"), "A", "osa_experiments.data (엑셀 읽기)"),
 (lambda v: v.startswith(("e11_", "r11_", "b_")) or v == "a_PAV_mm3", "H", "osa_newfactor.data"),
 (lambda v: v == "m_NAV", "F", "cbct_nav.py"),
 (lambda v: v.startswith("m_"), "B", "cbct_restore_extend.py → final_criteria(ANB 등)·환산(aligned)"),
 (lambda v: v in ("a_PMCA", "a_PAL", "a_PAV"), "C", "cbct_pmca.py, cbct_pav.py → osa_experiments.airway_raw"),
 (lambda v: v in ("a_PAV_h", "H_col"), "C", "osa_all83.data (a_PAV ÷ 기둥 높이)"),
 (lambda v: v.startswith("AW") and not v.startswith("AWshape") or v == "HD_mm", "C", "shape_extra_vectors.profile/airway, build_features.hd_at_pmca"),
 (lambda v: v.startswith("AWshape"), "G", "osa_all83.data (20구간 프로파일 PCA)"),
 (lambda v: v.startswith(("Hy_",)) or v in ("MP_H", "Hy_Me"), "C", "cbct_eb_detect.hyoid → shape_extra_vectors.relation"),
 (lambda v: "pharynx" in v or v.startswith(("soft_palate", "oro_naso")), "F", "shape_fine_vectors.soft_tissue"),
 (lambda v: v.startswith(("tongue_", "masseter", "lat_pterygoid", "med_pterygoid", "chew", "sinus_")) and v not in ("tongue_MMV", "tongue_MMVhull"), "F", "shape_tongue_sinus.measure → build_features"),
 (lambda v: v in ("SoftLoad_MMV", "tongue_MMVhull"), "F", "osa_all83.data"),
 (lambda v: v == "SDp_mm", "F", "cbct_extra.septum"), (lambda v: v == "m_NAV", "F", "cbct_nav.py"),
 (lambda v: v.startswith("roi_") or v == "MaxPost_Clivus_ant", "D", "shape_local_roi.main → build_features"),
 (lambda v: v.startswith("SIZE_"), "G", "osa_experiments.size_axis (고정 정의: validation/frozen.json)"),
 (lambda v: v.startswith(("MMV", "tongue_MMV")), "D", "build_features.mmv"),
 (lambda v: v.startswith("Condyle"), "D", "cbct_condyle.measure"), (lambda v: v.startswith("lever"), "D", "build_features.lever"),
 (lambda v: v in ("Overjet_mm",), "E", "build_features.overjet"), (lambda v: v in ("overbite_mm",), "E", "cbct_extra.measure"),
 (lambda v: v == "L1_MP_deg", "E", "build_features.l1_mp"), (lambda v: v.startswith(("upper_", "lower_", "Upper_Taper")), "E", "shape_fine_vectors.teeth"),
 (lambda v: v.startswith("EFA_"), "G", "build_features.fit_efa"), (lambda v: "_rPC" in v, "G", "osa_residual_shape.residual_pcs"),
 (lambda v: v.startswith(("F1_", "F2_", "F2MM_")), "G", "osa_joint_axes (main: combined → pca)"),
 (lambda v: v == "VERT_Z", "G", "osa_vertical_index.build (고정값: validation/frozen_vertical.csv)"),
 (lambda v: v.startswith("Atyp") or v in ("Yaxis", "FacialAxis_proxy"), "G" if True else "B", "osa_composite.atypicality / growth"),
 (lambda v: v in ("AlloShape", "logCS_all", "body_len_res"), "G", "osa_all83.data"),
 (lambda v: v.startswith("res_"), "G", "osa_addition (A3 잔차)"),
 (lambda v: v in ("PP_MP", "SN_MP", "ANS_Me", "LAFH_ratio"), "B", "osa_vertical.vertical"),
 (lambda v: v in ("Pog_PNS_ant", "PNS_Ba_depth", "SNA", "SNB", "S_N", "S_Ba", "ANS_PNS", "Co_Gn", "Co_A", "Go_Me", "Co_Go", "N_Me", "NSBa", "gonial", "Jarabak",
                  "McN_diff", "ramus_body", "palate_cb", "mand_cb"), "B", "osa_proportions.geometry"),
 (lambda v: v in ("body_len",), "B", "osa_shape_map.angles"),
 (lambda v: True, "B", "build_features (coords 이후 기하 계산)"),
]
GROUP_FIX = {"Yaxis": "B", "FacialAxis_proxy": "B"}                                     # 기준점 각도라 B로 분류


def where(v):
    for cond, g, code in CODE:
        if cond(v): return GROUP_FIX.get(v, g), code


def main():
    d = N.data(); cat = A.catalog()
    step = {}
    for k in N.R11: step.setdefault(N.X11[k], set()).add("Step 1"); step.setdefault(N.M11[k], set()).add("Step 1")
    for v in N.STEP2: step.setdefault(v, set()).add("Step 2")
    for v in N.STEP3: step.setdefault(v, set()).add("Step 3")
    for v in N.step4(d): step.setdefault(v, set()).add("Step 4")
    step.pop("a_PAV_mm3", None); step.setdefault("a_PAV", set()).add("Step 1")
    rows = {}
    for v in sorted(step):
        de = detail(v) or cat.get(v) or ("", "")
        g, code = where(v); rows.setdefault(g, []).append((v, de[0], de[1], code, ", ".join(sorted(step[v]))))
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    L = ["# 피처 가이드", "", "작성: 2026-11-18 (`python/osa_feature_guide.py`로 생성. 변수 설명은 보고서와 같은 원천인 `osa_var_detail.py`에서 가져온다).", "",
         "이 문서는 OSA 분석에 쓴 피처가 **어떤 원자료에서, 어떤 도구와 원리로, 어떤 계산을 거쳐** 만들어졌는지를 원천별로 정리한다. 값과 결과는 없고, 만드는 법만 담는다.", "",
         "## 읽는 순서", "",
         "1. 아래 '전체 흐름'과 '좌표계'로 큰 그림을 잡는다.",
         "2. 원천별 장(A~H)에서 도구의 원리를 읽고, 표에서 피처마다 뜻·만든 과정·코드 위치를 본다.",
         "3. 더 깊이 볼 때는 맨 아래 '관련 문서'의 문서와 코드로 넘어간다.", "",
         "## 전체 흐름", "",
         "```", "CBCT 원영상 (nrrd)",
         "  ├─ ALI-CBCT ──────────── 기준점 좌표 ─┐",
         "  ├─ 밝기 규칙 ─────────── 기도·설골·Eb ─┤",
         "  ├─ AMASSS ────────────── 하악·상악·두개저 분할 → 표면 대응점 2048개 ─┤",
         "  ├─ 치아 분할 ─────────── 치아별 중심·절단 끝 ─┤",
         "  └─ TotalSegmentator ──── 인두 구간·연구개·혀·근육·부비동 ─┤",
         "                                                  ▼",
         "              정중 좌표계로 옮김 → 1차 피처 (거리·각도·넓이·부피·위치)",
         "                                                  ▼",
         "              통계 요약 → 2차 피처 (PCA·요인분석·잔차·묶음 점수)",
         "                                                  ▼",
         "              분석 (osa_newfactor: Step 1~4 회귀)", "```", "",
         "- 1차 피처 모음: `build_features.py` → `Result_py/features/v1/features.csv` (변수 목록과 정의: `feature_catalog.csv`)",
         "- 분석용 자료 구성: `osa_all83.data` → `osa_newfactor.data`", "",
         "## 좌표계", "",
         "| 좌표계 | 원점·축 | 쓰는 곳 | 주의 |", "|---|---|---|---|",
         "| **정중 좌표계** | 원점 PNS, 축 = 좌·앞·위, 앞 축 = ANS–PNS 방향 (`cbct_eb_detect.midline_frame`) | 대부분의 3D 피처 | 귓구멍 점(Po)을 쓰지 않아 FH 오차의 영향이 없다 |",
         "| **머리 좌표계 (FH)** | FH 평면(귓구멍 점·눈밑점) 기준 (`osa.ceph.head_frame`) | Roh 지표 일부(MP-FH, Nperp), 표면 대응점 저장, 부위 중심 위치(`roi_*_cleft/cant/csup`) | 귓구멍 점이 틀리면 전체가 틀어진다(Step 1 작업일지) |", "",
         "같은 이름의 거리라도 3D 거리인지, 정중 시상면에 투영한 값인지는 표의 '만든 과정'에 적었다.", ""]
    for g in "ABCDEFGH":
        if g not in rows: continue
        t, why = SRC[g]
        L += [f"## {t}", "", why, "", "| 피처 | 뜻 | 만든 과정 | 코드 위치 | 쓰인 단계 |", "|---|---|---|---|---|"]
        L += [f"| `{v}` | {esc(a)} | {esc(b)} | `{esc(c)}` | {s} |" for v, a, b, c, s in rows[g]]
        L.append("")
    L += ["## 관련 문서", "",
          "| 더 알고 싶은 것 | 문서·코드 |", "|---|---|",
          "| 도구 설치·실행, 좌표계를 두 가지 쓰는 이유, Roh 지표 11개 측정법 | `Result_py/report_85/CBCT_측정기준_보고서_2026-11-15.html` (로컬) |",
          "| 자동 측정 정확도, 엑셀 방식 환산 | `Result_py/report_85/CBCT_종합보고서_2026-11-15.html` (로컬), `docs/CBCT_계측기준_개선기록.md` |",
          "| 표면 대응점(통계 형상 모델) 만드는 법 | `docs/4단계_형태_대응점.md`, `python/shape_ssm.py` |",
          "| 환자별 계측 파이프라인 | `docs/CBCT_환자별_계측_파이프라인.md` |",
          "| 변수 정의와 한계 | `docs/파생변수_정의서.md`, `docs/피처_원천별_분류표.md` |",
          "| 피처가 분석에 어떻게 쓰였고 결과가 어땠는지 | `docs/OSA_관련성_실험기록.md`, `docs/실험_가설_설계_결과.md`, `Result_py/analysis/all83/Step1-3_회귀_보고서.html` (로컬) |", ""]
    DOC.write_text("\n".join(L), encoding="utf-8")
    print("saved", DOC, "| 피처", sum(len(r) for r in rows.values()), "|", {g: len(r) for g, r in rows.items()})


if __name__ == "__main__":
    main()
