"""단계별 결과 그림을 목차 번호 이름으로 한 폴더에 복사 (원본은 그대로 둠). 각 스크립트를 다시 돌린 뒤 이것도 다시 실행.
결과: Result_py/EDA_figures/<단계>/ (목차는 docs/1단계_엑셀_EDA.md, 2단계_외부모듈.md, 3단계_CBCT기초_EDA.md)
원본 경로에 *가 있으면 맞는 파일을 모두 복사하고, 이름 앞에 목차 번호를 붙임.
2·3단계 그림 일부는 얼굴 윤곽이 보이므로 내부용.
"""
import shutil
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "Result_py"
FIGS = {
    "1_엑셀": [   # 모두 stage1_figs.py (86명 = 87 - P_39, Class III 표시). 변수마다 그림 하나, 상관은 Spearman 하나
        ("stage1_figs/fig_1-0_missingness.png", "1-0_결측_NAV_PAV_촬영연도.png"),
        ("stage1_figs/fig_1-0_spearman_all.png", "1-0_상관_Spearman_인적_CBCT25_PSG.png"),
        ("stage1_figs/fig_1-1_demographics.png", "1-1_인적_체격.png"),
        ("stage1_figs/fig_1-2_1-3_box_roh.png", "1-2_1-3_Roh변수_박스플롯_기준선.png"),
        ("stage1_figs/fig_1-2_roh_class_ncwr.png", "1-2_Roh분류_빈도_NCWR_촬영연도.png"),
        ("stage1_figs/fig_1-3_nav_pav_missing_compare.png", "1-3_NAV_PAV_측정군_vs_결측군.png"),
        ("stage1_figs/fig_1-4_box_added.png", "1-4_추가변수_박스플롯.png"),
        ("stage1_figs/fig_1-4_bilateral.png", "1-4_과두_좌우비교.png"),
        ("stage1_figs/fig_1-4_focus_pairs.png", "1-4_관심쌍_산점도.png"),
        ("stage1_figs/fig_1-4_binary.png", "1-4_이진변수_빈도.png"),
        ("stage1_figs/fig_1-5_noncbct.png", "1-5_비계측변수_빈도.png"),
        ("stage1_figs/fig_1-6_psg.png", "1-6_PSG_분포_중증도.png"),
        ("stage1_figs/fig_1-6_gap.png", "1-6_CBCT_PSG_검사간격.png"),
        ("stage1_figs/fig_1-6_replication_ahi.png", "1-6_분류별_AHI.png"),
        ("stage1_figs/fig_1-6_group_covariates.png", "1-6_분류별_공변량.png"),
        ("stage1_figs/fig_1-6_m0_scatter.png", "1-6_M0변수_vs_PSG_산점도.png"),
        ("stage1_figs/fig_1-6_m0_coef.png", "1-6_M0_표준화계수_83명.png"),
    ],
    "2_외부모듈": [
        ("CBCT/landmarks/P_*_check.png", "2-2_파일럿_랜드마크점검_"),
        ("pilot/cand_P_*.png", "2-3_파일럿_후보비교_"),
        ("eval/bland_altman.png", "2-4_평가71명_Bland_Altman.png"),
    ],
    "3_CBCT기초": [
        ("cbct_eda/fig_flag_items.png", "3-1_검수항목별_환자수.png"),
        ("cbct_eda/fig_1_A.png", "3-2①_촬영조건_A.png"),
        ("cbct_eda/fig_2_Q.png", "3-2②_영상품질_Q.png"),
        ("cbct_eda/fig_year.png", "3-2②_촬영연도별_연조직피크HU.png"),
        ("cbct_eda/fig_3_P.png", "3-3③_자세교합_P.png"),
        ("cbct_eda/fig_4_F.png", "3-4④_영상경계접촉_F.png"),
        ("cbct_eda/fig_5_S.png", "3-4⑤_분할품질_S.png"),
        ("cbct_eda/fig_uaw_vs_excel.png", "3-4⑥_상기도부피_vs_엑셀.png"),
        ("cbct_eda/montage_raw.png", "3-5_몽타주_원본.png"),
        ("cbct_eda/montage_mask.png", "3-5_몽타주_마스크_v1방식.png"),
        ("cbct_eda/montage_overlay_all.png", "3-5_몽타주_색분할_전체83명.png"),
        ("cbct_eda/montage_overlay_review.png", "3-5_몽타주_색분할_검수29명.png"),
        ("cbct_eda/qc_snapshots/rep_*.png", "3-6_대표환자_"),
    ],
    "4_형태_대응점": [   # PCA 이전까지만 (PCA·해석 그림은 피드백 뒤 단계)
        ("shape/MAND/review/corr_example.png", "4-2_하악_대응점_예시.png"),
        ("shape/CB/review/corr_example.png", "4-2_두개저_대응점_예시.png"),
        ("shape/MAND/review/qc_box.png", "4-5_하악_대응점_품질분포.png"),
        ("shape/CB/review/qc_box.png", "4-5_두개저_대응점_품질분포.png"),
        ("shape/MAND/review/corr_check.png", "4-6_하악_대응점_점검_사례.png"),
        ("shape/CB/review/corr_check.png", "4-6_두개저_대응점_점검_사례.png"),
    ],
}


def main():
    for folder, items in FIGS.items():
        out = R / "EDA_figures" / folder
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for src, name in items:
            if "*" in src:
                for f in sorted((R / Path(src).parent).glob(Path(src).name)):
                    shutil.copy2(f, out / (name + f.name))
                    n += 1
            else:
                shutil.copy2(R / src, out / name)
                n += 1
        print(f"{folder}: {n}개")


if __name__ == "__main__":
    main()
