# Step 1 작업일지: CBCT 자동 계측의 엑셀 값 복원과 복원값 기반 인자 재현

작성: 2026-11-16. 시간순 원장은 [OSA 관련성 실험기록](OSA_관련성_실험기록.md)(S1, S1b~S1g 절)이다. 이 문서는 결론 중심으로 다시 정리한 것이다.

> **그림 출처 표기 규칙 (HTML 변환 대비)**
> - `[그림]`은 이미 있는 그림 파일이다.
> - `[그림 예정]`은 아직 그리지 않은 그림으로, 만들 때 쓸 데이터 파일과 그릴 모양을 적었다.
> - 경로는 프로젝트 루트(`OSA/Project/`) 기준이다. `R/`은 `Result_py/`, `V2/`는 `Result_py/analysis/osa_experiments_v2/`, `PIPE/`는 `Result_py/CBCT/patient_pipeline/v2/`를 뜻한다.

---

## 0. 결론

1. **엑셀에서 AHI 신호가 있던 골격 지표는 3개다(ANB, Nperp-Pog, MP-FH).**
   - **ANB**: 자동 계측으로 복원했다.
   - **Nperp-Pog**: 3D 지표 **Pog-Ba**(Pog_Ba_ant)로 대체했다.
   - **MP-FH**: 재현되지 않았다. 다만 엑셀에서도 ANB와 겹치는 정보라 모델 손실은 없다.
2. **최종 모델은 "나이·성별·BMI + 최종 ANB + Pog-Ba"다.**
   - 주 분석에서 엑셀 조합(엑셀 ANB + Nperp-Pog)보다 높다. ΔR² H1 +0.155 대 +0.096, H2 +0.284 대 +0.103.
3. **기도 변수(PMCA, PAL)는 ANB 위에 더하는 것이 없고 오히려 골격 효과를 가린다.**
4. **나머지 8개 지표는 엑셀 값도 AHI와 관련이 없다.** 그래서 예측력으로는 복원 여부를 판정할 수 없고, 일치도로만 평가한다.
5. **FH를 쓰는 지표(MP-FH, Nperp-A, Nperp-Pog)는 FH 평면 방향 차이 때문에 엑셀과 어긋난다.**
   - 그래서 수직 분류와 CSP 같은 그룹 분석에는 자동값을 쓰지 않고, 엑셀 값으로 분류한다.

---

## 1. 목적과 범위

- **기준:** 논문 재현이 아니라 **우리 엑셀 값으로 나온 결과가 자동 계측값으로도 재현되는가**를 본다.
- **Step 1이 답하는 질문:**
  - (1) 자동 계측이 엑셀 값을 대신할 수 있는가
  - (2) 엑셀 기반 AHI 관련 결과가 자동값으로도 나오는가
  - (3) 안 되는 지표는 무엇으로 대체하는가
  - (4) 기도 변수가 골격 효과를 가리는가
- **결과 변수:**
  - H1 = log(AHI+1)
  - H2 = log(앙와위 AHI+1)
  - H2는 사용자 결정(2026-11-16)으로 주 가설에 포함했다.
- **비교 대상 논문:** Roh et al. 2025, Eur J Orthod 47:cjae041. 원문 PDF는 프로젝트 루트에 있다.

## 2. 입력 데이터와 처리

| 항목 | 파일 | 처리 |
|---|---|---|
| 엑셀 값 (`x_*`) | `PIPE/restore_ext/v1/compare_all.csv` `excel` 열 | `PIPE/develop85/final_criteria_v2/final_values.csv`의 `excel`과 340/340 일치 |
| 엑셀 0값 의심 | `final_values.csv` `excel_zero_suspect` | Nperp-A 17명, Nperp-Pog 5명을 결측 처리(`osa_experiments.data()`) |
| 자동 값, 현재판 (`m_*`) | `PIPE/restore_ext/v1/alignment/aligned_metrics.csv` | Nperp-A·Nperp-Pog·PHI·PMCA·NAV는 엑셀에 맞춘 직선 환산값. **예측력 비교에만 쓰고, 엑셀 − 자동 차이 분석에는 쓰지 않는다** |
| 자동 값, 환산 전 | 같은 파일의 `default` 열 (jung 3점 FH) | 차이 분석 민감도 |
| 자동 값, 최종표 (`f_*`) | `PIPE/develop85/final_criteria_v2/final_measurements_85.csv` | yao4 4점 FH, 골격 4개. 등급 E 제외(A~D), 민감도는 A·B만 |
| 기도 지표 | 기존 자동값 (최종표의 `_auto_unaudited`와 같은 출처) | 재확인 전 |
| 피처 (Pog_Ba_ant 등) | `R/features/v1/features.csv` (`python/build_features.py`) | Po를 쓰지 않는 정중 좌표계 |
| ICC·MAE | `PIPE/restore_ext/v1/experiments/final_comparison.csv` | 규칙 개발군 / 검증군 |
| PSG·공변량 | `R/phase1/patient_master.csv` | |

**표본**
- **주 분석:** PSG 확인이 끝난 59명.
- **85명 표본(민감도):** 측정한 85명 중 PSG 값이 있는 환자로, PSG 확인 중인 26명을 포함한다.
- 85명 모두 계측 규칙 개발에 쓴 자료라, 독립 검증군이 아니다.

## 3. 평가 방법

- **기준 모델 B0:** 나이 + 성별 + BMI.
- **성능:** 5겹 교차검증을 50회 반복한 **교차검증 R²의 증가(ΔR²)**. 범위는 반복 50회의 2.5~97.5 백분위다.
- **같이 보는 지표:** 50회 중 개선 비율, 전체 표본 중첩 F 검정 p, 조정 R² 증가, 교차검증 RMSE 변화.
- 모든 비교는 **같은 환자 안에서 1:1로** 한다.
- 코드: `python/osa_experiments.py` (`compare()`).

---

## 4. 지표별 결과 (11개)

| 지표 | ICC 개발 / 검증 | 엑셀 AHI 신호 (주 분석 ΔR² H1) | 자동 신호 | 판정 |
|---|---|---|---|---|
| **ANB** | 0.89 / 0.68 | +0.136 (p=0.001) | +0.134, 최종 +0.162 | ✅ **복원** |
| **Nperp-Pog** | 0.82 / 0.58 | +0.116 (p=0.006, 0값 제외) | ≈0 | ✅ **Pog-Ba로 대체** |
| **MP-FH** | 0.88 / 0.74 | +0.031 (p=0.042) | −0.025 | ❌ 재현 안 됨 (모델 손실 없음) |
| Nperp-A | 0.60 / **0.36** | 없음 | 없음 | 값 미복원, 신호 없음 |
| TDI | 0.74 / **0.42** | 없음 | 없음 | 값 미복원, 신호 없음 |
| PHI | 0.74 / **0.55** | 없음 | 없음 | 값 미복원, 신호 없음 |
| PMCA | 0.74 / 0.62 | 없음 | 없음 | 경계 (일정한 방향의 차이 → 환산) |
| NAV | 0.77 / 0.68 | 없음 | 없음 | 경계 |
| PAL | 0.78 / 0.78 | 없음 | 없음 | 값 복원 |
| NCWR | 0.57 / 0.84 | 없음 | 없음 | 값 복원 |
| PAV | 0.99 / 0.90 | 엑셀 16~25명 | — | 값 복원, 예측력은 판정 불가 |

- 데이터: `V2/S1_excel_vs_auto_per_metric.csv`, `R/analysis/osa_experiments_v2_final/S1_excel_vs_auto_per_metric.csv`, ICC는 `final_comparison.csv`.
- `[그림]` 지표별 엑셀 ↔ 자동 산점도: `PIPE/report/scatter/{지표}.svg` (11개), 요약 `PIPE/report/scatter/overview.png`, `R/report_85/figs/restore_scatter.png`
- `[그림 예정]` 지표별 엑셀 ΔR²와 자동 ΔR²를 나란히 놓은 점그림(지표 11개 × H1·H2). 데이터: 위 두 CSV의 `dR2_excel`, `dR2_auto`

---

## 5. 주요 비교 결과

### 5-1. ANB: 세 판 비교

| 판 | n | H1 엑셀 / 자동 | H2 엑셀 / 자동 |
|---|---|---|---|
| 현재 값 (jung) | 59 | +0.136 / +0.134 | +0.124 / +0.176 |
| 최종표 (yao4, A~D) | 55 | +0.132 / **+0.162** | +0.144 / **+0.278** |
| 최종표 A·B | 41 | +0.103 / +0.127 | +0.131 / **+0.267** |

- 엑셀–자동 ANB 상관은 0.88~0.90이다.
- 두 값을 같이 넣으면 서로 대체된다. 같은 정보라는 뜻이다.
- 앙와위 AHI에서는 자동 ANB가 엑셀 위에 더한다(+0.109, p=0.006).
- 데이터: `V2/S1_excel_vs_auto_per_metric.csv`, `R/analysis/osa_experiments_v2_final/`, `R/analysis/osa_experiments_v2_final_AB/`, `V2/S1d_excel_diff/diff_by_metric.csv`
- `[그림 예정]` 판별 ANB ΔR² 막대(엑셀·자동 × H1·H2 × 세 판). 데이터: 위 CSV들

### 5-2. Nperp-Pog → Pog-Ba 대체

**예측력 (주 분석, 공통 52명, 최종표 A~D, 엑셀 0값 제외)**

| 모델 | H1 | H2 |
|---|---|---|
| 엑셀 Nperp-Pog | +0.064 | +0.061 |
| 자동 Nperp-Pog | −0.041 | −0.051 |
| **Pog-Ba** | **+0.102** | **+0.152** |
| 엑셀 ANB + Nperp-Pog | +0.096 | +0.103 |
| **최종 ANB + Pog-Ba** | **+0.155** | **+0.284** |
| 엑셀 ANB 위 + 엑셀 Nperp-Pog | −0.032 | −0.040 |
| 최종 ANB 위 + Pog-Ba | −0.007 | −0.002 |

**크기 효과가 아님**
- S-Go(머리 크기 대리)를 먼저 넣어도 효과가 남는다. H1 +0.078 (p=0.010), H2 +0.113 (p=0.005).
- 크기로 나눈 비율은 약해진다(+0.02). 즉 절대적인 앞뒤 공간 길이가 중요하다.

**입 벌림 영향은 작음**
- overbite를 먼저 넣어도 효과가 남는다. H1 +0.075, H2 +0.068.

**도메인 근거**
- Pog-Ba는 혀와 연조직이 들어갈 뼈 울타리의 앞뒤 길이에 해당한다. 이는 해부학적 불균형 개념(Tsuiki·Isono 2008)과 맞는다.
- 성인 OSA 메타분석에서 두개저가 짧고 각이 작다(Neelapu 2017, J Sleep Res 2023).
- Ba–N을 기준으로 한 facial axis가 AHI와 관련이 있다(Hor 2024).
- Ba는 Ricketts 분석의 표준 계측점이다.
- **Ba–Pog 거리 자체를 OSA 예측변수로 보고한 연구는 없다.** 그래서 이 연구에서 정의한 3D 지표로 쓴다.

**주의**
- 순수한 하악 위치가 아니라 하악 위치와 두개저 길이가 섞인 값이다.
- 임상 기준값이 없다.
- Ba 자동 계측의 정확도는 엑셀 비교 대상이 없어, 영상 확인(5-8)으로 대신한다.

자료
- 데이터: `V2/S1d_excel_diff/model_compare.csv`, `V2/S1b_nperp_pog_check/replacement_compare.csv`
- `[그림 예정]` 모델별 ΔR² 막대(엑셀 조합 대 자동 대체 조합, H1·H2, 주 분석·85명). 데이터: `model_compare.csv`
- `[그림 예정]` 엑셀 Nperp-Pog 대 Pog-Ba 산점도(ρ 0.76). 데이터: `R/features/v1/features.csv` + `compare_all.csv`

### 5-3. MP-FH

| 모델 (주 분석) | H1 | H2 |
|---|---|---|
| 엑셀 MP-FH (59명) | +0.031 (p=0.042) | +0.035 |
| 엑셀 MP-FH (A·B 51명) | +0.116 | +0.151 |
| 자동 / 최종 MP-FH | −0.025 / −0.026 | −0.032 / −0.034 |
| FH 없는 SN-MP / PP-MP | −0.036 / +0.009 | −0.044 / +0.025 |
| 엑셀 ANB 위 + 엑셀 MP-FH | −0.026 | −0.036 |
| 최종 ANB + Pog-Ba 위 + MP-FH | 조합 −0.02 | 조합 −0.003 |

- 엑셀 MP-FH의 신호는 엑셀 ANB 위에서도 남지 않는다. 그래서 **모델에 넣지 않는다.**
- 데이터: `V2/S1c_mpfh_check/replacement_compare.csv`, `V2/S1d_excel_diff/model_compare.csv`, `R/analysis/osa_experiments_v2_final_AB/S1_excel_vs_auto_per_metric.csv`

### 5-4. 오차 원인: FH 평면 방향 차이

자동값은 "엑셀과 같은 정보 + 엑셀과 무관한 추가 오차" 형태다. 자동/엑셀 SD 비는 1.15~1.69이고, 차이는 자동값과 상관(−0.49~−0.82)이 있다. 그래서 엑셀 값은 자동값 위에 AHI 정보를 더하고, 반대는 아니다. 주 분석 H1 기준으로 Nperp-Pog +0.106, MP-FH +0.094다.

**엑셀 − 자동 차이끼리 상관 (주 분석)**

| 쌍 | 최종표 A~D | A·B | jung |
|---|---|---|---|
| Nperp-A ~ Nperp-Pog | +0.58 | +0.63 | +0.58 |
| Nperp-Pog ~ MP-FH | −0.52 | −0.54 | −0.57 |
| Nperp-A ~ MP-FH | −0.38 | −0.38 | −0.23 |
| ANB ~ 나머지 | ≈0 | ≈0 | ≈0 |

- FH를 쓰는 세 지표의 오차가 같이 움직이고, ANB만 따로 논다.
- 방향도 FH 기울기 차이와 맞는다.
- yao4 FH로 바꿔도 줄지 않는다.

자료
- 데이터: `V2/S1d_excel_diff/diff_by_metric.csv`, `V2/S1d_excel_diff/diff_cross.csv`
- `[그림 예정]` 차이 상관 열지도(4×4, 판 3개). 데이터: `diff_cross.csv`
- `[그림 예정]` 지표별 차이 대 자동값, 차이 대 엑셀 산점도(4지표). 데이터: `compare_all.csv` + `final_measurements_85.csv`
- **주의:** 차이 분석에는 환산값(`aligned`)을 쓰지 않는다. 엑셀에 맞춘 회귀라 순환이 된다.

### 5-5. 기도 변수는 골격 효과를 가림

| 모델 (B0 위) | H1 주 분석 | H2 주 분석 |
|---|---|---|
| + ANB | +0.134 | +0.176 |
| + ANB + PMCA·PAL | +0.075 | +0.120 |
| ANB 위 + PMCA·PAL | −0.038 (p=0.38) | −0.039 |
| + Roh 골격 3개 (ANB·MP-FH·TDI, n=46) | −0.006 | +0.066 |
| + 엑셀 Roh 5개 (기존 M0) | +0.037 | +0.014 |

- 최종표 판(−0.045)과 A·B 판(−0.060)에서도 같다.
- TDI를 넣으면 결측 때문에 표본이 줄고(59→46) ANB 효과가 사라진다.
- 데이터: `V2/S1_baseline.csv`, `R/analysis/osa_experiments_v2_final/S1_baseline.csv`, `R/analysis/osa_experiments_v2_final_AB/S1_baseline.csv`
- `[그림 예정]` 기준선 모델 ΔR² 막대(범위 표시). 데이터: `S1_baseline.csv` (`dR2`, `dR2_lo`, `dR2_hi`)

### 5-6. Roh 분류의 엑셀 ↔ 자동 일치도

| 분류 (주 분석) | 일치율 / κ, 현재 | 일치율 / κ, 최종표 |
|---|---|---|
| 시상 (ANB) | 86% / 0.72 | 87% / 0.74 |
| 수직 (MP-FH) | 75% / 0.61 | 75% / 0.61 |
| 횡 (TDI) | 72% / 0.48 | 72% / 0.48 |
| CSP 4분류 | 72% / 0.54 | 77% / 0.62 |

- 자동값은 고각형을 적게 잡는다. 엑셀 고각형 29명 중 21명만 자동으로도 고각형이다.
- 엑셀 MP-FH가 30° ± 3° 안인 환자가 19명이다.
- **그룹 비교(논문 표 4·5 방식)의 분류는 엑셀 값으로 한다.**

자료
- 데이터: `V2/S1e_class_agreement/agreement.csv`, `V2/S1e_class_agreement/crosstabs.csv`
- `[그림 예정]` 수직 분류·CSP 혼동표 열지도. 데이터: `crosstabs.csv`
- `[그림 예정]` 엑셀 MP-FH 분포와 25°·30° 경계선, 자동 분류 색 표시. 데이터: `compare_all.csv`

### 5-7. 논문(Roh 2025) 결론과의 관계

| 논문 결과 | 우리 엑셀 | 자동 |
|---|---|---|
| 표 3: ANB가 6개 PSG 지표 모두의 예측변수 | ✅ | ✅ (엑셀 이상) |
| 표 3: TDI가 6개 지표 모두의 예측변수 (핵심 주장) | ❌ | ❌ |
| 표 3: PMCA가 AHI·앙와위 AHI·RDI의 예측변수 | ❌ | ❌ |
| 표 3: MP-FH는 회귀에서 무관 (p 0.18~0.92) | 약함 | 없음 |
| Nperp-A, Nperp-Pog: 회귀에 넣지 않음 | Nperp-Pog 신호 있음 (논문에 없음) | Pog-Ba로 대체 |
| PAV: "예측에 유의하지 않음" | 판정 불가 | 판정 불가 |
| 표 4·5: 고각형, CSP가 더 중증 (그룹 비교) | 1차 재현에서 같은 방향, 유의하지 않음 | 분류 일치도 75% → 엑셀로 분류 |

- 엑셀 기반 1차 재현: `docs/논문재현_1차_결과.md`, `R/replication/*.csv`

### 5-8. Ba 점 영상 확인 (10명)

- 선택: Pog-Ba 최소 2, 최대 2, ANB 대비 잔차 최대 2, 주 분석 무작위 4.
- 1차 육안 판정(6명):
  - 5명은 사면 끝, 즉 대후두공 앞 가장자리에 있어 적절하다.
  - P_35는 몇 mm 아래일 수 있어 확인이 필요하다.
  - 4명은 미확인이다.
  - **사용자 판정 대기.**
- `[그림]` `V2/S1g_ba_review/P_{n}_Ba.png` (10장: Ba 중심 시상·축·관상 단면 + Ba–Pog 시상 단면), 판정 페이지 `V2/S1g_ba_review/index.html`, 목록 `selection.csv`

### 5-9. 85명 표본 H1 예외

- 85명 표본의 H1에서만 엑셀 ANB(+0.229)가 자동(+0.126)보다 높다.
- 원인은 엑셀–자동 차이가 3°를 넘는 **7명**이다. 이 중 6명은 추가 환자이고, 5명은 엑셀 검토 대상(`excel_review`)으로 표시돼 있다.
- 이 7명을 빼면 엑셀 +0.152, 자동 +0.154로 같다.
- 데이터: `V2/S1f_sens85_exception/groups.csv`, `V2/S1f_sens85_exception/trim.csv`
- `[그림 예정]` 85명 엑셀 ANB 대 자동 ANB 산점도에 7명 강조, 주 분석 / 추가 환자 색 구분. 데이터: `compare_all.csv` + `R/phase1/patient_master.csv`

---

## 6. 결정 사항

| 결정 | 근거 절 |
|---|---|
| Nperp-Pog 자리는 Pog-Ba로 대체 (사용자 결정 2026-11-16). 보고서에는 "같은 하악 후퇴 정보를 담은 3D 대체 지표"로 쓴다 | 5-2 |
| 최종 OSA 모델은 B0 + 최종 ANB + Pog-Ba. MP-FH, 수직 지표, 기도 변수는 넣지 않는다 | 5-3, 5-5 |
| 엑셀 Nperp-A·Nperp-Pog의 0값 의심은 결측으로 처리 | 2 |
| 환산값(`aligned`)은 예측력 비교 전용. 차이 분석에는 최종표나 `default`를 쓴다 | 5-4 |
| 수직 분류·CSP 그룹 분석의 분류는 엑셀 값으로 | 5-6 |
| 최종표는 등급 A~D를 기본으로 하고 E는 제외. A·B는 민감도 | 2 |

## 7. 한계와 남은 일

| 항목 | 상태 |
|---|---|
| 기도 지표 최종값 재확인 | **보류.** 기도 계측 포함 범위가 사용자·교수님 결정 사항([계측기준 개선기록](CBCT_계측기준_개선기록.md) "결정 필요 3"). Step 1 결론에는 영향 없음 |
| Ba 점 영상 판정 | **사용자 판정 대기** (`V2/S1g_ba_review/index.html`) |
| 85명 예외 7명의 엑셀 값 검토 | 다른 세션의 엑셀 검토 목록 결과 대기 |
| 표본 | 주 분석 59명으로 작고, 계측 규칙 개발에 쓴 자료라 독립 검증군이 아니다 |
| A·B 공통 표본 | 골격 4개를 동시에 A·B로 제한하면 29명이라 판정 불가 |
| PHI, NCWR의 FH 사용 여부 | 코드 미확인 |

## 8. 산출물

| 절 | 코드 | 출력 |
|---|---|---|
| S1 기준선·지표별 | `python/osa_experiments.py S1 [--source final_v2 [--grades AB]]` | `V2/`, `R/analysis/osa_experiments_v2_final/`, `R/analysis/osa_experiments_v2_final_AB/` |
| S1b Nperp-Pog 대체 (예측력 부분만 유효) | `python/osa_nperp_pog_check.py` | `V2/S1b_nperp_pog_check/` |
| S1c MP-FH (예측력 부분만 유효) | `python/osa_mpfh_check.py` | `V2/S1c_mpfh_check/` |
| S1d 차이 재계산·대체 비교 | `python/osa_excel_diff_check.py` | `V2/S1d_excel_diff/` |
| S1e 분류 일치도 | `python/osa_class_agreement.py` | `V2/S1e_class_agreement/` |
| S1f 85명 예외 | `python/osa_sens85_exception.py` | `V2/S1f_sens85_exception/` |
| S1g Ba 영상 확인 | `python/osa_ba_review.py` | `V2/S1g_ba_review/` |
| 1차 실행 (E1, E7, E7b) | `python/osa_experiments.py` (v1) | `R/analysis/osa_experiments_v1/` |
