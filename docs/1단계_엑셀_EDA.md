# 1단계: 엑셀 데이터 EDA와 Roh 1차 재현 (통합 정리)

> 작성: 2026-10-29. 엑셀 값만으로 한 분석을 목차 하나로 모았습니다. 세부 결과는 아래 원래 문서에 있습니다.
> - `Phase1_EDA_결과.md` (10/27): 표본 확정, Roh 변수 기초 통계, M0
> - `논문재현_1차_결과.md` (10/23): Roh 표2~5 재현
> - `엑셀_CBCT_EDA_결과.md` (10/29): 추가 변수 포함 30개 EDA
>
> **그림:** `Result_py/EDA_figures/1_엑셀/`에 목차 번호로 모았습니다 (`python/collect_eda_figures.py`가 원본을 복사).
> **표(CSV):** 각 절의 "결과 파일"에 경로를 적었습니다. 모두 `Result_py/` 아래에 있고 git에서 제외됩니다.
> 환자는 P_n 번호와 집계값으로만 표시합니다.

---

## 1-0. 공통: 표본, 결측, 분포, 이상값, 상관

**표본**
- 엑셀 87명에서 Class III 3명(P_31, P_75, P_85)과 CBCT 본인 아님 1명(P_39)을 빼 **주 분석 83명**으로 정했습니다.
- P_36과 P_59는 생년월일이 다르지만 등록번호가 같아 포함하고 표시했습니다.

**모든 변수에 공통으로 한 것**

| 항목 | 방법 | 대상 |
|---|---|---|
| 엑셀 열 대조 | 52개 열 머리글 전부 확인, 같은 열을 읽는지 Phase 1과 교차 확인 | 87명 |
| 결측 | 변수별 결측 수·비율 | 87명, 83명 |
| 분포 | 평균·SD·중앙값·IQR·범위·왜도, 히스토그램, 박스플롯(환자 점 포함) | 83명 |
| 이상값 | IQR 1.5배 밖, \|z\|>3 (Phase 1) / robust z>3.5 (추가 EDA). 표시만 하고 제외 안 함 | 83명 분포 기준 |
| 상관 | Pearson (Phase 1, 13개 변수) / Spearman + 쌍별 N (CBCT 25개 전부) | 83명 |

**결측:** NAV(83명 중 25명), PAV(58명), ESS(12명) 외에는 결측이 없습니다. NAV·PAV 결측은 촬영 연도로 갈립니다 (1-3).

| 그림 | 결과 파일 |
|---|---|
| `1-0_표본_흐름.png` | `phase1/patient_master.csv` (Valid_Main, Exclude_Main 열) |
| `1-0_결측_NAV_PAV_촬영연도.png` | `phase1/missingness_report.csv`, `excel_cbct_eda/cbct_excel_missingness.csv`, `cbct_excel_missing_pattern.csv` |
| `1-0_상관_CBCT25개_Spearman.png` | `excel_cbct_eda/cbct_excel_correlations.csv` |
| `1-0_상관_Roh변수_PSG_Pearson.png` | `phase1/correlation_matrix.csv` |
| `1-1_1-6_분포_히스토그램_Phase1.png` | `phase1/baseline_statistics.csv`, `baseline_composition.csv` |
| | `excel_cbct_eda/cbct_excel_dictionary.csv` (변수 사전), `cbct_excel_qc.csv` (이상값) |

## 1-1. 인적·체격 변수

| 변수 | 무엇을 봤나 | 결과 |
|---|---|---|
| 나이 | 분포, AHI와 상관, M0 공변량, 재현 표에서 그룹 간 비교 | 44.0±14.9세 (16–70). M0에서 가장 강한 예측변수 (β 0.44, p<.001). Class II와 hyper가 더 젊음 |
| 성별 | 빈도, M0 공변량, 재현 표2 (남녀 비교) | 남성 54명 (65%). 남성이 PSG 지표 6개 모두에서 더 중증 (재현됨) |
| BMI | 분포, 이상값, 키·몸무게로 계산한 값과 대조, M0 공변량 | 24.9±4.1. 계산값과 불일치 0건. 극단값 P_35·P_63·P_80. M0 p=.18 |
| 키·몸무게 | BMI 대조용 | 따로 분석하지 않음 |

결과 파일: `phase1/baseline_statistics.csv`, `replication/table2_sex.csv`

## 1-2. CBCT 골격 변수 (Roh)

Roh 논문의 분류 기준이 있는 변수는 ANB, MP-FH, TDI뿐입니다. 박스플롯에 빨간 점선으로 표시했습니다.

| 변수 | 무엇을 봤나 | 결과 |
|---|---|---|
| ANB | 분포, 시상 분류 (Class I: 0<ANB<4°, II: ≥4°), M0, 재현 표2~5, 민감도 분석 6조건 | 5.1±2.5°. Class I 26명 / II 57명. M0 p=.061, 조건에 따라 p .006~.23 (방향 일관) |
| Nperp_A | 분포, 재현 표2 | 0.7±2.0 mm. 이상값 없음 |
| Nperp_Pog | 분포, 재현 표2, 상관 | −7.7±5.6 mm. ANB와 Spearman −0.74 |
| MP-FH | 분포, 수직 분류 (hypo <25°, normo 25~30°, hyper ≥30°), M0, 재현 표2·4 | 29.6±5.4°. 19 / 21 / 43명. M0 p=.11. 여성에서 더 큼 (p=.048, 재현됨) |
| NCWR | 분포, 재현 표 비교 변수 (논문에서 TSD 보조 지표, 기준치 없음) | 51.2±7.1%. **촬영 연도와 상관 (r 0.42)** |
| PHI | 분포, 재현 표 비교 변수 (TSD 보조 지표, 기준치 없음) | 44.0±4.7%. 특이 소견 없음 |
| TDI | 분포, 횡방향 분류 (TSD <−2.5 mm, non-TSD >−2 mm), M0, 재현 표3~5, 논문과 분포 비교 | −1.0±2.5 mm. TSD 20 / 경계 11 / non-TSD 52명 (TSD 24%, 논문 34%). **어느 조건에서도 유의하지 않음 (재현 안 됨)** |

| 그림 | 결과 파일 |
|---|---|
| `1-2_1-3_Roh변수_박스플롯_기준선.png` | `excel_cbct_eda/cbct_excel_descriptive.csv` |
| `1-2_1-4_CBCT연속변수_히스토그램.png` | `replication/table2_sex.csv`, `table4_sagittal.csv`, `table4_vertical.csv`, `table4_transverse.csv` |

## 1-3. CBCT 기도 변수 (Roh)

| 변수 | 무엇을 봤나 | 결과 |
|---|---|---|
| NAV | 결측, 분포, 결측 패턴 (촬영 연도별, 측정 여부별 비교) | n=58. 2023년 이후 거의 측정되지 않음 |
| PAV | 결측, 분포, 결측 패턴, 재현 표2 | n=25. 2022년 이후 거의 측정되지 않음. PAV가 있으면 NAV도 항상 있음. 극단값 P_5·P_44 |
| PMCA | 분포 (왜도), M0, 재현 표3~5 | 124.6±68.4 mm², 오른쪽 꼬리 (왜도 0.93). Class II·CSP에서 더 작음 (재현됨). M0 p=.36 |
| PAL | 분포, M0, 재현 표2 | 71.5±6.0 mm. AHI와 단순상관 0.23 (유의). 남성에서 더 김 (재현됨). M0 p=.82 |

- NAV·PAV 결측은 해부학적 특성이 아니라 **측정 시기에 따라 갈립니다.** 측정 여부에 따라 ANB, MP-FH, PMCA, PAL은 차이가 없습니다.
- Roh 논문은 NAV 마스크에서 **인접 부비동 부피를 제거했다**고 적었습니다. 자동 기도 분할에서 상악동이 잡힌 사례(P_22)와 연결되는 점입니다.

그림: `1-0_결측_NAV_PAV_촬영연도.png`, `1-2_1-3_Roh변수_박스플롯_기준선.png` / 결과 파일: `excel_cbct_eda/cbct_excel_missing_pattern.csv`

## 1-4. 추가 CBCT 변수 14개 (Roh 논문에 없음)

논문 원문에 condyle, joint space, septal, asymmetry는 나오지 않습니다. 이상·정상을 가르는 기준치가 없어 그림에 기준선을 긋지 않았습니다.
Septal deviation, Nasal floor cant, Occlusal plane의 측정 정의는 인수인계 PPT 13~14쪽에 있습니다 (`CBCT_측정_계획.md` 4-4절).

| 변수 | 무엇을 봤나 | 결과 |
|---|---|---|
| Occlusal plane | 분포, 기존 변수와 상관 | 정수 6–19 (° 추정). 평면 = 좌우 상악 제1대구치 근심협측 교두정 + 앞니 겹침 중간점 (각도의 기준 평면은 미확인). MP-FH 0.48, ANB 0.49 |
| Chin dev | 빈도, 과두 좌우 차이와 비교 | 0/1 이진. 1이 38명 (46%). 과두 좌우 차이와 무관 (p>0.8) |
| Septal deviation | 빈도, 교차표 | 기준: MSP에서 비중격 최대 휜 점까지 > 2 mm면 1. 1이 34명. Nasal obstruction과 약하게 동반 (0.22) |
| Nasal floor cant | 빈도, 교차표 | 기준: 좌우 비강저 최하점 연결선과 MSP 각도가 90°±2° 밖이면 1. 1이 36명. Chin dev·Septal deviation과 관계 없음 |
| 과두 각도 4개 (Axial·Lat × R/L) | 분포, 좌우 상관, R−L 차이, Chin dev별 비교 | 좌우 같은 정의로 측정된 것으로 보임 (좌우 r 0.69–0.84). CondAx는 ANB와 0.45–0.49 |
| 관절 공간 6개 (Ant·Sup·Post × R/L) | 분포, 불가능한 값(≤0), 좌우 비교 | 불가능한 값 0건. Post만 좌우 차이 (0.14 mm, p=.01). 극단값 P_75·P_83 |

| 그림 | 결과 파일 |
|---|---|
| `1-4_추가변수_박스플롯.png` | `excel_cbct_eda/cbct_excel_descriptive.csv`, `cbct_excel_frequency.csv` |
| `1-4_과두_좌우비교.png` | `excel_cbct_eda/cbct_excel_bilateral.csv` |
| `1-2_1-4_CBCT연속변수_히스토그램.png` | `excel_cbct_eda/cbct_excel_correlations.csv` (관심 쌍 표가 맨 위) |

## 1-5. 비계측 변수 (영상 계측값이 아닌 열)

| 변수 | 성격 | 무엇을 봤나 | 결과 |
|---|---|---|---|
| TMJ pain history | 임상 병력 (0/1) | 빈도 | 19명 (23%) |
| Nasal obstruction | 임상 소견 (0/1/2, 인수인계 PPT: 측정 대상 아님) | 빈도, Septal deviation·상악동염과 교차표 | 39/28/16명. Septal deviation과 약하게 동반 (0.22) |
| Maxillary sinusitis | 임상 소견 (0/1, 측정 대상 아님) | 빈도, 교차표, 상기도 오분할 환자와 대조 | 4명뿐, 모두 코막힘 1~2. P_22(오분할)는 0 → 상악동염과 무관한 오분할로 봄 |
| Type (1~4) | 분류 코드, 정의 미확인 | 빈도만 (해석 안 함) | 12 / 53 / 17 / 1명 |
| Group (1/2) | "CII HA" 코드, 정의 미확인 | 빈도만 (해석 안 함) | 34 / 49명 |
| CBCT Slice 간격 | 촬영 조건 | DICOM 복셀 크기와 대조 | 불일치 0건. P_50 0.3 mm, P_78 0.2 mm, 나머지 0.39 mm |

결과 파일: `excel_cbct_eda/cbct_excel_frequency.csv`, `cbct_excel_qc.csv`

## 1-6. PSG 결과 변수, M0, Roh 1차 재현

**PSG 결과 변수** (M0와 재현에서 결과로만 사용)

| 변수 | 무엇을 봤나 | 결과 |
|---|---|---|
| AHI, RDI, ODI, LSaO2 | 분포, 중증도 구성, 리포트 PDF 값과 대조 | AHI 중앙값 28.9, 왜도 1.41. 전원 AHI≥5 (정상 대조군 없음). **P_65·P_67·P_69는 엑셀 값이 리포트와 다름** |
| AHI_SUP, AI, HI, CAI, ArI, ESS | 리포트에서 추출, 재현 표 | CAI 최댓값 0.8 → 중추성 무호흡 제외 기준을 사실상 충족 |
| CBCT–PSG 간격 | 분포, 1년 기준 민감도 분석 | 중앙값 14일. 1년 초과 15명 |

**M0:** `AHI ~ 나이 + 성별 + BMI (공변량) + ANB + MP-FH + TDI + PMCA + PAL (관심 변수)`
- 83명 기준 R² 0.35, adj R² 0.28입니다. 나이와 성별만 유의합니다.
- 민감도 분석은 6조건으로 했습니다: {전체, 간격 1년 이내, 리포트 불일치 3명 제외} × {AHI, log AHI}. ANB만 조건에 따라 유의와 비유의를 오갑니다.
- 단순상관 (보정 없음): p<0.05인 것은 BMI(AHI 0.26)와 PAL(0.23)뿐입니다.

**Roh 1차 재현**
- 시상(Class II)과 성별 효과는 재현됐습니다.
- 횡방향(TDI·TSD) 효과는 재현되지 않았습니다. CSP3D와 nonCSP 그룹이 작은 것이 한 원인입니다.

| 그림 | 결과 파일 |
|---|---|
| `1-6_M0변수_vs_PSG_산점도.png` | `phase1/M0_baseline_results.csv`, `phase1/m0_scatter_r.csv` |
| | `replication/table3_regression.csv`, `table5_csp.csv`, `table5_csp2d3d.csv`, `descriptive.csv`, `supp_correlation.csv` |

---

## 1단계에서 안 한 것

1. **Roh 변수 11개의 이상값 검수:** 극단값을 표시만 했고, 환자별 원인 확인은 하지 않았습니다.
2. **촬영 연도와의 관계:** 일부만 일회성으로 계산해 결과 파일이 없습니다. NCWR만 r 0.42였고 나머지는 |r|<0.2였습니다.
3. **자동 계측값과 대조:** ANB, MP-FH, Nperp는 2단계에서 했습니다. 나머지는 자동 계측이 아직 없습니다.
4. **정의 확인:** Occlusal plane·과두 각도의 기준 평면, 관절 공간 단위, Chin dev 판정 기준, Type·Group 정의는 데이터 담당 질문에 넣었습니다 (`엑셀_CBCT_EDA_결과.md` 9장).
