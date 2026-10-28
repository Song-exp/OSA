# 엑셀 CBCT·구강 변수 기본 EDA 결과

> 작성: 2026-10-29. 코드 `python/excel_cbct_eda.py`, 결과 `Result_py/excel_cbct_eda/`.
> 범위: 엑셀의 두개안면·구강 변수만 봅니다. PSG, SSM PCA, 예측 모형은 포함하지 않습니다.
> 분포와 상관은 주 분석 83명(Valid_Main) 기준이고, 결측은 87명과 83명 둘 다 냈습니다. p값은 탐색용이며 다중비교 보정을 하지 않았습니다.

---

## 1. 결론

- **엑셀의 CBCT 관련 변수는 빠짐없이 탐색했습니다.** 엑셀 열 52개의 머리글을 모두 대조해, 인적사항과 PSG를 뺀 CBCT 계측 25개, 임상 3개, 분류 2개, 촬영 1개, 체격 3개를 다뤘습니다.
- **정정 (10/29):** 처음에는 Nasal obstruction과 Maxillary sinusitis를 CBCT 변수로 묶고 판정 기준을 "미확인"으로 적었습니다. 그러나 인수인계 PPT 13~14쪽(`CBCT_측정_계획.md` 4-4절)에 따르면 이 둘은 **임상 소견**이고, Septal deviation·Nasal floor cant·Occlusal plane은 판정 기준이 있습니다. 묶음과 정의를 고쳤고, 상관 분석은 CBCT 25개로 다시 계산했습니다. 수치는 바뀌지 않았습니다.
- 입력 오류로 보이는 값은 없습니다.
  - 불가능한 값: 0건
  - 엑셀 BMI와 키·몸무게로 계산한 BMI의 불일치: 0건
  - 엑셀 Slice 간격과 DICOM 복셀 크기의 불일치: 0건
  - 극단값 9건은 실제 환자일 수 있어 검수 대상으로만 표시했습니다.
- **NAV·PAV 결측은 촬영 연도와 함께 갈립니다.** PAV는 2022년 이후 거의 측정되지 않았고, NAV는 2023년 이후 거의 측정되지 않았습니다. 결측이 해부학적 특성 때문이라기보다 측정 시기 때문일 가능성이 높습니다 (확정은 아님).
- **NCWR은 촬영 연도와 상관됩니다** (Spearman r 0.42). 측정자나 방법이 바뀌었는지, 실제로 환자 구성이 달라졌는지 확인이 필요합니다.
- Type과 Group은 정의를 몰라 빈도만 기록했습니다. 해석하지 않습니다.

## 2. 변수 정리 (`cbct_excel_dictionary.csv`)

| 묶음 | 변수 | 자료형 |
|---|---|---|
| CBCT 골격 (Roh) | ANB, Nperp_A, Nperp_Pog, MP-FH, NCWR, PHI, TDI | 연속 |
| CBCT 기도 (Roh) | NAV, PAV, PMCA, PAL | 연속 |
| CBCT 골격 (추가) | Occlusal plane (정수 6~19, 단위 ° 추정). 평면 = 좌우 상악 제1대구치 근심협측 교두정 + 앞니 겹침 중간점. 어느 평면과의 각도인지는 미확인 | 연속 |
| | Asymmetry (Chin dev) | **이진 0/1** (방향 정보 없음) |
| CBCT 비상악 | Septal deviation (MSP에서 비중격 최대 휜 점까지 > 2 mm면 1) | 이진 0/1 |
| | Nasal floor cant (좌우 비강저 최하점 연결선과 MSP 각도가 90°±2° 밖이면 1) | 이진 0/1 |
| CBCT 과두 | Condylar angle Axial·Lat × 좌우 (°) | 연속 |
| | Joint space Ant·Sup·Post × 좌우 (mm 추정) | 연속 |
| 임상 | TMJ pain history, Maxillary sinusitis | 이진 0/1 |
| | Nasal obstruction | 순서형 0/1/2 (등급 의미 미확인) |
| 분류 | Type (1~4), Group (1/2, "CII HA") | 코드, 정의 미확인 |
| 촬영 | CBCT Slice 간격 (mm) | 0.39 mm는 85명, P_50은 0.3 mm, P_78은 0.2 mm |

- 이름만 보고 범주형으로 짐작했던 Nasal floor cant는 실제로 0/1 이진 변수였습니다.
- Chin dev도 이진 변수라 방향 정보가 없습니다. 그래서 좌우 비대칭은 크기(|R−L|)로만 비교했습니다.

## 3. 결측 (`cbct_excel_missingness.csv`, `cbct_excel_missing_pattern.csv`)

- NAV(83명 중 25명)와 PAV(58명) 외에는 결측이 없습니다.
- 결측 조합:

| | PAV 있음 | PAV 결측 |
|---|---|---|
| NAV 있음 | 25 | 33 |
| NAV 결측 | 0 | 25 |

  PAV만 있고 NAV가 없는 환자는 없습니다. 측정 범위가 넓은 쪽부터 빠진 형태입니다.
- 촬영 연도별 (`fig_missingness.png`, 연도 정보가 있는 80명):
  - 2013~2021년: NAV는 전원 측정, PAV는 일부만 측정
  - 2022년: NAV만 있음 12명, 둘 다 결측 1명
  - 2023~2025년: 둘 다 결측 23명, NAV만 있음 2명, 둘 다 있음 2명
- 측정된 환자와 결측 환자 비교 (Mann-Whitney):
  - 촬영 연도: 차이가 큽니다 (NAV p<0.001, PAV p<0.001).
  - NCWR: NAV 결측 환자가 더 큽니다 (55.9 대 48.1, p<0.001).
  - 그 밖에 CondLat_R(p=0.013), JS_Sup_R(PAV, p=0.049)이 p<0.05입니다.
  - NCWR 자체가 촬영 연도와 상관(r 0.42)하므로, 이 차이는 연도를 통한 것일 수 있습니다.
  - ANB, MP-FH, PMCA, PAL은 측정 여부에 따라 차이가 없습니다.
- 해석: 결측은 무작위가 아니라 **측정 시기에 따라 갈립니다.** NAV·PAV를 쓰는 분석은 2022년 이전 촬영 환자로 사실상 한정됩니다.

## 4. 분포 (`cbct_excel_descriptive.csv`, `cbct_excel_frequency.csv`, `fig_distributions.png`)

연속 변수, 중앙값 [IQR]:

| 변수 | 중앙값 [IQR] | 범위 |
|---|---|---|
| Occlusal plane | 12 [10–14] | 6–19 |
| CondAx R / L | 26.2 [20.4–35.6] / 26.7 [16.5–33.5] | 9.1–57.8 |
| CondLat R / L | 71.7 [67.5–74.6] / 72.0 [66.9–75.6] | 58.2–82.2 |
| JS Ant R / L | 2.37 [1.91–2.79] / 2.34 [1.80–2.78] | 0.87–4.84 |
| JS Sup R / L | 3.19 [2.62–3.82] / 3.41 [2.84–3.78] | 1.10–5.20 |
| JS Post R / L | 2.14 [1.68–2.69] / 2.41 [1.90–2.85] | 1.10–4.62 |

- 기존 Roh 변수 11개의 분포는 Phase 1 결과와 같습니다. 같은 엑셀 열을 읽는지 코드에서 확인했습니다.
- 범주 빈도 (83명):

| 변수 | 빈도 |
|---|---|
| Chin dev | 1 = 38명 (46%) |
| Septal deviation | 1 = 34명 (41%) |
| Nasal floor cant | 1 = 36명 (43%) |
| Nasal obstruction | 0 / 1 / 2 = 39 / 28 / 16명 |
| Maxillary sinusitis | 1 = 4명 (5%) |
| TMJ pain | 1 = 19명 (23%) |
| Type | 1 / 2 / 3 / 4 = 12 / 53 / 17 / 1명 |
| Group | 1 / 2 = 34 / 49명 |

- Maxillary sinusitis는 4명뿐이어서 따로 분석하기 어렵습니다. Type 4도 1명뿐입니다.

## 5. 과두 좌우 비교 (`cbct_excel_bilateral.csv`, `fig_bilateral.png`)

| 쌍 | 좌우 Spearman r | R−L 중앙값 [IQR] | Wilcoxon p | Chin dev 0 vs 1의 \|R−L\| p |
|---|---|---|---|---|
| Condylar Axial | 0.84 | 1.90 [−3.60–5.55]° | 0.062 | 0.87 |
| Condylar Lat | 0.69 | −1.40 [−2.80–2.15]° | 0.41 | 0.98 |
| JS Ant | 0.48 | 0.01 [−0.39–0.33] mm | 0.92 | 0.89 |
| JS Sup | 0.70 | −0.06 [−0.49–0.34] mm | 0.48 | 0.85 |
| JS Post | 0.61 | −0.14 [−0.57–0.25] mm | **0.010** | 0.86 |

- 다섯 쌍 모두 좌우가 양의 상관이고 중앙값이 비슷합니다. **좌우를 같은 정의로 쟀을 가능성이 높습니다.** 반대 방향으로 쟀다는 단서는 없습니다.
- 좌우 체계적 차이는 JS Post에만 있고, 크기는 0.14 mm로 작습니다.
- **Chin dev는 과두 좌우 차이와 전혀 관련이 없습니다** (모든 쌍 p>0.8). 턱끝 편위와 과두 비대칭이 엑셀 안에서 서로 다른 정보이거나, Chin dev의 판정 기준이 느슨할 수 있습니다.

## 6. 변수 간 관계 (`cbct_excel_correlations.csv`, `fig_correlation.png`)

관심 쌍:

| 쌍 | n | 결과 |
|---|---|---|
| Occlusal plane ↔ MP-FH | 83 | Spearman 0.48 (p<0.001). 교합평면과 하악평면 기울기가 함께 움직임 |
| Septal deviation ↔ Nasal obstruction | 83 | 0.22 (p=0.046). 약한 동반 |
| Maxillary sinusitis ↔ Nasal obstruction | 83 | 0.23 (p=0.037). 상악동염 4명이 모두 코막힘 1 또는 2 |
| Nasal floor cant ↔ Chin dev | 83 | −0.02 (Fisher p=1.0). 관계 없음 |
| Septal deviation ↔ Nasal floor cant | 83 | 0.01 (p=1.0). 관계 없음 |

새 변수와 기존 Roh 변수 사이에서 |r| ≥ 0.4인 쌍:
- **골격:**
  - Occlusal plane: ANB 0.49, Nperp_Pog −0.51, MP-FH 0.48
  - CondAx R/L: ANB 0.49 / 0.45, Nperp_Pog −0.48 / −0.44
- **기도 (n이 작아 잠정):**
  - CondLat_L–PAV 0.69 (n=25)
  - CondLat_R–NAV 0.41 (n=58)
  - CondAx_L–NAV −0.40 (n=58)

새 변수가 기존 변수와 완전히 겹치는 경우(|r| > 0.8)는 같은 변수의 좌우 쌍(Condylar Axial 0.84) 외에는 없습니다.

## 7. QC (`cbct_excel_qc.csv`)

| 종류 | 건수 | 내용 |
|---|---|---|
| 불가능한 값 (관절 공간 ≤ 0, 비율 0~100% 밖) | 0 | |
| 엑셀 내부·DICOM 불일치 | 0 | 엑셀 BMI vs 키·몸무게, Slice 간격 vs DICOM 복셀 |
| 극단값 (robust z > 3.5, 87명 전원 대상) | 9 | BMI: P_35·P_63·P_80 / ANB: P_31 (−8.1, Class III로 주 분석 제외) / PAV: P_5·P_44 / PMCA: P_72 / JS_Ant_L: P_75·P_83 |

- P_44는 CBCT EDA에서도 상기도 부피 이상으로 표시된 환자입니다. 엑셀 PAV 극단값과 같은 방향이라 실제 큰 기도일 가능성이 있습니다.
- P_31과 P_75는 주 분석 대상(Valid_Main)이 아닙니다.

## 8. 앞선 결과와 연결되는 점

- **상기도 오분할과 상악동염:** 상기도 마스크가 상악동 안까지 잡힌 P_22는 엑셀 Maxillary sinusitis(임상 소견)가 0입니다. 상기도 여러 덩어리로 표시된 7명 중 sinusitis가 1인 환자는 P_31 한 명뿐입니다. 따라서 이 오분할은 **상악동염 때문이 아니라 공기로 찬 정상 상악동을 기도로 잡은 분할 오류**로 보입니다 (잠정).
- **과두 변수와 하악 SSM:** 하악 PC1·PC2의 변위가 관절돌기 주변에 몰려 있었습니다. 과두 각도·관절 공간 10개를 하악 PC와 비교하면 그 변위가 해부학적 차이인지 판단하는 근거가 됩니다. 이번 EDA 범위에서는 하지 않았고, 다음 단계 후보입니다.

## 9. 데이터 담당께 물을 것 (추가)

1. Occlusal plane 각도의 기준 평면 (평면 자체는 PPT 14에 있음), 과두 각도의 기준 평면과 단위
2. 관절 공간의 단위와 측정 단면
3. Chin dev의 판정 기준 (Septal deviation·Nasal floor cant는 인수인계 PPT 13에 있음)
4. Nasal obstruction 0/1/2의 의미, Maxillary sinusitis의 진단 근거 (둘 다 임상 소견)
5. NAV·PAV를 2022년 이후 측정하지 않은 이유 (측정 범위, 소프트웨어, 측정자 변경)
6. NCWR이 촬영 연도와 상관되는 이유 (측정자나 방법이 바뀌었는지)
7. Type 1~4와 Group(CII HA 1/2)의 정의
