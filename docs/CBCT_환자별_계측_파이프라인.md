# CBCT 환자별 계측·정확도 개선 파이프라인

작성: 2026-11-07. 환자가 추가되어도 같은 처리·QC 기준을 적용하기 위한 v2 실행 경로다. 원본과 기존 `Result_py/eval` 평가 결과는 변경하지 않는다.

2026-11-08 상태 정리: [작업 현황과 다음 순서](CBCT_작업현황_2026-11-08.md). 현재는 촬영 조건을 맞춘 비교 기준을 사용하며, 촬영 유형별로 다른 전처리나 자동 보정 규칙을 확정한 상태는 아니다.

## 범위와 현재 상태

- 실행 단위는 환자 1명이다. 그 환자의 입력 확인 → NRRD → ALI → 분할 → 계측 → QC → 엑셀 대조를 마친 뒤 다음 환자로 넘어간다.
- 기존 `batch_cbct_pipeline.py`는 변환·추론을 순차 처리하지만 계측·QC·개별 결과까지 묶지는 않는다. 새 진입점은 `python/cbct_patient_pipeline.py`다.
- 현재 구현된 엑셀 대조 변수는 ANB, Nperp-A, Nperp-Pog, MP-FH(Go–Me) 4개다. TDI·PHI·NCWR·NAV·PAV·PMCA·PAL은 아직 완료되지 않았으며 결과에 미구현 목록으로 표시한다.
- 이 구조를 만들었다고 정확도 향상이 입증된 것은 아니다. 첫 개선 대상은 기존 Po 실패와 기준면 오류이며, 실제 영상 검수 후 좌표를 수정하고 별도 결과로 비교한다.
- 사용자 지시에 따라 상세 문의 32명만 임시 제외해 55명을 처리 대상으로 고정했다. 명단은 `Result_py/CBCT/patient_pipeline/v2/cohort.csv`, 근거·문서 해시는 `cohort_selection.json`에 저장했다. M01~M10의 전체/변수별 결측 목록으로 제외 범위를 넓히지 않았다.
- 기존 산출물이 있는 포함 환자 54명으로 결과변수·엑셀 오차를 사용하지 않는 거리 QC 기준을 `qc_reference.json`에 고정했다. 개발용 탐색 기준이며 독립 검증 성능이나 임상 정상 범위가 아니다. 환자 추가 시 같은 기준을 재사용한다.
- 최초 실행은 54명 계측·대조 완료, P_78은 기존 NRRD/랜드마크/분할이 없어 미처리로 기록됐다. P_78도 Include=true이며 자동으로 제외하지 않았다. 완료 54명 중 거리 QC 표시 6명(P_7·P_12·P_16·P_20·P_27·P_37), 표시 없음 48명이다. 표시 없는 경우도 육안 검수 대기다.

## 한 명씩 실행과 환자 추가

프로젝트 루트에서 아래 명령을 사용한다. 기본 동작은 계획 출력이며 `--run`을 명시해야 실행한다.

```powershell
# 예시: 계획만 확인. 실제 포함 환자 지정과는 별개인 사용 예시다.
.venv/Scripts/python.exe python/cbct_patient_pipeline.py --ids 5

# 확정한 CSV 명단으로 계획 확인
.venv/Scripts/python.exe python/cbct_patient_pipeline.py --manifest Result_py/CBCT/patient_pipeline/v2/cohort.csv

# 기준을 고정한 후 실행. 결과가 없는 변환/추론 단계는 별도로 허용한다.
.venv/Scripts/python.exe python/cbct_patient_pipeline.py --manifest Result_py/CBCT/patient_pipeline/v2/cohort.csv --qc-reference Result_py/CBCT/patient_pipeline/v2/qc_reference.json --run

# 문의가 해결된 환자 한 명을 명단에 추가하거나 명시 실행
.venv/Scripts/python.exe python/cbct_patient_pipeline.py --ids 9 --qc-reference Result_py/CBCT/patient_pipeline/v2/qc_reference.json --run --infer-missing
```

명단의 필수 열은 `Patient,Include,Reason`이다. Include는 true/false/1/0만 허용하고 중복 번호는 오류 처리한다. 새 환자는 같은 QC 기준 파일·코드·모델 조건으로 추가해야 한다. 명단이 바뀌어도 QC 기준을 다시 추정하지 않는다. 확정된 명단만 실제 처리에 사용한다.

## 재개·기록·입력 오류 방지

- NRRD는 헤더·차원·gzip footer의 예상 크기를 검사한다. 이는 전체 CRC 검사나 분할의 해부학적 정확도 검증을 대신하지 않는다.
- 랜드마크는 파일 개수가 아닌 필수 16개 점의 존재, 좌표계, 유효 좌표, 위치 확정 여부, 중복 좌표 충돌을 검사한다. RAS는 LPS로 변환하고 미확인 좌표계는 거부한다.
- 실패한 단계 이후에는 같은 환자의 후속 처리를 하지 않는다. 실패는 환자별 status에 남기고 다음 환자로 진행한다.
- 계측 입력의 SHA256, 원본 DICOM 파일 목록·크기·수정시각의 서명, 원본 시리즈 UID, 분할 파일 상태, QC 기준, 코드, 엑셀, 수정 좌표의 지문을 기록한다. DICOM 서명은 픽셀 전체 해시가 아님을 구분한다.
- 지문이 같으면 기존 계측을 재사용한다. 변경되면 계측을 다시 수행하고 이전 결과는 환자별 history에 보존한다. 원본 DICOM 변경은 기존 파생물을 그대로 재사용하지 않도록 차단한다.
- 과거 추론 영수증이 없는 기존 ALI/분할을 재사용하면 당시 모델·환경을 확정할 수 없다는 한계를 결과에 표시한다. 모델을 바꾸는 재추론은 별도 버전·검증이 필요하다.

## QC와 정확도 개선

1. 개발에 사용할 환자를 먼저 정한다. 결과변수나 엑셀과의 오차로 환자를 골라 QC 기준을 만들지 않는다.
2. 개발 표본의 랜드마크 간 거리로 MAD 기준을 고정한다. 이는 큰 실패를 표시하는 보조 규칙이며 정상 해부학적 변이·미세한 위치 오류를 판별하는 정답은 아니다.
3. 기준 파일은 덮어쓰지 않는다. 기준 변경은 새 파일과 방법 버전으로 기록한다.

```powershell
# 개발 표본 번호를 명시해야 하며 최소 10명이 필요하다.
# 아래 명령의 번호는 예시다. 실제 표본 확정 후 사용한다.
.venv/Scripts/python.exe python/cbct_patient_pipeline.py --ids 1 2 3 4 5 6 7 8 9 10 --freeze-qc --qc-reference Result_py/CBCT/patient_pipeline/v2/qc_reference.json
```

4. Po를 쓰는 FH와 MSP에 문제가 있으면 MP-FH·Nperp뿐 아니라 MSP 투영 ANB도 영향을 받을 수 있다. QC 실패 값은 진단용으로 보존하고 분석용 확정값으로 자동 채택하지 않는다. 표시가 없어도 육안 검수 전 상태는 `qc_pass_pending_visual_review`다.
5. 원본 영상에서 검수한 수정 좌표만 별도 JSON으로 입력한다. 엑셀과 맞추기 위한 회전·평균 형태 치환을 자동 적용하지 않는다. 환자번호, LPS 좌표계, 검수자, 수정 이유를 필수로 기록한다.
6. 같은 환자의 automatic/reviewed 결과와 엑셀 차이를 비교한다. 기존 평가 성능을 덮어쓰지 않는다. 수정에 사용하지 않은 검증 사례에서 개선이 재현돼야 정확도 향상으로 보고한다.

수정 JSON 형식은 아래와 같다. `points` 값은 실제로 영상 검수한 좌표를 넣어야 한다.

```json
{
  "patient": "P_16",
  "coordinate_system": "LPS",
  "reviewer": "검수자",
  "reason": "원본 영상에서 확인한 수정 근거",
  "points": {}
}
```

환자별 파일을 폴더에 저장하고 `--corrections-dir <폴더>`를 지정한다. 기존 자동 좌표 파일은 수정하지 않는다.

## 결과

`Result_py/CBCT/patient_pipeline/v2/`

- `P_n/status.json`: 환자별 단계·실패·재개 상태.
- `P_n/result.json`: 입력 지문, 자동/검수 좌표별 계측, QC, 엑셀 차이, 미구현 변수.
- `P_n/history/`: 입력이나 수정 좌표가 바뀌기 전 결과.
- `P_n/*.log`: 해당 환자의 새 변환·추론 로그.
- `last_run.csv`: 마지막 실행 상태.
- `last_selection.json`: 실행 명단과 그 SHA256.
- `selected_measures_diagnostic.csv`: 이번 명단의 진단용 계측 대조. QC와 검수 상태를 무시해 최종 분석값으로 사용하지 않는다.

검증 명령:

```powershell
.venv/Scripts/python.exe -m unittest discover -s python -p test_cbct_patient_pipeline.py -v
```

테스트는 합성 데이터로 재개, 불완전 산출물 차단, 좌표계, 환자별 수정, 원본 변경 차단, 명단 제외를 검증한다. 실제 환자에서 계측 정확도가 개선됐다는 검증과는 다르다.

## Po 영상 검토 추가 (2026-11-07)

`python/cbct_landmark_review.py`는 명단에서 Include인 환자 한 명씩 원본 NRRD와 자동 랜드마크를 읽어 검토 자료를 만든다. 제외 환자는 영상을 읽기 전에 차단한다. 본 계측 파이프라인의 result.json, 자동 좌표, 기존 평가 결과를 변경하지 않는다.

```powershell
# 먼저 자동 위치와 주변 시상면을 확인한다. 엑셀은 이 단계에서 읽지 않는다.
.venv/Scripts/python.exe python/cbct_landmark_review.py --patient 16 --focus RPo

# 영상에서 정한 좌표 후보와 근거를 입력하면 비교·민감도·HTML을 추가한다.
# 아래 좌표는 P_16에만 해당하는 전문가 검수 전 후보다.
.venv/Scripts/python.exe python/cbct_landmark_review.py --patient 16 --focus RPo --candidate -58.11 20.67 104.13 --reason "Original image superior canal margin candidate; requires expert review"
```

다른 환자 추가 시 cohort.csv의 Include를 먼저 반영하고 동일한 명령을 환자번호에 맞춰 실행한다. `--focus LPo`도 지원한다. 후보 없이 탐색만 실행하면 `<focus>_search_mpr.png`, `<focus>_search_evidence.json`을 저장해 기존 후보 검토 자료와 구분한다.

산출물은 `P_n/review/<focus>_review.html`, `_mpr.png`, `_sagittal_search.png`, `_evidence.json`, `_candidate_comparison.csv`, `_sensitivity_grid.csv`다. HTML에서 이미지를 클릭하면 확대해서 볼 수 있다. 입력 해시, 원본 시리즈 식별자, LPS 좌표, 후보 선정 근거, 거리 QC, 좌표별 ±1 mm의 27개 격자 결과를 기록한다. 이 격자는 신뢰구간이나 해부학적 허용 오차가 아니다.

Po 탐색에 쓰는 반대편 반사 위치는 N·S·Ba·ANS·PNS로 구한 정중면을 사용한다. 잘못된 Po로 만든 FH나 엑셀 오차로 선택한 평균 형태에 의존하지 않는다. 반사 위치는 탐색 보조이며 수정값으로 자동 채택하지 않는다. Po 정의는 제공된 `cbct_인자추출.pdf` 4쪽 Table 1의 외이도 외측·최상방 경계다. AI가 선택한 영상 후보는 전문가 승인 좌표와 구분한다.

첫 두 환자 결과와 남은 검토 사항: [Po_1차_영상검토.md](../Result_py/CBCT/patient_pipeline/v2/Po_1차_영상검토.md).

추가 검증:

```powershell
.venv/Scripts/python.exe -m unittest discover -s python -p test_cbct_landmark_review.py -v
```

정중면 반사에서 잘못된 Po가 제외되는지, 실제 mm 좌표와 시상 단면·마커가 일치하는지, 제외 환자를 읽기 전에 차단하는지 검사한다.

## Po 주변 원본 복셀 비교 추가 (2026-11-07)

P_20 LPo 영상 후보와 P_16·P_37의 기존 후보를 대상으로 원본 밝기 분포·국소 경계·공기 유사 영역 연결 구조를 비교했다. [그림 보고서](../Result_py/CBCT/patient_pipeline/v2/po_pixel_audit/v1/index.html), [해석과 한계](../Result_py/CBCT/patient_pipeline/v2/po_pixel_audit/v1/해석_요약.md)를 참조한다.

`python/cbct_po_pixel_audit.py`는 Include 환자를 한 명씩 읽고 환자별 지표·NPZ 패치를 저장한다. 입력 영상·랜드마크·명단·후보·코드·EDA의 해시가 같으면 캐시를 재사용한다. 기존 계측 54명에서 111개 ROI(자동 양측 108개 + 후보 3개)를 계산했으며, P_78은 기존 파생물·계측 결과가 없어 처리하지 않았다. 참고군은 엑셀 오차와 무관하게 거리 QC 통과 48명으로 선정했다. 참고군도 Po 정답 검수 전 상태다.

```powershell
# 환자 1명 처리/재개
.venv/Scripts/python.exe python/cbct_po_pixel_audit.py --patient 20

# Include 명단을 순서대로 처리한 뒤 비교 보고서 생성
.venv/Scripts/python.exe python/cbct_po_pixel_audit.py --run

# 등록된 사례의 ROI 중심 이동 민감도 (후보 좌표 자체는 변경하지 않음)
.venv/Scripts/python.exe python/cbct_po_roi_sensitivity.py --patient 20

# 복셀·좌표 변환·제외 명단 방어 검증
.venv/Scripts/python.exe -m unittest discover -s python -p test_cbct_po_pixel_audit.py -v
```

환자 추가 시 Include 명단과 기존 계측 결과가 필요하며, 참고군 포함 여부는 같은 QC 기준으로 정한다. 현재 특별 사례는 pixel audit의 CASES에 환자번호와 검토할 Po를 등록하고 해당 영상 후보 evidence JSON을 준비한다. 추가 환자에서도 같은 반경·해상도·전처리·촬영조건 매칭을 유지한다. 참고군 추가로 기술 통계가 달라지면 새 실행 manifest로 구분하며, 이를 기존 고정 QC 기준의 재학습으로 사용하지 않는다.

픽셀 분포의 차이는 검출 실패 원인의 탐색 단서다. 후보·참고군 좌표, 국소 영상 품질, 개인차와 모델 탐색 실패를 분리하지 않은 상태에서 특정 골격 이상이 원인이라고 확정하지 않는다. P_20은 잘못된 자동 위치의 히스토그램이 오히려 참고군에 더 비슷해, 밝기 분포만으로 맞는 구조를 고를 수 없음을 보여준다.
