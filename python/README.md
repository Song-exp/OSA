# OSA Python 코드

`Matlab/` 코드를 Python으로 옮긴 것입니다. 원본 `Data/`, 기존 MATLAB 결과 `Result/`는 읽기만 합니다.

## 설치

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## MATLAB → Python 대응

| MATLAB | Python |
|---|---|
| `PSG_Read.m`, `PSG_readers/*`, `EbmRead.m` | `osa/psg_read.py` (`read_psg`, `read_ebm`, `read_ndf`, `read_edf`) |
| `PSG_Events.m` | `osa/psg_events.py` (`read_events`) |
| `PSG_Plot.m` | `osa/psg_plot.py` (`plot_psg`, 탭 하나 = Figure 하나) |
| `CBCT.m`, `drawCBCTFigure` | `osa/cbct.py` (`read_cbct`, `draw_cbct_figure`) |
| `main_YJ_PSG_All.m`, `PSG_Save.m` | `osa/batch.py` (일괄 처리, h5 저장/읽기) |
| `main_YJ_PSG.m` (한 명 보기) | 아래 "한 명 불러와서 보기" |
| `main_PSG_OpenFigure.m` | 불필요 (결과가 PNG) |

## GUI 뷰어

`viewer_app.py` (또는 루트의 `OSA_viewer.bat`). 사용법과 화면 정보의 출처는 `docs/00_진행상황.md` 4장에 있습니다.

## 사용 (`python/` 폴더에서)

일괄 처리 → `Result_py/<PSG|CBCT>/h5`, `Result_py/<PSG|CBCT>/Figure`

```
..\.venv\Scripts\python -m osa.batch psg
..\.venv\Scripts\python -m osa.batch psg --ids 4 7 12 --overwrite
..\.venv\Scripts\python -m osa.batch cbct
```

한 명 불러와서 보기 (스크립트: `..\.venv\Scripts\python view.py psg 1` / `view.py cbct 12`)

```python
import matplotlib.pyplot as plt
from osa.psg_read import read_psg
from osa.psg_plot import plot_psg

rec = read_psg("../Data/PSG/P_1_PSG_E")
figs, events, stages, plot_data, counts = plot_psg(rec, channels_per_tab=7)
plt.show()
```

저장된 결과 읽기

```python
from osa.batch import load_psg, load_cbct
r = load_psg("../Result_py/PSG/h5/P_1.h5")
r["Signals"]["SpO2"]["time"], r["Signals"]["SpO2"]["data"], r["Events"]
c = load_cbct("../Result_py/CBCT/h5/P_1.h5")
c["RescaledVolume"].shape   # (row, column, slice)
```

## 검증

- `verify_matlab.py psg|cbct [ID ...]`: 기존 MATLAB 결과(`Result/*/mat`)와 신호값, 시간, 이벤트, 수면단계, CBCT voxel을 비교합니다. 결과는 `Result_py/verify_*.csv`에 저장됩니다.
- `test_roundtrip.py`: h5로 저장했다가 다시 읽은 값이 원본과 같은지 확인합니다.

## MATLAB 버전과 달라진 점

- **다른 날 채널 제외 규칙:** MATLAB은 `P_43_PSG_E` 폴더 이름에 고정돼 있었는데, 환자 번호가 바뀌면서 작동하지 않았습니다. 이제는 채널 시작 시각이 전체 중앙값과 12시간 넘게 다르면 제외합니다. 현재 해당하는 환자는 P_23(채널 7개)뿐입니다.
- **저장 방식:** 신호 시간축은 segments에서, CBCT rescaled 볼륨은 slope/intercept에서 다시 계산하므로 저장하지 않습니다. 표시용 PlotData도 저장하지 않습니다.
- **EDF 리더:** 현재 `Data/`에 EDF 파일이 없어 실제 데이터로는 검증하지 못했습니다.

## CBCT 자동 랜드마크 (ALI-CBCT, Slicer 없이)

설치 (한 번만):
```
.venv\Scripts\python -m pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools.git third_party/sadt
git -C third_party/sadt sparse-checkout set ALI_CBCT ADT/ADTLib
```
- 확인한 커밋은 `9448c0d`(2026-10-16)입니다.
- 모델은 https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/tag/v0.1-v2.0_models 에서 받습니다.
  - 받을 파일: Cranial_Base, Upper_Bones_v2, Lower_Bones_1, Lower_Bones_2
  - `models/ali_cbct/`에 압축을 풉니다.
- `third_party/`와 `models/`는 git에서 제외됩니다.

실행:
```
python export_nrrd.py 5        # DICOM → NRRD (환자 정보 없음)
python run_ali.py 5            # 랜드마크 → Result_py/CBCT/landmarks/
python check_landmarks.py 5    # 좌표표, 해부학 점검, 검수 그림
```
