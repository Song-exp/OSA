"""환자 한 명의 PSG 또는 CBCT를 창으로 띄워 봅니다. (MATLAB main_YJ_PSG.m에 해당)

사용 (python 폴더에서):
  ..\\.venv\\Scripts\\python view.py psg 1
  ..\\.venv\\Scripts\\python view.py psg 1 --from 2013-08-15T22:51 --to 2013-08-15T23:17
  ..\\.venv\\Scripts\\python view.py cbct 12
PSG는 탭(채널 7개)마다 창이 하나씩 뜹니다. 창 아래 도구 막대로 확대·이동합니다.
CBCT는 단면 3개 + 3D 표면이며, 3D 패널은 마우스로 돌릴 수 있습니다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["psg", "cbct"])
    ap.add_argument("id", type=int, help="환자 번호 (P_n의 n)")
    ap.add_argument("--from", dest="start", help="PSG 구간 시작, 예: 2013-08-15T22:51")
    ap.add_argument("--to", dest="stop", help="PSG 구간 끝")
    a = ap.parse_args()
    folder = next((ROOT / "Data" / a.kind.upper()).glob(f"P_{a.id}_*"), None)
    assert folder, f"P_{a.id} 폴더가 Data/{a.kind.upper()}에 없습니다."
    if a.kind == "psg":
        from osa.psg_plot import plot_psg
        from osa.psg_read import read_psg
        window = (a.start, a.stop) if a.start and a.stop else None
        plot_psg(read_psg(folder), window=window, channels_per_tab=7)
    else:
        from osa.cbct import draw_cbct_figure, read_cbct
        draw_cbct_figure(read_cbct(folder), a.id)
    plt.show()


if __name__ == "__main__":
    main()
