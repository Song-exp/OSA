"""AMASSS 자동 분할(하악·상악·두개저·상기도 등)을 3D Slicer 없이 실행합니다.

코드: third_party/sadt/AMASSS_CLI (nnU-Net v2), 모델: models/amasss/ (AMASSS_Models.zip 압축 해제)
입력: export_nrrd.py가 만든 Result_py/CBCT/nrrd/P_n.nrrd
출력: Result_py/CBCT/seg/P_n_seg_<구조>.nrrd (구조별 따로, 원본과 같은 격자)
구조 코드: MAND 하악, MAX 상악, CB 두개저, UAW 상기도, CV 경추, SKIN 피부 (치아·하악관은 모델 없음)
사용: python run_amasss.py 5 [7 ...] [--struct MAND,MAX,UAW,CB]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_ali import ROOT, SADT, ascii_path

MODELS = ROOT / "models/amasss"
NRRD, OUT = ROOT / "Result_py/CBCT/nrrd", ROOT / "Result_py/CBCT/seg"


def model_root() -> Path:
    """압축 해제 위치가 한 단계 더 들어가 있을 수 있어 구조 폴더(MAND)가 있는 곳을 찾음."""
    return next((p.parent for p in MODELS.rglob("MAND") if p.is_dir()), MODELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--struct", default="MAND,MAX,UAW,CB")
    a = ap.parse_args()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SADT / "AMASSS_CLI"), str(SADT / "ADT")])}
    OUT.mkdir(parents=True, exist_ok=True)
    for pid in a.ids:
        scan = NRRD / f"P_{pid}.nrrd"
        assert scan.exists(), f"먼저 export_nrrd.py {pid}"
        tmp = ROOT / f"Result_py/CBCT/amasss_tmp/P_{pid}"
        cmd = [sys.executable, str(SADT / "AMASSS_CLI/AMASSS_CLI.py"), str(ascii_path(scan)), str(model_root()),
               a.struct, "SEPARATE", "false", "false", str(ascii_path(OUT)), "5", "seg", str(ascii_path(tmp)),
               "false", "false"]
        print(f"P_{pid}: {a.struct}", flush=True)
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
