"""ALI-CBCT 자동 랜드마크를 3D Slicer 없이 실행합니다.

코드: third_party/sadt (SlicerAutomatedDentalTools sparse clone: ALI_CBCT, ADT/ADTLib)
모델: models/ali_cbct/<그룹>/<랜드마크>/<스케일>/*.pth (GitHub release v0.1-v2.0_models 압축 해제)
입력: export_nrrd.py가 만든 Result_py/CBCT/nrrd/P_n.nrrd
출력: Result_py/CBCT/landmarks/P_n_*.mrk.json (Slicer 마크업 형식, LPS mm)
실행 인자는 Slicer 모듈(ALI/ALI_Method/CBCT.py)의 기본값과 같습니다.
사용: python run_ali.py 5 [7 9 ...] [--lm Ba S N]  (--lm 없으면 모델이 있는 랜드마크 전부)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SADT = ROOT / "third_party/sadt"
MODELS = ROOT / "models/ali_cbct"
NRRD, OUT = ROOT / "Result_py/CBCT/nrrd", ROOT / "Result_py/CBCT/landmarks"


def ascii_path(p: Path) -> Path:
    """ITK(SimpleITK)는 한글 경로를 열지 못합니다(fopen 'Illegal byte sequence').
    Result_py/CBCT를 가리키는 영문 경로 junction을 만들어 그 경로로 넘깁니다. 파일은 프로젝트 안에 그대로 있습니다."""
    if str(p).isascii():
        return p
    link, target = Path(r"C:\Users\Public\osa_cbct"), ROOT / "Result_py/CBCT"
    if not link.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    return link / p.relative_to(target)


def available() -> list[str]:
    """GetBrain과 같은 규칙: .pth의 두 단계 위 폴더 이름이 랜드마크."""
    return sorted({p.parent.parent.name for p in MODELS.rglob("*.pth")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--lm", nargs="*")
    a = ap.parse_args()
    lms = a.lm or available()
    assert lms, f"모델이 없습니다: {MODELS}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SADT / "ALI_CBCT"), str(SADT / "ADT")])}
    OUT.mkdir(parents=True, exist_ok=True)
    for pid in a.ids:
        scan = NRRD / f"P_{pid}.nrrd"
        assert scan.exists(), f"먼저 export_nrrd.py {pid}"
        tmp = ROOT / f"Result_py/CBCT/ali_tmp/P_{pid}"
        cmd = [sys.executable, str(SADT / "ALI_CBCT/ALI_CBCT.py"), str(ascii_path(scan)), str(MODELS),
               ",".join(f"'{x}'" for x in lms), str(ascii_path(OUT)), str(ascii_path(tmp)), "false",
               "[1,0.3]", "[1,1]", "[64,64,64]", "10"]
        print(f"P_{pid}: 랜드마크 {len(lms)}개", flush=True)
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
