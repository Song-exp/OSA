"""TotalSegmentator 공개 과제(Apache-2.0)를 CBCT에 실행. 기존 seg 결과는 건드리지 않는다.

입력: Result_py/CBCT/nrrd/P_n.nrrd → totalseg_work/P_n.nii.gz
출력: Result_py/CBCT/seg_totalseg/<task>/P_n.nii.gz (다중 라벨, 라벨 이름은 같은 폴더 labels.json)
가중치: Result_py/CBCT/totalseg_home (한글 경로 회피)
사용: python run_totalseg.py head_glands_cavities 5 7 30
"""
import json
import os
import sys

import SimpleITK as sitk

from cbct_patient_pipeline import CB
from run_ali import ascii_path

os.environ.setdefault("TOTALSEG_HOME_DIR", str(ascii_path(CB/"totalseg_home")))
from totalsegmentator.map_to_binary import class_map    # noqa: E402  (환경 변수 뒤에 불러야 함)
from totalsegmentator.python_api import totalsegmentator    # noqa: E402


def main():
    task, ids = sys.argv[1], sys.argv[2:]
    work, out = CB/"totalseg_work", CB/f"seg_totalseg/{task}"; work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    (out/"labels.json").write_text(json.dumps(class_map[task], ensure_ascii=False, indent=1), encoding="utf-8")
    for pid in ids:
        dst = out/f"P_{pid}.nii.gz"
        if dst.exists(): continue
        src = work/f"P_{pid}.nii.gz"
        if not src.exists(): sitk.WriteImage(sitk.ReadImage(str(ascii_path(CB/f"nrrd/P_{pid}.nrrd"))), str(ascii_path(src)), useCompression=True)
        totalsegmentator(str(ascii_path(src)), str(ascii_path(dst)), task=task, ml=True, device="gpu", quiet=True)
        print(f"P_{pid} done", flush=True)


if __name__ == "__main__":
    main()
