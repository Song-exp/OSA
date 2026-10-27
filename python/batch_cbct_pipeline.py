"""여러 환자에 CBCT 자동 처리를 차례로 실행: NRRD 변환 → ALI 랜드마크 → AMASSS 분할.

이미 결과가 있는 단계는 건너뛰므로, 중간에 멈춰도 다시 실행하면 이어서 합니다.
한 환자가 실패해도 다음 환자로 넘어가고, 실패는 로그에 남깁니다.
사용: python batch_cbct_pipeline.py --set eval      (Result_py/cbct_split.csv의 세트)
      python batch_cbct_pipeline.py 1 2 3
로그: Result_py/CBCT/batch_pipeline.log
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "Result_py/CBCT"
LMS = "N S Ba RPo LPo ROr LOr ANS PNS A B Pog Me Gn RGo LGo".split()
STRUCTS = ["MAND", "MAX", "UAW", "CB"]


def run(args, log):
    r = subprocess.run([sys.executable, *args], cwd=Path(__file__).parent, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode:
        log.write(f"    실패: {' '.join(args)}\n{r.stderr[-1500:]}\n")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--set")
    a = ap.parse_args()
    ids = a.ids
    if a.set:
        s = pd.read_csv(ROOT / "Result_py/cbct_split.csv")
        ids = s[s.Set == a.set].Patient.str[2:].astype(int).tolist()
    with open(CB / "batch_pipeline.log", "a", encoding="utf-8") as log:
        for n, p in enumerate(ids, 1):
            t0 = time.time()
            steps = []
            if not (CB / f"nrrd/P_{p}.nrrd").exists():
                steps.append(("nrrd", ["export_nrrd.py", str(p)]))
            if len(list((CB / "landmarks").glob(f"P_{p}_lm_Pred_*.mrk.json"))) < 3:
                steps.append(("ali", ["run_ali.py", str(p), "--lm", *LMS]))
            todo = [s for s in STRUCTS if not (CB / f"seg/P_{p}_seg_{s}.nrrd").exists()]
            if todo:
                steps.append(("amasss", ["run_amasss.py", str(p), "--struct", ",".join(todo)]))
            ok = True
            for name, args in steps:
                ok = run(args, log) and ok
                if name == "nrrd" and not ok:
                    break
            msg = f"[{n}/{len(ids)}] P_{p}: {'완료' if ok else '일부 실패'} ({', '.join(s for s, _ in steps) or '이미 있음'}) {time.time() - t0:.0f}s"
            log.write(msg + "\n")
            log.flush()
            print(msg, flush=True)


if __name__ == "__main__":
    main()
