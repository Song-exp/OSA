"""엑셀 추가 변수(cbct_extra: 턱끝 편위·비중격 편위·비강저 경사·교합평면)와 과두 각(cbct_condyle)을 동결 54명 밖의 환자에게 같은 규칙으로 계산.

기존 extra_review/v1·condyle_review/v1 결과는 읽기만 하고, 새 환자 결과와 합친 85명 표를 restore_ext/v1에 쓴다.
대구치 위치는 cbct_molar_section.measure로 그 환자 것을 계산한다(동결 molars.csv는 54명분).
사용: python cbct_extra_condyle_extend.py            (새 환자 전부)
      python cbct_extra_condyle_extend.py --check 3 28  (동결 환자 재계산 → 기존 CSV와 같은지)
"""
import gc
import sys

import numpy as np
import pandas as pd

import cbct_condyle as cd
import cbct_extra as ex
import cbct_molar_section as mo
from cbct_accuracy_experiments import wait_ram
from cbct_evidence_figs import excel_col
from cbct_patient_pipeline import OUT

EXT = OUT/"restore_ext/v1"


def finish_extra(d, xl):
    """cbct_extra.main의 파생 열과 같다."""
    d = d.copy(); d["Set"] = ["pilot" if int(p[2:]) in mo.PILOT else "eval" for p in d.Patient]
    for k in ex.XL: d[k+"_xl"] = [xl[k].get(int(p[2:]), np.nan) for p in d.Patient]
    d["Septal_auto"] = (d.SDp_mm > ex.SEPTAL_MM).where(d.SDp_mm.notna()); d["NFcant_auto"] = (d.NF_cant_deg > ex.CANT_DEG).where(d.NF_cant_deg.notna())
    d["Chin_auto2"] = (d.Chin_dev_mm > 2).astype(float); d["Chin_auto4"] = (d.Chin_dev_mm > 4).astype(float)
    return d


def finish_condyle(d, xl):
    d = d.copy(); d["Set"] = ["pilot" if int(p[2:]) in mo.PILOT else "eval" for p in d.Patient]
    for k in cd.XL: d[k+"_xl"] = [xl[k].get(int(p[2:]), np.nan) for p in d.Patient]
    return d


def one(pid):
    wait_ram(4, f"P_{pid}"); molars = pd.DataFrame(mo.measure(pid))
    e, c = ex.measure(pid, molars), cd.measure(pid); gc.collect(); return e, c


def main():
    old_e = pd.read_csv(OUT/"extra_review/v1/extra.csv", encoding="utf-8-sig")
    old_c = pd.read_csv(OUT/"condyle_review/v1/condyle.csv", encoding="utf-8-sig")
    xe = {k: excel_col(c) for k, c in ex.XL.items()}; xc = {k: excel_col(c) for k, c in cd.XL.items()}
    if sys.argv[1:2] == ["--check"]:
        for pid in map(int, sys.argv[2:]):
            e, c = one(pid); p = f"P_{pid}"
            for new, old in ((pd.Series(e), old_e.set_index("Patient").loc[p]), (pd.Series(c), old_c.set_index("Patient").loc[p])):
                num = [k for k in new.index if k in old.index and isinstance(new[k], (int, float, np.floating)) and pd.notna(old[k])]
                diff = {k: (round(float(new[k]), 4), round(float(old[k]), 4)) for k in num if not np.isclose(float(new[k]), float(old[k]), atol=1e-4)}
                print(p, "비교", len(num), "개 값, 다른 값:", diff or "없음")
        return
    done = {p.parent.name for p in OUT.glob("P_*/result.json")}
    new_ids = sorted(done-set(old_e.Patient), key=lambda p: int(p[2:])); E, C, failed = [], [], {}
    for p in new_ids:
        print(p, end=" ", flush=True)
        try: e, c = one(int(p[2:])); E.append(e); C.append(c)
        except Exception as err: failed[p] = f"{type(err).__name__}: {err}"
    print()
    ae = pd.concat([old_e, finish_extra(pd.DataFrame(E), xe)], ignore_index=True)
    ac = pd.concat([old_c, finish_condyle(pd.DataFrame(C), xc)], ignore_index=True)
    for d in (ae, ac): d.insert(1, "source", ["v1(동결 54명)" if p in set(old_e.Patient) else "restore_ext" for p in d.Patient])
    ae.to_csv(EXT/"extra_all.csv", index=False, encoding="utf-8-sig"); ac.to_csv(EXT/"condyle_all.csv", index=False, encoding="utf-8-sig")
    print("extra", ae.Patient.nunique(), "명 | condyle", ac.Patient.nunique(), "명 | 실패", failed or "없음")
    print("결측(새 환자):", {k: int(pd.DataFrame(E)[k].isna().sum()) for k in ("SDp_mm", "NF_cant_deg", "OP_FH_msp", "Chin_dev_mm") if k in pd.DataFrame(E)},
          {k: int(pd.DataFrame(C)[k].isna().sum()) for k in ("Lat_R", "Lat_L", "Axial_R", "Axial_L") if k in pd.DataFrame(C)})


if __name__ == "__main__":
    main()
