"""CBCT 측정 파이프라인용 파일럿(개발) / 평가 환자 분할. 측정 전에 한 번만 실행하고 결과를 고정합니다.

대상: 엑셀 ANB·MP-FH·TDI가 있고 CBCT 본인 확인(생년월일·성별 일치)된 환자
  - 제외: P_39 (본인 영상 없음)
  - FOV 확인용(어느 세트에도 안 넣음): P_78 (10.2 cm)
  - 파일럿 고정 포함: P_50 (15.4 cm)
파일럿 12명 = P_50 + 시상(Class)×수직 층별 무작위 11명. 아래 조건을 만족할 때까지 seed를 올려 다시 뽑습니다.
  - TDI 경계 근처(−3.5~−1 mm) 3명 이상, 엑셀 PAV 있음 3명 이상, NAV 있음 6명 이상
결과: Result_py/cbct_split.csv (Patient, Set), 화면에 seed와 구성 출력
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED, N_PILOT = 20261024, 12


def eligible() -> pd.DataFrame:
    x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None).iloc[3:]
    x = x[x[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    d = pd.DataFrame({"P": x[0].astype(int).values,
                      **{k: pd.to_numeric(x[c], errors="coerce").values
                         for k, c in {"ANB": 22, "MPFH": 25, "TDI": 28, "NAV": 29, "PAV": 30}.items()}})
    q = pd.read_csv(ROOT / "Result_py/quality_flags.csv")
    d["Identity"] = d.P.map(dict(zip(q.Patient.str[2:].astype(int), q.CBCT_Identity)))
    d["Sag"] = np.select([d.ANB >= 4, d.ANB > 0], ["II", "I"], "III")
    d["Ver"] = np.select([d.MPFH >= 30, d.MPFH >= 25], ["hyper", "normo"], "hypo")
    ok = d[["ANB", "MPFH", "TDI"]].notna().all(axis=1) & (d.Identity == "match") & ~d.P.isin([39, 78])
    return d[ok].reset_index(drop=True)


def draw(d: pd.DataFrame, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    pool = d[d.P != 50]
    strata = pool.groupby(["Sag", "Ver"]).P.apply(list)
    # 층 크기에 비례해 배분 (최대 잉여 방식), 합 = N_PILOT - 1
    share = strata.map(len) / len(pool) * (N_PILOT - 1)
    k = np.floor(share).astype(int)
    k[(share - k).sort_values(ascending=False).index[: N_PILOT - 1 - k.sum()]] += 1
    return {50} | {p for s, n in k.items() for p in rng.choice(strata[s], n, replace=False)}


def good(d: pd.DataFrame, pick: set[int]) -> bool:
    s = d[d.P.isin(pick)]
    return (s.TDI.between(-3.5, -1).sum() >= 3) and (s.PAV.notna().sum() >= 3) and (s.NAV.notna().sum() >= 6)


def main():
    d = eligible()
    seed = BASE_SEED
    while not good(d, pick := draw(d, seed)):
        seed += 1
    d["Set"] = np.where(d.P.isin(pick), "pilot", "eval")
    out = pd.concat([d[["P", "Set"]], pd.DataFrame({"P": [78, 39], "Set": ["fov_check", "excluded"]})])
    out = out.assign(Patient="P_" + out.P.astype(str)).sort_values("P")[["Patient", "Set"]]
    out.to_csv(ROOT / "Result_py/cbct_split.csv", index=False, encoding="utf-8-sig")

    s = d[d.Set == "pilot"]
    print(f"대상 {len(d)}명 → 파일럿 {len(s)}, 평가 {len(d) - len(s)} (seed {seed}, 시도 {seed - BASE_SEED + 1}회)")
    print("파일럿:", ", ".join(f"P_{p}" for p in sorted(s.P)))
    print("층 구성:", s.groupby(["Sag", "Ver"]).size().to_dict())
    print(f"TDI 경계 근처 {s.TDI.between(-3.5, -1).sum()}, NAV 있음 {s.NAV.notna().sum()}, PAV 있음 {s.PAV.notna().sum()}")
    print("제외(엑셀 값 없음/본인 불확실):", sorted(set(range(1, 88)) - set(d.P) - {39, 78}))


if __name__ == "__main__":
    main()
