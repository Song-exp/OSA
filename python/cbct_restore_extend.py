"""동결된 54명 이후 계측한 환자에게 11개 복원 지표를 같은 규칙으로 계산하고, 87명 기준 복원값·엑셀값 비교표를 만든다.

기존 결과(fh_repair/v2, tdi/phi_ncwr/nav/pav/pmca_review/v1, eb_review/v2)와 그림은 읽기만 한다.
새 환자에게 쓰는 고정값: FH 보정 기준 fh_repair/v1/reference.json, QC 기준 qc_reference.json,
Eb 설골 보정량 eb_review/v2/summary.json(offset_mm). 새 환자로 기준을 다시 맞추지 않는다.
입력 추론(새 환자): seg_totalseg/head_glands_cavities, seg_teeth/UNIVERSALLAB_MODEL, landmarks_teeth.
출력: Result_py/CBCT/patient_pipeline/v2/restore_ext/v1/{new_metrics.csv, compare_all.csv, summary.json}
사용: python cbct_restore_extend.py            (result.json이 있고 동결 54명이 아닌 환자 전부)
      python cbct_restore_extend.py --check 3 7  (동결 환자 재계산 → 기존 CSV와 같은지 확인)
      python cbct_restore_extend.py --nav-only  (Rh 규칙 변경 시 NAV만 갱신, 나머지 계측 보존)
      python cbct_restore_extend.py --excel-only  (엑셀만 바뀐 경우 계측은 그대로, 엑셀값만 다시 읽어 비교표 갱신)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cbct_eb_detect as eb
import cbct_maxilla_section as mx
import cbct_molar_section as mo
import cbct_nav as nv
import cbct_pav as pv
import cbct_pmca as pm
from cbct_evidence_figs import excel_col
from cbct_fh_repair import DEST as FH_REF, repair
from cbct_patient_pipeline import OUT, digest, landmarks, qc, save
from osa.ceph import measures

DEST = OUT/"restore_ext/v1"
SKELETAL = {"ANB": ("ANB", "°"), "Nperp-A": ("Nperp_A", "mm"), "Nperp-Pog": ("Nperp_Pog", "mm"), "MP-FH": ("MPFH_GoMe", "°")}
# 지표: (동결 CSV, 복원값 열, 엑셀 열 이름 또는 엑셀 열 번호, 단위) — cbct_report_scatter.load_pairs와 같은 선택
OTHER = {"TDI": ("tdi_review/v1/tdi.csv", "TDI_rc", "TDI_xl", "mm"),
         "PHI": ("phi_ncwr_review/v1/phi_ncwr.csv", "PHI_rc", "PHI_xl", "%"),
         "NCWR": ("phi_ncwr_review/v1/phi_ncwr.csv", "NCWR_ts", "NCWR_xl", "%"),
         "PAL": ("eb_review/v2/eb_candidates.csv", "PAL_mm", 32, "mm"),
         "PAV": ("pav_review/v1/pav.csv", "PAV_fh", "PAV_xl", "mm³"),
         "NAV": ("nav_review/v1/nav.csv", "NAV_ts", "NAV_xl", "mm³"),
         "PMCA": ("pmca_review/v1/pmca_candidates.csv", "PMCA_center", 31, "mm²")}
XL_COL = {"TDI": mo.XL_TDI, "PHI": mx.XL["PHI"], "NCWR": mx.XL["NCWR"], "PAL": 32, "PAV": pv.XL_PAV, "NAV": nv.XL_NAV, "PMCA": 31}


def frozen_ids():
    return set(pd.read_csv(OUT/"fh_repair/v2/comparison.csv", encoding="utf-8-sig").Patient)


def skeletal(pid, ref, profile):
    p = f"P_{pid}"; res = json.loads((OUT/p/"result.json").read_text(encoding="utf-8"))
    assert "reviewed" not in res["variants"], f"수정 좌표 결과는 별도 확인 필요: {p}"
    auto = res["variants"]["automatic"]; points = landmarks(pid)[0]
    fixed, log = repair(points, ref["points"]); before, after = measures(points, "jung"), measures(fixed, "jung")
    qc_after = "needs_review" if qc(fixed, profile) else "qc_pass_pending_visual_review"
    rows = []
    for metric, (key, _) in SKELETAL.items():
        assert np.isclose(before[key], auto["excel_comparison"][key]["auto"], atol=1e-8), f"result.json과 재계산 불일치: {p}/{metric}"
        rows.append(dict(Patient=p, metric=metric, excel=auto["excel_comparison"][key]["excel"], restored=after[key],
                         before_fh_repair=before[key], qc=qc_after, repaired=",".join(k for k, _, a in log if a)))
    return rows


def eb_row(pid, offset):
    """cbct_eb_detect.main의 환자별 처리와 같다. 설골 보정량은 동결값을 쓴다."""
    r, _, (o, R), (shape, dirs, org) = eb.analyse(pid)
    r = {"Eb_ant": np.nan, "Eb_sup": np.nan, **r}    # 미계측이어도 열이 있어야 PAV·PMCA가 'none'으로 처리한다 (동결 CSV와 같게)
    if r["method"] == "none" and r.get("hy_ant") == r.get("hy_ant") and "hy_ant" in r:
        r.update(Eb_ant=r["hy_ant"]+offset["ant"], Eb_sup=r["hy_sup"]+offset["sup"], method="hyoid_offset")
    if r.get("Eb_ant") is not None and r["method"] != "none":
        q = o+np.array([0, r["Eb_ant"], r["Eb_sup"]])@R; ijk = np.linalg.solve(dirs.T, q-org)
        if np.any(ijk < 0) or np.any(ijk > np.array(shape[::-1])-1):
            r.update(method="none", reason="Eb_outside_image", Eb_ant=np.nan, Eb_sup=np.nan)
        else: r.update(Eb_x=q[0], Eb_y=q[1], Eb_z=q[2])
    r["PAL_mm"] = float(np.hypot(r["Eb_ant"], r["Eb_sup"])) if r.get("Eb_ant") == r.get("Eb_ant") and "Eb_ant" in r else np.nan
    # 동결 CSV처럼 소수 둘째 자리로 반올림한 Eb를 PAV·PMCA에 넘긴다
    return pd.Series({k: round(float(v), 2) if isinstance(v, (float, np.floating)) else v for k, v in r.items()})


ALT = {}    # 환자별 대안 계산값 (정확도 실험 1단계용). 동결 CSV의 같은 이름 열과 대응


def other(pid, offset, xl):
    p = f"P_{pid}"; out = {}; nan = np.nan
    molars = pd.DataFrame(mo.measure(pid)); w = mo.widths(molars)
    out["TDI"] = w["MxAW_rc"]-w["MnAW_rc"]
    m = mx.measure(pid, molars); g = lambda k: m.get(k, nan)
    out["PHI"] = g("PH")/g("MxOW_rc")*100
    out["NCWR"] = g("NCW_ts")/g("MxBW_jugal")*100
    e = eb_row(pid, offset); out["PAL"] = e.PAL_mm
    pav = pv.measure(e, xl["PAV"])[0]; out["PAV"] = pav.get("PAV_fh", nan)
    pmc = pm.measure(e)[0]; out["PMCA"] = pmc.get("PMCA_center", nan)
    nav = nv.measure(pid, keep_imgs=False)[0]; out["NAV"] = nav.get("NAV_ts", nan)
    ALT[p] = {"TDI_surface": w["MxAW_surface"]-w["MnAW_surface"], "TDI_rc": out["TDI"],
              "PHI_oc": g("PH")/g("MxOW_oc")*100, "PHI_mb": g("PH")/g("MxOW_mb")*100, "PHI_rc": out["PHI"],
              "NCWR_jugal": g("NCW")/g("MxBW_jugal")*100, "NCWR_wall": g("NCW_wall")/g("MxBW_jugal")*100, "NCWR_ts": out["NCWR"],
              "PAL_mm": out["PAL"], "Eb_method": e.method, "PAV_fh": out["PAV"], "PAV_midline": pav.get("PAV_midline", nan),
              "PMCA_center": out["PMCA"], "PMCA_axial": pmc.get("PMCA_axial", nan),
              "NAV": nav.get("NAV", nan), "NAV_500": nav.get("NAV_500", nan), "NAV_ts": out["NAV"], "NAV_ts_500": nav.get("NAV_ts_500", nan)}
    notes = {"molars": ";".join(f"{t}:{s}" for t, s in zip(molars.tooth, molars.status)), "Eb_method": e.method,
             "maxilla_reason": m.get("reason", "")}
    return [dict(Patient=p, metric=k, excel=xl[k].get(pid, np.nan), restored=v, qc="", **notes) for k, v in out.items()]


def frozen_rows(ids):
    rows = []
    fh = pd.read_csv(OUT/"fh_repair/v2/comparison.csv", encoding="utf-8-sig")
    for _, r in fh.iterrows():
        rows.append(dict(Patient=r.Patient, metric=r.metric, excel=r.excel, restored=r.after, before_fh_repair=r.before, qc=r.qc_after, source="fh_repair/v2"))
    for metric, (path, col, _, _) in OTHER.items():
        # 엑셀값은 동결 CSV의 *_xl 사본이 아니라 현재 엑셀에서 읽는다 (2026-11-25 NAV·PAV 결측 보충 반영)
        d = pd.read_csv(OUT/path, encoding="utf-8-sig"); xl = excel_col(XL_COL[metric])
        for _, r in d[d.Patient.isin(ids)].iterrows():
            rows.append(dict(Patient=r.Patient, metric=metric, excel=xl.get(int(r.Patient[2:]), np.nan), restored=r[col], qc="", source=path))
    return rows


def status_map():
    c = pd.read_csv(OUT/"cohort.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    inc = set(c.Patient[c.Include.str.lower().isin(["true", "1"])])
    # 2026-11-13 사용자 결정: CBCT 성별 또는 생년월일만 다르고 등록번호(보조 포함)가 일치하면 같은 사람으로 보고 복원
    rule = {"P_36", "P_59", "P_85"}
    return lambda p: "분석 명단" if p in inc else ("등록번호 일치 기준 포함" if p in rule else "계측만(PSG 회신 대기)")


def main():
    ref = json.loads((FH_REF/"reference.json").read_text(encoding="utf-8"))
    profile = json.loads((OUT/"qc_reference.json").read_text(encoding="utf-8"))
    offset = json.loads((OUT/"eb_review/v2/summary.json").read_text(encoding="utf-8"))["offset_mm"]
    xl = {k: excel_col(c) for k, c in XL_COL.items()}
    frozen = frozen_ids(); assert len(frozen) == 54

    if sys.argv[1:2] == ["--check"]:    # 함수 분리·동결값 사용이 기존 결과를 재현하는지
        old = pd.DataFrame(frozen_rows(frozen))
        for pid in map(int, sys.argv[2:]):
            assert f"P_{pid}" in frozen
            new = pd.DataFrame(skeletal(pid, ref, profile)+other(pid, offset, xl))
            m = new.merge(old, on=["Patient", "metric"], suffixes=("", "_old"))
            diff = (m.restored-m.restored_old).abs()
            print(f"P_{pid}", "최대 차이", float(diff.max()), m.loc[diff > 1e-6, ["metric", "restored", "restored_old"]].to_dict("records"))
        return

    done = {p.parent.name for p in OUT.glob("P_*/result.json")}
    new_ids = sorted((done-frozen), key=lambda p: int(p[2:]))
    only = [f"P_{a}" for a in sys.argv[2:]] if sys.argv[1:2] == ["--only"] else None    # 지정 환자만 다시 계산해 기존 결과에 합침
    DEST.mkdir(parents=True, exist_ok=True); rows, failed = [], {}
    nav_only = sys.argv[1:2] == ["--nav-only"]
    if sys.argv[1:2] == ["--excel-only"]:    # 엑셀만 바뀐 경우: 계측은 그대로 두고 excel 열만 다시 읽음
        rows = pd.read_csv(DEST/"new_metrics.csv", encoding="utf-8-sig").to_dict("records")
        for r in rows:
            if r["metric"] in xl: r["excel"] = xl[r["metric"]].get(int(r["Patient"][2:]), np.nan)
        ALT.update(pd.read_csv(DEST/"new_alternatives.csv", encoding="utf-8-sig").set_index("Patient").to_dict("index"))
        failed = json.loads((DEST/"summary.json").read_text(encoding="utf-8"))["failed"]
    elif nav_only:
        previous = pd.read_csv(DEST/"new_metrics.csv", encoding="utf-8-sig")
        alternatives = pd.read_csv(DEST/"new_alternatives.csv", encoding="utf-8-sig").set_index("Patient")
        assert set(previous.Patient) == set(new_ids) == set(alternatives.index)
        assert not previous.duplicated(["Patient", "metric"]).any()
        assert previous[previous.metric == "NAV"].Patient.nunique() == len(new_ids)
        # 모든 환자를 성공적으로 계산한 뒤 저장한다. 다른 지표와 기존 결측은 보존한다.
        for p in new_ids:
            print(p, end=" ", flush=True)
            nav, _ = nv.measure(int(p[2:]), keep_imgs=False)
            previous.loc[(previous.Patient == p) & (previous.metric == "NAV"), "restored"] = nav.get("NAV_ts", np.nan)
            for key in ("NAV", "NAV_500", "NAV_ts", "NAV_ts_500"):
                alternatives.loc[p, key] = nav.get(key, np.nan)
        rows = previous.to_dict("records")
        ALT.update(alternatives.to_dict("index"))
        failed = json.loads((DEST/"summary.json").read_text(encoding="utf-8"))["failed"]
    else:
        for p in (only or new_ids):
            assert p in new_ids, p
            pid = int(p[2:]); print(p, end=" ", flush=True)
            try: rows += skeletal(pid, ref, profile)+other(pid, offset, xl)
            except Exception as e: failed[p] = f"{type(e).__name__}: {e}"
    print()
    new = pd.DataFrame(rows); new["source"] = "restore_ext/v1"
    alt = pd.DataFrame.from_dict(ALT, orient="index").rename_axis("Patient").reset_index()
    if only:
        prev = json.loads((DEST/"summary.json").read_text(encoding="utf-8"))["failed"]
        failed = {**{k: v for k, v in prev.items() if k not in only}, **failed}
        old_m, old_a = (pd.read_csv(DEST/f, encoding="utf-8-sig") for f in ("new_metrics.csv", "new_alternatives.csv"))
        new = pd.concat([old_m[~old_m.Patient.isin(only)], new], ignore_index=True)
        alt = pd.concat([old_a[~old_a.Patient.isin(only)], alt], ignore_index=True)
    new.to_csv(DEST/"new_metrics.csv", index=False, encoding="utf-8-sig")
    alt.to_csv(DEST/"new_alternatives.csv", index=False, encoding="utf-8-sig")

    units = {k: u for k, (_, u) in SKELETAL.items()} | {k: v[3] for k, v in OTHER.items()}
    allr = pd.concat([pd.DataFrame(frozen_rows(frozen)), new[[c for c in new if c in ("Patient", "metric", "excel", "restored", "before_fh_repair", "qc", "source")]]])
    status = status_map()
    allr.insert(1, "status", allr.Patient.map(status)); allr.insert(3, "unit", allr.metric.map(units))
    allr[["excel", "restored"]] = allr[["excel", "restored"]].apply(pd.to_numeric, errors="coerce")
    allr["difference"] = allr.restored-allr.excel
    allr["paired"] = np.isfinite(allr.excel) & np.isfinite(allr.restored)
    allr = allr.sort_values(["Patient", "metric"], key=lambda s: s.map(lambda p: int(p[2:])) if s.name == "Patient" else s)
    assert not allr.duplicated(["Patient", "metric"]).any()
    allr.to_csv(DEST/"compare_all.csv", index=False, encoding="utf-8-sig")

    summ = allr[allr.paired].groupby(["metric", "status"]).difference.agg(n="size", bias="mean", MAE=lambda e: e.abs().mean()).round(2)
    print(summ.to_string())
    save(DEST/"summary.json", {"frozen_patients": 54, "new_patients": new_ids, "failed": failed,
         "patients_total": int(allr.Patient.nunique()), "status_counts": allr.drop_duplicates("Patient").status.value_counts().to_dict(),
         "fixed_inputs_sha256": {k: digest(v) for k, v in {"fh_reference": FH_REF/"reference.json", "qc_reference": OUT/"qc_reference.json",
                                                           "eb_summary": OUT/"eb_review/v2/summary.json"}.items()},
         "script_sha256": digest(Path(__file__)), "status": "exploratory; 전문가 확인 전; 기존 동결 결과 미변경"})
    print("failed:", failed or "없음")


if __name__ == "__main__":
    main()
