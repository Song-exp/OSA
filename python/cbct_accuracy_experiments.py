"""복원값 정확도 개선 실험 체인. 개발(동결 54명)에서만 후보를 고르고, 홀드아웃(새 31명)은 final에서 한 번만 연다.

단계 (python cbct_accuracy_experiments.py <단계>):
  0     기준선: 개발 세트 지표별 오차·일치도, 큰 오차 사례
  1     이미 계산된 대안 비교 (TDI·PHI·NCWR·NAV·PAV·PMCA)
  2     골격 4지표: FH 정의(jung / yao4) × A·B·Pog 윤곽 보정(osa.refine) 유무. Po·Or 이상값 대체는 공통
  3     엑셀 정의 환산(직선 보정): 개발 세트 LOOCV로 이득이 있을 때만 채택. 측정 개선이 아니라 환산이므로 따로 표시
  4     두개 랜드마크 좌우 반전 추론 앙상블(TTA): 환자마다 별도 프로세스, 여유 RAM 확인 후 실행, 이어서 하기 가능
  final 채택 설정을 홀드아웃에 한 번 적용 → 기준선과 비교, 보고서
  export 분석용 표: 머리 좌표계 랜드마크 좌표(골격 벡터)·최종 지표
기존 결과는 읽기만 한다. 출력: Result_py/CBCT/patient_pipeline/v2/restore_ext/v1/experiments/
"""
from __future__ import annotations

import ctypes
import gc
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from batch_cbct_pipeline import LMS
from cbct_evidence_figs import excel_col, teeth_points
from cbct_fh_repair import DEST as FH_REF, repair
from cbct_patient_pipeline import CB, OUT, landmarks, save
from osa.ceph import measures, to_head

EXT = OUT/"restore_ext/v1"; EXP = EXT/"experiments"
SKEL = {"ANB": "ANB", "Nperp-A": "Nperp_A", "Nperp-Pog": "Nperp_Pog", "MP-FH": "MPFH_GoMe"}
ALT = {"TDI": ("tdi_review/v1/tdi.csv", ["TDI_rc", "TDI_surface"], "TDI_xl"),
       "PHI": ("phi_ncwr_review/v1/phi_ncwr.csv", ["PHI_rc", "PHI_oc", "PHI_mb"], "PHI_xl"),
       "NCWR": ("phi_ncwr_review/v1/phi_ncwr.csv", ["NCWR_ts", "NCWR_wall", "NCWR_jugal"], "NCWR_xl"),
       # 2026-11-26 사용자 확정: NAV 공기 임계값은 Roh 수동 범위(-400~-500 HU) 안의 -400 HU로 고정(-500 변형 제외), 최종값 NAV_ts 원값
       "NAV": ("nav_review/v1/nav.csv", ["NAV_ts", "NAV"], "NAV_xl"),
       "PAV": ("pav_review/v1/pav.csv", ["PAV_fh", "PAV_midline"], "PAV_xl"),
       "PMCA": ("pmca_review/v1/pmca_candidates.csv", ["PMCA_center", "PMCA_axial"], 31)}    # 첫 후보 = 현재 기본값
VARIANTS = [("jung", False), ("yao4", False), ("jung", True), ("yao4", True)]    # 첫 항목 = 현재 기본값
SWAP = {"RPo": "LPo", "LPo": "RPo", "ROr": "LOr", "LOr": "ROr", "RGo": "LGo", "LGo": "RGo"}
MIN_FREE_GB = 8.0


# ---------- 공통 ----------
def free_gb():
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]+[(n, ctypes.c_ulonglong) for n in
                    ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    m = M(); m.dwLength = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys/2**30


def wait_ram(need=MIN_FREE_GB, label=""):
    t0 = time.time()
    while free_gb() < need:
        if time.time()-t0 > 3600: raise RuntimeError(f"여유 RAM {free_gb():.1f} GB가 1시간 동안 {need} GB 미만 {label}")
        time.sleep(30)


def sets():
    dev = set(pd.read_csv(OUT/"fh_repair/v2/comparison.csv", encoding="utf-8-sig").Patient)
    allp = set(pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig").Patient)
    hold = allp-dev; assert len(dev) == 54 and len(hold) == 31, (len(dev), len(hold))
    return sorted(dev, key=lambda p: int(p[2:])), sorted(hold, key=lambda p: int(p[2:]))


def stats(x, y):
    """x = 복원값, y = 엑셀. 오차·Bland-Altman·ICC(2,1, 절대일치)."""
    ok = np.isfinite(x) & np.isfinite(y); x, y = np.asarray(x)[ok], np.asarray(y)[ok]; n = len(x)
    if n < 3: return dict(n=n)
    d = x-y; Y = np.c_[x, y]; gm = Y.mean(); k = 2
    msr = k*((Y.mean(1)-gm)**2).sum()/(n-1); msc = n*((Y.mean(0)-gm)**2).sum()/(k-1)
    mse = ((Y-Y.mean(1, keepdims=True)-Y.mean(0)+gm)**2).sum()/((n-1)*(k-1))
    icc = (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
    return dict(n=n, bias=d.mean(), MAE=np.abs(d).mean(), RMSE=np.sqrt((d**2).mean()), r=np.corrcoef(x, y)[0, 1],
                ICC=icc, LoA_low=d.mean()-1.96*d.std(ddof=1), LoA_high=d.mean()+1.96*d.std(ddof=1))


def excel_skel(p):
    res = json.loads((OUT/p/"result.json").read_text(encoding="utf-8"))["variants"]["automatic"]["excel_comparison"]
    return {m: res[k]["excel"] for m, k in SKEL.items()}


def log(stage, payload):
    f = EXP/"experiment_log.json"; L = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    L[stage] = {**payload, "time": time.strftime("%Y-%m-%d %H:%M:%S")}; save(f, L)


# ---------- 0. 기준선 ----------
def stage0():
    dev, _ = sets(); d = pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig"); d = d[d.Patient.isin(dev)]
    st = pd.DataFrame([{"metric": m, **stats(g.restored.to_numpy(float), g.excel.to_numpy(float))} for m, g in d.groupby("metric")])
    st.round(3).to_csv(EXP/"stage0_dev_stats.csv", index=False, encoding="utf-8-sig")
    d = d.assign(abs_err=(d.restored-d.excel).abs()).dropna(subset=["abs_err"])
    top = d.sort_values("abs_err", ascending=False).groupby("metric").head(5)
    top[["metric", "Patient", "excel", "restored", "abs_err", "qc"]].round(2).to_csv(EXP/"stage0_dev_top_errors.csv", index=False, encoding="utf-8-sig")
    log("stage0", {"dev_n": len(dev)}); print(st.round(2).to_string())


# ---------- 1. 대안 비교 ----------
def alt_table(hold=False):
    """{지표: DataFrame(Patient, excel, 후보 열...)}. 개발은 동결 CSV, 홀드아웃은 new_alternatives.csv."""
    out = {}
    na = pd.read_csv(EXT/"new_alternatives.csv", encoding="utf-8-sig") if hold else None
    for m, (path, cols, x) in ALT.items():
        # 엑셀값은 동결 CSV의 *_xl 사본이 아니라 현재(전문가 최종) 엑셀에서 읽는다 (2026-11-26 NAV·PAV 보충 반영)
        ex = excel_col(x) if isinstance(x, int) else excel_col({"TDI": 28, "PHI": 27, "NCWR": 26, "NAV": 29, "PAV": 30}[m])
        t = (na if hold else pd.read_csv(OUT/path, encoding="utf-8-sig"))[["Patient"]+cols].copy()
        t["excel"] = t.Patient.map(lambda p: ex.get(int(p[2:]), np.nan))
        out[m] = t
    return out


def stage1():
    rows, choice = [], {}
    for m, t in alt_table().items():
        cols = ALT[m][1]; s = {c: stats(t[c].to_numpy(float), t.excel.to_numpy(float)) for c in cols}
        for c in cols: rows.append({"metric": m, "candidate": c, **s[c]})
        # 같은 환자 집합에서 비교 (후보마다 결측이 다를 수 있음)
        both = t.dropna(subset=cols+["excel"]); mae = {c: float((both[c]-both.excel).abs().mean()) for c in cols}
        best = min(cols, key=lambda c: (round(mae[c], 6), cols.index(c)))
        choice[m] = {"chosen": best, "default": cols[0], "common_n": len(both), "MAE_common": mae}
    pd.DataFrame(rows).round(3).to_csv(EXP/"stage1_dev_alternatives.csv", index=False, encoding="utf-8-sig")
    save(EXP/"stage1_choices.json", choice); log("stage1", {m: c["chosen"] for m, c in choice.items()})
    print(json.dumps({m: (c["chosen"], c["common_n"]) for m, c in choice.items()}, ensure_ascii=False))


# ---------- 2. 골격 4지표 변형 ----------
_ref = None


def skeletal(pid, P, fh, use_refine):
    """Po·Or 이상값 대체 → (선택) A·B·Pog 윤곽 보정 → 4지표."""
    global _ref
    import SimpleITK as sitk
    from osa.refine import refine
    from run_ali import ascii_path
    _ref = _ref or json.loads((FH_REF/"reference.json").read_text(encoding="utf-8"))["points"]
    Q = repair(P, _ref)[0]; info = {}
    if use_refine:
        rd = lambda s: sitk.ReadImage(str(ascii_path(CB/f"seg/P_{pid}_seg_{s}.nrrd")))
        mand = rd("MAND"); dirs = np.array(mand.GetDirection()).reshape(3, 3).T*np.array(mand.GetSpacing())[:, None]
        Q, info = refine(Q, sitk.GetArrayFromImage(mand), sitk.GetArrayFromImage(rd("MAX")), dirs, np.array(mand.GetOrigin()), fh)
        del mand
    m = measures(Q, fh)
    return {k: m[v] for k, v in SKEL.items()}, Q, info


def stage2():
    dev, hold = sets(); rows = []; f = EXP/"stage2_values.csv"
    done = set(pd.read_csv(f, encoding="utf-8-sig").Patient) if f.exists() else set()
    for p in dev+hold:
        if p in done: continue
        wait_ram(4, p); pid = int(p[2:]); P = landmarks(pid)[0]; ex = excel_skel(p); r = []
        for fh, ref in VARIANTS:
            m, _, _ = skeletal(pid, P, fh, ref)
            r += [dict(Patient=p, variant=f"{fh}{'+refine' if ref else ''}", metric=k, value=v, excel=ex[k]) for k, v in m.items()]
        pd.DataFrame(r).to_csv(f, mode="a", header=not f.exists(), index=False, encoding="utf-8-sig"); gc.collect(); print(p, end=" ", flush=True)
    d = pd.read_csv(f, encoding="utf-8-sig"); dd = d[d.Patient.isin(dev)]; choice = {}; rows = []
    # 안전장치: 기본 설정(jung, 보정 없음)은 기존 동결 복원값(fh_repair/v2)과 같아야 한다
    base = pd.read_csv(OUT/"fh_repair/v2/comparison.csv", encoding="utf-8-sig").set_index(["Patient", "metric"]).after
    j = dd[dd.variant == "jung"].set_index(["Patient", "metric"]).value
    assert np.allclose(j.sort_index(), base.reindex(j.index).sort_index(), atol=1e-6), "기본 설정 재계산이 동결값과 다름"
    names = [f"{fh}{'+refine' if r else ''}" for fh, r in VARIANTS]
    for m, g in dd.groupby("metric"):
        mae = {}
        for v in names:
            h = g[g.variant == v]; s = stats(h.value.to_numpy(float), h.excel.to_numpy(float)); rows.append({"metric": m, "variant": v, **s}); mae[v] = s["MAE"]
        choice[m] = {"chosen": min(names, key=lambda v: (round(mae[v], 6), names.index(v))), "default": names[0], "MAE": mae}
    pd.DataFrame(rows).round(3).to_csv(EXP/"stage2_dev_variants.csv", index=False, encoding="utf-8-sig")
    save(EXP/"stage2_choices.json", choice); log("stage2", {m: c["chosen"] for m, c in choice.items()})
    print("\n", json.dumps({m: c["chosen"] for m, c in choice.items()}, ensure_ascii=False))


# ---------- 4. 좌우 반전 TTA ----------
TTA = CB/"landmarks_tta"; TTA_WORK = CB/"tta_work"


def flip_axis(img):
    D = np.array(img.GetDirection()).reshape(3, 3)    # 열 = 각 영상 축의 물리 방향
    return int(np.argmax(np.abs(D[0])))                 # LPS x(좌우)에 가장 가까운 영상 축


def tta_one(pid):
    """반전 영상으로 ALI 추론 → 원래 공간으로 되돌리고 좌우 이름을 바꿔 저장. 별도 프로세스에서 실행."""
    import shutil
    import SimpleITK as sitk
    from run_ali import MODELS, SADT, ascii_path
    out = TTA/f"P_{pid}.json"
    if out.exists(): return
    TTA_WORK.mkdir(parents=True, exist_ok=True); pred = TTA_WORK/f"pred_{pid}"; tmp = TTA_WORK/f"tmp_{pid}"; scan = TTA_WORK/f"P_{pid}.nrrd"
    img = sitk.ReadImage(str(ascii_path(CB/f"nrrd/P_{pid}.nrrd"))); ax = flip_axis(img)
    flip = [i == ax for i in range(3)]; sitk.WriteImage(sitk.Flip(img, flip, False), str(ascii_path(scan)), True)
    D = np.array(img.GetDirection()).reshape(3, 3); u = D[:, ax]/np.linalg.norm(D[:, ax])
    c = np.array(img.TransformContinuousIndexToPhysicalPoint([(img.GetSize()[i]-1)/2 for i in range(3)])); del img
    pred.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SADT/"ALI_CBCT"), str(SADT/"ADT")])}
    cmd = [sys.executable, str(SADT/"ALI_CBCT/ALI_CBCT.py"), str(ascii_path(scan)), str(MODELS), ",".join(f"'{x}'" for x in LMS),
           str(ascii_path(pred)), str(ascii_path(tmp)), "false", "[1,0.3]", "[1,1]", "[64,64,64]", "10"]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pts = {}
    for f in pred.glob(f"P_{pid}_*.mrk.json"):
        for mk in json.loads(f.read_text(encoding="utf-8"))["markups"]:
            for cp in mk["controlPoints"]:
                q = np.asarray(cp["position"], float)*(np.array([-1, -1, 1]) if mk.get("coordinateSystem") == "RAS" else 1)
                q = q-2*((q-c)@u)*u                                   # 반전 평면에 대해 되돌림
                pts[SWAP.get(cp["label"], cp["label"])] = q.tolist()  # 반전 영상의 오른쪽 = 원래 왼쪽
    TTA.mkdir(parents=True, exist_ok=True); save(out, {"points": pts, "flip_axis": ax})
    shutil.rmtree(pred, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True); scan.unlink(missing_ok=True)


def tta_points(pid):
    P = landmarks(pid)[0]; f = TTA/f"P_{pid}.json"
    if not f.exists(): return None, {}
    T = {k: np.asarray(v) for k, v in json.loads(f.read_text(encoding="utf-8"))["points"].items()}
    dist = {k: float(np.linalg.norm(T[k]-P[k])) for k in P if k in T}
    return {k: (P[k]+T[k])/2 if k in T else P[k] for k in P}, dist


def stage4():
    dev, hold = sets()
    for p in dev+hold:    # 개발 먼저
        pid = int(p[2:])
        if (TTA/f"P_{pid}.json").exists(): continue
        wait_ram(MIN_FREE_GB, p); t0 = time.time()
        r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--tta-one", str(pid)], timeout=3600)
        print(p, "ok" if r.returncode == 0 else f"실패({r.returncode})", f"{time.time()-t0:.0f}s", flush=True)
    c2 = json.loads((EXP/"stage2_choices.json").read_text(encoding="utf-8")); rows, dists = [], []
    for p in dev+hold:
        pid = int(p[2:]); A, dist = tta_points(pid)
        if A is None: continue
        dists += [dict(Patient=p, point=k, mm=v) for k, v in dist.items()]; ex = excel_skel(p)
        for m in SKEL:
            fh, ref = c2[m]["chosen"].split("+")[0], c2[m]["chosen"].endswith("+refine")
            rows.append(dict(Patient=p, metric=m, tta=skeletal(pid, A, fh, ref)[0][m], excel=ex[m]))
        gc.collect()
    dist = pd.DataFrame(dists); dist.to_csv(EXP/"stage4_tta_point_distance.csv", index=False, encoding="utf-8-sig")
    sane = bool(len(dist)) and float(dist.mm.median()) < 6.0      # 되돌림 계산이 틀리면 거리가 크게 벌어진다
    t = pd.DataFrame(rows); t.to_csv(EXP/"stage4_values.csv", index=False, encoding="utf-8-sig")
    s2 = pd.read_csv(EXP/"stage2_values.csv", encoding="utf-8-sig"); choice = {}
    for m in SKEL:
        base = s2[(s2.metric == m) & (s2.variant == c2[m]["chosen"]) & s2.Patient.isin(dev)].set_index("Patient")
        g = t[(t.metric == m) & t.Patient.isin(dev)].set_index("Patient").join(base.value)
        a, b = stats(g.value.to_numpy(float), g.excel.to_numpy(float)), stats(g.tta.to_numpy(float), g.excel.to_numpy(float))
        choice[m] = {"use_tta": bool(sane and b.get("MAE", 1e9) < a.get("MAE", 0)), "MAE_no_tta": a.get("MAE"), "MAE_tta": b.get("MAE"), "n": a.get("n")}
    save(EXP/"stage4_choices.json", {"sane": sane, "median_point_mm": float(dist.mm.median()) if len(dist) else None, "metrics": choice})
    log("stage4", {"sane": sane, **{m: c["use_tta"] for m, c in choice.items()}}); print(json.dumps(choice, ensure_ascii=False, default=float))


# ---------- 최종 값 (설정 적용) ----------
def final_values(patients, which):
    """which = 'baseline' 또는 'chosen'. 반환: Patient, metric, value, excel."""
    rows = []; ca = pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig")
    if which == "baseline":
        d = ca[ca.Patient.isin(patients)]
        return d.rename(columns={"restored": "value"})[["Patient", "metric", "value", "excel"]]
    c1 = json.loads((EXP/"stage1_choices.json").read_text(encoding="utf-8"))
    c2 = json.loads((EXP/"stage2_choices.json").read_text(encoding="utf-8"))
    c4 = json.loads((EXP/"stage4_choices.json").read_text(encoding="utf-8")) if (EXP/"stage4_choices.json").exists() else {"metrics": {}}
    s2 = pd.read_csv(EXP/"stage2_values.csv", encoding="utf-8-sig")
    s4 = pd.read_csv(EXP/"stage4_values.csv", encoding="utf-8-sig") if (EXP/"stage4_values.csv").exists() else None
    for m in SKEL:
        if c4["metrics"].get(m, {}).get("use_tta") and s4 is not None:
            d = s4[(s4.metric == m) & s4.Patient.isin(patients)].rename(columns={"tta": "value"})
        else:
            d = s2[(s2.metric == m) & (s2.variant == c2[m]["chosen"]) & s2.Patient.isin(patients)]
        rows.append(d[["Patient", "metric", "value", "excel"]])
    dev, _ = sets(); hold_needed = any(p not in dev for p in patients)
    tabs = {**alt_table(False)}; th = alt_table(True) if hold_needed else {}
    for m in ALT:
        col = c1[m]["chosen"]; t = pd.concat([tabs[m]]+([th[m]] if th else []))
        t = t[t.Patient.isin(patients)]; rows.append(pd.DataFrame({"Patient": t.Patient, "metric": m, "value": t[col], "excel": t.excel}))
    pal = ca[(ca.metric == "PAL") & ca.Patient.isin(patients)].rename(columns={"restored": "value"})    # PAL은 대안 없음
    rows.append(pal[["Patient", "metric", "value", "excel"]])
    return pd.concat(rows, ignore_index=True)


# ---------- 3. 엑셀 정의 환산 ----------
def stage3():
    dev, _ = sets(); d = final_values(dev, "chosen"); out = {}
    for m, g in d.dropna(subset=["value", "excel"]).groupby("metric"):
        x, y = g.value.to_numpy(float), g.excel.to_numpy(float)
        if len(x) < 8: out[m] = {"adopt": False, "n": len(x)}; continue
        loo = [abs(np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i])-y[i]) for i in range(len(x))]
        b, a = np.polyfit(x, y, 1)
        out[m] = {"adopt": bool(np.mean(loo) < np.mean(np.abs(x-y))), "a": float(a), "b": float(b), "n": len(x),
                  "MAE_raw": float(np.mean(np.abs(x-y))), "MAE_loocv_calibrated": float(np.mean(loo))}
    save(EXP/"stage3_calibration.json", out); log("stage3", {m: v["adopt"] for m, v in out.items()})
    print(json.dumps({m: (v["adopt"], round(v.get("MAE_raw", 0), 2), round(v.get("MAE_loocv_calibrated", 0), 2)) for m, v in out.items()}, ensure_ascii=False))


def calibrate(d):
    cal = json.loads((EXP/"stage3_calibration.json").read_text(encoding="utf-8")); d = d.copy()
    d["value_cal"] = [v*cal[m]["b"]+cal[m]["a"] if cal.get(m, {}).get("adopt") else np.nan for m, v in zip(d.metric, d.value)]
    return d


# ---------- final: 홀드아웃 1회 ----------
def final():
    dev, hold = sets(); rows = []
    for name, P in (("dev", dev), ("holdout", hold)):
        b, c = final_values(P, "baseline"), calibrate(final_values(P, "chosen"))
        for m in sorted(set(c.metric)):
            gb, gc_ = b[b.metric == m], c[c.metric == m]
            for label, x, y in (("기준선", gb.value, gb.excel), ("개선 설정", gc_.value, gc_.excel), ("개선+엑셀 환산", gc_.value_cal, gc_.excel)):
                rows.append({"set": name, "metric": m, "config": label, **stats(x.to_numpy(float), y.to_numpy(float))})
    r = pd.DataFrame(rows); r.round(3).to_csv(EXP/"final_comparison.csv", index=False, encoding="utf-8-sig")
    log("final", {"holdout_opened": True}); report(r); print(r[r.set == "holdout"].pivot(index="metric", columns="config", values="MAE").round(2).to_string())


def report(r):
    L = json.loads((EXP/"experiment_log.json").read_text(encoding="utf-8"))
    piv = lambda s: r[r.set == s].pivot(index="metric", columns="config", values="MAE").round(2)
    tab = lambda df: df.to_html(border=0, na_rep="—")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>복원 정확도 실험</title><style>body{{font:15px/1.6 "Malgun Gothic",sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}th{{background:#eef1f4}}</style></head><body>
<h1>복원값 정확도 개선 실험</h1><p>개발 세트(동결 54명)에서 후보를 고르고, 홀드아웃(규칙 고정 후 처음 계측한 31명)에는 한 번만 적용했습니다. 값은 엑셀 대비 평균 절대오차(MAE). '엑셀 환산'은 측정 개선이 아니라 엑셀 정의로 옮긴 값입니다.</p>
<h2>홀드아웃 31명</h2>{tab(piv("holdout"))}<h2>개발 54명</h2>{tab(piv("dev"))}
<h2>단계별 선택</h2><pre>{escape(json.dumps(L, ensure_ascii=False, indent=1, default=float))}</pre>
<p>전체 지표(bias, r, ICC, Bland-Altman): final_comparison.csv</p></body></html>"""
    (EXP/"report.html").write_text(html, encoding="utf-8")


# ---------- export: 골격 벡터 ----------
def export():
    dev, hold = sets(); allp = dev+hold; X = EXP/"export"; X.mkdir(parents=True, exist_ok=True)
    c2 = json.loads((EXP/"stage2_choices.json").read_text(encoding="utf-8"))
    c4 = json.loads((EXP/"stage4_choices.json").read_text(encoding="utf-8")) if (EXP/"stage4_choices.json").exists() else {"metrics": {}}
    fh_m = c2["MP-FH"]["chosen"]; fh, use_ref = fh_m.split("+")[0], fh_m.endswith("+refine")    # 좌표계는 MP-FH에서 고른 FH 정의
    use_tta = c4["metrics"].get("MP-FH", {}).get("use_tta", False)
    status = pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig").drop_duplicates("Patient").set_index("Patient").status
    wide, lps = [], []
    for p in allp:
        pid = int(p[2:]); P = tta_points(pid)[0] if use_tta and (TTA/f"P_{pid}.json").exists() else landmarks(pid)[0]
        _, Q, _ = skeletal(pid, P, fh, use_ref); Q = {**Q, **{k: v for k, v in teeth_points(pid).items() if k[2:3] == "6"}}
        H = to_head(Q, fh); row = {"Patient": p, "set": "dev" if p in dev else "holdout", "status": status[p]}
        for k in LMS+sorted(k for k in H if k not in LMS):
            if k in H: row |= {f"{k}_left": H[k][0], f"{k}_ant": H[k][1], f"{k}_sup": H[k][2]}
            if k in Q: lps.append({"Patient": p, "point": k, "x": Q[k][0], "y": Q[k][1], "z": Q[k][2]})
        wide.append(row); gc.collect()
    pd.DataFrame(wide).round(3).to_csv(X/"skeletal_landmarks_head.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lps).round(3).to_csv(X/"landmarks_lps.csv", index=False, encoding="utf-8-sig")
    v = calibrate(final_values(allp, "chosen"))
    w = v.pivot(index="Patient", columns="metric", values="value").add_suffix("_restored")
    w = w.join(v.pivot(index="Patient", columns="metric", values="excel").add_suffix("_excel"))
    w = w.join(v.pivot(index="Patient", columns="metric", values="value_cal").dropna(axis=1, how="all").add_suffix("_excel_scale"))
    w.insert(0, "status", w.index.map(status)); w.insert(0, "set", ["dev" if p in dev else "holdout" for p in w.index])
    w.round(3).to_csv(X/"final_metrics.csv", encoding="utf-8-sig")
    (X/"README.md").write_text(f"""# 분석용 내보내기

- skeletal_landmarks_head.csv: 환자별 랜드마크 좌표, 머리 좌표계(FH {fh}, 원점 N, 축 left/ant/sup, mm). Po·Or 이상값 대체{', A·B·Pog 윤곽 보정' if use_ref else ''}{', 좌우 반전 TTA 평균' if use_tta else ''} 적용. 상·하악 제1대구치 ALI 점 포함.
- landmarks_lps.csv: 같은 점의 원래 영상 좌표(LPS mm).
- final_metrics.csv: 11개 지표의 채택 설정 복원값(_restored), 엑셀값(_excel), 엑셀 정의 환산값(_excel_scale, 채택된 지표만).
- set: dev = 규칙 개발에 쓴 54명, holdout = 규칙 고정 후 처음 계측한 31명. status: 분석 명단 / 계측만(PSG 회신 대기) / 등록번호 일치 기준 포함.
- PSG 지표와 합칠 때: '계측만' 환자는 PSG 검사 선택·리포트 회신 전이다. 좌표는 크기 정렬(Procrustes)을 하지 않은 실제 mm다.
""", encoding="utf-8")
    log("export", {"patients": len(allp), "fh": fh, "refine": use_ref, "tta": use_tta}); print("export", X)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if sys.argv[1:2] == ["--tta-one"]: tta_one(int(sys.argv[2])); sys.exit()
    EXP.mkdir(parents=True, exist_ok=True)
    {"0": stage0, "1": stage1, "2": stage2, "3": stage3, "4": stage4, "final": final, "export": export}[sys.argv[1]]()
