"""CBCT 기본 EDA: 환자별 결과표와 육안 검수용 몽타주. 읽기 전용이며, 환자를 자동으로 제외하지 않습니다.

범주 (결과표 열 접두사)
  Q_ 영상 품질 : 잡음(고주파 잔차 SD), 금속(≥3000 HU) 복셀, 최대값 포화(3071) 비율, 연조직 피크, 하악 뼈 99백분위
  A_ 촬영 조건 : 장비, 관전압·관전류·노출, 복셀 크기, 행렬, 슬라이스 수, 촬영 범위, 촬영 연도 (개인정보 없는 헤더만)
  P_ 자세·교합 : 정중면 기울기, S–N 앞뒤 기울기, 구개면 기울기 (Po를 쓰지 않음), 상·하악 최소 간격(전체, 앞니 부위)
  F_ 구조별 범위: 마스크가 영상 경계면에 닿는 방향 (예: UAW 아래 → 하인두 잘림)
  S_ 분할 품질 : 구조별 부피, 연결 덩어리 수, 가장 큰 덩어리 비율
  M_ 메모리    : 환자별 numpy 최대 할당량(tracemalloc), 프로세스 메모리 전후 (검수 표시 대상 아님)
검수 표시: 연속형 지표는 로버스트 z(중앙값·MAD) |z|>3.5, 규칙형은 RULES. 표시는 육안 검수 대상일 뿐 제외가 아닙니다.
메모리: 연결 덩어리·거리 계산은 마스크가 있는 영역(경계 상자)만 잘라서 하고, 마스크는 하나씩 읽음.
재개: 환자별 결과를 cache/에 바로 저장 (임시 파일에 다 쓴 뒤 이름 바꾸기 → 최종 이름의 파일은 항상 완전).
      다시 실행하면 JSON과 단면 파일이 둘 다 있고 정상적으로 읽히는 환자만 건너뜀.
결과: Result_py/cbct_eda/ (cbct_eda_patient.csv, cbct_eda_flags.csv, montage_raw.png, montage_mask.png, cache/)
사용: python cbct_eda.py
"""
import ctypes
import json
import os
import tracemalloc
from ctypes import wintypes
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, label, median_filter

from check_landmarks import read_points
from run_ali import ascii_path

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "Result_py/CBCT"
OUT = ROOT / "Result_py/cbct_eda"
STRUCTS = ["MAND", "MAX", "UAW", "CB"]
COLORS = {"MAND": "tab:red", "MAX": "tab:blue", "UAW": "tab:green", "CB": "gold"}
HDR = ["Manufacturer", "ManufacturerModelName", "KVP", "XRayTubeCurrent", "ExposureTime", "Exposure"]
FACE = {(0, 1): "좌", (0, -1): "우", (1, 1): "뒤", (1, -1): "앞", (2, 1): "위", (2, -1): "아래"}   # LPS 축, 부호
# v2 (첫 실행 결과를 보고 QC 규칙만 조정, 2026-10-28): 83명 전원이 표시되어 검수 우선순위를 구분하지 못했기 때문
#  - 경계 접촉은 가장 큰 덩어리 기준 (상악의 경추 위 조각 같은 떨어진 조각 제외)
#  - 상기도 아래 접촉(83명 전원)과 두개저 접촉(구조상 영상 밖으로 이어짐)은 환자별 표시가 아니라 코호트 수준 결과로 보고
#  - 가장 큰 덩어리 비율은 대부분 0.99 이상이라 로버스트 z가 과민 → 절대 기준 < 0.95
EDA_VERSION = 2
RULES = {"F_MAND_touch_main": "", "F_MAX_touch_main": ""}   # 이 방향 글자가 있으면 표시 ("" = 어느 방향이든)
LARGEST_FRAC_MIN = 0.95


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
               [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def working_set_mb():
    """현재 프로세스 메모리 (Windows working set, MB)."""
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE            # 64비트 핸들이 잘리지 않게 형식 지정
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2 ** 20 if ok else float("nan")


def atomic_save(path: Path, write):
    """임시 파일에 다 쓴 뒤에만 최종 이름으로 바꿈 → 최종 이름의 파일은 항상 완전한 파일."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def bbox(m, pad=2):
    k, j, i = np.nonzero(m)
    lo = np.maximum(np.array([k.min(), j.min(), i.min()]) - pad, 0)
    hi = np.minimum(np.array([k.max(), j.max(), i.max()]) + pad + 1, m.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi)), lo


def largest(m):
    """가장 큰 연결 덩어리만 남긴 마스크 (원래 크기). 떨어진 조각(경추 위 오분할 등)의 경계 접촉을 빼려고 씀."""
    out = np.zeros_like(m)
    if m.any():
        sl, _ = bbox(m)
        lab, n = label(m[sl])
        out[sl] = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return out


def components(m):
    """연결 덩어리 수, 가장 큰 덩어리 비율 (경계 상자 안에서 계산)."""
    if not m.any():
        return 0, np.nan
    sl, _ = bbox(m)
    lab, n = label(m[sl])
    sizes = np.bincount(lab.ravel())[1:]
    return n, sizes.max() / sizes.sum()


def index_to_lps(img, kji):
    """(k, j, i) 정수 인덱스 배열 → LPS mm (SimpleITK TransformIndexToPhysicalPoint와 같은 식, 한 번에)."""
    D = np.array(img.GetDirection()).reshape(3, 3)
    return np.array(img.GetOrigin()) + (kji[:, ::-1] * np.array(img.GetSpacing())) @ D.T


def header(p):
    folder = next((ROOT / "Data/CBCT").glob(f"P_{p}_*"))
    files = sorted(f for f in folder.iterdir() if f.is_file())          # 상위 폴더만 (하위 폴더는 다른 영상)
    d = pydicom.dcmread(files[0], force=True, stop_before_pixels=True)
    year = str(getattr(d, "StudyDate", "") or getattr(d, "AcquisitionDate", ""))[:4]
    return {f"A_{k}": getattr(d, k, None) for k in HDR} | {"A_StudyYear": year or None, "A_NFiles": len(files)}


def faces(m, img):
    """마스크가 닿는 영상 경계면의 방향 이름."""
    D = np.array(img.GetDirection()).reshape(3, 3)   # 열 = sitk 축 (i, j, k)의 LPS 방향
    out = []
    for ax_arr, ax_itk in ((2, 0), (1, 1), (0, 2)):
        lps = int(np.argmax(np.abs(D[:, ax_itk])))
        sgn = np.sign(D[lps, ax_itk])
        for end, s in ((0, -sgn), (-1, sgn)):
            if np.take(m, end, axis=ax_arr).any():
                out.append(FACE[(lps, int(s))])
    return "".join(sorted(set(out)))


def pose(P):
    mid = np.array([P[k] for k in ("N", "S", "Ba", "ANS", "PNS")])
    n = np.linalg.svd(mid - mid.mean(0))[2][-1]
    sn, pal = P["N"] - P["S"], P["ANS"] - P["PNS"]
    return {"P_midplane_tilt_deg": np.degrees(np.arccos(abs(n[0]))),
            "P_SN_pitch_deg": np.degrees(np.arctan2(sn[2], -sn[1])),      # 앞이 위로 +
            "P_palatal_pitch_deg": np.degrees(np.arctan2(pal[2], -pal[1]))}


def gaps(mand, mx, img, P):
    """상·하악 마스크 최소 간격 (mm): 전체, 앞니 부위(B 주변 정중 ±15 mm, B보다 앞 또는 10 mm 뒤까지)."""
    sl, lo = bbox(mand | mx, pad=0)
    sp = np.array(img.GetSpacing())[::-1]
    md = mand[sl]
    dist = distance_transform_edt(~mx[sl], sampling=sp).astype(np.float32)[md]
    out = {"P_gap_min_mm": float(dist.min())}
    if "B" in P:
        pts = index_to_lps(img, np.argwhere(md) + lo)
        front = (np.abs(pts[:, 0] - P["B"][0]) < 15) & (pts[:, 1] < P["B"][1] + 10)
        out["P_gap_incisor_mm"] = float(dist[front].min()) if front.any() else np.nan
    return out


def thumb(a, i_mid, size=160):
    s = a[:, :, i_mid][::-1]                                # 시상 단면, 위가 위로
    step = max(1, max(s.shape) // size)
    return s[::step, ::step].copy()                     # 복사: view로 두면 전체 볼륨이 메모리에 계속 남음


def analyze(p):
    """환자 한 명: 결과 행(dict), 원본 단면, 마스크 단면."""
    img = sitk.ReadImage(str(ascii_path(CB / f"nrrd/P_{p}.nrrd")))
    a = sitk.GetArrayFromImage(img)                         # (k, j, i), HU
    r = {"Patient": f"P_{p}", "eda_version": EDA_VERSION, **header(p)}
    sp = img.GetSpacing()
    r |= {"A_voxel_mm": round(sp[0], 3), "A_matrix": f"{img.GetSize()[0]}x{img.GetSize()[1]}x{img.GetSize()[2]}",
          "A_FOV_x_mm": img.GetSize()[0] * sp[0], "A_FOV_z_mm": img.GetSize()[2] * sp[2]}
    # 영상 품질
    mid = a[a.shape[0] // 2 - 2:a.shape[0] // 2 + 3].astype(np.float32)
    soft = (mid > -100) & (mid < 200)
    r["Q_noise_SD"] = float((mid - median_filter(mid, size=(1, 3, 3)))[soft].std())
    del mid, soft
    r["Q_metal_vox"] = int((a >= 3000).sum())
    r["Q_sat3071_frac"] = float((a >= 3071).mean())
    h, e = np.histogram(a[(a > -300) & (a < 300)], bins=60)
    r["Q_soft_peak_HU"] = float(e[np.argmax(h)] + (e[1] - e[0]) / 2)
    P = None
    try:
        P = {k: v[["x", "y", "z"]].to_numpy(float) for k, v in read_points(p).iterrows()}
        r |= pose(P)
    except KeyError:
        pass
    i_mid = a.shape[2] // 2
    if P:
        c = np.mean([P[k] for k in ("N", "S", "ANS", "PNS")], 0)
        i_mid = int(round(img.TransformPhysicalPointToContinuousIndex(tuple(c))[0]))
    mthumbs, keep = {}, {}
    for s in STRUCTS:                                        # 마스크는 하나씩 읽고, 간격 계산에 필요한 MAND·MAX만 유지
        f = CB / f"seg/P_{p}_seg_{s}.nrrd"
        if not f.exists():
            continue
        m = sitk.GetArrayFromImage(sitk.ReadImage(str(ascii_path(f)))) > 0
        n, frac = components(m)
        r |= {f"S_{s}_vol_cc": m.sum() * np.prod(sp) / 1000, f"S_{s}_ncomp": n,
              f"S_{s}_largest_frac": frac, f"F_{s}_touch": faces(m, img), f"F_{s}_touch_main": faces(largest(m), img)}
        mthumbs[s] = thumb(m, i_mid)
        if s in ("MAND", "MAX"):
            keep[s] = m
    if "MAND" in keep:
        r["Q_MAND_bone_p99_HU"] = float(np.percentile(a[keep["MAND"]], 99))
        jaw = keep["MAND"] | keep.get("MAX", False)
        r["Q_metal_jaw_frac"] = float((a[jaw] >= 3000).mean())
        del jaw
    if {"MAND", "MAX"} <= keep.keys():
        r |= gaps(keep["MAND"], keep["MAX"], img, P or {})
    return r, thumb(a, i_mid), mthumbs


def load_cached(p, cache):
    """JSON과 단면 파일이 둘 다 있고 정상적으로 읽히면 (행, 단면) 반환, 아니면 None."""
    fj, ft = cache / f"P_{p}.json", cache / f"P_{p}_thumb.npz"
    if not (fj.exists() and ft.exists()):
        return None
    try:
        r = json.loads(fj.read_text(encoding="utf-8"))
        with np.load(ft) as z:
            th = (z["raw"], {k[2:]: z[k] for k in z.files if k.startswith("m_")})
        assert r["Patient"] == f"P_{p}"
        if r.get("eda_version") != EDA_VERSION:              # 계산 항목이 바뀐 이전 버전 캐시는 쓰지 않음
            return None
        return r, th
    except Exception as ex:                                  # 손상된 캐시는 버리고 다시 계산
        print(f"P_{p} 캐시 손상 ({ex}) → 다시 계산", flush=True)
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    ids = sorted(int(f.stem.split("_")[1]) for f in (CB / "nrrd").glob("P_*.nrrd"))
    rows, thumbs = [], {}
    tracemalloc.start()
    for p in ids:
        c = load_cached(p, cache)
        if c:
            rows.append(c[0])
            thumbs[p] = c[1]
            print(f"P_{p} 캐시 사용", flush=True)
            continue
        tracemalloc.reset_peak()
        ws0 = working_set_mb()
        r, raw, mth = analyze(p)
        r |= {"M_numpy_peak_MB": tracemalloc.get_traced_memory()[1] / 2 ** 20, "M_ws_before_MB": ws0,
              "M_ws_after_MB": working_set_mb()}
        r = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}

        def write_npz(t):
            with open(t, "wb") as fh:
                np.savez_compressed(fh, raw=raw, **{f"m_{k}": v for k, v in mth.items()})
        atomic_save(cache / f"P_{p}_thumb.npz", write_npz)                       # 단면 먼저, JSON은 마지막에
        atomic_save(cache / f"P_{p}.json", lambda t: t.write_text(json.dumps(r, ensure_ascii=False, default=str), encoding="utf-8"))
        thumbs[p] = (raw, mth)
        rows.append(r)
        print(f"P_{p} 완료 (numpy 최대 {r['M_numpy_peak_MB']:.0f} MB, 프로세스 {r['M_ws_after_MB']:.0f} MB)", flush=True)
    tracemalloc.stop()

    d = pd.DataFrame(rows).set_index("Patient")
    flags = []
    for c in d.columns:
        x = pd.to_numeric(d[c], errors="coerce")
        if c.endswith("_largest_frac"):
            for pid in x.index[x < LARGEST_FRAC_MIN]:
                flags.append({"Patient": pid, "Category": "S", "Item": c, "Value": x[pid], "Reason": f"< {LARGEST_FRAC_MIN}"})
            continue
        if c.startswith(("Q_", "P_", "S_", "A_FOV")) and x.notna().sum() > 10:
            mad = 1.4826 * (x - x.median()).abs().median()
            if mad > 0:
                z = (x - x.median()) / mad
                for pid in z.index[z.abs() > 3.5]:
                    flags.append({"Patient": pid, "Category": c.split("_")[0], "Item": c, "Value": x[pid],
                                  "Reason": f"로버스트 z {z[pid]:.1f}"})
    for c, ch in RULES.items():
        for pid, v in d.get(c, pd.Series(dtype=str)).fillna("").items():
            if v and (ch == "" or ch in v):
                flags.append({"Patient": pid, "Category": "F", "Item": c, "Value": v, "Reason": "영상 경계에 닿음"})
    for c in ("A_Manufacturer", "A_ManufacturerModelName", "A_KVP", "A_voxel_mm", "A_matrix"):
        mode = d[c].mode().iloc[0]
        for pid in d.index[d[c] != mode]:
            flags.append({"Patient": pid, "Category": "A", "Item": c, "Value": d.loc[pid, c], "Reason": f"다수값 {mode}와 다름"})
    fl = pd.DataFrame(flags)
    d["review_items"] = fl.groupby("Patient").Item.apply(lambda s: ", ".join(sorted(set(s)))) if len(fl) else ""
    d["review_items"] = d["review_items"].fillna("")
    d.round(3).to_csv(OUT / "cbct_eda_patient.csv", encoding="utf-8-sig")
    fl.round(3).to_csv(OUT / "cbct_eda_flags.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "Malgun Gothic"
    ncol = 10
    nrow = int(np.ceil(len(ids) / ncol))
    for kind in ("raw", "mask"):
        fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 2.1))
        for k, axx in enumerate(ax.ravel()):
            axx.axis("off")
            if k >= len(ids):
                continue
            p = ids[k]
            s, ms = thumbs[p]
            axx.imshow(s, cmap="gray", vmin=-1000, vmax=2000)
            if kind == "mask":
                for name, m in ms.items():
                    if m.any():
                        axx.contour(m, levels=[0.5], colors=COLORS[name], linewidths=0.6)
            rev = d.loc[f"P_{p}", "review_items"]
            axx.set_title(f"P_{p}" + (" 검수" if rev else ""), fontsize=8, color="red" if rev else "black")
        title = "원본 정중 시상 단면" if kind == "raw" else "AMASSS 마스크 (빨강 하악, 파랑 상악, 초록 상기도, 노랑 두개저)"
        fig.suptitle(f"{title}  |  빨간 제목 = 자동 QC 검수 대상 (제외 아님)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"montage_{kind}.png", dpi=110)
        plt.close(fig)

    print(f"\n환자 {len(d)}명, 검수 표시 {d.review_items.ne('').sum()}명")
    print("코호트 수준 — 가장 큰 덩어리의 경계 접촉 방향:",
          {s: d[f"F_{s}_touch_main"].fillna("").replace("", "없음").value_counts().to_dict() for s in STRUCTS if f"F_{s}_touch_main" in d})
    print("코호트 수준 — 촬영 연도별 연조직 피크(HU) 중앙값:", d.groupby("A_StudyYear").Q_soft_peak_HU.median().round(0).to_dict())
    print("메모리: numpy 최대 할당 " + d.M_numpy_peak_MB.describe().round(0).to_dict().__repr__()
          + f" | 프로세스 최대 {d.M_ws_after_MB.max():.0f} MB" if "M_numpy_peak_MB" in d else "")
    if len(fl):
        print(fl.groupby(["Category", "Item"]).Patient.apply(lambda s: ", ".join(s)).to_string())
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
