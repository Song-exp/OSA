"""수치 검증에서 걸린 환자만 모은 육안 점검 페이지. 이유별로 필요한 그림만 그린다(환자 한 명씩 읽고 버림).

이유 → 그림
  기도 프로파일 불일치(곡선 r<0.7): 정중 단면 + TotalSegmentator 인두(청록)·AMASSS 상기도(노랑) 윤곽
  치아 라벨 오류(제1대구치 ALI와 10 mm 초과): 제1대구치 높이 관상 단면 + 치아 분할 윤곽 + ALI 저항중심(+)
  연구개 이상치: 정중 단면 + 연구개(빨강) 윤곽 + PNS
  SSM 형태 이상치(하악·상악·기도): 환자 대응점(빨강)과 85명 평균(회색)의 옆·앞 투영
판단은 페이지에서 JSON으로 내려받는다.
출력: Result_py/shape/validation/visual_review/ (figs, visual_review.html)
"""
import gc
from html import escape
import os
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from cbct_eb_detect import midline_frame
from cbct_evidence_figs import teeth_points
from cbct_patient_pipeline import CB, landmarks
from check_landmarks import read_nrrd
from run_ali import ascii_path
from shape_fine_vectors import ALI_R
from shape_vector_validation import DEST as VAL, SH

DEST = VAL/"visual_review"; FIG = DEST/"figs"


def reasons():
    R = {}
    add = lambda p, k, txt: R.setdefault(p, {}).setdefault(k, []).append(txt)
    ph = pd.read_csv(VAL/"pharynx_agreement.csv", encoding="utf-8-sig")
    for _, r in ph[ph.profile_r < .7].iterrows(): add(r.Patient, "airway", f"기도 프로파일 곡선 상관 {r.profile_r:.2f} (Dice {r.dice:.2f})")
    t = pd.read_csv(SH/"fine/teeth_long.csv", encoding="utf-8-sig")
    for _, r in t[t.tooth.isin(list(ALI_R)) & t.present & (t.ali_gap_mm > 10)].iterrows():
        add(r.Patient, "teeth", f"{ALI_R[r.tooth][:3]} 치아 분할 중심이 ALI 저항중심과 {r.ali_gap_mm:.1f} mm")
    for _, r in pd.read_csv(VAL/"softpalate_flags.csv", encoding="utf-8-sig").iterrows():
        add(r.Patient, "palate", f"{r.variable} = {r.value:.1f} (robust z {r.robust_z:.1f})")
    so = pd.read_csv(VAL/"ssm_outliers.csv", encoding="utf-8-sig")
    for _, r in so[(so.variable == "procrustes_dist") & (so.struct != "CB")].iterrows():
        add(r.Patient, f"ssm_{r.struct}", f"{r.struct} SSM 평균 형태와의 거리 이상 (robust z {r.robust_z:.1f})")
    return R


def lab_sample(f, Q):
    im = sitk.ReadImage(str(ascii_path(f))); a = sitk.GetArrayFromImage(im)
    org, sp, D = np.array(im.GetOrigin()), np.array(im.GetSpacing()), np.array(im.GetDirection()).reshape(3, 3)
    k = np.rint(((Q-org)@np.linalg.inv(D).T)/sp).astype(int)[:, ::-1]; ok = ((k >= 0) & (k < a.shape)).all(1)
    out = np.zeros(len(Q), a.dtype); out[ok] = a[tuple(k[ok].T)]; return out


def plane(v, dirs, org, o, R, Lv, Av, Sv):
    G = np.stack(np.meshgrid(Lv, Av, Sv, indexing="ij"), -1).reshape(-1, 3); Q = o+G@R
    ct = map_coordinates(v, np.linalg.solve(dirs.T, (Q-org).T)[::-1], order=1, cval=-1000)
    return ct, Q


def figure(p, kinds):
    pid = int(p[2:]); f = FIG/f"{p}.png"
    P = landmarks(pid)[0]; o, R, _ = midline_frame(P); v, dirs, org = read_nrrd(CB/f"nrrd/{p}.nrrd"); v = v.astype(np.float32)
    panels = []
    if {"airway", "palate"} & set(kinds):
        A, S = np.arange(-70, 40, .5), np.arange(-100, 30, .5); ct, Q = plane(v, dirs, org, o, R, np.array([0.]), A, S)
        ts = lab_sample(CB/f"seg_totalseg/head_glands_cavities/{p}.nii.gz", Q).reshape(len(A), len(S))
        uaw = lab_sample(CB/f"seg/{p}_seg_UAW.nrrd", Q).reshape(len(A), len(S)) > 0
        panels.append(("mid", ct.reshape(len(A), len(S)), A, S, ts, uaw))
    if "teeth" in kinds:
        tp = {k: R@(x-o) for k, x in teeth_points(pid).items()}
        a0 = float(np.mean([tp[k][1] for k in ALI_R.values() if k in tp])); Lv, Sv = np.arange(-45, 45.1, .4), np.arange(-70, -5, .4)
        ct, Q = plane(v, dirs, org, o, R, Lv, np.array([a0]), Sv)
        tl = lab_sample(CB/f"seg_teeth/UNIVERSALLAB_MODEL/{p}.nii.gz", Q).reshape(len(Lv), len(Sv))
        panels.append(("teeth", ct.reshape(len(Lv), len(Sv)), Lv, Sv, tl, tp))
    del v
    ssm = [k[4:] for k in kinds if k.startswith("ssm_")]
    n = len(panels)+2*len(ssm)
    fig, ax = plt.subplots(1, max(n, 1), figsize=(6*max(n, 1), 6), squeeze=False); ax = ax[0]; i = 0
    for kind, ct, X, Y, lab, extra in panels:
        a = ax[i]; i += 1
        a.imshow(ct.T, origin="lower", cmap="gray", vmin=-1000 if kind == "mid" else -500, vmax=1500 if kind == "mid" else 2500, extent=[X[0], X[-1], Y[0], Y[-1]])
        if kind == "mid":
            for l, c in ((11, "c"), (12, "c"), (13, "c"), (18, "r"), (19, "y")):
                if (lab == l).any(): a.contour(X, Y, (lab == l).T.astype(float), [.5], colors=c, linewidths=1.2)
            if extra.any(): a.contour(X, Y, extra.T.astype(float), [.5], colors="gold", linewidths=1, linestyles="--")
            a.plot(0, 0, "w+", ms=12, mew=2); a.annotate("PNS", (0, 0), color="w", xytext=(4, 4), textcoords="offset points")
            a.set_title(f"{p} 정중 단면: 인두 TotalSeg(청록) · 상기도 AMASSS(금색 점선) · 연구개(빨강) · 경구개(노랑)", fontsize=9)
            a.set_xlabel("앞 (mm, PNS 기준)")
        else:
            for t in (2, 3, 4, 13, 14, 15, 18, 19, 20, 29, 30, 31):
                if (lab == t).any(): a.contour(X, Y, (lab == t).T.astype(float), [.5], colors=("lime" if t in ALI_R else "orange"), linewidths=1)
            for t, k in ALI_R.items():
                if k in extra: a.plot(extra[k][0], extra[k][2], "c+", ms=13, mew=2); a.annotate(k[:3], (extra[k][0], extra[k][2]), color="c", fontsize=8, xytext=(3, 3), textcoords="offset points")
            a.set_title(f"{p} 제1대구치 관상 단면: 분할 제1대구치(연두)·인접 치아(주황), ALI 저항중심(청록 +)", fontsize=9); a.set_xlabel("좌우 (mm, + 왼쪽)")
    for s in ssm:
        z = np.load(SH/s/"points.npz"); ids = [f"P_{x}" for x in z["ids"]]; X = z["frame"]; m = X.mean(0); x = X[ids.index(p)]
        for j, (c1, c2, nm) in enumerate(((1, 2, "옆(앞–위)"), (0, 2, "앞(좌–위)"))):
            a = ax[i]; i += 1
            a.scatter(m[:, c1], m[:, c2], s=2, c="0.6", label="85명 평균"); a.scatter(x[:, c1], x[:, c2], s=2, c="r", label=p)
            a.set_aspect("equal"); a.set_title(f"{s} SSM 대응점 {nm} 투영", fontsize=9); a.legend(fontsize=8, markerscale=4)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(f, dpi=65); plt.close(fig); gc.collect()
    return f


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    R = reasons(); cards = []
    for p in sorted(R, key=lambda x: int(x[2:])):
        f = figure(p, list(R[p])); print(p, end=" ", flush=True); src = quote(os.path.relpath(f, DEST).replace("\\", "/"))
        items = "".join(f"<li>{escape(t)}</li>" for ts in R[p].values() for t in ts)
        opts = "".join(f'<label><input type="radio" name="{p}|{k}" value="{v}"> {t}</label> ' for k in R[p]
                       for v, t in (("ok", f"{k}: 분할·값 정상"), ("bad", "분할 오류 → 해당 값 제외"), ("hold", "보류")))
        cards.append(f'<section class="card" id="{p}"><h2>{p}</h2><ul>{items}</ul><a href="{src}" target="_blank"><img src="{src}" alt="{p}"></a>'
                     f'<div>{opts}</div><textarea name="{p}|note" placeholder="메모"></textarea></section>')
    nav = " ".join(f'<a href="#{p}">{p}</a>' for p in sorted(R, key=lambda x: int(x[2:])))
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>세밀 벡터 육안 점검</title>
<style>body{{font:15px/1.55 "Malgun Gothic",sans-serif;margin:0;background:#f6f6f3;color:#1d1d1b}}main{{max-width:1400px;margin:0 auto;padding:20px 16px 60px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px 16px;margin:16px 0}}img{{max-width:100%;border:1px solid #ccc}}label{{margin-right:12px;white-space:nowrap}}
textarea{{width:100%;height:44px}}nav{{position:sticky;top:0;background:#f6f6f3;padding:6px 0}}nav a{{margin-right:8px}}button{{font:inherit;padding:6px 14px}}</style></head><body><main>
<h1>세밀 벡터 육안 점검 ({len(R)}명)</h1>
<p>수치 검증(서로 독립적인 두 방법 비교·이상치)에서 걸린 환자만 모았습니다. 두개저 SSM은 방법 자체의 정합 문제로 개별 점검 대상에서 뺐습니다.
분할이 해부학적으로 틀렸으면 <b>분할 오류</b>를 골라 주세요. 해당 값은 분석에서 뺍니다. 판단은 아래 버튼으로 JSON 저장됩니다.</p>
<nav>{nav} <button onclick="save()">판단 저장 (JSON)</button></nav>{"".join(cards)}
<script>function save(){{const d={{}};document.querySelectorAll('input[type=radio]:checked,textarea').forEach(e=>{{if(e.value)d[e.name]=e.value}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,1)],{{type:'application/json'}}));a.download='visual_review_decisions.json';a.click()}}</script>
</main></body></html>"""
    (DEST/"visual_review.html").write_text(html, encoding="utf-8"); print("\nsaved", DEST/"visual_review.html", len(R), "명")


if __name__ == "__main__":
    main()
