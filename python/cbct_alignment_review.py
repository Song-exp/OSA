"""엑셀 정렬에서 뺀 의심 사례 검토 페이지. 사례마다 원본 CBCT 단면에 우리 랜드마크를 겹쳐 그려, 오차가 우리 랜드마크 탓인지
엑셀 쪽인지 사람이 판단하게 한다. 판단은 페이지에서 JSON으로 내려받는다(파이프라인 수정 좌표·문의 목록 입력용).

입력: restore_ext/v1/alignment/excel_flags.csv, compare_all.csv, NRRD, ALI 랜드마크(Po·Or 이상값 대체 적용)
출력: restore_ext/v1/alignment/review/ (figs/P_n.png, review.html). 원본은 읽기만 한다. 환자 한 명씩 읽고 버린다.
"""
import gc
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from cbct_evidence_figs import teeth_points
from cbct_fh_repair import DEST as FH_REF, repair
from cbct_patient_pipeline import CB, OUT, landmarks
from check_landmarks import read_nrrd
from osa.ceph import head_frame

EXT = OUT/"restore_ext/v1"; DEST = EXT/"alignment/review"; FIG = DEST/"figs"
MID = ["N", "S", "Ba", "ANS", "PNS", "A", "B", "Pog", "Me", "Gn"]
SIDE = {"RPo": "c", "ROr": "c", "RGo": "m", "LPo": "y", "LOr": "y", "LGo": "y"}


def sample(v, dirs, org, o, R, L, A, S):
    """머리 좌표계 격자(좌 L, 앞 A, 위 S)에서 CT 값."""
    G = np.stack(np.meshgrid(L, A, S, indexing="ij"), -1).reshape(-1, 3)
    return map_coordinates(v, np.linalg.solve(dirs.T, (o+G@R-org).T)[::-1], order=1, cval=-1000).reshape(len(L), len(A), len(S))


def figure(p, metrics):
    pid = int(p[2:]); f = FIG/f"{p}.png"
    if f.exists(): return f
    ref = json.loads((FH_REF/"reference.json").read_text(encoding="utf-8"))["points"]
    P = repair(landmarks(pid)[0], ref)[0]; o, R = head_frame(P); H = {k: R@(v-o) for k, v in P.items()}
    v, dirs, org = read_nrrd(CB/f"nrrd/{p}.nrrd"); v = v.astype(np.float32)
    A, S = np.arange(-115, 30, .5), np.arange(-145, 35, .5)
    sag = sample(v, dirs, org, o, R, np.array([0.]), A, S)[0]                       # 정중 시상 단면
    lat = sample(v, dirs, org, o, R, np.arange(-65, 66, 1.), A, S).max(0)           # 측면 뼈 투영 (최대값)
    tdi = "TDI" in metrics; tp = {k: R@(x-o) for k, x in teeth_points(pid).items()} if tdi else {}
    n = 3 if tdi and tp else 2
    fig, ax = plt.subplots(1, n, figsize=(6.2*n, 6.6))
    ext = [A[0], A[-1], S[0], S[-1]]
    ax[0].imshow(sag.T, origin="lower", cmap="gray", vmin=-1000, vmax=1800, extent=ext)
    for k in MID:
        if k in H: ax[0].plot(H[k][1], H[k][2], "r+", ms=11, mew=2); ax[0].annotate(k, (H[k][1], H[k][2]), color="yellow", fontsize=9, xytext=(3, 3), textcoords="offset points")
    ax[0].set_title(f"{p} 정중 시상 단면 (공기·뼈)")
    ax[1].imshow(lat.T, origin="lower", cmap="gray", vmin=0, vmax=2500, extent=ext)
    for k, c in SIDE.items():
        if k in H: ax[1].plot(H[k][1], H[k][2], "+", color=c, ms=12, mew=2); ax[1].annotate(k, (H[k][1], H[k][2]), color=c, fontsize=9, xytext=(3, 3), textcoords="offset points")
    for k in ("N", "Me", "Pog", "A", "B"): ax[1].plot(H[k][1], H[k][2], "r+", ms=9, mew=1.5)
    ax[1].plot([H["RPo"][1], H["ROr"][1]], [H["RPo"][2], H["ROr"][2]], "c-", lw=1)          # FH (오른쪽)
    ax[1].plot([H["RGo"][1], H["Me"][1]], [H["RGo"][2], H["Me"][2]], "m-", lw=1)            # 하악평면 (Go–Me)
    ax[1].set_title("측면 뼈 투영: FH(청록)·하악평면(자홍), 노랑=왼쪽 점")
    for a in ax[:2]: a.set_xlabel("앞 (mm, N 기준)"); a.set_ylabel("위 (mm)")
    if n == 3:
        mol = [tp[k] for k in ("UR6R", "UL6R", "LR6R", "LL6R") if k in tp]
        a0 = float(np.mean([m[1] for m in mol])); L = np.arange(-45, 45.1, .4); Sz = np.arange(-110, -40, .4)
        cor = sample(v, dirs, org, o, R, L, np.array([a0]), Sz)[:, 0, :]
        ax[2].imshow(cor.T, origin="lower", cmap="gray", vmin=-500, vmax=2500, extent=[L[0], L[-1], Sz[0], Sz[-1]])
        for k, q in tp.items():
            if k[2:3] == "6" and k.endswith(("R", "O")): ax[2].plot(q[0], q[2], "c+" if k.endswith("R") else "y+", ms=10, mew=2); ax[2].annotate(k, (q[0], q[2]), color="w", fontsize=8)
        ax[2].set_title("제1대구치 관상 단면 (청록=저항중심 R, 노랑=교두 O)"); ax[2].set_xlabel("좌우 (mm, + 왼쪽)")
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(f, dpi=70); plt.close(fig)
    del v, sag, lat; gc.collect(); return f


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    fl = pd.read_csv(EXT/"alignment/excel_flags.csv", encoding="utf-8-sig")
    st = pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig").drop_duplicates("Patient").set_index("Patient").status
    unit = pd.read_csv(EXT/"compare_all.csv", encoding="utf-8-sig").drop_duplicates("metric").set_index("metric").unit
    out_ = fl[fl.reason == "오차 이상치"]; zero = fl[fl.reason != "오차 이상치"]
    cards = []
    for p, g in out_.groupby("Patient", sort=False):
        f = figure(p, set(g.metric)); print(p, end=" ", flush=True)
        rows = "".join(f"<tr><td>{m}</td><td>{x:g} {escape(str(unit[m]))}</td><td>{r:.2f}</td><td>{r-x:+.2f}</td><td>{z:.1f}</td></tr>"
                       for m, x, r, z in zip(g.metric, g.excel, g.restored, g.robust_z))
        opts = "".join(f'<label><input type="radio" name="{p}|{m}" value="{v}"> {t}</label> ' for m in g.metric
                       for v, t in (("ours", f"{m}: 우리 랜드마크 오류"), ("excel", "엑셀 의심"), ("hold", "보류")))
        cards.append(f'<section class="card" id="{p}"><h2>{p} <small>{escape(st.get(p, ""))}</small></h2>'
                     f'<table><tr><th>지표</th><th>엑셀</th><th>복원</th><th>차이</th><th>robust z</th></tr>{rows}</table>'
                     f'<a href="{quote(os.path.relpath(f, DEST).replace(chr(92), "/"))}" target="_blank"><img src="{quote(os.path.relpath(f, DEST).replace(chr(92), "/"))}" alt="{p} 단면"></a>'
                     f'<div class="pick">{opts}</div><textarea name="{p}|note" placeholder="메모 (어느 점이 틀렸는지 등)"></textarea></section>')
    zl = "".join(f"<tr><td>{p}</td><td>{m}</td><td>{x:g}</td><td>{r:.2f}</td></tr>" for p, m, x, r in zip(zero.Patient, zero.metric, zero.excel, zero.restored))
    nav = " ".join(f'<a href="#{p}">{p}</a>' for p in out_.Patient.unique())
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>엑셀 오차 이상치 검토</title><style>
body{{font:15px/1.55 "Malgun Gothic",sans-serif;margin:0;background:#f6f6f3;color:#1d1d1b}}main{{max-width:1300px;margin:0 auto;padding:20px 16px 60px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px 16px;margin:16px 0}}img{{max-width:100%;border:1px solid #ccc;margin:8px 0}}
table{{border-collapse:collapse;font-size:.9rem}}td,th{{border:1px solid #ddd;padding:3px 8px}}th{{background:#eef1f4}}
.pick label{{margin-right:12px;white-space:nowrap}}textarea{{width:100%;height:46px}}nav{{position:sticky;top:0;background:#f6f6f3;padding:6px 0;z-index:2}}
nav a{{margin-right:8px}}button{{font:inherit;padding:6px 14px}}small{{color:#666;font-weight:normal}}</style></head><body><main>
<h1>엑셀 오차 이상치 검토 ({out_.Patient.nunique()}명, {len(out_)}건)</h1>
<p>엑셀과의 오차가 지표 분포에서 크게 벗어나(robust z &gt; 3.5) 정렬 학습에서 뺀 사례입니다. 그림의 + 표시가 우리 랜드마크(Po·Or 이상값 대체 적용)입니다.
점이 해부학적으로 틀린 곳에 있으면 <b>우리 랜드마크 오류</b>(좌표 수정 후 재계측), 점이 맞는데 엑셀만 동떨어지면 <b>엑셀 의심</b>(연구실 문의)으로 골라 주세요.
고른 결과는 아래 버튼으로 JSON 파일로 저장됩니다.</p>
<nav>{nav} <button onclick="save()">판단 저장 (JSON)</button></nav>
{"".join(cards)}
<h2>엑셀 값이 0인 사례 ({len(zero)}건, 입력 누락 의심)</h2><p>학습에서 뺐습니다. 실제 0인지 미측정인지 연구실 확인 대상입니다.</p>
<table><tr><th>환자</th><th>지표</th><th>엑셀</th><th>복원</th></tr>{zl}</table>
<script>function save(){{const d={{}};document.querySelectorAll('input[type=radio]:checked,textarea').forEach(e=>{{if(e.value)d[e.name]=e.value}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,1)],{{type:'application/json'}}));a.download='alignment_review_decisions.json';a.click()}}</script>
</main></body></html>"""
    (DEST/"review.html").write_text(html, encoding="utf-8"); print("\nsaved", DEST/"review.html")


if __name__ == "__main__":
    main()
