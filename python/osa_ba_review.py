"""S1g: Pog_Ba_ant에 쓰는 ALI Ba 점 영상 확인 (10명).
선택: Pog_Ba_ant 최소 2·최대 2, ANB로 예측한 Pog_Ba_ant와 잔차가 가장 큰 2, 주 분석에서 무작위 4(seed 0).
그림: Ba 중심 시상·축·관상 단면(±28 mm) + Ba–Pog 넓은 시상 단면. 출력 Result_py/analysis/osa_experiments_v2/S1g_ba_review/
판정은 사람이 index.html에서 한다(정의: 대후두공 앞 가장자리의 가장 아래 점).
"""
import base64, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import osa_experiments as X
from cbct_landmark_review import panel
from cbct_patient_pipeline import CB, landmarks
from check_landmarks import read_nrrd

OUT = X.R_/"analysis/osa_experiments_v2/S1g_ba_review"; OUT.mkdir(parents=True, exist_ok=True)
d = X.data(); s = d[d.restored.astype(bool)].dropna(subset=["Pog_Ba_ant", "m_ANB"])
res = s.Pog_Ba_ant-np.poly1d(np.polyfit(s.m_ANB, s.Pog_Ba_ant, 1))(s.m_ANB)
pick = {p: "Pog_Ba 최소" for p in s.Pog_Ba_ant.nsmallest(2).index} | {p: "Pog_Ba 최대" for p in s.Pog_Ba_ant.nlargest(2).index}
for p in res.abs().sort_values(ascending=False).index:
    if len([v for v in pick.values() if v == "ANB 대비 잔차 큼"]) == 2: break
    pick.setdefault(p, "ANB 대비 잔차 큼")
rest = [p for p in s.index[s.analysis_list.astype(bool)] if p not in pick]
for p in np.random.default_rng(0).choice(rest, 4, replace=False): pick[p] = "무작위(주 분석)"

rows, cards = [], []
for p, why in pick.items():
    pid = int(p.split("_")[1]); P, _ = landmarks(pid); vol, D, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd")
    marks = {"Ba": (P["Ba"], "red", "+"), "Pog": (P["Pog"], "lime", "+"), "S": (P["S"], "cyan", "x")}
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    for a, view in zip(ax[:3], ("sagittal", "axial", "coronal")): panel(a, vol, D, org, P["Ba"], view, {"Ba": marks["Ba"]})
    mid = (P["Ba"]+P["Pog"])/2; panel(ax[3], vol, D, org, mid, "sagittal", marks, half=70)
    ax[3].plot([P["Ba"][1], P["Pog"][1]], [P["Ba"][2], P["Pog"][2]], "y--", lw=1)
    fig.suptitle(f"{p}  Ba 확인 ({why})  Pog_Ba_ant {s.Pog_Ba_ant[p]:.1f} mm, ANB {s.m_ANB[p]:.1f}°")
    fig.tight_layout(); png = OUT/f"{p}_Ba.png"; fig.savefig(png, dpi=110); plt.close(fig)
    rows.append({"Patient": p, "reason": why, "Pog_Ba_ant": s.Pog_Ba_ant[p], "m_ANB": s.m_ANB[p], "resid": res[p], "figure": png.name, "judgement": ""})
    cards.append(f'<h3>{p} — {why}</h3><img src="data:image/png;base64,{base64.b64encode(png.read_bytes()).decode()}" style="width:100%">'
                 f'<p>판정: <label><input type="radio" name="{p}"> 적절</label> <label><input type="radio" name="{p}"> 수정 필요</label> 메모 <input size="60"></p>')
pd.DataFrame(rows).round(2).to_csv(OUT/"selection.csv", index=False, encoding="utf-8-sig")
(OUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Ba 점 확인</title><body style="font-family:sans-serif;max-width:1400px;margin:auto">'
                              "<h1>Ba 점 영상 확인 (10명)</h1><p>정의: 대후두공 앞 가장자리의 가장 아래 점. 빨간 + = ALI Ba. 마지막 칸은 Ba(빨강)–Pog(초록) 시상 단면.</p>"
                              + "".join(cards), encoding="utf-8")
print(pd.DataFrame(rows).round(2).to_string(index=False))
