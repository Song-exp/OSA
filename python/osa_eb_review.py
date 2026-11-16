"""S2a: Eb(후두개곡 최하점) 영상 확인 — 설골 기준 추정(hyoid_offset)·실패(none) 환자 전원.
Eb 좌표는 features.csv의 정중 좌표계 값(c_Eb_ant/sup, 좌우 0)을 LPS로 되돌려 표시한다.
출력 Result_py/analysis/osa_experiments_v2/S2a_eb_review/ (P_n_Eb.png, index.html 판정 페이지, targets.csv)
판정은 사람이 한다: A 적절 / B 1~3 mm 차이 / C 틀림 / D 판정 불가. 페이지의 "결과 CSV" 버튼으로 내보낸다.
"""
import base64, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import osa_experiments as X
from cbct_eb_detect import midline_frame
from cbct_landmark_review import panel
from cbct_patient_pipeline import CB, landmarks
from check_landmarks import read_nrrd

V2 = X.R_/"CBCT/patient_pipeline/v2"
OUT = X.R_/"analysis/osa_experiments_v2/S2a_eb_review"; OUT.mkdir(parents=True, exist_ok=True)


def eb_methods():
    a = pd.read_csv(V2/"eb_review/v2/eb_candidates.csv", encoding="utf-8-sig")[["Patient", "method"]]
    b = pd.read_csv(V2/"restore_ext/v1/new_alternatives.csv", encoding="utf-8-sig")[["Patient", "Eb_method"]].rename(columns={"Eb_method": "method"})
    return pd.concat([a, b]).drop_duplicates("Patient").set_index("Patient").method


if __name__ == "__main__":
    d = X.data(); m = eb_methods(); todo = m[m.isin(["hyoid_offset", "none"])].index
    rows, cards = [], []
    for p in todo:
        pid = int(p.split("_")[1]); P, _ = landmarks(pid); o, R, _ = midline_frame(P); vol, D, org = read_nrrd(CB/f"nrrd/P_{pid}.nrrd")
        lps = lambda a, s: o+R.T@np.array([0., a, s]) if np.isfinite([a, s]).all() else None
        eb, hy = lps(d.at[p, "c_Eb_ant"], d.at[p, "c_Eb_sup"]), lps(d.at[p, "c_Hy_ant"], d.at[p, "c_Hy_sup"])
        marks = {k: (v, c, mk) for k, v, c, mk in (("Eb", eb, "red", "+"), ("설골", hy, "yellow", "x"), ("PNS", P["PNS"], "cyan", "x")) if v is not None}
        c0 = eb if eb is not None else (hy if hy is not None else P["PNS"])
        fig, ax = plt.subplots(1, 3, figsize=(20, 7))
        panel(ax[0], vol, D, org, c0, "sagittal", marks, half=25); panel(ax[1], vol, D, org, c0, "axial", marks, half=25)
        panel(ax[2], vol, D, org, (P["PNS"]+c0)/2, "sagittal", marks, half=55)
        fig.suptitle(f"{p}  Eb 확인 (방법: {m[p]})  PAL {d.at[p, 'm_PAL'] if 'm_PAL' in d else np.nan:.1f} mm"); fig.tight_layout()
        png = OUT/f"{p}_Eb.png"; fig.savefig(png, dpi=100); plt.close(fig)
        rows.append({"Patient": p, "method": m[p], "Eb_found": eb is not None, "figure": png.name})
        cards.append(f'<section data-p="{p}"><h3>{p} — {m[p]}</h3><img src="data:image/png;base64,{base64.b64encode(png.read_bytes()).decode()}">'
                     + "".join(f'<label><input type="radio" name="{p}" value="{g}"> {g}</label> ' for g in ("A 적절", "B 1~3mm 차이", "C 틀림", "D 판정 불가"))
                     + ' 메모 <input class="memo" size="60"></section>')
    pd.DataFrame(rows).to_csv(OUT/"targets.csv", index=False, encoding="utf-8-sig")
    js = """<script>
document.getElementById('dl').onclick=()=>{let r=['Patient,grade,memo'];document.querySelectorAll('section').forEach(s=>{
let g=s.querySelector('input[type=radio]:checked');r.push([s.dataset.p,g?g.value:'',JSON.stringify(s.querySelector('.memo').value)].join(','))});
let a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+r.join('\\n')],{type:'text/csv'}));a.download='eb_judgement.csv';a.click()};
</script>"""
    (OUT/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Eb 점 확인</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:0 16px}img{width:100%}section{border-bottom:1px solid #ccc;padding:12px 0}</style>'
                                  f"<h1>Eb(후두개곡 최하점) 확인 — {len(rows)}명</h1><p>정의: 후두개 앞쪽 홈(후두개곡) 공기의 가장 아랫점. 빨간 + = 현재 Eb, 노란 x = 설골, 하늘색 x = PNS. "
                                  "'none'은 Eb가 없어 설골 주변을 보여 준다. 판정 후 아래 버튼으로 CSV를 저장해 Claude에게 전달.</p>"
                                  '<button id="dl">결과 CSV 저장</button>' + "".join(cards) + js, encoding="utf-8")
    print(pd.DataFrame(rows).method.value_counts().to_dict(), "Eb 없음", int((~pd.DataFrame(rows).Eb_found).sum()))
