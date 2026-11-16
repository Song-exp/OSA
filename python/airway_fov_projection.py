"""Eb 미계측 4명 vs 대조 3명: 이전 6절(cbct_evidence_figs.fov)과 같은 투영 방식에 기도 분할을 색으로 겹친 비교 그림.

환자마다 3칸: ① 정중(PNS 좌우 위치) ±3 mm 최소투영(공기) + AMASSS 상기도(주황)
             ② 같은 두께 최대투영(뼈) — 설골·경추·하악 확인
             ③ 영상 맨 아래 6 mm 축상 최소투영 + 같은 높이의 기도 분할 — 기도가 영상 아래 끝에서 잘렸는지
NRRD와 AMASSS 분할은 같은 격자(512×512×460, 0.39 mm, 단위 방향)라 인덱스로 바로 자른다. 원본은 읽기만 한다.
출력: Result_py/shape/validation/airway_fov/airway_fov_projection.png (+ index.html에 추가)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import SimpleITK as sitk

from airway_fov_compare import CUT, DEST, REF
from cbct_patient_pipeline import CB, landmarks
from run_ali import ascii_path

AIR = ListedColormap(["#eb6834"])


def rows(p):
    img = sitk.ReadImage(str(ascii_path(CB/f"nrrd/{p}.nrrd"))); seg = sitk.ReadImage(str(ascii_path(CB/f"seg/{p}_seg_UAW.nrrd")))
    assert img.GetSize() == seg.GetSize() and np.allclose(img.GetOrigin(), seg.GetOrigin()) and np.allclose(img.GetDirection(), np.eye(3).ravel())
    v = sitk.GetArrayFromImage(img).astype(np.float32); m = sitk.GetArrayFromImage(seg) > 0       # (z, y, x)
    sp = np.array(img.GetSpacing()); org = np.array(img.GetOrigin()); P = landmarks(int(p[2:]))[0]
    ix = int(round((P["PNS"][0]-org[0])/sp[0])); h = int(round(3/sp[0]))
    slab, mslab = v[:, :, ix-h:ix+h+1], m[:, :, ix-h:ix+h+1].any(2)
    y = org[1]+np.arange(v.shape[1])*sp[1]; z = org[2]+np.arange(v.shape[0])*sp[2]; x = org[0]+np.arange(v.shape[2])*sp[0]
    n = int(round(6/sp[2]))
    return dict(minip=slab.min(2), maxip=slab.max(2), air=mslab, y=y, z=z, x=x, bottom=v[:n].min(0), bottom_air=m[:n].any(0), P=P,
                air_at_bottom=bool(m[:n].any()), n_slices=v.shape[0])


def zoom(ids):
    """영상 아래쪽 45 mm만 확대: 공기 최소투영 + 기도 윤곽, 뼈 최대투영. 모듈이 찾은 Eb·설골을 LPS로 되돌려 표시."""
    import json
    from cbct_eb_detect import midline_frame
    from cbct_patient_pipeline import OUT
    from cbct_restore_extend import eb_row
    offset = json.loads((OUT/"eb_review/v2/summary.json").read_text(encoding="utf-8"))["offset_mm"]
    fig, ax = plt.subplots(2, len(ids), figsize=(3.4*len(ids), 8.2))
    for c, p in enumerate(ids):
        d = rows(p); P = d["P"]; y, z = d["y"], d["z"]; ext = [y[-1], y[0], z[0], z[-1]]
        o, R, _ = midline_frame(P); e = eb_row(int(p[2:]), offset); pts = []
        if np.isfinite(e.get("Eb_sup", np.nan)): pts.append(("Eb", o+np.array([0, e.Eb_ant, e.Eb_sup])@R, "#1baf7a"))
        if "hy_ant" in e and np.isfinite(e.get("hy_ant", np.nan)): pts.append(("설골", o+np.array([0, e.hy_ant, e.hy_sup])@R, "#eda100"))
        for r, (im, lo, hi) in enumerate(((d["minip"], -1000, 300), (d["maxip"], -300, 1500))):
            a = ax[r, c]; a.imshow(im[:, ::-1], origin="lower", cmap="gray", vmin=lo, vmax=hi, extent=ext, aspect="equal")
            if r == 0: a.contour(y, z, d["air"].astype(float), [.5], colors="#eb6834", linewidths=1)
            for t, q, cl in pts: a.plot(q[1], q[2], "+", color=cl, ms=14, mew=2.4); a.annotate(t, (q[1], q[2]), color=cl, fontsize=10, fontweight="bold", xytext=(4, 4), textcoords="offset points")
            a.axhline(z[0], color="#e34948", lw=2)
            a.set_xlim(P["PNS"][1]+48, P["PNS"][1]-22); a.set_ylim(z[0]-1, z[0]+45); a.set_xticks([]); a.set_yticks([])    # PNS 뒤 48 mm ~ 앞 22 mm (기도·설골 구간; LPS y 감소 = 앞); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{p} {'Eb 미계측' if p in CUT else '대조'}" if r == 0 else "뼈 최대투영", fontsize=11)
    ax[0, 0].set_ylabel("공기 최소투영 + 기도 윤곽", fontsize=11); ax[1, 0].set_ylabel("뼈 최대투영", fontsize=11)
    fig.suptitle("영상 아래쪽 45 mm 확대 (빨간 선 = 영상 아래 끝, 녹색 = Eb, 노랑 = 설골)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(DEST/"airway_fov_zoom.png", dpi=80); plt.close(fig)


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; ids = CUT+REF
    fig, ax = plt.subplots(len(ids), 3, figsize=(16, 4.3*len(ids)))
    for r, p in enumerate(ids):
        d = rows(p); P = d["P"]; y, z, x = d["y"], d["z"], d["x"]
        ext = [y[-1], y[0], z[0], z[-1]]                                    # 얼굴이 오른쪽(앞 = LPS y 감소)
        lim = dict(xlim=(P["PNS"][1]+50, P["PNS"][1]-60), ylim=(z[0]-2, P["PNS"][2]+15))
        for c, (im, lo, hi, nm) in enumerate(((d["minip"], -1000, 500, "정중 ±3 mm 최소투영 (공기) + 기도 분할"), (d["maxip"], -500, 1500, "정중 ±3 mm 최대투영 (뼈)"))):
            a = ax[r, c]; a.imshow(im[:, ::-1], origin="lower", cmap="gray", vmin=lo, vmax=hi, extent=ext, aspect="equal")
            if c == 0: a.imshow(np.where(d["air"], 1, np.nan)[:, ::-1], origin="lower", cmap=AIR, alpha=.55, extent=ext, aspect="equal")
            a.axhline(z[0], color="#e34948", lw=2); a.annotate("영상 아래 끝", (lim["xlim"][0]-3, z[0]+1.5), color="#e34948", fontsize=9)
            for k, cl in (("PNS", "w"), ("Me", "c")): a.plot(P[k][1], P[k][2], "+", color=cl, ms=13, mew=2); a.annotate(k, (P[k][1], P[k][2]), color=cl, fontsize=9, xytext=(4, 3), textcoords="offset points")
            a.set(**lim); a.set_title(f"{p} · {nm}", fontsize=10); a.set_xticks([]); a.set_yticks([])
        a = ax[r, 2]; a.imshow(d["bottom"], origin="lower", cmap="gray", vmin=-1000, vmax=500, extent=[x[0], x[-1], y[0], y[-1]])
        a.imshow(np.where(d["bottom_air"], 1, np.nan), origin="lower", cmap=AIR, alpha=.6, extent=[x[0], x[-1], y[0], y[-1]])
        a.set_title(f"{p} · 영상 맨 아래 6 mm 축상 최소투영 — 기도 분할 {'있음 (기도가 아래 끝에서 잘림)' if d['air_at_bottom'] else '없음'}", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel("Eb 미계측" if p in CUT else "대조", fontsize=12)
        print(p, "아래 6 mm 기도:", d["air_at_bottom"], flush=True)
    fig.suptitle("기도 촬영 범위 확인 (이전 6절과 같은 투영): 위 4명 = Eb 미계측, 아래 3명 = 대조. 주황 = AMASSS 기도 분할, 빨간 선 = 영상 아래 끝", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(DEST/"airway_fov_projection.png", dpi=60); plt.close(fig)
    zoom(ids)
    h = DEST/"index.html"; s = h.read_text(encoding="utf-8")
    add = """<h2 id="proj">원본 투영으로 다시 확인 (이전 6절 방식)</h2><p>환자마다 정중 ±3 mm 공기 최소투영(기도 분할 주황), 뼈 최대투영, 영상 맨 아래 6 mm 축상 최소투영입니다.
<b>7명 모두 영상 아래 끝에서 기도가 잘립니다</b>(맨 아래 6 mm에 기도 분할이 걸림). 두 그룹의 차이는 기도가 잘렸느냐가 아니라 <b>후두개곡(후두개 앞의 홈)과 설골이 영상 끝보다 위에 들어오느냐</b>입니다. 아래 확대 그림에서 확인할 수 있습니다.</p>
<a href="airway_fov_projection.png" target="_blank"><img src="airway_fov_projection.png" alt="투영 비교"></a>
<h3>영상 아래쪽 45 mm 확대 (공기 최소투영 · 뼈 최대투영)</h3><p>녹색 + = 모듈이 찾은 Eb(후두개곡), 노랑 + = 설골. 대조 3명은 영상 끝 위에서 후두개가 기도를 앞뒤로 가르는 홈이 보이고 그 아래 끝에 Eb가 있습니다. Eb 미계측 4명은 영상 끝까지 홈 없이 하나의 공기 기둥입니다.</p>
<a href="airway_fov_zoom.png" target="_blank"><img src="airway_fov_zoom.png" alt="아래쪽 확대"></a>"""
    s = s.split('<h2 id="proj">')[0].split("<h2>원본 투영으로 다시 확인")[0] + (("<h2>수치</h2>"+s.split("<h2>수치</h2>")[1]) if "<h2>수치</h2>" in s else "")
    h.write_text(s.replace("<h2>수치</h2>", add+"<h2>수치</h2>"), encoding="utf-8")
    print("saved", DEST/"airway_fov_projection.png")


if __name__ == "__main__":
    main()
