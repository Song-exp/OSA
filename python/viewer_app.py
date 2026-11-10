"""OSA 데이터 뷰어 (GUI). 환자 번호와 검사 유형을 고르면 PSG 그래프 또는 CBCT 단면·3D를 보여줍니다.

실행 (python 폴더에서):  ..\\.venv\\Scripts\\python viewer_app.py
- 왼쪽: 환자 선택, 검사 유형(PSG/CBCT), PSG 구간, 환자 요약(엑셀 값: 나이·성별·BMI·AHI 등, Roh CBCT 계측값)
- PSG : 탭마다 채널 7개 + 수면단계 + 이벤트 색칠. 그림 아래 도구 막대로 확대·이동
        채널 그래프를 더블클릭하거나 왼쪽 "채널 확대"에서 고르면 그 채널만 크게 그린 [확대] 탭이 열림
- CBCT: 축상·관상·시상 단면을 슬라이더로 넘겨 보기, 창 설정(뼈/연조직/공기)
        [단면] 탭에 ALI 측정점·FH·MSP 선 겹쳐 보기, 측정점으로 이동
        [3D 분할 (AMASSS)] 탭: 구조별 분할 표면 + SSM 대응점. 마우스로 회전, 시점 버튼
원본 읽기는 백그라운드에서 하고, 마지막으로 연 환자 1명만 메모리에 둡니다.
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from osa.cbct import read_cbct
from osa.psg_plot import _draw_channel, _draw_stages, plot_psg
from osa.psg_read import read_psg

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 7             # 창 안에 여러 축이 들어가므로 작게
WINDOWS = {"뼈 (−500~2000)": (-500, 2000), "연조직 (−300~400)": (-300, 400), "공기·기도 (−1000~0)": (-1000, 0),
           "전체 (−1000~3000)": (-1000, 3000)}
SUMMARY = {"Age": (6, "나이"), "Sex": (10, "성별"), "BMI": (9, "BMI"), "AHI": (12, "AHI"), "RDI": (13, "RDI"),
           "ODI": (14, "ODI"), "LSaO2": (15, "최저 SpO2"), "ANB": (22, "ANB (°)"), "MPFH": (25, "MP-FH (°)"),
           "TDI": (28, "TDI (mm)"), "PMCA": (31, "PMCA (mm²)"), "PAL": (32, "PAL (mm)"),
           "NAV": (29, "NAV (mm³)"), "PAV": (30, "PAV (mm³)")}


def patients() -> list[int]:
    ids = {int(p.name.split("_")[1]) for k in ("PSG", "CBCT") for p in (ROOT / "Data" / k).glob("P_*")}
    return sorted(ids)


def excel_table() -> pd.DataFrame | None:
    try:
        x = pd.read_excel(ROOT / "Patient_Info_Data_87명_261019.xlsx", header=None).iloc[3:]
    except Exception:
        return None
    x = x[x[0].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
    return pd.DataFrame({k: pd.to_numeric(x[c], errors="coerce").values for k, (c, _) in SUMMARY.items()},
                        index=x[0].astype(int).values)


def summary_text(xl, pid) -> str:
    if xl is None or pid not in xl.index:
        return "엑셀 값 없음"
    r = xl.loc[pid]
    sev = "" if pd.isna(r.AHI) else ("정상" if r.AHI < 5 else "경증" if r.AHI < 15 else "중등도" if r.AHI < 30 else "중증")
    headers = {"Age": "── 기본", "AHI": "── PSG (엑셀 판독)", "ANB": "── CBCT (엑셀 수작업)"}
    lines = []
    for k, (_, name) in SUMMARY.items():
        if k in headers:
            lines.append(headers[k])
        v = r[k]
        if pd.isna(v):
            v = "-"
        elif k == "Sex":
            v = {1: "남", 2: "여"}.get(int(v), v)
        else:
            v = f"{v:,.0f}" if k in ("NAV", "PAV") else f"{v:g}"
        lines.append(f"{name:<10} {v}" + (f"  ({sev})" if k == "AHI" and sev else ""))
    return "\n".join(lines)


FRAME_PTS = ("RPo", "LPo", "ROr", "N", "Ba")                # 머리 좌표계 (Jung FH·MSP)에 쓰는 점
MEASURE_PTS = ("A", "B", "Pog", "Me", "Gn", "RGo", "LGo", "ANS", "PNS", "S")
ALI_COLOR = lambda k: "red" if k in FRAME_PTS else "gold" if k in MEASURE_PTS else "deepskyblue"


def ali_points(data, pid):
    """ALI 랜드마크(LPS mm)를 이 DICOM 배열의 번호 (열, 행, 슬라이스)로 바꾸고, FH·MSP 평면도 같은 번호 공간으로.
    반환: ({이름: (열, 행, 슬라이스)}, {"FH"|"MSP": (법선, d)}) 또는 None (ALI 결과 없음)."""
    from check_landmarks import read_points
    from osa.ceph import head_frame
    try:
        lm = read_points(pid)
    except Exception:
        return None
    if lm.empty:
        return None
    t = data.slice_table
    o = t[[f"Orient{i}" for i in range(1, 7)]].iloc[0].to_numpy(float)
    rs, cs, z = data.voxel_size_mm
    A = np.column_stack([o[:3] * cs, o[3:] * rs, np.cross(o[:3], o[3:]) * z])   # export_nrrd.py와 같은 기하
    p0 = t[["PosX", "PosY", "PosZ"]].to_numpy(float)[0]
    to_idx = lambda x: np.linalg.solve(A, np.asarray(x, float) - p0)
    P = {k: r[["x", "y", "z"]].to_numpy(float) for k, r in lm.iterrows()}
    idx = {k: to_idx(x) for k, x in P.items()}
    plane = lambda a, b, c: (np.cross(b - a, c - a), np.cross(b - a, c - a) @ a)
    planes = {}
    if all(k in P for k in FRAME_PTS):                       # 번호 공간도 아핀이라 평면은 평면으로 남음
        sup = head_frame(P)[1][2]
        planes = {"FH": plane(*(idx[k] for k in ("RPo", "LPo", "ROr"))),
                  "MSP": plane(idx["N"], idx["Ba"], to_idx(P["N"] + 50 * sup))}
    return idx, planes


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OSA 데이터 뷰어")
        self.geometry("1500x900")
        self.state("zoomed")                 # 최대화로 시작
        self.xl = excel_table()
        self.cbct = None
        self.vtk_win = None                  # 3D 탭의 VTK 창 (환자 바꿀 때 정리)
        self.psg = None                      # 마지막 PSG: 원신호, 이벤트, 수면단계, 구간, 확대 탭
        self._build_controls()
        self.nb = ttk.Notebook(self)
        self.nb.pack(side="right", fill="both", expand=True)

    # ---------------------------------------------------------- 왼쪽 패널
    def _build_controls(self):
        f = ttk.Frame(self, padding=10)
        f.pack(side="left", fill="y")
        ttk.Label(f, text="환자 번호").pack(anchor="w")
        self.pid = ttk.Combobox(f, values=[f"P_{i}" for i in patients()], width=12, state="readonly")
        self.pid.current(0)
        self.pid.pack(anchor="w", pady=(0, 8))
        self.pid.bind("<<ComboboxSelected>>", lambda e: self._show_summary())
        ttk.Label(f, text="검사 유형").pack(anchor="w")
        self.kind = tk.StringVar(value="psg")
        ttk.Radiobutton(f, text="PSG (수면다원검사)", variable=self.kind, value="psg",
                        command=self._toggle_psg_ctl).pack(anchor="w")
        ttk.Radiobutton(f, text="CBCT (3D 영상)", variable=self.kind, value="cbct",
                        command=self._toggle_psg_ctl).pack(anchor="w", pady=(0, 8))
        ttk.Label(f, text="PSG 구간 (비우면 전체)\n예: 2013-08-15T22:51").pack(anchor="w")
        self.t0, self.t1 = ttk.Entry(f, width=20), ttk.Entry(f, width=20)
        self.t0.pack(anchor="w")
        self.t1.pack(anchor="w", pady=(0, 8))
        self.btn = ttk.Button(f, text="보기", command=self.load)
        self.btn.pack(anchor="w", fill="x")
        self.status = ttk.Label(f, text="", foreground="#555", wraplength=220)
        self.status.pack(anchor="w", pady=8)
        # PSG를 골랐을 때만 보이는 채널 확대 선택
        self.psg_ctl = ttk.Frame(f)
        ttk.Label(self.psg_ctl, text="채널 확대 (그래프 더블클릭도 가능)").pack(anchor="w")
        self.ch = ttk.Combobox(self.psg_ctl, values=[], width=28, state="disabled")
        self.ch.pack(anchor="w", pady=(0, 8))
        self.ch.bind("<<ComboboxSelected>>", lambda e: self._zoom(self.ch.current()))
        self.psg_anchor = ttk.Separator(f)
        self.psg_anchor.pack(fill="x", pady=4)
        self._toggle_psg_ctl()
        ttk.Label(f, text="환자 요약").pack(anchor="w")
        self.info = tk.Text(f, width=30, height=30, font=("Consolas", 10), relief="flat", background="#f6f6f6")
        self.info.pack(anchor="w")
        self._show_summary()

    def _id(self) -> int:
        return int(self.pid.get()[2:])

    def _show_summary(self):
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", summary_text(self.xl, self._id()))
        self.info.configure(state="disabled")

    # ---------------------------------------------------------- 불러오기
    def load(self):
        pid, kind = self._id(), self.kind.get()
        folder = next((ROOT / "Data" / kind.upper()).glob(f"P_{pid}_*"), None)
        if folder is None:
            return self._msg(f"P_{pid}의 {kind.upper()} 폴더가 없습니다.")
        self.btn.state(["disabled"])
        self._msg(f"P_{pid} {kind.upper()} 읽는 중… (PSG는 수십 초)")

        def work():
            try:
                data = read_psg(folder) if kind == "psg" else read_cbct(folder)
                self.after(0, lambda: self._show(kind, pid, data))
            except Exception as e:
                self.after(0, lambda e=e: self._msg(f"읽기 실패: {e}"))
            finally:
                self.after(0, lambda: self.btn.state(["!disabled"]))

        threading.Thread(target=work, daemon=True).start()

    def _toggle_psg_ctl(self):
        if self.kind.get() == "psg":
            self.psg_ctl.pack(anchor="w", fill="x", before=self.psg_anchor)
        else:
            self.psg_ctl.pack_forget()

    # ---------------------------------------------------------- PSG 채널 확대
    def _zoom(self, i: int):
        if self.psg is None or i < 0:
            return
        rec, events, (w0, w1) = self.psg["rec"], self.psg["events"], self.psg["window"]
        ch = rec.channels[i]
        fig = Figure(figsize=(8, 5), constrained_layout=True)
        a0, a1 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 5]})
        _draw_stages(a0, self.psg["stages"], w0, w1)
        _draw_channel(a1, ch, events, w0, w1, 400_000, 1.0, legend=True)   # 더 촘촘하게, 세로축은 자르지 않음
        for ax in (a0, a1):
            ax.grid(True, alpha=0.4)
        a1.set_xlim(w0, w1)
        a1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
        a1.set_xlabel("Recorded date/time")
        fig.suptitle(f"{Path(rec.source_folder).name} | {i + 1}. {ch.name} | fs {ch.fs:g} Hz")
        old = self.psg["zoom"]
        if old is not None:                 # 확대 탭은 하나만 두고 내용만 바꿈
            self.nb.forget(old)
        self._tab(fig, f"확대: {ch.name}")
        self.psg["zoom"] = self.nb.tabs()[-1]
        self.nb.select(self.psg["zoom"])
        self.ch.current(i)

    def _msg(self, s):
        self.status.configure(text=s)

    def _clear(self):
        for t in self.nb.tabs():
            self.nb.forget(t)
        plt.close("all")
        if self.vtk_win is not None:
            self.vtk_win.Finalize()
            self.vtk_win = None
        self.psg = None
        self.ch.configure(values=[], state="disabled")
        self.ch.set("")

    def _tab(self, fig, title):
        frame = ttk.Frame(self.nb)
        fig.set_size_inches(8, 5)           # 작게 만든 뒤 창 크기에 맞춰 늘어나게 (큰 그림이 창을 밀어내지 않게)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        NavigationToolbar2Tk(canvas, frame).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self.nb.add(frame, text=title)
        return canvas

    def _show(self, kind, pid, data):
        self._clear()
        if kind == "psg":
            window = (self.t0.get(), self.t1.get()) if self.t0.get() and self.t1.get() else None
            try:
                figs, events, stages, plot_data, counts = plot_psg(data, window=window, channels_per_tab=7)
            except Exception as e:
                return self._msg(f"그리기 실패: {e}")
            self.psg = {"rec": data, "events": events, "stages": stages, "window": plot_data["Window"], "zoom": None}
            for i, fig in enumerate(figs, 1):
                canvas = self._tab(fig, f"채널 {(i - 1) * 7 + 1}–{min(i * 7, len(data.channels))}")
                # 축 순서: [수면단계, 채널 1..7] → 더블클릭한 축의 채널 번호
                ax_ch = {ax: (i - 1) * 7 + r - 1 for r, ax in enumerate(fig.axes[:8]) if r > 0}
                canvas.mpl_connect("button_press_event",
                                   lambda e, m=ax_ch: e.dblclick and e.inaxes in m and self._zoom(m[e.inaxes]))
            self.ch.configure(values=[f"{i + 1}. {c.name}" for i, c in enumerate(data.channels)], state="readonly")
            self.ch.set("")
            self._msg(f"P_{pid} PSG: 채널 {len(data.channels)}개, 탭 {len(figs)}개")
        else:
            self.cbct = data
            self._cbct_slices(pid)
            self._cbct_3d(pid)
            self._msg(f"P_{pid} CBCT: {' × '.join(map(str, data.rescaled.shape))}, voxel {data.voxel_size_mm[0]:.2f} mm")

    # ---------------------------------------------------------- CBCT 단면
    def _cbct_slices(self, pid):
        v = self.cbct.rescaled
        nr, nc, ns = v.shape
        ali = ali_points(self.cbct, pid)                     # 없으면 None (자동 처리 안 된 환자)
        fig = Figure(figsize=(8, 4), constrained_layout=True)
        axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
        lo, hi = WINDOWS["뼈 (−500~2000)"]
        pos = {"axial": ns // 2, "coronal": nr // 2, "sagittal": nc // 2}
        ims = [axes[0].imshow(v[:, :, pos["axial"]], cmap="gray", vmin=lo, vmax=hi),
               axes[1].imshow(v[pos["coronal"], :, :].T, cmap="gray", vmin=lo, vmax=hi, origin="lower"),
               axes[2].imshow(v[:, pos["sagittal"], :].T, cmap="gray", vmin=lo, vmax=hi, origin="lower")]
        names = {"axial": "축상 (위에서)", "coronal": "관상 (앞에서)", "sagittal": "시상 (옆에서)"}
        for ax, k in zip(axes, pos):
            ax.set_title(f"{names[k]}  {pos[k] + 1}")
            ax.axis("off")
        fig.suptitle(f"P_{pid} CBCT")
        frame = ttk.Frame(self.nb)
        ctl = ttk.Frame(frame)
        ctl.pack(fill="x")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # ALI 겹쳐 그리기: 단면마다 (고정 축, 가로 축, 세로 축), 배열 번호 축 = (열, 행, 슬라이스)
        view_ax = {"axial": (2, 0, 1), "coronal": (1, 0, 2), "sagittal": (0, 1, 2)}
        dims, sp = (nc, nr, ns), (self.cbct.voxel_size_mm[1], self.cbct.voxel_size_mm[0], self.cbct.voxel_size_mm[2])
        show_pts, show_lines = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        drawn = {k: [] for k in pos}

        def overlay(k):
            ax = axes[list(pos).index(k)]
            for a in drawn[k]:
                a.remove()
            drawn[k] = []
            if ali is None:
                return
            f, xa, ya = view_ax[k]
            pts, planes = ali
            if show_lines.get():
                for name, (n, d) in planes.items():
                    if abs(n[f]) / np.linalg.norm(n) > 0.9:      # 단면과 거의 나란한 평면(시상의 MSP, 축상의 FH)은 교선이 무의미
                        continue
                    c = "tomato" if name == "FH" else "deepskyblue"
                    if abs(n[ya]) > 1e-9:
                        xs = np.array([0, dims[xa] - 1])
                        line = ax.plot(xs, (d - n[f] * pos[k] - n[xa] * xs) / n[ya], "--", color=c, lw=1)
                    else:
                        x = (d - n[f] * pos[k]) / n[xa]
                        line = ax.plot([x, x], [0, dims[ya] - 1], "--", color=c, lw=1)
                    drawn[k] += line + [ax.text(0.01, 0.97 - 0.05 * len(drawn[k]), name, color=c, fontsize=8,
                                                transform=ax.transAxes, va="top")]
            if show_pts.get():
                for name, q in pts.items():
                    if abs(q[f] - pos[k]) * sp[f] <= 3:          # 지금 단면에서 3 mm 안의 점만
                        c = ALI_COLOR(name)
                        drawn[k] += ax.plot(q[xa], q[ya], "o", ms=5, mfc=c, mec="black", mew=0.5)
                        drawn[k].append(ax.text(q[xa] + 6, q[ya], name, color=c, fontsize=8))
            ax.set_xlim(-0.5, dims[xa] - 0.5)
            ax.set_ylim((dims[ya] - 0.5, -0.5) if k == "axial" else (-0.5, dims[ya] - 0.5))

        def update(k, val):
            pos[k] = int(float(val))
            img = {"axial": lambda i: v[:, :, i], "coronal": lambda i: v[i, :, :].T,
                   "sagittal": lambda i: v[:, i, :].T}[k](pos[k])
            idx = list(pos).index(k)
            ims[idx].set_data(img)
            axes[idx].set_title(f"{names[k]}  {pos[k] + 1}")
            overlay(k)
            canvas.draw_idle()

        sliders = {}
        for k, n in (("axial", ns), ("coronal", nr), ("sagittal", nc)):
            ttk.Label(ctl, text=names[k]).pack(side="left", padx=(10, 2))
            s = ttk.Scale(ctl, from_=0, to=n - 1, length=220, command=lambda val, k=k: update(k, val))
            s.set(pos[k])
            s.pack(side="left")
            sliders[k] = s

        ctl2 = ttk.Frame(frame)
        ctl2.pack(fill="x", before=canvas.get_tk_widget())
        if ali is None:
            ttk.Label(ctl2, text="ALI 측정점: 자동 처리 결과 없음", foreground="#555").pack(side="left", padx=10)
        else:
            redraw = lambda: ([overlay(k) for k in pos], canvas.draw_idle())
            ttk.Checkbutton(ctl2, text="ALI 측정점 (단면 ±3 mm)", variable=show_pts, command=redraw).pack(side="left", padx=10)
            ttk.Checkbutton(ctl2, text="FH·MSP 선", variable=show_lines, command=redraw).pack(side="left", padx=4)
            ttk.Label(ctl2, text="측정점으로 이동").pack(side="left", padx=(16, 2))
            jump = ttk.Combobox(ctl2, values=list(ali[0]), width=10, state="readonly")
            jump.pack(side="left")

            def go(_):
                q = ali[0][jump.get()]                       # 세 단면이 모두 이 점을 지나게
                for k, (f, _, _) in view_ax.items():
                    sliders[k].set(int(round(np.clip(q[f], 0, dims[f] - 1))))
            jump.bind("<<ComboboxSelected>>", go)
            ttk.Label(ctl2, foreground="#555", text="빨강 = 좌표계 점 (RPo·LPo·ROr = FH, N·Ba = MSP), 노랑 = 계측점, "
                      "하늘 = 나머지 | 점선: 빨강 FH, 파랑 MSP").pack(side="left", padx=16)
            for k in pos:
                overlay(k)
        ttk.Label(ctl, text="창 설정").pack(side="left", padx=(20, 2))
        wsel = ttk.Combobox(ctl, values=list(WINDOWS), width=18, state="readonly")
        wsel.current(0)
        wsel.pack(side="left")

        def set_window(_):
            lo, hi = WINDOWS[wsel.get()]
            for im in ims:
                im.set_clim(lo, hi)
            canvas.draw_idle()

        wsel.bind("<<ComboboxSelected>>", set_window)
        NavigationToolbar2Tk(canvas, frame).update()
        canvas.draw()
        self.nb.add(frame, text="단면")

    # ---------------------------------------------------------- CBCT 3D (AMASSS 분할 + SSM 대응점, VTK)
    def _cbct_3d(self, pid):
        """VTK 화면을 탭 안의 Frame에 자식 창으로 붙임. 처음 탭을 열 때 만들고 데이터는 백그라운드에서 읽음.
        ALI 측정점은 [단면] 탭에 겹쳐 그림."""
        import vtk
        from cbct3d_viewer import NAMES, QC_COLORS, VIEWS, Scene, patients as processed
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="3D 분할 (AMASSS)")
        if pid not in processed():
            ttk.Label(frame, text=f"P_{pid}는 자동 분할(AMASSS) 결과가 없습니다 (Result_py/CBCT).").pack(pady=20)
            return
        st = {"scene": None}
        sc = lambda: st["scene"]

        row1, row2, row3 = ttk.Frame(frame), ttk.Frame(frame), ttk.Frame(frame)
        for r in (row1, row2, row3):
            r.pack(fill="x", padx=6, pady=2)
        head = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="머리 좌표계로 정렬 (Jung FH·MSP)", variable=head,
                        command=lambda: sc() and (setattr(sc(), "head", head.get()), sc().apply(), sc().view("비스듬히"))
                        ).pack(side="left", padx=(4, 4))
        ttk.Label(row1, text="시점").pack(side="left", padx=(16, 2))
        for name in VIEWS:
            ttk.Button(row1, text=name, width=8, command=lambda n=name: sc() and sc().view(n)).pack(side="left")

        def check(key, text):
            v = tk.BooleanVar(value=key not in ("ssm_CB", "ct"))

            def toggle():
                if sc():
                    sc().show[key] = v.get()
                    sc().apply()
            ttk.Checkbutton(row2, text=text, variable=v, command=toggle).pack(side="left", padx=4)

        for s in QC_COLORS:
            check(s, NAMES[s])
        check("ssm_MAND", "하악 대응점")
        check("ssm_CB", "두개저 대응점")
        check("ct", "CT 단면")
        ttk.Label(row2, text="창 설정").pack(side="left", padx=(16, 2))
        wsel = ttk.Combobox(row2, values=list(WINDOWS), width=18, state="readonly")
        wsel.current(0)
        wsel.pack(side="left")
        wsel.bind("<<ComboboxSelected>>", lambda e: sc() and sc().set_window(*WINDOWS[wsel.get()]))
        sliders = []
        for axis, t in enumerate(("CT 시상 (좌우)", "관상 (앞뒤)", "축상 (위아래)")):
            ttk.Label(row3, text=t).pack(side="left", padx=(10, 2))
            s = ttk.Scale(row3, from_=0, to=1, length=220, command=lambda val, a=axis: sc() and sc().set_slice(a, val))
            s.pack(side="left")
            sliders.append(s)
        ttk.Label(frame, foreground="#555", text="하악 빨강 · 상악 파랑 · 두개저 초록 · 상기도 노랑 | "
                  "대응점 색 = 평균 형태 위치 (같은 색 = 사람마다 같은 대응점)").pack(fill="x", padx=8)
        qc = ttk.Label(frame, foreground="#a15c00", justify="left")
        qc.pack(fill="x", padx=8)

        view = tk.Frame(frame, background="black")
        view.pack(fill="both", expand=True)

        def start(_):
            if st["scene"] is not None or "rw" in st:
                return
            rw = vtk.vtkRenderWindow()
            rw.SetParentInfo(str(view.winfo_id()))              # 탭 안 Frame의 자식 창으로
            rw.SetSize(max(view.winfo_width(), 100), max(view.winfo_height(), 100))
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(rw)
            iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            iren.Initialize()                                    # 이벤트는 tk 메인 루프가 함께 돌림 (Start 안 함)
            st.update(rw=rw, iren=iren)
            self.vtk_win = rw
            self._msg(f"P_{pid} 3D 읽는 중… (몇 초)")

            def work():
                try:
                    data = Scene.read(pid)
                    self.after(0, lambda: ready(data))
                except Exception as e:
                    self.after(0, lambda e=e: self._msg(f"3D 읽기 실패: {e}"))
            threading.Thread(target=work, daemon=True).start()

        def ready(data):
            if st.get("rw") is not self.vtk_win:                 # 그사이 다른 환자를 열었으면 버림
                return
            s = Scene(st["rw"])
            s.head = head.get()
            s.build(data)
            s.set_window(*WINDOWS[wsel.get()])
            for axis, sl in enumerate(sliders):
                sl.configure(to=s.dims[axis] - 1)
                sl.set(s.pos[axis])
            qc.configure(text=s.qc_text())
            st["scene"] = s
            self._msg(f"P_{pid} 3D 준비됨. 마우스 왼쪽 = 회전, 오른쪽·휠 = 확대, 가운데 = 이동")

        def resize(e):
            if "rw" in st:
                st["rw"].SetSize(e.width, e.height)
                st["rw"].Render()

        view.bind("<Map>", start)
        view.bind("<Configure>", resize)


if __name__ == "__main__":
    try:                                    # Windows 화면 배율에서 흐릿하거나 잘리지 않게
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
