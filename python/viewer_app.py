"""OSA 데이터 뷰어 (GUI). 환자 번호와 검사 유형을 고르면 PSG 그래프 또는 CBCT 단면·3D를 보여줍니다.

실행 (python 폴더에서):  ..\\.venv\\Scripts\\python viewer_app.py
- 왼쪽: 환자 선택, 검사 유형(PSG/CBCT), PSG 구간, 환자 요약(엑셀 값: 나이·성별·BMI·AHI 등, Roh CBCT 계측값)
- PSG : 탭마다 채널 7개 + 수면단계 + 이벤트 색칠. 그림 아래 도구 막대로 확대·이동
- CBCT: 축상·관상·시상 단면을 슬라이더로 넘겨 보기, 창 설정(뼈/연조직/공기), [3D 표면] 탭은 마우스로 회전
원본 읽기는 백그라운드에서 하고, 마지막으로 연 환자 1명만 메모리에 둡니다.
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from osa.cbct import read_cbct
from osa.psg_plot import plot_psg
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OSA 데이터 뷰어")
        self.geometry("1500x900")
        self.state("zoomed")                 # 최대화로 시작
        self.xl = excel_table()
        self.cbct = None
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
        ttk.Radiobutton(f, text="PSG (수면다원검사)", variable=self.kind, value="psg").pack(anchor="w")
        ttk.Radiobutton(f, text="CBCT (3D 영상)", variable=self.kind, value="cbct").pack(anchor="w", pady=(0, 8))
        ttk.Label(f, text="PSG 구간 (비우면 전체)\n예: 2013-08-15T22:51").pack(anchor="w")
        self.t0, self.t1 = ttk.Entry(f, width=20), ttk.Entry(f, width=20)
        self.t0.pack(anchor="w")
        self.t1.pack(anchor="w", pady=(0, 8))
        self.btn = ttk.Button(f, text="보기", command=self.load)
        self.btn.pack(anchor="w", fill="x")
        self.status = ttk.Label(f, text="", foreground="#555", wraplength=220)
        self.status.pack(anchor="w", pady=8)
        ttk.Separator(f).pack(fill="x", pady=4)
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

    def _msg(self, s):
        self.status.configure(text=s)

    def _clear(self):
        for t in self.nb.tabs():
            self.nb.forget(t)
        plt.close("all")

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
                figs, _, _, _, counts = plot_psg(data, window=window, channels_per_tab=7)
            except Exception as e:
                return self._msg(f"그리기 실패: {e}")
            for i, fig in enumerate(figs, 1):
                self._tab(fig, f"채널 {(i - 1) * 7 + 1}–{min(i * 7, len(data.channels))}")
            self._msg(f"P_{pid} PSG: 채널 {len(data.channels)}개, 탭 {len(figs)}개")
        else:
            self.cbct = data
            self._cbct_slices(pid)
            frame = ttk.Frame(self.nb)
            ttk.Button(frame, text="3D 표면 만들기 (10~30초)", command=lambda: self._cbct_3d(frame)).pack(pady=20)
            self.nb.add(frame, text="3D 표면")
            self._msg(f"P_{pid} CBCT: {' × '.join(map(str, data.rescaled.shape))}, voxel {data.voxel_size_mm[0]:.2f} mm")

    # ---------------------------------------------------------- CBCT 단면
    def _cbct_slices(self, pid):
        v = self.cbct.rescaled
        nr, nc, ns = v.shape
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

        def update(k, val):
            pos[k] = int(float(val))
            img = {"axial": lambda i: v[:, :, i], "coronal": lambda i: v[i, :, :].T,
                   "sagittal": lambda i: v[:, i, :].T}[k](pos[k])
            idx = list(pos).index(k)
            ims[idx].set_data(img)
            axes[idx].set_title(f"{names[k]}  {pos[k] + 1}")
            canvas.draw_idle()

        for k, n in (("axial", ns), ("coronal", nr), ("sagittal", nc)):
            ttk.Label(ctl, text=names[k]).pack(side="left", padx=(10, 2))
            s = ttk.Scale(ctl, from_=0, to=n - 1, length=220, command=lambda val, k=k: update(k, val))
            s.set(pos[k])
            s.pack(side="left")
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

    # ---------------------------------------------------------- CBCT 3D
    def _cbct_3d(self, frame, threshold=400, max_faces=250_000):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes
        for w in frame.winfo_children():
            w.destroy()
        d = self.cbct
        sp = d.voxel_size_mm
        step = 2
        small = d.rescaled[::step, ::step, ::step]
        s = 1
        verts, faces, _, _ = marching_cubes(small, threshold, step_size=s, spacing=tuple(sp * step))
        while len(faces) > max_faces:
            s += 1
            verts, faces, _, _ = marching_cubes(small, threshold, step_size=s, spacing=tuple(sp * step))
        # (row=뒤쪽+, col=왼쪽+, slice=위쪽+) → 화면 (x=좌우, y=앞뒤, z=위아래)
        xyz = verts[:, [1, 0, 2]]
        fig = Figure(figsize=(6, 5), constrained_layout=True)
        ax = fig.add_subplot(projection="3d")
        ax.add_collection3d(Poly3DCollection(xyz[faces], facecolors=[(0.90, 0.86, 0.76)], linewidths=0,
                                             shade=True, lightsource=matplotlib.colors.LightSource(azdeg=225, altdeg=45)))
        lo, hi = xyz.min(0), xyz.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=10, azim=-60)     # 앞쪽(−y)에서 비스듬히: 얼굴이 보임, 환자 왼쪽이 화면 오른쪽
        ax.set_axis_off()
        ax.set_title(f"뼈 표면 (≥ {threshold} HU) · 마우스로 회전")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()


if __name__ == "__main__":
    try:                                    # Windows 화면 배율에서 흐릿하거나 잘리지 않게
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
