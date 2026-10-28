r"""CBCT 분할 QC 뷰어: 환자를 고르면 축상·관상·시상 단면 위에 구조별 색 분할(하악 빨강, 상악 파랑, 두개저 초록, 상기도 노랑)을 겹쳐 봄.
- 왼쪽: 환자 목록 (전체 / EDA 검수 대상 / SSM 검수 대상), 고른 환자의 자동 QC 사유
- 위: 구조 켜기·끄기, 표시 방식(윤곽 / 채우기+윤곽), 투명도, 창 설정, ALI 랜드마크, PNG 저장
- 단면을 클릭하면 그 점을 지나는 다른 두 단면으로 이동. 슬라이더로도 넘김
읽기만 하며, 저장은 Result_py/cbct_eda/qc_snapshots/ (얼굴이 보이므로 내부용)
실행: QC_viewer.bat 더블클릭, 또는 python 폴더에서 ..\.venv\Scripts\python qc_viewer.py [--patient 7]
"""
import argparse
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from check_landmarks import read_points
from qc_montage import NAMES, QC_COLORS, overlay
from run_ali import ascii_path

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "Result_py/CBCT"
EDA = ROOT / "Result_py/cbct_eda"
WINDOWS = {"전체 (−1000~2000)": (-1000, 2000), "뼈 (−500~2000)": (-500, 2000), "연조직 (−300~400)": (-300, 400), "기도 (−1000~0)": (-1000, 0)}
VIEWS = ("축상", "관상", "시상")


def qc_tables():
    d = pd.read_csv(EDA / "cbct_eda_patient.csv", index_col=0, encoding="utf-8-sig")
    eda = d.review_items.fillna("")
    ssm = {}
    for s in ("MAND", "CB"):
        f = ROOT / f"Result_py/shape/{s}/review/qc_review.csv"
        if f.exists():
            q = pd.read_csv(f, index_col=0, encoding="utf-8-sig")
            for p, r in q.iterrows():
                items = [f"정합 잔차 {r.fit_mean_mm:.2f} mm" + (" (3배 초과)" if r.fit_flag else ""),
                         f"전달 오차 중앙값 {r.te_median_mm:.1f} mm"]
                if isinstance(r.te_flag_items, str) and r.te_flag_items:
                    items.append(f"전달 오차 큼: {r.te_flag_items}")
                if r.po_flag:
                    items.append("Po 실패")
                if isinstance(r.note, str) and r.note:
                    items.append(r.note)
                ssm.setdefault(p, []).append((s, bool(r.review or r.po_flag), items))
    return eda, ssm


class Viewer(tk.Tk):
    def __init__(self, first=None):
        super().__init__()
        self.title("CBCT 분할 QC 뷰어 (내부용)")
        self.geometry("1500x820")
        self.eda, self.ssm = qc_tables()
        self.ids = sorted(self.eda.index, key=lambda p: int(p[2:]))
        self.vol = None
        self._controls()
        self.fig = Figure(figsize=(12, 6), constrained_layout=True)
        self.ax = [self.fig.add_subplot(1, 3, i + 1) for i in range(3)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._click)
        self._fill_list()
        if first:
            self.load(f"P_{first}")

    # ------------------------------------------------------------ 화면 구성
    def _controls(self):
        left = ttk.Frame(self, width=300)
        left.pack(side="left", fill="y", padx=4, pady=4)
        self.filter = ttk.Combobox(left, values=["전체", "EDA 검수 대상", "SSM 검수 대상"], state="readonly", width=20)
        self.filter.current(0)
        self.filter.bind("<<ComboboxSelected>>", lambda _: self._fill_list())
        self.filter.pack(fill="x")
        self.lst = tk.Listbox(left, height=22, exportselection=False)
        self.lst.pack(fill="y")
        self.lst.bind("<<ListboxSelect>>", lambda _: self.load(self.lst.get(self.lst.curselection()[0]).split()[0]))
        self.info = tk.Text(left, width=40, height=22, wrap="word", font=("Malgun Gothic", 9))
        self.info.pack(fill="both", expand=True)

        self.right = ttk.Frame(self)
        self.right.pack(side="left", fill="both", expand=True)
        top = ttk.Frame(self.right)
        top.pack(fill="x")
        self.on = {s: tk.BooleanVar(value=True) for s in QC_COLORS}
        for s in QC_COLORS:
            tk.Checkbutton(top, text=NAMES[s], variable=self.on[s], fg=QC_COLORS[s] if s != "UAW" else "goldenrod",
                           command=self.redraw).pack(side="left")
        self.mode = tk.StringVar(value="채우기+윤곽")
        for m in ("윤곽", "채우기+윤곽"):
            ttk.Radiobutton(top, text=m, value=m, variable=self.mode, command=self.redraw).pack(side="left", padx=(8, 0))
        ttk.Label(top, text="  투명도").pack(side="left")
        self.alpha = tk.DoubleVar(value=0.3)
        ttk.Scale(top, from_=0.05, to=0.8, variable=self.alpha, length=90, command=lambda _: self.redraw()).pack(side="left")
        self.win = ttk.Combobox(top, values=list(WINDOWS), state="readonly", width=16)
        self.win.current(0)
        self.win.bind("<<ComboboxSelected>>", lambda _: self.redraw())
        self.win.pack(side="left", padx=8)
        self.show_lm = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="ALI 랜드마크(±2 mm)", variable=self.show_lm, command=self.redraw).pack(side="left")
        ttk.Button(top, text="PNG 저장", command=self.save).pack(side="left", padx=8)
        self.status = ttk.Label(top, text="")
        self.status.pack(side="left")
        bar = ttk.Frame(self.right)
        bar.pack(fill="x", side="bottom")
        self.sl = []
        for v in VIEWS:
            ttk.Label(bar, text=v).pack(side="left", padx=(12, 2))
            s = ttk.Scale(bar, from_=0, to=1, length=300, command=lambda val, v=v: self._slide(v, val))
            s.pack(side="left")
            self.sl.append(s)

    def _fill_list(self):
        f = self.filter.get()
        self.lst.delete(0, "end")
        for p in self.ids:
            e = bool(self.eda[p])
            s = any(flag for _, flag, _ in self.ssm.get(p, []))
            if f == "전체" or (f == "EDA 검수 대상" and e) or (f == "SSM 검수 대상" and s):
                self.lst.insert("end", f"{p}  {'E' if e else ''}{'S' if s else ''}")

    # ------------------------------------------------------------ 불러오기
    def load(self, pid):
        self.status.config(text=f"{pid} 불러오는 중…")
        self.pid = pid

        def work():
            img = sitk.ReadImage(str(ascii_path(CB / f"nrrd/{pid}.nrrd")))
            a = sitk.GetArrayFromImage(img)                       # (k, j, i) = (위, 뒤, 왼)
            lab = np.zeros(a.shape, np.uint8)                     # 구조마다 비트 하나 (겹쳐도 보이게)
            for b, s in enumerate(QC_COLORS):
                f = CB / f"seg/{pid}_seg_{s}.nrrd"
                if f.exists():
                    lab |= (sitk.GetArrayFromImage(sitk.ReadImage(str(ascii_path(f)))) > 0).astype(np.uint8) << b
            try:
                lm = {k: np.array(img.TransformPhysicalPointToContinuousIndex(tuple(r[["x", "y", "z"]].astype(float))))
                      for k, r in read_points(int(pid[2:])).iterrows()}          # (i, j, k)
            except Exception:
                lm = {}
            self.after(0, lambda: self._loaded(pid, a, lab, lm, img.GetSpacing()))

        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, pid, a, lab, lm, sp):
        if pid != self.pid:
            return
        self.vol, self.lab, self.lm, self.sp = a, lab, lm, sp
        self.pos = [a.shape[0] // 2, a.shape[1] // 2, a.shape[2] // 2]   # 축상 k, 관상 j, 시상 i
        P = {k: v for k, v in lm.items()}
        if "N" in P:                                               # 시상 단면은 N을 지나게 (정중면 근처)
            self.pos[2] = int(round(P["N"][0]))
        for s, n, p in zip(self.sl, a.shape, self.pos):
            s.config(to=n - 1)
            s.set(p)
        self._info(pid)
        self.status.config(text=f"{pid}  크기 {a.shape[2]}×{a.shape[1]}×{a.shape[0]}, 복셀 {sp[0]:.2f} mm")
        self.redraw()

    def _info(self, pid):
        t = self.info
        t.delete("1.0", "end")
        t.insert("end", f"{pid}\n\n[EDA 자동 QC]\n{self.eda.get(pid, '') or '표시 없음'}\n")
        for s, flag, items in self.ssm.get(pid, []):
            t.insert("end", f"\n[SSM {s}]{' 검수 대상' if flag else ''}\n" + "\n".join("· " + x for x in items) + "\n")
        t.insert("end", "\n자동 QC 표시는 검수 대상일 뿐 제외가 아닙니다.")

    # ------------------------------------------------------------ 그리기
    def _slice(self, v):
        k, j, i = self.pos
        if v == 0:
            return self.vol[k], self.lab[k]
        if v == 1:
            return self.vol[:, j, :][::-1], self.lab[:, j, :][::-1]
        return self.vol[:, :, i][::-1], self.lab[:, :, i][::-1]

    def redraw(self):
        if self.vol is None:
            return
        lo_hi = WINDOWS[self.win.get()]
        fill = self.alpha.get() if self.mode.get() == "채우기+윤곽" else 0.0
        nk = self.vol.shape[0]
        for v, ax in enumerate(self.ax):
            ax.clear()
            raw, lab = self._slice(v)
            masks = {s: (lab >> b) & 1 > 0 for b, s in enumerate(QC_COLORS) if self.on[s].get()}
            overlay(ax, raw, masks, alpha=fill, lw=0.9, window=lo_hi)
            ax.set_aspect(1 if v == 0 else self.sp[2] / self.sp[0])
            ax.set_title(f"{VIEWS[v]} {self.pos[v] + 1}/{self.vol.shape[v]}", fontsize=10)
            ax.axis("off")
            if self.show_lm.get():
                tol = 2.0 / self.sp[0]
                for name, (i, j, k) in self.lm.items():
                    d, x, y = ((k, i, j), (j, i, nk - 1 - k), (i, j, nk - 1 - k))[v]
                    if abs(d - self.pos[v]) < tol:
                        ax.plot(x, y, "c+", ms=8, mew=1.5)
                        ax.text(x + 4, y, name, color="cyan", fontsize=7)
        self.fig.suptitle(f"{self.pid}  (클릭하면 그 점으로 다른 단면 이동)", fontsize=11)
        self.canvas.draw_idle()

    def _slide(self, v, val):
        if self.vol is None:
            return
        n = int(float(val))
        if n != self.pos[VIEWS.index(v)]:
            self.pos[VIEWS.index(v)] = n
            self.redraw()

    def _click(self, e):
        if self.vol is None or e.inaxes not in self.ax or e.xdata is None:
            return
        v, x, y = self.ax.index(e.inaxes), int(round(e.xdata)), int(round(e.ydata))
        nk = self.vol.shape[0]
        if v == 0:
            self.pos[1], self.pos[2] = y, x
        elif v == 1:
            self.pos[0], self.pos[2] = nk - 1 - y, x
        else:
            self.pos[0], self.pos[1] = nk - 1 - y, x
        self.pos = [int(np.clip(p, 0, n - 1)) for p, n in zip(self.pos, self.vol.shape)]
        for s, p in zip(self.sl, self.pos):
            s.set(p)
        self.redraw()

    def save(self):
        if self.vol is None:
            return
        out = EDA / "qc_snapshots"
        out.mkdir(exist_ok=True)
        f = out / f"{self.pid}_k{self.pos[0]}_j{self.pos[1]}_i{self.pos[2]}.png"
        self.fig.savefig(f, dpi=130)
        self.status.config(text=f"저장: {f.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=int)
    Viewer(ap.parse_args().patient).mainloop()
