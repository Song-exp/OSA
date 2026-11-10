"""CBCT 3D 장면 (VTK): AMASSS 분할 표면 + SSM 대응점. viewer_app.py의 [3D 분할 (AMASSS)] 탭이 씀. 읽기만 함.
- 구조별 분할 표면: 하악 빨강, 상악 파랑, 두개저 초록, 상기도 노랑 (표시용으로 2배 축소 후 표면 추출)
- SSM 대응점(하악·두개저): 색 = 평균 형태에서의 위치(좌우·앞뒤·위아래 → R·G·B). 사람마다 같은 색 = 같은 대응점 번호
- 선택: CT 직교 단면 3장, Jung FH·MSP 머리 좌표계로 정렬 (원점 N, Po가 틀린 환자는 정렬도 틀어짐)
- ALI 측정점은 viewer_app.py [단면] 탭에 2D로 겹쳐 그림
입력: Result_py/CBCT/nrrd, seg, landmarks, Result_py/shape/<구조>/points.npz
점검용 단독 실행 (창 없이 PNG): python cbct3d_viewer.py --patient 5 --shot a.png [--head] [--ct]
얼굴 윤곽이 보이므로 내부용.
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from matplotlib.colors import to_rgb
from vtk.util import numpy_support

from check_landmarks import read_points
from osa.ceph import head_frame
from qc_montage import NAMES, QC_COLORS
from qc_viewer import qc_tables
from run_ali import ascii_path

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "Result_py/CBCT"
FRAME_PTS = ("RPo", "LPo", "ROr", "N", "Ba")                     # 머리 좌표계에 필요한 점
VIEWS = {"비스듬히": ((-0.6, -0.6, 0.5), (0, 0, 1)), "정면": ((0, -1, 0), (0, 0, 1)), "오른쪽": ((-1, 0, 0), (0, 0, 1)), "왼쪽": ((1, 0, 0), (0, 0, 1)),
         "위": ((0, 0, 1), (0, -1, 0))}                           # 카메라 방향 (LPS: 앞 = −y, 오른쪽 = −x), 위쪽 벡터


def patients():
    return sorted(int(f.stem[2:]) for f in (CB / "nrrd").glob("P_*.nrrd"))


def read_array(f):
    img = sitk.ReadImage(str(ascii_path(f)))
    assert np.allclose(img.GetDirection(), np.eye(3).ravel()), "방향 행렬이 단위가 아닌 영상은 지원 안 함"   # 83명 모두 단위
    return sitk.GetArrayFromImage(img), img.GetSpacing(), img.GetOrigin()


def to_vtk_image(a, spacing, origin):
    d = vtk.vtkImageData()
    d.SetDimensions(*a.shape[::-1])                               # (k, j, i) 배열 → VTK (i, j, k)
    d.SetSpacing(*spacing)
    d.SetOrigin(*origin)
    d.GetPointData().SetScalars(numpy_support.numpy_to_vtk(np.ascontiguousarray(a).ravel(), deep=True))
    return d


def matrix(R=np.eye(3), o=np.zeros(3)):
    """x' = R (x - o) 를 4x4 vtkMatrix로."""
    vm = vtk.vtkMatrix4x4()
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, -R @ o
    for i in range(4):
        for j in range(4):
            vm.SetElement(i, j, m[i, j])
    return vm


def points_actor(xyz, colors, radius):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.asarray(xyz, float), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().SetScalars(numpy_support.numpy_to_vtk((np.asarray(colors) * 255).astype(np.uint8), deep=True))
    s = vtk.vtkSphereSource()
    s.SetRadius(radius)
    s.SetThetaResolution(10)
    s.SetPhiResolution(10)
    g = vtk.vtkGlyph3DMapper()
    g.SetInputData(poly)
    g.SetSourceConnection(s.GetOutputPort())
    g.SetColorModeToDirectScalars()
    g.ScalingOff()
    a = vtk.vtkActor()
    a.SetMapper(g)
    return a



def surface_actor(img, color, opacity):
    shrink = vtk.vtkImageShrink3D()                               # 표시용이라 2배 줄여 빠르게
    shrink.SetInputData(img)
    shrink.SetShrinkFactors(2, 2, 2)
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputConnection(shrink.GetOutputPort())
    fe.SetValue(0, 0.5)
    sm = vtk.vtkWindowedSincPolyDataFilter()
    sm.SetInputConnection(fe.GetOutputPort())
    sm.SetNumberOfIterations(15)
    sm.NormalizeCoordinatesOn()
    nm = vtk.vtkPolyDataNormals()
    nm.SetInputConnection(sm.GetOutputPort())
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(nm.GetOutputPort())
    m.ScalarVisibilityOff()
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetOpacity(opacity)
    return a



class Scene:
    """렌더 창 하나에 환자 한 명의 분할·대응점(·CT)을 그림. 표시 상태는 show/head로 바꾸고 apply()."""

    def __init__(self, rw):
        self.rw = rw
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.12, 0.12, 0.14)
        rw.AddRenderer(self.ren)
        self.head = False
        self.show = {"ct": False, **{s: True for s in QC_COLORS}, "ssm_MAND": True, "ssm_CB": False}
        self.ssm = {}
        for s in ("MAND", "CB"):
            f = ROOT / f"Result_py/shape/{s}/points.npz"
            if f.exists():
                z = np.load(f, allow_pickle=True)
                mean = z["frame"].mean(0)
                self.ssm[s] = dict(ids=[int(str(x).replace("P_", "")) for x in z["ids"]], pts=z["frame"],
                                   o=z["head_origin"], R=z["head_R"], rgb=(mean - mean.min(0)) / np.ptp(mean, 0))
        self.eda, self.ssm_qc = qc_tables()
        self.actors, self.pid, self.frame = {}, None, None

    @staticmethod
    def read(pid):
        """무거운 읽기 (백그라운드 스레드용, VTK 객체 안 만듦)."""
        p = f"P_{pid}"
        segs = {s: read_array(CB / f"seg/{p}_seg_{s}.nrrd") for s in QC_COLORS if (CB / f"seg/{p}_seg_{s}.nrrd").exists()}
        lm = read_points(pid)
        return dict(pid=pid, ct=read_array(CB / f"nrrd/{p}.nrrd"), segs=segs,
                    lm={k: r[["x", "y", "z"]].to_numpy(float) for k, r in lm.iterrows()})

    def build(self, data):
        """읽은 데이터로 장면을 새로 만듦 (메인 스레드)."""
        for key, a in self.actors.items():
            for x in (a if isinstance(a, list) else [a]):
                self.ren.RemoveViewProp(x)
        self.actors = {}
        self.pid, self.lm = data["pid"], data["lm"]
        self.frame = head_frame(self.lm) if all(k in self.lm for k in FRAME_PTS) else None
        self.ct = to_vtk_image(*data["ct"])
        dims = self.ct.GetDimensions()
        self.dims = dims
        self.pos = [d // 2 for d in dims]
        if "N" in self.lm:                                        # 시상 단면은 N을 지나게
            self.pos[0] = int(round((self.lm["N"][0] - self.ct.GetOrigin()[0]) / self.ct.GetSpacing()[0]))
        self.slices = []
        for axis in range(3):
            m = vtk.vtkImageSliceMapper()
            m.SetInputData(self.ct)
            m.SetOrientation(axis)
            m.SetSliceNumber(self.pos[axis])
            s = vtk.vtkImageSlice()
            s.SetMapper(m)
            self.slices.append(s)
        self.actors["ct"] = self.slices

        for s, arr in data["segs"].items():
            self.actors[s] = surface_actor(to_vtk_image(*arr), to_rgb(QC_COLORS[s]), 0.45 if s == "UAW" else 0.85)
        for s, z in self.ssm.items():                             # 저장된 머리 좌표계 값 → LPS (x = y R + o)
            if self.pid in z["ids"]:
                i = z["ids"].index(self.pid)
                self.actors[f"ssm_{s}"] = points_actor(z["pts"][i] @ z["R"][i] + z["o"][i], z["rgb"], 0.9)
        for key, a in self.actors.items():
            for x in (a if isinstance(a, list) else [a]):
                self.ren.AddViewProp(x)
        self.apply()
        self.view("비스듬히")

    # ------------------------------------------------------------ 표시
    def visible(self, key):
        return self.show.get(key, True)

    def apply(self):
        # 화면용 축 (좌, 뒤, 위) = LPS와 같은 손잡이라 순수 회전. (좌, 앞, 위)를 그대로 쓰면 거울상이 되어 법선이 뒤집힘
        Rd = self.frame[1] * [[1], [-1], [1]] if self.head and self.frame is not None else None
        T = matrix(Rd, self.frame[0]) if Rd is not None else matrix()
        for key, a in self.actors.items():
            for x in (a if isinstance(a, list) else [a]):
                x.SetVisibility(self.visible(key))
                x.SetUserMatrix(T)
        self.rw.Render()

    def view(self, name):
        d, up = VIEWS[name]
        self.ren.ResetCamera()
        cam = self.ren.GetActiveCamera()
        f = np.array(cam.GetFocalPoint())
        cam.SetPosition(*(f + 500 * np.array(d) / np.linalg.norm(d)))
        cam.SetViewUp(*up)
        self.ren.ResetCamera()
        self.rw.Render()

    def set_slice(self, axis, v):
        self.pos[axis] = int(round(float(v)))
        self.slices[axis].GetMapper().SetSliceNumber(self.pos[axis])
        self.rw.Render()

    def set_window(self, lo, hi):
        for s in self.slices:
            s.GetProperty().SetColorWindow(hi - lo)
            s.GetProperty().SetColorLevel((hi + lo) / 2)
        self.rw.Render()

    def qc_text(self):
        p = f"P_{self.pid}"
        lines = [f"EDA 자동 QC: {self.eda.get(p, '') or '표시 없음'}"]
        for s, flag, items in self.ssm_qc.get(p, []):
            lines.append(f"SSM {NAMES[s]}{' [검수 대상]' if flag else ''}: " + " / ".join(items))
        missing = [k for k in FRAME_PTS if k not in self.lm]
        if missing:
            lines.append("좌표계 점 없음: " + ", ".join(missing))
        return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="점검용: 창 없이 장면을 PNG로 저장")
    ap.add_argument("--patient", type=int, default=5)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--ct", action="store_true")
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--view", default="비스듬히", choices=list(VIEWS))
    a = ap.parse_args()
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(True)
    rw.SetSize(1200, 900)
    sc = Scene(rw)
    sc.build(Scene.read(a.patient))
    sc.set_window(-500, 2000)
    sc.head, sc.show["ct"] = a.head, a.ct
    sc.apply()
    sc.view(a.view)
    f = vtk.vtkWindowToImageFilter()
    f.SetInput(rw)
    f.Update()
    w = vtk.vtkPNGWriter()
    w.SetFileName(a.shot)
    w.SetInputConnection(f.GetOutputPort())
    w.Write()
