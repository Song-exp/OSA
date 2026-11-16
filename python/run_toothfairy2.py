"""ToothFairy2 42구조 nnU-Net(ResEnc-L, 공개 커뮤니티 체크포인트)을 턱·상악동·인두 영역에 실행한다. 기존 결과는 건드리지 않는다.

모델: models/toothfairy2_hf/model/{plans.json, dataset.json, fold_0/checkpoint_best.pth}
      trainer(nnUNetTrainer_onlyMirror01)가 설치된 nnU-Net에 없어, plans로 네트워크를 만들고 가중치를 직접 넣는다.
입력: Result_py/CBCT/nrrd/P_n.nrrd를 ALI 좌표로 잘라 사용(0.3 mm 재표본이라 머리 전체는 메모리 초과).
      좌우 Go 중점 ±55 mm, 앞 = ANS·Pog보다 15 mm 앞, 뒤 = Ba보다 15 mm 뒤, 위 = 눈밑점보다 5 mm 위, 아래 = Me보다 30 mm 아래(후두개곡 포함).
좌우: 이 모델은 배열 x축이 우리(LPS)와 반대로 학습돼, 넣을 때 x를 뒤집고 결과를 다시 뒤집는다(P_5에서 뒤집지 않으면 오른쪽·왼쪽 치아 번호가 바뀜).
      8가지 축 뒤집기를 시험해도 상악동·인두 부피는 같아 위아래·앞뒤 방향 문제는 없다.
한계(P_5 시험): 상악동은 바닥 일부만 칠함(좌우 2~3 mL, 라벨 안 공기 48%) → 상악동 부피 지표로 쓰지 않는다.
      인두 라벨은 입안 공기까지 포함한 공기 영역(라벨 안 공기 96%).
메모리: 로짓은 CPU float16에 쌓는다(perform_everything_on_device=False). 환자마다 여유 RAM을 확인한다.
출력: Result_py/CBCT/seg_toothfairy2/P_n.nii.gz (자른 영상의 물리 좌표 유지, 라벨 번호는 dataset.json)
사용: python run_toothfairy2.py 5 7 9
"""
import json
import sys
import time

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from cbct_patient_pipeline import CB, landmarks
from run_ali import ROOT, ascii_path
from run_teeth_seg import free_gb, predict_labels

MODEL = ROOT/"models/toothfairy2_hf/model"
OUT, WORK = CB/"seg_toothfairy2", CB/"toothfairy2_work"
SPACING, N_CLASSES = .3, 49


def need_gb(pid):
    """CPU float16 로짓(클래스 × 0.3 mm 복셀) + 여유 2.5 GB."""
    lo, hi = crop_box(landmarks(pid)[0]); return float(np.prod((hi-lo)/SPACING))*N_CLASSES*2/2**30 + 2.5


def crop_box(P):
    """LPS mm (최소, 최대). y는 앞쪽이 작은 값, z는 위쪽이 큰 값."""
    cx = (P["RGo"][0]+P["LGo"][0])/2
    y0 = min(P[k][1] for k in ("ANS", "Pog", "A")) - 15; y1 = P["Ba"][1] + 15
    z1 = max(P["ROr"][2], P["LOr"][2]) + 5; z0 = P["Me"][2] - 30
    return np.array([cx-55, y0, z0]), np.array([cx+55, y1, z1])


def crop(pid, out):
    img = sitk.ReadImage(str(ascii_path(CB/f"nrrd/P_{pid}.nrrd")))
    lo, hi = crop_box(landmarks(pid)[0]); size = np.array(img.GetSize())
    a = np.array(img.TransformPhysicalPointToContinuousIndex(tuple(map(float, lo))))
    b = np.array(img.TransformPhysicalPointToContinuousIndex(tuple(map(float, hi))))
    i0 = np.maximum(np.floor(np.minimum(a, b)).astype(int), 0); i1 = np.minimum(np.ceil(np.maximum(a, b)).astype(int), size-1)
    c = img[int(i0[0]):int(i1[0])+1, int(i0[1]):int(i1[1])+1, int(i0[2]):int(i1[2])+1]
    sitk.WriteImage(flip_x(c), str(out))


def flip_x(img):
    """배열 x축(가장 빠른 축)을 뒤집고 물리 정보는 그대로 둔다."""
    out = sitk.GetImageFromArray(np.ascontiguousarray(sitk.GetArrayFromImage(img)[:, :, ::-1])); out.CopyInformation(img); return out


def predictor():
    plans = json.loads((MODEL/"plans.json").read_text(encoding="utf-8")); ds = json.loads((MODEL/"dataset.json").read_text(encoding="utf-8"))
    pm = PlansManager(plans); cm = pm.get_configuration("3d_fullres"); lm = pm.get_label_manager(ds)
    ck = torch.load(MODEL/"fold_0/checkpoint_best.pth", map_location="cpu", weights_only=False)
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
                                 determine_num_input_channels(pm, cm, ds), lm.num_segmentation_heads, allow_init=True, deep_supervision=False)
    net.load_state_dict(ck["network_weights"])
    p = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=False,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=False, allow_tqdm=False)
    p.manual_initialization(net, pm, cm, [ck["network_weights"]], ds, ck["trainer_name"], ck.get("inference_allowed_mirroring_axes"))
    return p


def main():
    ids = sys.argv[1:]; OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    pending = [pid for pid in ids if not (OUT/f"P_{pid}.nii.gz").exists()]
    if not pending: print("이미 완료"); return
    p = predictor()
    for pid in pending:
        if free_gb() < need_gb(pid): print(f"P_{pid} 건너뜀: 여유 메모리 {free_gb():.1f} GB", flush=True); continue
        f = WORK/f"P_{pid}_0000.nii.gz"
        if not f.exists(): crop(pid, ascii_path(f))
        t = time.time(); tmp = ascii_path(OUT)/f"P_{pid}.tmp.nii.gz"
        predict_labels(p, ascii_path(f), tmp); sitk.WriteImage(flip_x(sitk.ReadImage(str(tmp))), str(ascii_path(OUT)/f"P_{pid}.nii.gz"), useCompression=True); tmp.unlink()
        print(f"P_{pid} {time.time()-t:.0f}초, 여유 메모리 {free_gb():.1f} GB, GPU 최대 {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
