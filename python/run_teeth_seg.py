"""공개 nnU-Net 치아 분할 모델을 턱 영역에 실행한다. 기존 seg/ALI 결과는 건드리지 않는다.

모델: models/downloads_teeth/<이름>/{checkpoint_final.pth, dataset.json, plans.json}
입력: Result_py/CBCT/nrrd/P_n.nrrd를 ALI 좌표로 턱 영역만 잘라 사용(메모리 절약).
출력: Result_py/CBCT/seg_teeth/<이름>/P_n.nii.gz (자른 영상의 물리 좌표 유지)
사용: python run_teeth_seg.py UNIVERSALLAB_MODEL 33 5 12 38
"""
import ctypes
from pathlib import Path
import shutil
import sys

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from cbct_patient_pipeline import CB, landmarks
from run_ali import ROOT, ascii_path

FOLDS = {"UNIVERSALLAB_MODEL": ("all",), "NASOMAXILLADENTSEG_MODEL": (0,)}
MARGIN = {"x": 60.0, "ant": 15.0, "post": 10.0, "below": 10.0, "above": 35.0}   # mm


def model_folder(name):
    """nnU-Net이 요구하는 폴더 구조를 만든다. 체크포인트는 옮기지 않고 하드링크한다."""
    src = ROOT/"models/downloads_teeth"/name; fold = FOLDS[name][0]
    dst = ROOT/"models/nnunet_teeth"/name/"nnUNetTrainer__nnUNetPlans__3d_fullres"; (dst/f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    for f in ("dataset.json", "plans.json"):
        shutil.copy2(src/f, dst/f)
    ckpt = dst/f"fold_{fold}/checkpoint_final.pth"
    if not ckpt.exists(): ckpt.hardlink_to(src/"checkpoint_final.pth")
    return dst


def crop_box(P):
    """LPS mm (최소, 최대). y는 앞쪽이 작은 값이다."""
    cx = (P["RGo"][0]+P["LGo"][0])/2
    y = [P[k][1] for k in ("ANS", "Pog", "Me", "PNS", "RGo", "LGo")]; z = [P["Me"][2], P["ANS"][2]]
    return (np.array([cx-MARGIN["x"], min(y)-MARGIN["ant"], min(z)-MARGIN["below"]]),
            np.array([cx+MARGIN["x"], max(y)+MARGIN["post"], max(z)+MARGIN["above"]]))


def crop(pid, out):
    img = sitk.ReadImage(str(ascii_path(CB/f"nrrd/P_{pid}.nrrd")))
    lo, hi = crop_box(landmarks(pid)[0]); size = np.array(img.GetSize())
    a = np.array(img.TransformPhysicalPointToContinuousIndex(tuple(map(float, lo))))
    b = np.array(img.TransformPhysicalPointToContinuousIndex(tuple(map(float, hi))))
    i0 = np.maximum(np.floor(np.minimum(a, b)).astype(int), 0); i1 = np.minimum(np.ceil(np.maximum(a, b)).astype(int), size-1)
    sitk.WriteImage(img[int(i0[0]):int(i1[0])+1, int(i0[1]):int(i1[1])+1, int(i0[2]):int(i1[2])+1], str(out))


def free_gb():
    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)]+[(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tf", "af", "tv", "av", "x")]
    m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap/2**30


def predict_labels(p, in_file, out_file):
    """추론 후 GPU에서 바로 클래스 번호로 줄인다. nnU-Net의 전 클래스 재표본·프로세스 간 복사를 피한다."""
    cm = p.configuration_manager
    data, _, props = cm.preprocessor_class(verbose=False).run_case([str(in_file)], None, p.plans_manager, cm, p.dataset_json)
    with torch.no_grad():
        logits = p.predict_sliding_window_return_logits(torch.from_numpy(data).float())
        seg = logits.argmax(0).to(torch.uint8); del logits
        target = tuple(int(x) for x in props["shape_after_cropping_and_before_resampling"])
        seg = F.interpolate(seg[None, None].float(), size=target, mode="nearest-exact")[0, 0].to(torch.uint8).cpu().numpy()
    torch.cuda.empty_cache()
    full = insert_crop_into_image(np.zeros(props["shape_before_cropping"], np.uint8), seg, props["bbox_used_for_cropping"])
    img = sitk.GetImageFromArray(full.transpose(p.plans_manager.transpose_backward))
    st = props["sitk_stuff"]; img.SetSpacing(st["spacing"]); img.SetOrigin(st["origin"]); img.SetDirection(st["direction"])
    sitk.WriteImage(img, str(out_file), useCompression=True)


def main():
    name, ids = sys.argv[1], sys.argv[2:]
    work = CB/"teeth_work"/name/"in"; out = CB/"seg_teeth"/name; work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    pending = []
    for pid in ids:
        if (out/f"P_{pid}.nii.gz").exists(): continue
        f = work/f"P_{pid}_0000.nii.gz"
        if not f.exists(): crop(pid, ascii_path(f))
        pending.append(pid)
    if not pending: print("이미 완료"); return
    p = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(model_folder(name)), use_folds=FOLDS[name], checkpoint_name="checkpoint_final.pth")
    p.network.load_state_dict(p.list_of_parameters[0])
    for pid in pending:
        predict_labels(p, ascii_path(work/f"P_{pid}_0000.nii.gz"), ascii_path(out)/f"P_{pid}.nii.gz")
        print(f"P_{pid} 저장, 여유 메모리 {free_gb():.1f} GB, GPU 최대 {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
    print("done", pending, flush=True)


if __name__ == "__main__":
    main()
