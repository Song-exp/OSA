"""P_5 기도 마스크와 원본의 동일 좌표 단면 대조. 원본·분할은 읽기만 한다."""
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cbct_patient_pipeline import CB, landmarks
from cbct_method_report import FIG
from run_ali import ascii_path


def main():
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    image = sitk.ReadImage(str(ascii_path(CB/'nrrd/P_5.nrrd')))
    seg = sitk.ReadImage(str(ascii_path(CB/'seg/P_5_seg_UAW.nrrd')))
    assert image.GetSize() == seg.GetSize()
    for method in ('GetOrigin', 'GetSpacing', 'GetDirection'):
        assert np.allclose(getattr(image, method)(), getattr(seg, method)())
    v = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(seg) > 0
    x = image.TransformPhysicalPointToIndex(tuple(landmarks(5)[0]['PNS']))[0]
    z = int(mask[:, :, x].sum(1).argmax())
    panels = [
        (v.max(2), mask.any(2), '전체 깊이 뼈 최대투영 + 기도 투영', 0, 2500),
        (v[:, :, x], mask[:, :, x], 'PNS 좌우 위치의 실제 시상 단면 + 기도', -1000, 1000),
        (v[z], mask[z], '같은 기도를 가로지르는 실제 축상 단면', -1000, 1000),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (ct, m, title, lo, hi) in zip(axes, panels):
        ax.imshow(ct, origin='lower', cmap='gray', vmin=lo, vmax=hi)
        rgba = np.zeros((*m.shape, 4))
        rgba[m] = (1., .65, 0., .3)
        ax.imshow(rgba, origin='lower')
        ax.contour(m, [.5], colors=['#ffab00'], linewidths=.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].axhline(z, color='#00c8ff', linewidth=1)
    fig.suptitle('P_5 원본–AMASSS 기도 좌표 대조: 동일 격자 · 이동/벡터 변형 없음', fontsize=14)
    fraction = float(np.mean(v[mask] < -400))
    fig.text(.5, .025, f'기도 마스크 내부 원본 밝기 < −400: {fraction:.2%} (해부학적 정확도의 정답 검증은 아님) · 하늘색 = 오른쪽 축상 단면 높이', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(FIG/'airway_alignment_check.png', dpi=130)
    plt.close(fig)
    print('Saved airway_alignment_check.png')


if __name__ == '__main__':
    main()
