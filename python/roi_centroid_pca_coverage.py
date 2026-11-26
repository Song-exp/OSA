"""Render P_2 CBCT alongside the cohort's first four centroid PCA axes."""

from itertools import combinations
from os.path import relpath

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label
from scipy.spatial import ConvexHull, cKDTree
from skimage.measure import marching_cubes
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from roi_centroid_pca import (
    AXES, COLORS, NAMES, OUTPUT, REGIONS, RESULTS, configure_plot_font, plt,
)


PATIENT = 2
PAPER, INK, MUTED, RULE = "#f5f3ef", "#182832", "#50616a", "#cbd6d4"


def region_meshes():
    meshes, centers, volumes = [], [], []
    for structure, regions in REGIONS.items():
        with np.load(RESULTS / "shape" / structure / "points.npz") as files:
            index = np.flatnonzero(files["ids"] == PATIENT).item()
            points = files["frame"][index]
            origin = files["head_origin"][index]
            rotation = files["head_R"][index]
        labels = np.load(RESULTS / "shape/local" / f"{structure}_roi_labels.npy")
        path = RESULTS / "CBCT/seg" / f"P_{PATIENT}_seg_{structure}.nrrd"
        image = sitk.ReadImage(relpath(path))
        mask = sitk.GetArrayFromImage(image) > 0
        components, count = label(mask)
        if count > 1:
            mask = components == (np.bincount(components.ravel())[1:].argmax() + 1)
        del components
        voxels = np.argwhere(mask)
        del mask
        direction = np.array(image.GetDirection()).reshape(3, 3)
        spacing = np.asarray(image.GetSpacing())
        offset = (np.array(image.GetOrigin()) - origin) @ rotation.T
        transform = (direction.T @ rotation.T) * spacing[:, None]
        _, nearest = cKDTree(points).query(voxels[:, ::-1] @ transform + offset, workers=4)
        for region in regions:
            selected = voxels[labels[nearest] == region]
            low, high = selected.min(axis=0) - 2, selected.max(axis=0) + 3
            crop = np.zeros(high - low, dtype=np.uint8)
            crop[tuple((selected - low).T)] = 1
            vertices, faces, _, _ = marching_cubes(crop, level=.5, step_size=2)
            meshes.append(((vertices + low)[:, ::-1] @ transform + offset)[faces])
            centers.append(points[labels == region].mean(axis=0))
            volumes.append(len(selected) * spacing.prod() / 1000)
    return meshes, np.asarray(centers), volumes, origin, rotation


def draw_ct(ax, centers, origin, rotation):
    image = sitk.ReadImage(relpath(RESULTS / "CBCT/nrrd" / f"P_{PATIENT}.nrrd"))
    ant = np.linspace(-122, 36, 530)
    sup = np.linspace(-137, 33, 570)
    lateral = (centers[:, 0].min() + centers[:, 0].max()) / 2
    physical_origin = np.array([lateral, ant[0], sup[0]]) @ rotation + origin
    slice_image = sitk.Resample(image, [1, len(ant), len(sup)], sitk.Transform(),
                               sitk.sitkLinear, physical_origin.tolist(),
                               [1., float(ant[1] - ant[0]), float(sup[1] - sup[0])],
                               rotation.T.ravel().tolist(), -1000., sitk.sitkFloat32)
    section = sitk.GetArrayFromImage(slice_image)[:, :, 0]
    ax.imshow(section, cmap="gray", origin="lower", extent=[ant[0], ant[-1], sup[0], sup[-1]],
              vmin=np.percentile(section, 24), vmax=np.percentile(section, 96))
    ax.set_axis_off()
    ax.text(.06, .05, "뒤", color="white", transform=ax.transAxes, weight="bold")
    ax.text(.94, .05, "앞", color="white", transform=ax.transAxes, weight="bold", ha="right")


def setup_3d(ax, triangles):
    low = triangles.min(axis=(0, 1)) - 3
    high = triangles.max(axis=(0, 1)) + 3
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=15, azim=28)
    ax.set_box_aspect(high - low, zoom=1.13)
    for start, end, set_limit in zip(low, high, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        set_limit(start, end)


def face_lighting(triangles):
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    return np.abs(normal @ np.array([.65, .46, .61]))


def draw_mesh(ax, meshes, triangles, fill, lighting, ambient):
    colors = np.repeat(fill[:, :3], [len(mesh) for mesh in meshes], axis=0)
    shade = ambient + (1 - ambient) * lighting
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors * shade[:, None],
                                         linewidths=0, antialiased=False))


def draw_hull(ax, centers):
    hull = ConvexHull(centers)
    edges = {tuple(sorted(pair)) for face in hull.simplices for pair in combinations(face, 2)}
    ax.add_collection3d(Poly3DCollection(centers[hull.simplices], facecolor="#83aab6",
                                         alpha=.13, linewidths=0))
    for a, b in sorted(edges):
        ax.plot(*centers[[a, b]].T, color="#355969", linewidth=1.35, alpha=.86)
    for index, (center, color) in enumerate(zip(centers, COLORS)):
        outer = index in hull.vertices
        ax.scatter(*center, s=72, color=color if outer else PAPER,
                   edgecolors="white" if outer else color, linewidth=1.45, depthshade=False)
    ax.figure.canvas.draw()
    offsets = [[9, -10], [-12, 4], [-9, -10], [-14, 0], [13, 3], [12, -5],
               [7, 11], [12, 0], [-8, 12], [10, 11], [-15, -5], [-10, 12]]
    for index, (center, offset) in enumerate(zip(centers, offsets)):
        x, y, _ = proj3d.proj_transform(*center, ax.get_proj())
        location = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
        ax.annotate(str(index + 1), location, xycoords="axes fraction", xytext=offset,
                    textcoords="offset points", ha="center", va="center", size=9,
                    weight="bold", color=INK, zorder=20)


def draw_loadings(ax, coefficients, ratios):
    row_ids = np.arange(36)
    directions = ("좌우", "앞뒤", "상하")
    colors = np.repeat(COLORS, len(AXES))
    for i in range(12):
        if i % 2 == 0:
            ax.axhspan(3 * i - .5, 3 * i + 2.5, color="#eef2f1")
        ax.axhline(3 * i - .5, color=RULE, linewidth=.6)
    for index, column in enumerate(coefficients.T):
        ax.axvline(index - .5, color=RULE, linewidth=.6)
        ax.axvline(index, color="#9dacb0", linewidth=.55)
        shares = 100 * np.square(column)
        widths = np.sign(column) * shares * .33 / 25
        ax.barh(row_ids, widths, left=index, height=.56, color=colors)
        for i in row_ids[shares >= 5]:
            sign = np.sign(widths[i])
            ax.text(index + widths[i] + .015 * sign, i, f"{shares[i]:.1f}",
                    color=INK, size=9, ha="left" if sign > 0 else "right", va="center")
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(35.5, -.5)
    ax.set_xticks(np.arange(4), [f"PC{i+1}  {r:.1%}" for i, r in enumerate(ratios)], size=10)
    ax.xaxis.tick_top()
    ax.set_yticks(row_ids, [f"{i//3+1:02d} {directions[i%3]}" for i in row_ids], size=9)
    ax.tick_params(axis="both", length=0, pad=5)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(np.array(to_rgb(color)) * .68)
    for spine in ax.spines.values():
        spine.set_color(RULE)


def main():
    configure_plot_font()
    variance = pd.read_csv(OUTPUT / "variance.csv")
    axes = pd.read_csv(OUTPUT / "loadings.csv")
    rows = [f"{structure}_{region}" for structure, regions in REGIONS.items() for region in regions]
    assert axes["region"].tolist() == np.repeat(rows, len(AXES)).tolist()
    assert axes["axis"].tolist() == list(AXES) * len(rows)
    all_loadings = axes[variance["PC"].tolist()].to_numpy()
    eigenvalues = variance["explained_variance"].to_numpy()
    region_weights = np.square(all_loadings.reshape(12, 3, -1)).sum(axis=1)
    retained = region_weights[:, :4] @ eigenvalues[:4]
    total = region_weights @ eigenvalues
    coverage = 100 * retained / total
    overall = 100 * variance.loc[3, "cumulative_fraction"]
    assert np.isclose(np.average(coverage, weights=total), overall)

    meshes, centers, volumes, origin, rotation = region_meshes()
    triangles = np.concatenate(meshes)
    lighting = face_lighting(triangles)
    color_scale = ScalarMappable(norm=Normalize(0, 100), cmap=colormaps["viridis"])
    fill = color_scale.to_rgba(coverage)

    fig = plt.figure(figsize=(28, 10.2), facecolor=PAPER)
    fig.text(.032, .936, "12점 골격 도형과 PC1~PC4", size=31, weight="bold", color=INK)
    fig.text(.033, .901, f"P_{PATIENT} CBCT에 80명 PCA 결과를 표시  ·  PC1~PC4의 전체 누적 설명분산 {overall:.2f}%",
             size=14, color=MUTED)
    for x, title, subtitle in [
        (.034, "원본 CBCT", "정중 부근 시상 단면"),
        (.204, "12개 부위의 3D 부피", "P_2 골격 마스크의 부위별 분할"),
        (.389, "12개 중심점의 외곽", "P_2의 부위 번호와 고유 색상"),
        (.561, "부위별 배치 설명률", "PC1~PC4에 남는 각 부위의 좌표 변동 비율"),
        (.750, "PC1~PC4 로딩", "36개 좌표 × 4개 축"),
    ]:
        fig.text(x, .850, title, size=18, weight="bold", color=INK)
        fig.text(x, .821, subtitle, size=11, color=MUTED)

    ct_ax = fig.add_axes([.029, .292, .153, .51])
    draw_ct(ct_ax, centers, origin, rotation)
    region_ax = fig.add_axes([.192, .29, .175, .50], projection="3d", facecolor=PAPER)
    hull_ax = fig.add_axes([.378, .268, .165, .525], projection="3d", facecolor=PAPER)
    coverage_ax = fig.add_axes([.547, .29, .181, .50], projection="3d", facecolor=PAPER)
    for ax in (region_ax, hull_ax, coverage_ax):
        setup_3d(ax, triangles)
    draw_hull(hull_ax, centers)
    draw_mesh(region_ax, meshes, triangles, np.array([to_rgb(c) for c in COLORS]), lighting, .57)
    draw_mesh(coverage_ax, meshes, triangles, fill, lighting, .84)
    bar_ax = fig.add_axes([.565, .273, .143, .014])
    colorbar = fig.colorbar(color_scale, cax=bar_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    colorbar.ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    colorbar.ax.tick_params(length=0, labelsize=10)
    colorbar.outline.set_visible(False)
    hull_ax.text2D(.10, .06, "● 외곽점    ○ 내부점", transform=hull_ax.transAxes, color=MUTED, size=10)
    loading_ax = fig.add_axes([.775, .267, .200, .515], facecolor="white")
    draw_loadings(loading_ax, all_loadings[:, :4], variance["explained_fraction"].iloc[:4])
    fig.text(.750, .239, "왼쪽 음수 / 오른쪽 양수  ·  길이 = 로딩² 기여율  ·  5% 이상 수치 표시", size=10, color=MUTED)

    for group, heading in enumerate(("하악", "상악", "두개저")):
        x = (.034, .267, .500)[group]
        fig.text(x, .209, heading, size=13, weight="bold", color=INK)
        fig.text(x + .157, .209, "부피", size=10, color=MUTED, ha="right")
        fig.text(x + .206, .209, "설명률", size=10, color=MUTED, ha="right")
        for row in range(4):
            i = group * 4 + row
            y = .176 - row * .027
            fig.text(x, y, "●", color=COLORS[i], size=10, va="center")
            fig.text(x + .012, y, f"{i+1:02d}  {NAMES[i]}", color=INK, size=11, va="center")
            fig.text(x + .157, y, f"{volumes[i]:.2f} mL", color=MUTED, size=11, va="center", ha="right")
            fig.text(x + .170, y, "■", color=fill[i], size=11, va="center")
            fig.text(x + .206, y, f"{coverage[i]:.1f}%", color=INK, size=11, va="center", ha="right")
    fig.text(.750, .197, "색칠 기준", size=13, weight="bold", color=INK)
    fig.text(.750, .167, "80명의 각 부위 좌우·앞뒤·상하 좌표에서", size=11, color=MUTED)
    fig.text(.750, .140, "PC1~PC4가 설명하는 분산 ÷ 전체 분산.", size=11, color=MUTED)
    fig.text(.750, .107, "같은 부위에 속한 복셀에는 같은 설명률을 적용.", size=11, color=MUTED)
    fig.text(.033, .056, "부위별 부피는 기존 골격 마스크를 나눈 시각화용 추정치입니다. 중심점은 기존 표면 대응점의 좌표 평균입니다.",
             size=10.5, color=MUTED)
    path = OUTPUT / "p2_centroid_pc1_to_pc4_five_panels.png"
    fig.savefig(path, dpi=185, facecolor=PAPER)
    plt.close(fig)
    print(path)
    print(f"Overall {overall:.4f}%; regional range {coverage.min():.2f}-{coverage.max():.2f}%")


if __name__ == "__main__":
    main()
