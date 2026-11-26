"""PCA of the size-normalized configuration of the 12 existing CBCT ROI centroids."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


RESULTS = Path(__file__).resolve().parents[1] / "Result_py"
OUTPUT = RESULTS / "analysis" / "roi_centroid_pca"
REGIONS = {
    "MAND": ("symphysis", "body", "gonial", "ramus_condyle"),
    "MAX": ("anterior_alveolar", "palate", "posterior", "orbital_floor"),
    "CB": ("anterior_base", "sella", "clivus", "temporal"),
}
NAMES = ("턱끝", "하악체", "하악각", "하악지·과두", "앞 치조부", "구개",
         "상악 후방부", "안와저", "앞 두개저", "터키안", "사대", "측두골")
COLORS = ("#a00b52", "#df4551", "#f78638", "#932d18", "#005967", "#008687",
          "#3ba930", "#33bfb3", "#243ea1", "#8234b1", "#4961dc", "#8975df")
AXES = ("left", "ant", "sup")


def normalize(points):
    centered = points - points.mean(axis=1, keepdims=True)
    size = np.sqrt(np.square(centered).sum(axis=2).mean(axis=1))
    if not np.isfinite(points).all() or np.any(size <= 0):
        raise ValueError("Expected finite coordinates and a positive RMS radius for every patient")
    return centered / size[:, None, None], size


def fit_pca(configurations, use):
    flat = configurations.reshape(len(configurations), -1)
    mean = flat[use].mean(axis=0)
    _, singular, vectors = np.linalg.svd(flat[use] - mean, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(np.count_nonzero(use), flat.shape[1])
                            * np.finfo(flat.dtype).eps)
    vectors, singular = vectors[:rank], singular[:rank]
    # Fix the arbitrary sign so the largest absolute loading is positive.
    vectors *= np.sign(vectors[np.arange(rank), np.abs(vectors).argmax(axis=1)])[:, None]
    scores = (flat - mean) @ vectors.T
    variance = np.square(singular) / (np.count_nonzero(use) - 1)
    return mean, vectors, scores, variance


def configure_plot_font():
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def plot_summary(mean, pc1, variance, count, path):
    paper, ink, muted = "#f5f3ef", "#182832", "#52656d"
    fig = plt.figure(figsize=(15, 7.5), facecolor=paper)
    fig.text(.05, .91, "12개 중심점의 골격 배치 PCA", fontsize=26, weight="bold", color=ink)
    fig.text(.051, .86, f"축 적합 {count}명  ·  12점 중심 정렬 및 크기 정규화  ·  공통 머리 좌표축 유지",
             fontsize=12, color=muted)

    modes = (mean - pc1, mean, mean + pc1)
    extrema = np.asarray(modes).reshape(-1, 3)
    low, high = extrema.min(axis=0) - .06, extrema.max(axis=0) + .06
    for column, (points, title) in enumerate(zip(modes, ("PC1 -1 SD", "평균 배치", "PC1 +1 SD"))):
        ax = fig.add_axes([.045 + .237 * column, .22, .233, .58], projection="3d", facecolor=paper)
        ax.set_title(title, fontsize=15, weight="bold", color=ink, pad=0)
        ax.view_init(elev=18, azim=24)
        ax.set_proj_type("ortho")
        ax.set_box_aspect(high - low, zoom=1.16)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])
        ax.set_axis_off()
        hull = ConvexHull(points)
        ax.add_collection3d(Poly3DCollection(points[hull.simplices], facecolor="#b5cbd1",
                                            edgecolor="#5b7a86", alpha=.21, linewidth=.8))
        ax.scatter(*points.T, c=COLORS, s=49, edgecolors="white", linewidths=.9, depthshade=False)

    ax = fig.add_axes([.775, .33, .18, .36], facecolor=paper)
    shown = np.arange(1, min(6, len(variance)) + 1)
    percent = 100 * variance / variance.sum()
    ax.bar(shown, percent[:len(shown)], color=["#25576c"] + ["#a5b9be"] * (len(shown) - 1))
    ax.set_title("설명분산", loc="left", fontsize=14, weight="bold", color=ink)
    ax.set_ylabel("%")
    ax.set_xticks(shown, [f"PC{i}" for i in shown], fontsize=8)
    ax.set_ylim(0, max(percent) * 1.22)
    ax.grid(axis="y", color="#d7dedc", linewidth=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in shown:
        ax.text(i, percent[i - 1] + max(percent) * .028, f"{percent[i - 1]:.1f}", ha="center", fontsize=8)
    for index, (name, color) in enumerate(zip(NAMES, COLORS)):
        x = .052 + .232 * (index // 4)
        y = .173 - .034 * (index % 4)
        fig.text(x, y, "●", color=color, fontsize=10, va="center")
        fig.text(x + .016, y, name, color=ink, fontsize=10, va="center")
    fig.savefig(path, dpi=160, facecolor=paper)
    plt.close(fig)


def plot_loadings(vectors, ratios, path, annotate=False):
    """Use bar direction for loading sign and bar length for squared-loading share."""
    paper, ink, muted = "#f5f3ef", "#182832", "#52656d"
    rows = np.arange(vectors.shape[1])
    count = len(vectors)
    shares = 100 * np.square(vectors)
    maximum = shares.max()
    # Round up to an axis scale that applies to every cell in this image.
    ceiling = 5 * np.ceil(maximum / 5)
    width_per_percent = .34 / ceiling
    row_colors = np.repeat(COLORS, len(AXES))
    directions = ("좌우", "앞뒤", "상하")
    labels = [f"{name}  ·  {direction}" for name in NAMES for direction in directions]
    column_width = 1.21 if annotate else .68
    fig = plt.figure(figsize=(max(11, 2.4 + column_width * count), 13), facecolor=paper)
    fig.text(.047, .965, "12점 골격 배치 PCA 로딩", size=25, weight="bold", color=ink)
    fig.text(.049, .928, "행 36개 = 부위 12개 × 좌우·앞뒤·상하   |   열 = 주성분   |   같은 색 = 같은 부위",
             size=11.5, color=muted)

    left = 1.65 / fig.get_figwidth()
    ax = fig.add_axes([left, .10, 1 - left - .04, .765], facecolor="white")
    for region in range(len(NAMES)):
        if region % 2 == 0:
            ax.axhspan(region * 3 - .5, region * 3 + 2.5, color="#f0f3f2", zorder=0)
    for row in range(3, len(rows), 3):
        ax.axhline(row - .5, color="#cbd6d4", linewidth=.65, zorder=1)
    for column in range(count):
        ax.axvline(column - .5, color="#cbd6d4", linewidth=.7, zorder=1)
        ax.axvline(column, color="#a4b4b8", linewidth=.6, zorder=1)
        bar_width = np.sign(vectors[column]) * shares[column] * width_per_percent
        ax.barh(rows, bar_width, left=column, height=.55, color=row_colors, zorder=2)
        if annotate:
            for row in rows[shares[column] >= 2.5]:
                sign = np.sign(bar_width[row])
                ax.text(column + bar_width[row] + .015 * sign, row,
                        f"{shares[column, row]:.1f}", va="center",
                        ha="left" if sign > 0 else "right", size=8, color=ink)

    ax.set_xlim(-.5, count - .5)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_yticks(rows, labels, fontsize=9)
    ax.tick_params(axis="y", length=0, pad=8)
    for label, color in zip(ax.get_yticklabels(), row_colors):
        label.set_color(np.array(matplotlib.colors.to_rgb(color)) * .68)
    ax.set_xticks(np.arange(count), [f"PC{i + 1}\n{r:.1%}" for i, r in enumerate(ratios[:count])],
                  fontsize=10 if annotate else 8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_color("#cbd6d4")
    fig.text(.049, .065, f"왼쪽은 음의 로딩, 오른쪽은 양의 로딩. 막대 길이는 로딩² × 100의 기여율이며, 각 열의 합은 100%입니다. 한쪽 최대 눈금 {ceiling:g}%.",
             size=10, color=ink)
    note = "열 제목 아래 수치는 각 PC의 설명분산입니다."
    if annotate:
        note += " 기여율 2.5% 이상은 막대 옆에 수치로 표시했습니다."
    fig.text(.049, .04, note, size=10, color=muted)
    fig.savefig(path, dpi=170, facecolor=paper)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loadings-only", action="store_true",
                        help="Draw the loading matrices from the saved PCA results")
    parser.add_argument("--components", type=int,
                        help="Export this many PCs and their loading matrix from the saved fit")
    args = parser.parse_args()
    configure_plot_font()
    if args.loadings_only or args.components is not None:
        variance_table = pd.read_csv(OUTPUT / "variance.csv")
        axes = pd.read_csv(OUTPUT / "loadings.csv")
        if args.components is not None:
            if not 1 <= args.components <= len(variance_table):
                parser.error(f"--components must be between 1 and {len(variance_table)}")
            export_components(args.components, variance_table, axes)
            return
        draw_loading_matrices(axes[variance_table["PC"]].to_numpy().T,
                              variance_table["explained_fraction"].to_numpy())
        return

    regions = [f"{structure}_{region}" for structure, group in REGIONS.items() for region in group]
    columns = [f"{region}_c{axis}" for region in regions for axis in AXES]
    table = pd.read_csv(RESULTS / "shape/local/local_roi_features.csv").set_index("Patient")
    table = table.sort_index(key=lambda values: values.str.split("_").str[-1].astype(int))
    if not table.index.is_unique:
        raise ValueError("Expected exactly one row per patient")
    failed = set()
    for structure in REGIONS:
        qc = pd.read_csv(RESULTS / "shape" / structure / "review/qc_review.csv").set_index("Patient")
        if not table.index.isin(qc.index).all():
            raise ValueError(f"Missing QC records for {structure}")
        reasons = qc["confirmed_fail_frame"].astype("string").fillna("").str.strip()
        failed.update(qc.index[reasons.ne("")])
    use = ~table.index.isin(failed)
    configurations, size = normalize(table[columns].to_numpy().reshape(len(table), len(regions), 3))
    mean, vectors, scores, variance = fit_pca(configurations, use)
    names = [f"PC{i}" for i in range(1, len(variance) + 1)]
    ratios = variance / variance.sum()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    score_table = pd.DataFrame(scores, index=table.index, columns=names)
    score_table.insert(0, "rms_radius_mm", size)
    score_table.insert(0, "in_pca_axes", use)
    score_table.to_csv(OUTPUT / "scores.csv", encoding="utf-8-sig")
    pd.DataFrame({"PC": names, "explained_variance": variance, "explained_fraction": ratios,
                  "cumulative_fraction": np.cumsum(ratios)}).to_csv(OUTPUT / "variance.csv", index=False)
    axes = pd.DataFrame(vectors.T, columns=names)
    axes.insert(0, "mean_coordinate", mean)
    axes.insert(0, "axis", np.tile(AXES, len(regions)))
    axes.insert(0, "region", np.repeat(regions, 3))
    axes.to_csv(OUTPUT / "loadings.csv", index=False)
    mean = mean.reshape(-1, 3)
    pc1 = vectors[0].reshape(-1, 3) * np.sqrt(variance[0])
    plot_summary(mean, pc1, variance, use.sum(), OUTPUT / "pc1_shape.png")
    draw_loading_matrices(vectors, ratios)
    print(f"Fit {use.sum()} of {len(use)} patients; projected excluded: {table.index[~use].tolist()}")
    print(f"Explained fractions PC1-5: {np.round(ratios[:5], 4)}")
    print("PC1 +1 SD components (left, anterior, superior):")
    for region, delta in zip(regions, pc1):
        print(f"  {region:25s} {np.round(delta, 4)}")


def draw_loading_matrices(vectors, ratios):
    plot_loadings(vectors[:6], ratios, OUTPUT / "loadings_pc1_to_pc6.png", annotate=True)
    plot_loadings(vectors, ratios, OUTPUT / "loadings_all.png")


def export_components(count, variance, axes):
    selected = variance.iloc[:count]
    names = selected["PC"].tolist()
    scores = pd.read_csv(OUTPUT / "scores.csv", usecols=["Patient", "in_pca_axes", *names])
    scores.to_csv(OUTPUT / f"scores_pc1_to_pc{count}.csv", index=False, encoding="utf-8-sig")
    plot_loadings(axes[names].to_numpy().T, selected["explained_fraction"].to_numpy(),
                  OUTPUT / f"loadings_pc1_to_pc{count}.png", annotate=True)
    print(f"{count} PCs explain {selected['cumulative_fraction'].iloc[-1]:.4%} of the fitted variance.")


if __name__ == "__main__":
    main()
