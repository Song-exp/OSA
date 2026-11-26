"""Build a self-contained report of the 95% PCA models for local and joint ROI shape."""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from roi_centroid_pca import AXES, COLORS, NAMES, OUTPUT, REGIONS, RESULTS, normalize
from shape_ssm import gpa


TARGET = .95
PATIENT = 2
DIRECTIONS = ("좌우", "앞뒤", "상하")
NEGATIVE = ("오른쪽", "뒤쪽", "아래쪽")
POSITIVE = ("왼쪽", "앞쪽", "위쪽")


def fit(configurations):
    flat = configurations.reshape(len(configurations), -1)
    model = PCA(n_components=TARGET, svd_solver="full").fit(flat)
    components = model.components_
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    model.components_ *= signs[:, None]
    ratios = model.explained_variance_ratio_
    assert ratios.sum() > TARGET and ratios[:-1].sum() <= TARGET
    return model


def encode_points(points, digits=5):
    return np.round(points, digits).tolist()


def display_rotation(points, aligned, sample_index):
    # Put the normalized PCA shapes in the same anatomical orientation as P_2.
    original, _ = normalize(points)
    source = aligned[sample_index]
    target = original[sample_index]
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    return u @ correction @ vt


def describe_local_axis(mean, delta):
    coordinate_energy = np.square(delta).sum(axis=0)
    axis = coordinate_energy.argmax()
    share = coordinate_energy[axis] / coordinate_energy.sum()
    point = np.abs(delta[:, axis]).argmax()
    low, high = np.quantile(mean, (1 / 3, 2 / 3), axis=0)
    location = []
    for i, coordinate in enumerate(mean[point]):
        if coordinate <= low[i]:
            location.append(NEGATIVE[i])
        elif coordinate >= high[i]:
            location.append(POSITIVE[i])
    site = "·".join(location) + " 표면" if location else "부위 중앙 쪽 표면"
    direction = POSITIVE[axis] if delta[point, axis] > 0 else NEGATIVE[axis]
    return (f"{DIRECTIONS[axis]} 성분이 가장 큼({share:.1%}). "
            f"점수가 높은 쪽에서 {site}의 대표점({point + 1}번)이 평균보다 {direction}에 놓임.")


def describe_joint_axis(delta):
    coordinate_energy = np.square(delta).sum(axis=0)
    axis = coordinate_energy.argmax()
    share = coordinate_energy[axis] / coordinate_energy.sum()
    high = delta[:, axis].argmax()
    low = delta[:, axis].argmin()
    return (f"{DIRECTIONS[axis]} 성분이 가장 큼({share:.1%}). 점수가 높은 쪽에서 "
            f"{NAMES[high]} 중심은 {POSITIVE[axis]}, {NAMES[low]} 중심은 {NEGATIVE[axis]}에 놓이는 배치.")


def local_data():
    regions = []
    reference_cloud = []
    for structure, names in REGIONS.items():
        with np.load(RESULTS / "shape" / structure / "points.npz") as data:
            ids, points = data["ids"], data["frame"]
        reference = np.flatnonzero(ids == PATIENT).item()
        labels = np.load(RESULTS / "shape/local" / f"{structure}_roi_labels.npy")
        qc = pd.read_csv(RESULTS / "shape" / structure / "review/qc_review.csv").set_index("Patient")
        reasons = qc.loc[[f"P_{i}" for i in ids], "confirmed_fail_procrustes"]
        use = reasons.astype("string").fillna("").str.strip().eq("").to_numpy()
        for name in names:
            selected = labels == name
            raw = points[:, selected]
            # GPA removes position, rotation and centroid size before shape PCA.
            aligned, _ = gpa(raw[use])
            reference_fit_index = np.flatnonzero(np.flatnonzero(use) == reference).item()
            axis = display_rotation(raw[use], aligned, reference_fit_index)
            model = fit(aligned)
            mean = model.mean_.reshape(-1, 3) @ axis
            displacements = ((model.components_ * np.sqrt(model.explained_variance_)[:, None])
                             .reshape(model.n_components_, -1, 3) @ axis)
            # The GPA scale is centroid size 1. Multiply by sqrt(number of points)
            # so the display units correspond to the RMS normalization used jointly.
            factor = np.sqrt(raw.shape[1])
            mean *= factor
            displacements *= factor
            assert np.allclose(mean.mean(axis=0), 0, atol=1e-12)
            assert np.allclose(displacements.mean(axis=1), 0, atol=1e-12)
            index = len(regions)
            reference_cloud.extend(np.column_stack([raw[reference], np.full(raw.shape[1], index)]))
            regions.append({
                "name": NAMES[index], "color": COLORS[index], "structure": structure,
                "points": raw.shape[1], "patients": int(use.sum()),
                "ratios": model.explained_variance_ratio_.tolist(),
                "features": [f"{structure}_{name}_PC{i + 1}" for i in range(model.n_components_)],
                "meanings": [describe_local_axis(mean, delta) for delta in displacements],
            })
    return regions, encode_points(np.asarray(reference_cloud), 3)


def joint_data():
    cols = [f"{structure}_{name}_c{axis}" for structure, names in REGIONS.items()
            for name in names for axis in AXES]
    table = pd.read_csv(RESULTS / "shape/local/local_roi_features.csv").set_index("Patient")
    qc = pd.read_csv(OUTPUT / "scores.csv").set_index("Patient")
    eligible = qc.index[qc.in_pca_axes]
    configs, _ = normalize(table.loc[eligible, cols].to_numpy().reshape(-1, 12, 3))
    model = fit(configs)
    deltas = (model.components_ * np.sqrt(model.explained_variance_)[:, None]).reshape(-1, 12, 3)
    previous = pd.read_csv(OUTPUT / "variance.csv")
    assert np.allclose(model.explained_variance_ratio_, previous.explained_fraction.iloc[:model.n_components_])
    return {
        "patients": len(eligible), "ratios": model.explained_variance_ratio_.tolist(),
        "features": [f"layout_PC{i + 1}" for i in range(model.n_components_)],
        "meanings": [describe_joint_axis(delta) for delta in deltas],
    }


def soft_data():
    """연조직 3부위의 크기 제거 프로파일 PCA와 예시 CT."""
    import io
    import matplotlib.pyplot as plt
    import softtissue_profile_pca as ST
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    profiles, excluded = ST.profiles(); ex = ST.example_masks(PATIENT)
    size = {"airway": ("PAV", "인두 기도 부피 PAV_fh (mm³)"), "nasal": ("NAV", "비강 기도 부피 NAV_ts (mm³)"), "tongue": ("tongue_mL", "혀 부피 (mL)")}
    out = []
    for key, X in profiles.items():
        model, scores, _ = ST.fit(X)
        fig, ax = plt.subplots(figsize=(5.4, 4.8)); ST.ct_panel(ax, key, ex); fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90); plt.close(fig)
        name, prefix, lo, hi = ST.REGIONS[key]
        features = [f"{prefix.removesuffix('_p')}_PC{i + 1}" for i in range(model.n_components_)]
        meanings = [
            f"{int(axis.argmax()) + 1}번 구간의 상대 단면적은 평균보다 크고, "
            f"{int(axis.argmin()) + 1}번 구간은 작은 쪽으로 나타나는 패턴."
            for axis in model.components_
        ]
        out.append({"key": key, "name": name, "color": ST.COLORS[key], "lo": lo, "hi": hi, "patients": len(scores),
                    "ratios": model.explained_variance_ratio_.tolist(), "features": features, "meanings": meanings,
                    "size_var": size[key][0], "size_label": size[key][1], "excluded": excluded.get(key, []),
                    "ct": base64.b64encode(buf.getvalue()).decode("ascii")})
    return out


def main():
    local, reference = local_data()
    joint = joint_data()
    soft = soft_data()
    source = Path(__file__).with_name("roi_shape_pca_report_template.html")
    template = source.read_text(encoding="utf-8")
    overview = (OUTPUT / "p2_centroid_pc1_to_pc4_five_panels.png").read_bytes()
    data = json.dumps({"local": local, "reference": reference, "joint": joint, "soft": soft},
                      ensure_ascii=False, separators=(",", ":"))
    report = template.replace("__PCA_DATA__", data).replace("__OVERVIEW__", base64.b64encode(overview).decode("ascii"))
    path = OUTPUT / "roi_and_joint_pca_report.html"
    path.write_text(report, encoding="utf-8")
    for region in local:
        print(f"{region['structure']:4s} | {region['name']:10s} | n={region['patients']} | "
              f"k={len(region['ratios']):2d} | cumulative={sum(region['ratios']):.2%}")
    print(f"Joint: k={len(joint['ratios'])} | cumulative={sum(joint['ratios']):.2%}")
    for region in soft:
        print(f"SOFT | {region['name']:6s} | n={region['patients']} | k={len(region['ratios']):2d} | cumulative={sum(region['ratios']):.2%}")
    print(path)


if __name__ == "__main__":
    main()
