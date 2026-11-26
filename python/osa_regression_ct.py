"""Representative P_2 CBCT and aligned overlays for the regression feature picker."""

import base64
from functools import cache
from io import BytesIO
from os.path import relpath

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, map_coordinates
from scipy.spatial import cKDTree
import SimpleITK as sitk

from osa.ceph import head_frame
from roi_centroid_pca import COLORS, NAMES, REGIONS, RESULTS
from softtissue_profile_pca import CB, example_masks, landmarks, midline_frame, read_nrrd


PATIENT = 2
STEP = .55
ANT = np.arange(-76, 77, STEP)
SUP = np.arange(66, -104, -STEP)
GOLD = "#ffd066"


def feature_layers(key, domains):
    if key in domains:
        domain = domains[key]["d"]
        return [domain]
    if key.startswith("soft_palate_"):
        return ["soft_palate"]
    if key.startswith("tongue_"):
        return ["tongue"]
    if key.startswith("nasal_") or key == "NAV_ts":
        return ["nasal"]
    if key.startswith("[기본] "):
        known = {
            "NAV": ["nasal"], "ANB": ["anb"], "Nperp-A": ["nperp_a"], "Nperp-Pog": ["nperp_pog"], "MP-FH": ["mpfh"],
            "TDI": ["dental"], "PHI": ["palate"], "NCWR": ["nasal", "palate"],
        }
        return known.get(key.removeprefix("[기본] "), ["airway"])
    return ["airway"]


def png_url(image):
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_image(mask, color):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, :3] = tuple(bytes.fromhex(color[1:]))
    rgba[mask, 3] = 99
    rgba[mask & ~binary_erosion(mask), 3] = 229
    return Image.fromarray(rgba)


def resample_voxels(image, world, left):
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    return (((world - origin) @ direction) / spacing)[:, ::-1].T, ((left @ direction) / spacing)[::-1]


def sample_mask(array, coords, offset, match):
    return match(map_coordinates(array, coords + offset[:, None], order=0, mode="constant", cval=0))


def read_mask(path):
    image = sitk.ReadImage(relpath(path))
    return image, sitk.GetArrayFromImage(image)


def bone_layers(world, left, origin, frame):
    layers, centers = {}, []
    number = 0
    for structure, regions in REGIONS.items():
        with np.load(RESULTS / "shape" / structure / "points.npz") as packed:
            k = np.flatnonzero(packed["ids"] == PATIENT).item()
            surface = packed["frame"][k] @ packed["head_R"][k] + packed["head_origin"][k]
        surface = (surface - origin) @ frame.T
        labels = np.load(RESULTS / "shape/local" / f"{structure}_roi_labels.npy")
        tree = cKDTree(surface)
        image, values = read_mask(CB / "seg" / f"P_{PATIENT}_seg_{structure}.nrrd")
        coords, step = resample_voxels(image, world, left)
        silhouettes = {name: np.zeros(len(world), dtype=bool) for name in regions}
        for lateral in np.arange(-46, 46.1, 1.5):
            present = sample_mask(values, coords, lateral * step, lambda voxels: voxels > 0)
            if not present.any():
                continue
            region_at_voxel = labels[tree.query((world[present] - origin) @ frame.T + [lateral, 0, 0])[1]]
            for name in regions:
                silhouettes[name][present] |= region_at_voxel == name
        for name in regions:
            # Patches come from the existing 3D ROI labels transferred to the bone mask.
            mask = silhouettes[name].reshape(len(SUP), len(ANT))
            centers.append(surface[labels == name].mean(axis=0)[1:])
            layers[f"sk_{name}"] = (NAMES[number], COLORS[number], mask_image(mask, COLORS[number]))
            number += 1
    return layers, centers


def point_pixels(point):
    ant, sup = point
    return ((ant - ANT[0]) / STEP, (SUP[0] - sup) / STEP)


def landmark_overlay(lines=(), points=(), color=GOLD):
    image = Image.new("RGBA", (len(ANT), len(SUP)))
    draw = ImageDraw.Draw(image)
    for path in lines:
        draw.line([point_pixels(p) for p in path], fill=color, width=2)
    for point, name in points:
        x, y = point_pixels(point)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="#242a2a")
        if name:
            draw.text((x + 4, y - 9), name, fill=color, stroke_width=1, stroke_fill="#242a2a")
    return image


def ceph_layers(landmarks_lps, origin, frame):
    def project(p):
        return ((p - origin) @ frame.T)[1:]

    p = {key: project(value) for key, value in landmarks_lps.items()}
    _, fh = head_frame(landmarks_lps)
    up = (fh[2] @ frame.T)[1:]
    dot = lambda *names: [(p[name], name) for name in names]
    layers = {"anb": ("ANB 기준점", GOLD,
                     landmark_overlay([(p["A"], p["N"], p["B"])], dot("A", "N", "B")))}
    for name, key in (("A", "nperp_a"), ("Pog", "nperp_pog")):
        foot = p["N"] + up * ((p[name] - p["N"]) @ up) / (up @ up)
        lines = [(p["N"] + up * 8, foot - up * 7), (foot, p[name])]
        layers[key] = (f"Nperp-{name} 기준점", GOLD, landmark_overlay(lines, dot("N", name)))
    po = (p["RPo"] + p["LPo"]) / 2
    go = (p["RGo"] + p["LGo"]) / 2
    anterior = (fh[1] @ frame.T)[1:]
    lines = [(po, po + 85 * anterior), (go, p["Me"])]
    layers["mpfh"] = ("FH·하악평면 위치", GOLD,
                      landmark_overlay(lines, [(po, "Po"), (go, "Go"), (p["Me"], "Me")]))
    return layers


def soft_layers(world, left):
    _, _, masks, _ = example_masks(PATIENT)
    color = {"airway": "#35b7ff", "nasal": "#53d77b", "tongue": "#fc697d",
             "soft_palate": "#ef9bfa", "palate": "#7be0eb", "dental": "#f6ad58"}
    titles = {"airway": "인두 기도", "nasal": "비강", "tongue": "혀",
              "soft_palate": "연구개", "palate": "경구개", "dental": "치열"}
    aa, ss = np.meshgrid(ANT, SUP)
    projected = {}
    for key, (values, ant, sup) in masks.items():
        coords = [(aa - ant[0]) / (ant[1] - ant[0]), (ss - sup[0]) / (sup[1] - sup[0])]
        projected[key] = map_coordinates(values, coords, order=0, cval=0)
    paths = [
        (CB / "seg_totalseg/head_glands_cavities" / f"P_{PATIENT}.nii.gz",
         [("soft_palate", 18, (-1., 0., 1.)), ("palate", 19, (-12., -6., 0., 6., 12.))]),
        (CB / "seg_teeth/UNIVERSALLAB_MODEL" / f"P_{PATIENT}.nii.gz",
         [("dental", None, np.arange(-30., 30.1, 1.5))]),
    ]
    for path, regions in paths:
        image, array = read_mask(path)
        coords, direction = resample_voxels(image, world, left)
        for key, value, offsets in regions:
            match = (lambda x: x > 0) if value is None else (lambda x: x == value)
            visible = np.zeros(len(world), dtype=bool)
            for offset in offsets:
                visible |= sample_mask(array, coords, direction * offset, match)
            projected[key] = visible.reshape(len(SUP), len(ANT))
    return {key: (titles[key], color[key], mask_image(mask, color[key])) for key, mask in projected.items()}


@cache
def atlas():
    landmarks_lps, _ = landmarks(PATIENT)
    origin, frame, _ = midline_frame(landmarks_lps)
    aa, ss = np.meshgrid(ANT, SUP)
    world = np.column_stack([np.zeros(aa.size), aa.ravel(), ss.ravel()]) @ frame + origin
    voxels, direction, start = read_nrrd(CB / "nrrd" / f"P_{PATIENT}.nrrd")
    coords = np.linalg.solve(direction.T, (world - start).T)[::-1]
    sagittal = map_coordinates(voxels.astype(np.float32), coords, order=1, cval=-1000.).reshape(aa.shape)
    del voxels
    base = Image.fromarray((np.clip((sagittal + 1000) / 2500, 0, 1) * 255).astype(np.uint8))
    layers, centers = bone_layers(world, frame[0], origin, frame)
    layers.update(soft_layers(world, frame[0]))
    layers.update(ceph_layers(landmarks_lps, origin, frame))
    layout = landmark_overlay(points=[(center, str(i + 1)) for i, center in enumerate(centers)])
    layers["sk_layout"] = ("12점 골격 배치", GOLD, layout)
    assert all(image.getbbox() for _, _, image in layers.values())
    markers = {"anb", "nperp_a", "nperp_pog", "mpfh", "sk_layout"}
    return {"base": png_url(base), "patient": f"P_{PATIENT}",
            "layers": {key: {"label": title, "color": color, "image": png_url(image), "marker": key in markers}
                       for key, (title, color, image) in layers.items()}}
