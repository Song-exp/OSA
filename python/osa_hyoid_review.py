"""Render CBCT crops for a visual review of the current H landmark."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

import cbct_eb_detect as eb


def main():
    out = eb.OUT.parents[2] / "analysis/all83/hyoid_review"
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(p.parent.name[2:]) for p in eb.OUT.glob("P_*/result.json"))
    assert len(ids) == 85
    ant = np.arange(-45, 33, eb.STEP)
    detection_sup = np.arange(*eb.SUP, eb.STEP)
    sup = np.r_[np.arange(-115, -95, eb.STEP)[:50], detection_sup, np.arange(-35, -30, eb.STEP)]
    left = np.arange(-6, 6+1e-9, eb.STEP)
    grid = np.stack(np.meshgrid(left, ant, sup, indexing="ij"), -1)
    center = np.argmin(abs(left))
    rows = []
    for start in range(0, len(ids), 12):
        fig, axes = plt.subplots(4, 6, figsize=(18, 10), layout="constrained")
        for ax in axes.flat:
            ax.axis("off")
        for i, pid in enumerate(ids[start:start+12]):
            origin, frame, _ = eb.midline_frame(eb.landmarks(pid)[0])
            volume, dirs, offset = eb.read_nrrd(eb.CB / f"nrrd/P_{pid}.nrrd")
            coords = np.linalg.solve(dirs.T, (origin + grid.reshape(-1, 3) @ frame-offset).T)[::-1]
            slab = map_coordinates(volume.astype(np.float32), coords, order=1, cval=-3000).reshape(grid.shape[:-1])
            del volume, coords
            mip = slab[abs(left) <= 3].max(0)
            n_ant = len(np.arange(*eb.ANT, eb.STEP))
            hy = eb.hyoid(mip[:n_ant, 50:50+len(detection_sup)], ant[:n_ant], detection_sup)
            point = hy[3:5] if hy else (np.nan, np.nan)
            found = np.isfinite(point).all()
            a, z = point if found else (-10, -77)
            views = [mip, slab[center]]
            for j, (ax, view) in enumerate(zip(axes.flat[2*i:2*i+2], views)):
                ax.imshow(view.T, origin="lower", cmap="gray", vmin=-500, vmax=1050,
                          extent=[ant[0]-.2, ant[-1]+.2, sup[0]-.2, sup[-1]+.2])
                ax.set_xlim(a-21, a+21)
                ax.set_ylim(z-25, z+21)
                ax.set_title(f"P_{pid} {'MIP' if j == 0 else 'midline'}{' (no H)' if not found else ''}", fontsize=10)
                if found:
                    ax.plot(*point, "o", ms=5, mfc="none", mec="#00ffff", mew=1)
                else:
                    ax.axhline(-95, color="cyan", lw=.6, ls=":")
            rows.append({"Patient": f"P_{pid}", "H_ant": point[0], "H_sup": point[1],
                         "sheet": f"sheet_{start//12+1:02}.png"})
        fig.savefig(out / f"sheet_{start//12+1:02}.png", dpi=105)
        plt.close(fig)
        print(f"{min(start+12,len(ids))}/{len(ids)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "coordinates.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
