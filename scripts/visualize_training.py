"""
训练数据可视化 — 渲染压力场 GIF 动画

用法:
    uv run --extra geostat python scripts/visualize_training.py --samples 0 5 42
    uv run --extra geostat python scripts/visualize_training.py --samples 0 5 42 --fps 10
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json


def render_gif(sample_idx: int, npz, configs, time_days, output_dir: str, fps: int = 8):
    pressure = npz["pressure_MPa"][sample_idx]
    perm = npz["permeability_mD"][sample_idx]
    well_mask = npz["well_mask"][sample_idx]
    well_value = npz["well_value"][sample_idx]
    td = time_days[sample_idx]
    cfg = configs[sample_idx]

    ny, nx = pressure.shape[1], pressure.shape[2]
    n_slices = pressure.shape[0]

    p_min, p_max = pressure.min(), pressure.max()
    well_y, well_x = np.where(well_mask > 0)
    is_inj = well_mask[well_y[0], well_x[0]] == 1 if len(well_y) > 0 else False

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Sample {sample_idx} | Well ({well_x[0]},{well_y[0]}) "
        f"{'Injector' if is_inj else 'Producer'} BHP={cfg['bhp_MPa']:.1f} MPa",
        fontsize=13,
    )

    ax0.imshow(perm, cmap="viridis", origin="lower", aspect="equal")
    ax0.set_title("Permeability (mD)")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    fig.colorbar(ax0.images[0], ax=ax0, shrink=0.8)
    if len(well_x) > 0:
        ax0.plot(well_x[0], well_y[0], "rv", markersize=10)

    im1 = ax1.imshow(pressure[0], cmap="RdYlBu_r", origin="lower", aspect="equal",
                      vmin=p_min, vmax=p_max)
    title1 = ax1.set_title(f"Pressure | t = {td[0]:.1f} d")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="MPa")
    if len(well_x) > 0:
        ax1.plot(well_x[0], well_y[0], "kv", markersize=10)

    plt.tight_layout()

    def update(frame):
        im1.set_data(pressure[frame])
        title1.set_text(f"Pressure | t = {td[frame]:.1f} d")
        return [im1]

    ani = animation.FuncAnimation(fig, update, frames=n_slices, interval=1000 // fps, blit=True)

    gif_path = os.path.join(output_dir, f"sample_{sample_idx:03d}.gif")
    ani.save(gif_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training data as GIF")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 5, 42],
                        help="Sample indices to visualize")
    parser.add_argument("--fps", type=int, default=8, help="GIF frame rate")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, args.data_dir)

    npz = np.load(os.path.join(data_dir, "training_32x32.npz"))
    with open(os.path.join(data_dir, "training_32x32_configs.json")) as f:
        configs = json.load(f)["configs"]

    time_days = npz["time_days"]
    output_dir = os.path.join(data_dir, "gifs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Visualizing samples: {args.samples}")
    for idx in args.samples:
        render_gif(idx, npz, configs, time_days, output_dir, args.fps)
    print("Done!")


if __name__ == "__main__":
    main()
