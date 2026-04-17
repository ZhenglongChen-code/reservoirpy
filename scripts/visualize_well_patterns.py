"""
两相流井网模拟可视化

1. 两点井网：1注1采（对角）
2. 五点井网：1注4采（中心注入 + 四角生产）

均使用非均质渗透率场，初始 Sw=0，初始 P=30MPa，生成 GIF 动画对比

用法:
    uv run --extra geostat python scripts/visualize_well_patterns.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import TwoPhaseProperties
from reservoirpy.core.well_model import WellManager
from reservoirpy.models.two_phase_impes import TwoPhaseIMPES
from reservoirpy.geostatistics import PermeabilityGenerator
from reservoirpy.utils.units import uc

NX = NY = 32
NZ = 1
DX = DY = 50.0
DZ = 10.0
TOTAL_TIME_DAYS = 1095.0
TOTAL_TIME = uc.d_to_s(TOTAL_TIME_DAYS)
N_SNAPS = 50


def make_snap_times():
    fracs = np.concatenate([
        np.geomspace(0.001, 0.25, 34, endpoint=False),
        np.linspace(0.25, 1.0, 16),
    ])
    return fracs * TOTAL_TIME


def generate_perm(seed=42):
    gen = PermeabilityGenerator(nx=NX, ny=NY, dx=DX, dy=DY)
    return gen.generate(
        major_range=300, minor_range=150, azimuth=45,
        sill=0.6, nugget=0.02, vtype="exponential",
        n_realizations=1, seed=seed,
        mean_log_perm=2.0, std_log_perm=0.4,
    ).squeeze()


def run_simulation(perm, wells_config, seed=0):
    rng = np.random.default_rng(seed)
    mesh = StructuredMesh(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ)

    porosity = rng.uniform(0.15, 0.25)
    compressibility = 10 ** rng.uniform(-10, -8)
    oil_visc = uc.mpas_to_pas(rng.uniform(2.0, 8.0))
    water_visc = uc.mpas_to_pas(rng.uniform(0.5, 1.5))

    physics = TwoPhaseProperties(mesh, {
        "type": "two_phase_impes",
        "permeability": perm.reshape(1, NY, NX),
        "porosity": porosity,
        "compressibility": compressibility,
        "oil_viscosity": oil_visc,
        "water_viscosity": water_visc,
    })

    wm = WellManager(mesh, wells_config)
    k = physics.property_manager.properties["permeability"]
    if isinstance(k, float):
        k = np.full((1, NY, NX), k)
    wm.initialize_wells(k, physics.viscosity)

    model = TwoPhaseIMPES(mesh, physics, {"cfl_factor": 0.8})
    state = model.initialize_state({
        "initial_pressure": uc.mpa_to_pa(30.0),
        "initial_saturation": 0.0,
    })

    P_snaps = [state["pressure"].reshape(NY, NX).copy()]
    Sw_snaps = [state["saturation"].reshape(NY, NX).copy()]
    time_list = [0.0]
    current_time = 0.0

    while current_time < TOTAL_TIME:
        cfl_dt = model.compute_cfl_timestep(
            state["pressure"], state["saturation"], wm
        )
        actual_dt = min(cfl_dt, uc.d_to_s(10), TOTAL_TIME - current_time)
        if actual_dt <= 0:
            break
        state = model.solve_timestep(actual_dt, state, wm)
        model.update_properties(state)
        current_time += actual_dt
        P_snaps.append(state["pressure"].reshape(NY, NX).copy())
        Sw_snaps.append(state["saturation"].reshape(NY, NX).copy())
        time_list.append(current_time)

    td = np.array(time_list) / 86400
    return np.stack(P_snaps), np.stack(Sw_snaps), td, wm


def render_gif(perm, P, Sw, td, wm, title, gif_path, well_positions):
    P_MPa = uc.pa_to_mpa(P)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(title, fontsize=13)

    ax0 = axes[0]
    ax0.imshow(np.log10(perm), cmap="viridis", origin="lower", aspect="equal")
    ax0.set_title("log10 Perm (mD)")
    for wx, wy, is_inj in well_positions:
        ax0.plot(wx, wy, "b^" if is_inj else "rv", markersize=9)

    ax1 = axes[1]
    im1 = ax1.imshow(P_MPa[0], cmap="RdYlBu_r", origin="lower", aspect="equal",
                      vmin=P_MPa.min(), vmax=P_MPa.max())
    title1 = ax1.set_title(f"Pressure | t={td[0]:.1f} d")
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="MPa")
    for wx, wy, is_inj in well_positions:
        ax1.plot(wx, wy, "b^" if is_inj else "kv", markersize=7)

    ax2 = axes[2]
    im2 = ax2.imshow(Sw[0], cmap="Blues", origin="lower", aspect="equal",
                      vmin=0.0, vmax=1.0)
    title2 = ax2.set_title(f"Water Sat | t={td[0]:.1f} d")
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Sw")
    for wx, wy, is_inj in well_positions:
        ax2.plot(wx, wy, "b^" if is_inj else "rv", markersize=7)

    plt.tight_layout()

    def update(frame):
        im1.set_data(P_MPa[frame])
        title1.set_text(f"Pressure | t={td[frame]:.1f} d")
        im2.set_data(Sw[frame])
        title2.set_text(f"Water Sat | t={td[frame]:.1f} d")
        return [im1, im2]

    ani = animation.FuncAnimation(fig, update, frames=len(td), interval=80, blit=True)
    ani.save(gif_path, writer="pillow", fps=8)
    plt.close()
    print(f"  Saved {gif_path}")


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gifs"
    )
    os.makedirs(output_dir, exist_ok=True)

    perm = generate_perm(seed=42)

    inj_bhp = 40.0
    prod_bhp = 20.0

    # === 两点井网 ===
    print("=== 两点井网 (1注1采) ===")
    wells_2pt = [
        {"location": [0, 2, 2], "control_type": "bhp",
         "value": uc.mpa_to_pa(inj_bhp), "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY - 3, NX - 3], "control_type": "bhp",
         "value": uc.mpa_to_pa(prod_bhp), "rw": 0.1, "skin_factor": 0},
    ]
    well_pos_2pt = [(2, 2, True), (NX - 3, NY - 3, False)]
    P2, Sw2, td2, wm2 = run_simulation(perm, wells_2pt, seed=0)
    print(f"  P: [{uc.pa_to_mpa(P2).min():.1f}, {uc.pa_to_mpa(P2).max():.1f}] MPa")
    print(f"  Sw: [{Sw2.min():.3f}, {Sw2.max():.3f}]")
    render_gif(
        perm, P2, Sw2, td2, wm2,
        f"Two-Point | Inj(2,2) BHP={inj_bhp}MPa + Prod({NX-3},{NY-3}) BHP={prod_bhp}MPa",
        os.path.join(output_dir, "two_point_pattern.gif"),
        well_pos_2pt,
    )

    # === 五点井网 ===
    print("\n=== 五点井网 (1注4采) ===")
    cx, cy = NX // 2, NY // 2
    wells_5pt = [
        {"location": [0, cy, cx], "control_type": "bhp",
         "value": uc.mpa_to_pa(inj_bhp), "rw": 0.1, "skin_factor": 0},
        {"location": [0, 2, 2], "control_type": "bhp",
         "value": uc.mpa_to_pa(prod_bhp), "rw": 0.1, "skin_factor": 0},
        {"location": [0, 2, NX - 3], "control_type": "bhp",
         "value": uc.mpa_to_pa(prod_bhp), "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY - 3, 2], "control_type": "bhp",
         "value": uc.mpa_to_pa(prod_bhp), "rw": 0.1, "skin_factor": 0},
        {"location": [0, NY - 3, NX - 3], "control_type": "bhp",
         "value": uc.mpa_to_pa(prod_bhp), "rw": 0.1, "skin_factor": 0},
    ]
    well_pos_5pt = [
        (cx, cy, True),
        (2, 2, False),
        (NX - 3, 2, False),
        (2, NY - 3, False),
        (NX - 3, NY - 3, False),
    ]
    P5, Sw5, td5, wm5 = run_simulation(perm, wells_5pt, seed=1)
    print(f"  P: [{uc.pa_to_mpa(P5).min():.1f}, {uc.pa_to_mpa(P5).max():.1f}] MPa")
    print(f"  Sw: [{Sw5.min():.3f}, {Sw5.max():.3f}]")
    render_gif(
        perm, P5, Sw5, td5, wm5,
        f"Five-Point | Inj({cx},{cy}) BHP={inj_bhp}MPa + 4 Prods(corners) BHP={prod_bhp}MPa",
        os.path.join(output_dir, "five_point_pattern.gif"),
        well_pos_5pt,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
