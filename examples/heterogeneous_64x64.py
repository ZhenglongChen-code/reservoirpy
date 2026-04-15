"""
64×64 非均质渗透率场 — 单相流 & 两相流模拟

流程：
1. 使用 PermeabilityGenerator 生成 64×64 非均质渗透率场
2. 单相流模拟：五点法注采，观察压力场分布
3. 两相流 IMPES 模拟：水驱过程，观察饱和度前缘推进

井网：五点法
    - 注入井 (中心): BHP = 40 MPa
    - 生产井 (四角): BHP = 20 MPa
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.utils.units import uc


def generate_perm_field():
    """生成 64×64 非均质渗透率场"""
    from reservoirpy.geostatistics import PermeabilityGenerator

    print("=" * 60)
    print("  Step 1: 生成 64×64 非均质渗透率场")
    print("=" * 60)

    gen = PermeabilityGenerator(nx=64, ny=64, dx=50, dy=50)

    perm_field = gen.generate(
        major_range=120,
        minor_range=80,
        azimuth=30,
        sill=1.0,
        nugget=0.05,
        vtype='exponential',
        n_realizations=1,
        seed=2024,
        mean_log_perm=2.0,
        std_log_perm=0.6,
    )

    print(f"      渗透率范围: {perm_field.min():.2f} ~ {perm_field.max():.0f} mD")
    print(f"      几何平均:   {np.exp(np.mean(np.log(perm_field))):.1f} mD")
    print(f"      算术平均:   {np.mean(perm_field):.1f} mD")
    return perm_field


def run_single_phase(perm_field):
    """单相流模拟"""
    print("\n" + "=" * 60)
    print("  Step 2: 单相流模拟")
    print("=" * 60)

    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import SinglePhaseProperties
    from reservoirpy.core.well_model import WellManager
    from reservoirpy.models.single_phase.single_phase_model import SinglePhaseModel

    nx, ny = 64, 64
    mesh = StructuredMesh(nx=nx, ny=ny, nz=1, dx=50, dy=50, dz=10)

    physics = SinglePhaseProperties(mesh, {
        'type': 'single_phase',
        'permeability': perm_field.reshape(1, ny, nx),
        'porosity': 0.2,
        'viscosity': 1e-3,
        'compressibility': 1e-9,
    })

    wells_config = [
        {'location': [0, 32, 32], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(40), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 0, 0], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 0, 63], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 63, 0], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 63, 63], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
    ]

    well_manager = WellManager(mesh, wells_config)
    k_field = physics.property_manager.properties['permeability']
    if isinstance(k_field, float):
        k_field = np.full((1, ny, nx), k_field)
    well_manager.initialize_wells(k_field, physics.viscosity)

    model = SinglePhaseModel(mesh, physics, {})
    state = model.initialize_state({'initial_pressure': uc.mpa_to_pa(30)})

    total_time = uc.d_to_s(365)
    dt = uc.d_to_s(10)
    current_time = 0.0
    step = 0

    while current_time < total_time:
        state = model.solve_timestep(dt, state, well_manager)
        model.update_properties(state)
        current_time += dt
        step += 1
        if step % 5 == 0:
            print(f"      t={uc.s_to_d(current_time):.0f}d | "
                  f"P=[{uc.pa_to_mpa(state['pressure'].min()):.2f}, "
                  f"{uc.pa_to_mpa(state['pressure'].max()):.2f}] MPa")

    print(f"      单相流模拟完成！共 {step} 步")
    return state, mesh, well_manager


def run_two_phase(perm_field):
    """两相流 IMPES 模拟"""
    print("\n" + "=" * 60)
    print("  Step 3: 两相流 IMPES 水驱模拟")
    print("=" * 60)

    from reservoirpy.mesh.mesh import StructuredMesh
    from reservoirpy.physics.physics import TwoPhaseProperties
    from reservoirpy.core.well_model import WellManager
    from reservoirpy.models.two_phase_impes import TwoPhaseIMPES

    nx, ny = 64, 64
    mesh = StructuredMesh(nx=nx, ny=ny, nz=1, dx=50, dy=50, dz=10)

    physics = TwoPhaseProperties(mesh, {
        'type': 'two_phase_impes',
        'permeability': perm_field.reshape(1, ny, nx),
        'porosity': 0.2,
        'compressibility': 1e-9,
        'oil_viscosity': 5e-3,
        'water_viscosity': 1e-3,
    })

    wells_config = [
        {'location': [0, 32, 32], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(40), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 0, 0], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 0, 63], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 63, 0], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
        {'location': [0, 63, 63], 'control_type': 'bhp',
         'value': uc.mpa_to_pa(20), 'rw': 0.1, 'skin_factor': 0},
    ]

    well_manager = WellManager(mesh, wells_config)
    k_field = physics.property_manager.properties['permeability']
    if isinstance(k_field, float):
        k_field = np.full((1, ny, nx), k_field)
    well_manager.initialize_wells(k_field, physics.viscosity)

    model = TwoPhaseIMPES(mesh, physics, {'cfl_factor': 0.8})
    state = model.initialize_state({
        'initial_pressure': uc.mpa_to_pa(30),
        'initial_saturation': 0.2,
    })

    total_time = uc.d_to_s(5 * 365)
    dt_max = uc.d_to_s(30)
    current_time = 0.0
    step = 0

    history_time = []
    history_sw_avg = []
    history_sw_inj = []

    inj_cell = mesh.get_cell_index(0, 32, 32)

    snap_times = [30, 180, 365, 540, 730, 1095]
    snap_days_set = set()
    snapshots = {}

    while current_time < total_time:
        cfl_dt = model.compute_cfl_timestep(
            state['pressure'], state['saturation'], well_manager)
        actual_dt = min(dt_max, cfl_dt, total_time - current_time)
        actual_dt = max(actual_dt, 3600.0)

        state = model.solve_timestep(actual_dt, state, well_manager)
        model.update_properties(state)
        current_time += actual_dt
        step += 1

        days = uc.s_to_d(current_time)
        sw_avg = np.mean(state['saturation'])

        history_time.append(days)
        history_sw_avg.append(sw_avg)
        history_sw_inj.append(state['saturation'][inj_cell])

        for st in snap_times:
            if days >= st and st not in snap_days_set:
                snap_days_set.add(st)
                snapshots[st] = state['saturation'].copy()

        if step % 10 == 0:
            print(f"      t={days:>7.1f}d | dt={uc.s_to_d(actual_dt):.2f}d | "
                  f"<Sw>={sw_avg:.4f} | Sw_inj={state['saturation'][inj_cell]:.4f}")

    print(f"      两相流模拟完成！共 {step} 步")
    print(f"      最终 <Sw>={state['saturation'].mean():.4f}")

    return state, mesh, well_manager, history_time, history_sw_avg, history_sw_inj, snapshots


def plot_results(
    perm_field, sp_state, sp_mesh, tp_state, tp_mesh,
    history_time, history_sw_avg, history_sw_inj, snapshots,
):
    """绘制综合结果图"""
    print("\n" + "=" * 60)
    print("  Step 4: 绘制结果图")
    print("=" * 60)

    nx, ny = 64, 64

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.30)
    fig.suptitle('64×64 Heterogeneous Reservoir: Single-Phase & Two-Phase Simulation',
                 fontsize=16, fontweight='bold', y=0.98)

    well_kw = dict(markersize=8, zorder=5)
    well_positions = [(32, 32), (0, 0), (0, 63), (63, 0), (63, 63)]
    well_labels = ['Inj', 'P1', 'P2', 'P3', 'P4']
    well_colors = ['red', 'blue', 'blue', 'blue', 'blue']
    well_markers = ['^', 'v', 'v', 'v', 'v']

    # Row 0: 渗透率 + 单相流压力
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(np.log10(perm_field), origin='lower', cmap='YlOrBr_r', aspect='equal')
    ax0.set_title('log₁₀(K) [mD]')
    ax0.set_xlabel('X'); ax0.set_ylabel('Y')
    for (wx, wy), c, m in zip(well_positions, well_colors, well_markers):
        ax0.plot(wx, wy, m, color=c, **well_kw)
    plt.colorbar(im, ax=ax0, shrink=0.75)

    ax1 = fig.add_subplot(gs[0, 1])
    p_map = uc.pa_to_mpa(sp_state['pressure'].reshape(ny, nx))
    im = ax1.imshow(p_map, origin='lower', cmap='RdYlBu_r', aspect='equal')
    ax1.set_title('Single-Phase: Pressure (MPa)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    for (wx, wy), c, m in zip(well_positions, well_colors, well_markers):
        ax1.plot(wx, wy, m, color=c, **well_kw)
    plt.colorbar(im, ax=ax1, shrink=0.75)

    # 单相流压力等值线
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(p_map, origin='lower', cmap='RdYlBu_r', aspect='equal', alpha=0.6)
    cs = ax2.contour(p_map, levels=12, colors='k', linewidths=0.6, origin='lower')
    ax2.clabel(cs, inline=True, fontsize=7, fmt='%.1f')
    ax2.set_title('Pressure Contours (MPa)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    for (wx, wy), c, m in zip(well_positions, well_colors, well_markers):
        ax2.plot(wx, wy, m, color=c, **well_kw)

    # 单相流压力剖面
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(np.arange(ny), p_map[32, :], 'r-', lw=2, label='y=32 (through injector)')
    ax3.plot(np.arange(ny), p_map[:, 32], 'b--', lw=2, label='x=32')
    ax3.set_xlabel('Grid index')
    ax3.set_ylabel('Pressure (MPa)')
    ax3.set_title('Pressure Profiles')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Row 1: 两相流饱和度快照
    snap_keys = sorted(snapshots.keys())
    for idx, key in enumerate(snap_keys[:4]):
        ax = fig.add_subplot(gs[1, idx])
        sw_map = snapshots[key].reshape(ny, nx)
        im = ax.imshow(sw_map, origin='lower', cmap='Blues',
                       vmin=0.2, vmax=1.0, aspect='equal')
        ax.set_title(f'Sw @ {key}d')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        for (wx, wy), c, m in zip(well_positions, well_colors, well_markers):
            ax.plot(wx, wy, m, color=c, **well_kw)
        plt.colorbar(im, ax=ax, shrink=0.75)

    # Row 2: 最终结果 + 历史曲线
    ax_final_sw = fig.add_subplot(gs[2, 0])
    final_sw = tp_state['saturation'].reshape(ny, nx)
    im = ax_final_sw.imshow(final_sw, origin='lower', cmap='Blues',
                            vmin=0.2, vmax=1.0, aspect='equal')
    ax_final_sw.set_title('Final Sw (Two-Phase)')
    ax_final_sw.set_xlabel('X'); ax_final_sw.set_ylabel('Y')
    for (wx, wy), c, m in zip(well_positions, well_colors, well_markers):
        ax_final_sw.plot(wx, wy, m, color=c, **well_kw)
    plt.colorbar(im, ax=ax_final_sw, shrink=0.75)

    ax_dsw = fig.add_subplot(gs[2, 1])
    dsw = final_sw - 0.2
    im = ax_dsw.imshow(dsw, origin='lower', cmap='Reds',
                       vmin=0, vmax=max(dsw.max(), 0.01), aspect='equal')
    ax_dsw.set_title('ΔSw (Sweep Map)')
    ax_dsw.set_xlabel('X'); ax_dsw.set_ylabel('Y')
    for (wx, wy), c, m in zip(well_positions, well_colors, well_markers):
        ax_dsw.plot(wx, wy, m, color=c, **well_kw)
    plt.colorbar(im, ax=ax_dsw, shrink=0.75)

    ax_hist = fig.add_subplot(gs[2, 2])
    ax_hist.plot(history_time, history_sw_avg, 'b-', lw=2, label='Avg $S_w$')
    ax_hist.plot(history_time, history_sw_inj, 'r--', lw=1.5, label='$S_w$ Injector')
    ax_hist.axhline(y=0.2, color='gray', ls=':', alpha=0.5)
    ax_hist.set_xlabel('Time (days)')
    ax_hist.set_ylabel('Water Saturation')
    ax_hist.set_title('Saturation History')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    ax_tp_p = fig.add_subplot(gs[2, 3])
    tp_p = uc.pa_to_mpa(tp_state['pressure'].reshape(ny, nx))
    im = ax_tp_p.imshow(tp_p, origin='lower', cmap='RdYlBu_r', aspect='equal')
    ax_tp_p.set_title('Two-Phase: Pressure (MPa)')
    ax_tp_p.set_xlabel('X'); ax_tp_p.set_ylabel('Y')
    for (wx, wy), c, m in zip(well_positions, well_colors, well_markers):
        ax_tp_p.plot(wx, wy, m, color=c, **well_kw)
    plt.colorbar(im, ax=ax_tp_p, shrink=0.75)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'heterogeneous_64x64_result.png',
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"      结果图已保存至: {output_path}")
    plt.show()


def main():
    perm_field = generate_perm_field()
    perm_2d = perm_field.squeeze()
    sp_state, sp_mesh, sp_wm = run_single_phase(perm_field)
    tp_state, tp_mesh, tp_wm, ht, hsw, hsw_inj, snaps = run_two_phase(perm_field)
    plot_results(perm_2d, sp_state, sp_mesh, tp_state, tp_mesh,
                 ht, hsw, hsw_inj, snaps)


if __name__ == '__main__':
    main()
