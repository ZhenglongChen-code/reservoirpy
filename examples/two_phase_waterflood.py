"""
两相流水驱模拟示例 — 20×20 非均质渗透率场

使用 SGSIM（序贯高斯模拟）生成的非均质渗透率场，
通过 IMPES 方法模拟水驱过程。

网格：20 × 20 × 1（二维油藏）
渗透率场：permeability-sgsim20x20-2100samples.txt 第一个样例
井位：
    - 注入井 (左上角, x=0, y=0): BHP = 40 MPa
    - 生产井 (右下角, x=19, y=19): BHP = 20 MPa

物理参数：
    - 孔隙度 φ = 0.2
    - 水粘度 μ_w = 1.0 mPa·s
    - 油粘度 μ_o = 5.0 mPa·s
    - 初始含水饱和度 Sw = 0.2（束缚水）
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def load_sgsim_perm(filepath: str, sample_index: int = 0) -> np.ndarray:
    data = np.loadtxt(filepath)
    n_samples = data.shape[0]
    if sample_index >= n_samples:
        raise ValueError(f"sample_index {sample_index} out of range (total {n_samples})")
    return data[sample_index].reshape(20, 20)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    perm_file = os.path.join(base_dir, '..', 'permeability-sgsim20x20-2100samples.txt')

    print("=" * 60)
    print("  两相流 IMPES 模拟 — 非均质渗透率场水驱")
    print("=" * 60)

    # ── 1. 加载非均质渗透率场 ──────────────────────────────
    print("\n[1/6] 加载 SGSIM 渗透率场...")
    perm_mD = load_sgsim_perm(perm_file, sample_index=0)
    perm_3d = perm_mD.reshape(1, 20, 20)
    print(f"      渗透率范围: {perm_mD.min():.2f} ~ {perm_mD.max():.2f} mD")
    print(f"      几何平均:   {np.exp(np.mean(np.log(perm_mD))):.2f} mD")
    print(f"      算术平均:   {np.mean(perm_mD):.2f} mD")

    # ── 2. 创建网格 ────────────────────────────────────────
    print("\n[2/6] 创建结构化网格...")
    from reservoirpy.mesh.mesh import StructuredMesh

    nx, ny, nz = 20, 20, 1
    dx, dy, dz = 10.0, 10.0, 10.0

    mesh = StructuredMesh(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    print(f"      网格尺寸: {nx} × {ny} × {nz}")
    print(f"      油藏面积: {nx*dx:.0f}m × {ny*dy:.0f}m")

    # ── 3. 创建物理属性（非均质渗透率）──────────────────────
    print("\n[3/6] 创建两相流物理属性...")
    from reservoirpy.physics.physics import TwoPhaseProperties

    physics_config = {
        'type': 'two_phase_impes',
        'permeability': perm_3d,
        'porosity': 0.2,
        'compressibility': 1e-9,
        'oil_viscosity': 5e-3,
        'water_viscosity': 1e-3,
    }

    physics = TwoPhaseProperties(mesh, physics_config)
    print(f"      水粘度 μ_w: {physics.mu_w * 1000:.1f} mPa·s")
    print(f"      油粘度 μ_o: {physics.mu_o * 1000:.1f} mPa·s")

    # ── 4. 设置井（注入井 + 生产井）─────────────────────────
    print("\n[4/6] 配置井模型...")
    from reservoirpy.core.well_model import WellManager

    inj_x, inj_y = 0, 0
    prod_x, prod_y = nx - 1, ny - 1

    wells_config = [
        {
            'location': [0, inj_y, inj_x],
            'control_type': 'bhp',
            'value': 40e6,
            'rw': 0.1,
            'skin_factor': 0,
        },
        {
            'location': [0, prod_y, prod_x],
            'control_type': 'bhp',
            'value': 20e6,
            'rw': 0.1,
            'skin_factor': 0,
        },
    ]

    well_manager = WellManager(mesh, wells_config)
    permeability_field = physics.property_manager.properties['permeability']
    if isinstance(permeability_field, float):
        permeability_field = np.full((nz, ny, nx), permeability_field)
    well_manager.initialize_wells(permeability_field, physics.viscosity)

    print(f"      注入井: ({inj_x}, {inj_y}), BHP=40 MPa, "
          f"k={perm_mD[inj_y, inj_x]:.2f} mD, WI={well_manager.wells[0].well_index:.4e}")
    print(f"      生产井: ({prod_x}, {prod_y}), BHP=20 MPa, "
          f"k={perm_mD[prod_y, prod_x]:.2f} mD, WI={well_manager.wells[1].well_index:.4e}")

    # ── 5. 运行 IMPES 模拟 ─────────────────────────────────
    print("\n[5/6] 运行 IMPES 水驱模拟...")
    from reservoirpy.models.two_phase_impes import TwoPhaseIMPES

    model = TwoPhaseIMPES(mesh, physics, {'cfl_factor': 0.8})

    state = model.initialize_state({
        'initial_pressure': 30e6,
        'initial_saturation': 0.2,
    })

    total_time = 5 * 365 * 86400.0
    dt_max = 30 * 86400.0
    current_time = 0.0
    step_count = 0

    history_time = []
    history_sw_avg = []
    history_sw_inj = []
    history_sw_prod = []

    snap_times = [30, 180, 365, 730, 1095, 1825]
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
        step_count += 1

        days_elapsed = current_time / 86400.0
        sw_avg = np.mean(state['saturation'])

        inj_cell = mesh.get_cell_index(0, inj_y, inj_x)
        prod_cell = mesh.get_cell_index(0, prod_y, prod_x)

        history_time.append(days_elapsed)
        history_sw_avg.append(sw_avg)
        history_sw_inj.append(state['saturation'][inj_cell])
        history_sw_prod.append(state['saturation'][prod_cell])

        for st in snap_times:
            if days_elapsed >= st and st not in snap_days_set:
                snap_days_set.add(st)
                snapshots[st] = state['saturation'].copy()

        if step_count % 10 == 0:
            print(f"      t={days_elapsed:>7.1f}d | "
                  f"dt={actual_dt/86400:.1f}d | "
                  f"<Sw>={sw_avg:.4f} | "
                  f"Sw_inj={state['saturation'][inj_cell]:.4f} | "
                  f"Sw_prod={state['saturation'][prod_cell]:.4f}")

    final_sw = state['saturation']
    final_p = state['pressure']

    print(f"\n      模拟完成！共 {step_count} 个时间步")
    print(f"      最终平均含水饱和度: <Sw> = {final_sw.mean():.4f}")
    print(f"      最终含水饱和度范围: [{final_sw.min():.4f}, {final_sw.max():.4f}]")
    print(f"      最终压力范围: [{final_p.min()/1e6:.2f}, {final_p.max()/1e6:.2f}] MPa")

    # ── 6. 可视化结果 ──────────────────────────────────────
    print("\n[6/6] 绘制结果图...")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    fig.suptitle('Two-Phase Waterflood (IMPES, 20×20 Heterogeneous K)',
                 fontsize=15, fontweight='bold')

    well_marker_kw = dict(markersize=10, zorder=5)

    ax_perm = fig.add_subplot(gs[0, 0])
    im = ax_perm.imshow(perm_mD, origin='lower', cmap='YlOrBr_r', aspect='equal')
    ax_perm.set_title('Permeability (mD)')
    ax_perm.set_xlabel('X'); ax_perm.set_ylabel('Y')
    ax_perm.plot(inj_x, inj_y, 'r^', label='Inj', **well_marker_kw)
    ax_perm.plot(prod_x, prod_y, 'bv', label='Prod', **well_marker_kw)
    ax_perm.legend(fontsize=8, loc='upper right')
    plt.colorbar(im, ax=ax_perm, shrink=0.8)

    snap_keys = sorted(snapshots.keys())
    for idx, key in enumerate(snap_keys[:3]):
        ax = fig.add_subplot(gs[0, idx + 1])
        sw_map = snapshots[key].reshape(ny, nx)
        im = ax.imshow(sw_map, origin='lower', cmap='Blues',
                       vmin=0.2, vmax=1.0, aspect='equal')
        ax.set_title(f'Sw @ {key}d')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.plot(inj_x, inj_y, 'r^', **well_marker_kw)
        ax.plot(prod_x, prod_y, 'bv', **well_marker_kw)
        plt.colorbar(im, ax=ax, shrink=0.8)

    for idx, key in enumerate(snap_keys[3:]):
        ax = fig.add_subplot(gs[1, idx])
        sw_map = snapshots[key].reshape(ny, nx)
        im = ax.imshow(sw_map, origin='lower', cmap='Blues',
                       vmin=0.2, vmax=1.0, aspect='equal')
        ax.set_title(f'Sw @ {key}d')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.plot(inj_x, inj_y, 'r^', **well_marker_kw)
        ax.plot(prod_x, prod_y, 'bv', **well_marker_kw)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax_p = fig.add_subplot(gs[1, 2])
    im = ax_p.imshow(final_p.reshape(ny, nx) / 1e6, origin='lower',
                     cmap='RdYlBu_r', aspect='equal')
    ax_p.set_title('Pressure (MPa)')
    ax_p.set_xlabel('X'); ax_p.set_ylabel('Y')
    ax_p.plot(inj_x, inj_y, 'r^', **well_marker_kw)
    ax_p.plot(prod_x, prod_y, 'bv', **well_marker_kw)
    plt.colorbar(im, ax=ax_p, shrink=0.8)

    ax_sw_final = fig.add_subplot(gs[1, 3])
    im = ax_sw_final.imshow(final_sw.reshape(ny, nx), origin='lower',
                            cmap='Blues', vmin=0.2, vmax=1.0, aspect='equal')
    ax_sw_final.set_title(f'Sw Final ({days_elapsed:.0f}d)')
    ax_sw_final.set_xlabel('X'); ax_sw_final.set_ylabel('Y')
    ax_sw_final.plot(inj_x, inj_y, 'r^', **well_marker_kw)
    ax_sw_final.plot(prod_x, prod_y, 'bv', **well_marker_kw)
    plt.colorbar(im, ax=ax_sw_final, shrink=0.8)

    ax_hist = fig.add_subplot(gs[2, :2])
    ax_hist.plot(history_time, history_sw_avg, 'b-', lw=2, label='Avg $S_w$')
    ax_hist.plot(history_time, history_sw_inj, 'r--', lw=1.5, label='$S_w$ Injector')
    ax_hist.plot(history_time, history_sw_prod, 'g--', lw=1.5, label='$S_w$ Producer')
    ax_hist.axhline(y=0.2, color='gray', ls=':', alpha=0.5)
    ax_hist.set_xlabel('Time (days)')
    ax_hist.set_ylabel('Water Saturation')
    ax_hist.set_title('Saturation History')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    ax_dsw = fig.add_subplot(gs[2, 2:])
    dsw = final_sw - 0.2
    im = ax_dsw.imshow(dsw.reshape(ny, nx), origin='lower', cmap='Reds',
                       vmin=0, vmax=max(dsw.max(), 0.01), aspect='equal')
    ax_dsw.set_title('$\\Delta S_w$ (Sweep Map)')
    ax_dsw.set_xlabel('X'); ax_dsw.set_ylabel('Y')
    ax_dsw.plot(inj_x, inj_y, 'r^', **well_marker_kw)
    ax_dsw.plot(prod_x, prod_y, 'bv', **well_marker_kw)
    plt.colorbar(im, ax=ax_dsw, shrink=0.8)

    output_path = os.path.join(base_dir, 'two_phase_waterflood_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"      结果图已保存至: {output_path}")
    plt.show()


if __name__ == '__main__':
    main()
