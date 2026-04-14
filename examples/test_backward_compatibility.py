"""
向后兼容性测试

测试新架构是否与旧的使用方式兼容
"""
import numpy as np
from reservoirpy import (
    ReservoirSimulator,
    SinglePhaseProperties,
    StructuredMesh,
    WellManager,
    validate_well_config,
)
from reservoirpy.core.discretization import FVMDiscretizer

def test_old_style_usage():
    """测试旧风格的使用方法"""
    print("=== 向后兼容性测试 ===")
    
    # 1. 旧风格配置
    config = {
        'mesh': {
            'nx': 10, 'ny': 10, 'nz': 1,
            'dx': 10, 'dy': 10, 'dz': 10
        },
        'physics': {
            'type': 'single_phase',
            'permeability': 10.0,  # mD
            'porosity': 0.2,
            'viscosity': 0.001,     # Pa·s
            'compressibility': 1e-9  # 1/Pa
        },
        'wells': [
            {'location': [0, 0, 0],      # 井位置
            'control_type': 'bhp',      # 定井底流压控制
            'value': 100000,            # 井底流压值（Pa），通常低于地层压力
            'rw': 0.05,                 # 井筒半径
            'skin_factor': 0            # 表皮因子
             }  # 只保留一口生产井
        ],
        'simulation': {
            'dt': 3600,           # 时间步长(秒)
            'total_time': 36000,  # 减少模拟时间
            'initial_pressure': 300000 # 初始压力(Pa)
        },
        'output': {
            'output_interval': 2
        }
    }

    # 2. 使用新架构的ReservoirSimulator（但保持旧的接口）
    print("\n1. 创建模拟器（新架构，旧接口）")
    simulator = ReservoirSimulator(config_dict=config)
    
    # 3. 验证旧的属性仍然可用
    print(f"Wells: {len(simulator.wells)}")
    print(f"Mesh: {simulator.mesh.nx}x{simulator.mesh.ny}x{simulator.mesh.nz}")
    
    # 4. 旧风格的方法调用
    print("\n2. 测试旧方法")
    pressure_field = simulator.get_pressure_field()
    print(f"Initial pressure field shape: {pressure_field.shape}")
    
    cell_props = simulator.get_cell_properties(0, 5, 5)
    print(f"Cell properties keys: {list(cell_props.keys())}")
    
    # 5. 运行模拟
    print("\n3. 运行模拟")
    results = simulator.run_simulation()
    
    # 6. 验证结果格式兼容
    print(f"Result keys: {list(results.keys())}")
    print(f"Time steps: {len(results['time_history'])}")
    
    # 7. 旧风格的结果访问
    final_pressure = simulator.get_pressure_field()
    print(f"Final pressure range: {np.min(final_pressure)/1000:.1f} - {np.max(final_pressure)/1000:.1f} kPa")
    
    well_data = simulator.get_well_production(0)
    print(f"Well production data keys: {list(well_data.keys())}")
    
    print("\n✓ 向后兼容性测试通过")

def test_direct_component_usage():
    """测试直接使用组件的旧方法"""
    print("\n=== 直接组件使用测试 ===")
    
    # 1. 直接创建网格
    mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
    print(f"Created mesh: {mesh}")
    
    # 2. 直接创建物理属性
    physics_config = {
        'permeability': 10.0,
        'porosity': 0.2,
        'viscosity': 0.001,
        'compressibility': 1e-9
    }
    physics = SinglePhaseProperties(mesh, physics_config)
    print(f"Created physics: {physics}")
    
    # 3. 直接创建井管理器
    wells_config = [
        {'location': [0, 2, 2], 'control_type': 'bhp', 'value': 100000, 'rw': 0.05, 'skin_factor': 0}
    ]
    
    # 验证井配置
    for well_config in wells_config:
        valid = validate_well_config(well_config, mesh)
        print(f"Well config valid: {valid}")
    
    well_manager = WellManager(mesh, wells_config)
    
    # 初始化井
    permeability = physics.property_manager.properties['permeability']
    if isinstance(permeability, float):
        import numpy as np
        nx, ny, nz = mesh.grid_shape
        permeability = np.full((nz, ny, nx), permeability)
    
    well_manager.initialize_wells(permeability, physics.viscosity)
    print(f"Created well manager: {well_manager}")
    
    # 4. 直接使用FVM离散化器
    discretizer = FVMDiscretizer(mesh, physics)
    pressure = np.ones(mesh.n_cells) * 300000  # 300 kPa
    
    dt = 3600.0
    A, b = discretizer.discretize_single_phase(dt, pressure, well_manager)
    print(f"Assembled system: A shape {A.shape}, b shape {b.shape}")
    
    # 5. 求解
    from reservoirpy.core.linear_solver import LinearSolver
    
    solver = LinearSolver()
    pressure_new = solver.solve(A, b)
    print(f"Solved pressure range: {np.min(pressure_new)/1000:.1f} - {np.max(pressure_new)/1000:.1f} kPa")
    
    print("✓ 直接组件使用测试通过")

if __name__ == "__main__":
    try:
        test_old_style_usage()
        test_direct_component_usage()
        print("\n🎉 所有向后兼容性测试通过！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()