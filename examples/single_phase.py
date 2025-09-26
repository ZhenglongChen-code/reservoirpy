"""
单相流1D算例, 问题模型，瞬态达西单相流
网格数量：$(nx, ny, nz) = (100, 1, 1)$；
该网格表示在一个长度方向上的 100 个单元；
单元长度：$\Delta x = 1$ m；
时间步长：$\Delta t = 0.1$ s；
初始压力分布：$P(x, 0) = 10^6$ Pa（均匀分布）；
左边界为定压边界：$P_{left}(t) = 10^6$ Pa；
右边界为定压边界：$P_{right}(t) = 10^5$ Pa；
渗透率：$k = 10^{-12}$ m²；
黏度：$\mu = 10^{-3}$ Pa·s；
孔隙度：$\phi = 0.2$；
源汇项：$q = 0$（无注入或生产）。
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from reservoirpy import (
    ReservoirSimulator,
    SinglePhaseProperties,
    StructuredMesh,
    WellManager,
    validate_well_config,
)
from reservoirpy.core.discretization import FVMDiscretizer
from reservoirpy.visualization.plot_3d import create_3d_plotter
# 1. 使用字典配置
config = {
    'mesh': {
        'nx': 10, 'ny': 1, 'nz': 1,
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
        {'location': [0, 0, 0], 'control_type': 'bhp', 'value': 2000000}  # 只保留一口注入井
    ],
    'simulation': {
        'dt': 720,           # 时间步长(秒)
        'total_time': 3600,  # 减少模拟时间
        'initial_pressure': 3000000 # 初始压力(Pa)
    }
}

# 2. 生成网格
mesh_config = config['mesh']
mesh = StructuredMesh(**mesh_config)
print(mesh)

# 3. 生成物理场
physics_config = config['physics']
physics = SinglePhaseProperties(mesh, physics_config)
print(physics)

# plotter_3d = create_3d_plotter(mesh)
# permeability_plotter = plotter_3d.plot_permeability_field_3d(physics.permeability[:, :, :, 0], "3D Permeability Field")
# permeability_plotter.show()

# 4. 初始化井管理器 (WellManager)
# 验证井配置是否有效
for well_config in config['wells']:
    if not validate_well_config(well_config, mesh):
        print("Invalid well configuration:", well_config)
    else:
        print("Valid well configuration:", well_config)

well_manager = WellManager(mesh, config['wells'])
print(well_manager)

# 初始化井（计算产能指数）
# 修改为使用属性管理器获取渗透率
permeability = physics.property_manager.properties['permeability']
if isinstance(permeability, float):
    # 如果是均匀渗透率场，创建一个合适的数组
    import numpy as np
    nx, ny, nz = mesh.grid_shape
    permeability = np.full((nz, ny, nx), permeability)

well_manager.initialize_wells(permeability, physics.viscosity)

# 5. FVM 离散化
discretizer = FVMDiscretizer(mesh, physics)
pressure = np.ones(mesh.n_cells) * config['simulation']['initial_pressure']
A, b = discretizer.discretize_single_phase(config['simulation']['dt'], pressure, well_manager)
print(A.shape, b.shape)

# 简化模拟循环，只运行几个时间步
for t in range(100):  # 只运行5个时间步进行测试
    print(f"Time step {t}")
    # 6. 模拟
    pressure_new = discretizer.solve_linear_system(A, b)
    # 7. 输出结果
    print("Pressure:", pressure_new)
    
    # 检查是否有nan值
    if np.any(np.isnan(pressure_new)):
        print("NaN detected! Breaking simulation.")
        break
        
    # 8. A, b 更新
    A, b = discretizer.discretize_single_phase(config['simulation']['dt'], pressure_new, well_manager)
    pressure = pressure_new.copy()

plotter_3d = create_3d_plotter(mesh)
pv_plotter = plotter_3d.plot_pressure_field_3d(pressure_new, "3D Pressure Field")
pv_plotter.show()



