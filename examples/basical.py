"""
基础功能演示
1. 网格生成与可视化
"""
from src.reservoirpy import StructuredMesh
from src.reservoirpy.visualization.plot_3d import create_3d_plotter
from src.reservoirpy.utils.io import mesh_to_vtk
import numpy as np

# 创建网格
mesh = StructuredMesh(nz=5, ny=10, nx=10, dx=1.0, dy=1.0, dz=1.0)
print(mesh)

# 生成示例数据
np.random.seed(42)
pressure = np.random.rand(mesh.n_cells) * 100 + 1000  # 压力数据 (1000-1100 Pa)
saturation = np.random.rand(mesh.n_cells)  # 饱和度数据 (0-1)
permeability = np.random.rand(mesh.n_cells) * 100 + 50  # 渗透率数据 (50-150 mD)

# 创建3D可视化器
plotter_3d = create_3d_plotter(mesh)

# 绘制3D压力场（交互式）
pressure_plotter = plotter_3d.interactive_plot(pressure, "Pressure (Pa)", "viridis")
pressure_plotter.add_title("Interactive 3D Pressure Field")

# 显示交互式窗口
pressure_plotter.show()
# pressure_plotter = plotter_3d.plot_pressure_field_3d(
#     pressure,
#     title="3D Pressure Field",
#     save_path="pressure_field.png"  # 指定保存路径
# )

# 绘制3D饱和度场
saturation_plotter = plotter_3d.plot_saturation_field_3d(saturation)
saturation_plotter.show()

# 绘制3D渗透率场
permeability_plotter = plotter_3d.plot_permeability_field_3d(permeability, "3D Permeability Field")
permeability_plotter.show()

# 保存油藏数值到VTK文件
# 保存为ASCII格式（可用文本编辑器查看）
mesh_to_vtk(mesh, pressure, saturation, permeability, "results/simulation_results.vtk", file_format="ascii")
