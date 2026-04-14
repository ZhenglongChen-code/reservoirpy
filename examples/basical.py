"""
基础功能演示
1. 网格生成与可视化
"""
from reservoirpy import StructuredMesh
from reservoirpy.visualization.plot_3d import create_3d_plotter
from reservoirpy.utils.io import mesh_to_vtk
import numpy as np

mesh = StructuredMesh(nz=5, ny=10, nx=10, dx=1.0, dy=1.0, dz=1.0)
print(mesh)

np.random.seed(42)
pressure = np.arange(mesh.n_cells) * 100 + 1000
saturation = np.arange(mesh.n_cells)
permeability = np.random.rand(mesh.n_cells) * 100 + 50

plotter_3d = create_3d_plotter(mesh)

pressure_plotter = plotter_3d.interactive_plot(pressure, "Pressure (Pa)", "viridis")
pressure_plotter.add_title("Interactive 3D Pressure Field")
pressure_plotter.show()

saturation_plotter = plotter_3d.plot_saturation_field_3d(saturation)
saturation_plotter.show()

permeability_plotter = plotter_3d.plot_permeability_field_3d(permeability, "3D Permeability Field")
permeability_plotter.show()

mesh_to_vtk(mesh, pressure, saturation, permeability, "results/simulation_results.vtk", file_format="ascii")
