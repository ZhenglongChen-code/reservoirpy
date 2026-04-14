import logging
logging.basicConfig(level=logging.WARNING)

from reservoirpy import ReservoirSimulator
from reservoirpy.models.two_phase_impes import TwoPhaseIMPES
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import TwoPhaseProperties
from reservoirpy.core.well_model import WellManager
from reservoirpy.core.output_manager import OutputManager
import numpy as np

mesh = StructuredMesh(nx=10, ny=10, nz=1, dx=10, dy=10, dz=10)
physics = TwoPhaseProperties(mesh, {
    'permeability': 100.0,
    'porosity': 0.2,
    'compressibility': 1e-9,
    'oil_viscosity': 5e-3,
    'water_viscosity': 1e-3
})

wells_config = [
    {'location': [0, 1, 1], 'control_type': 'bhp', 'value': 35e6, 'rw': 0.05, 'skin_factor': 0},
    {'location': [0, 8, 8], 'control_type': 'bhp', 'value': 25e6, 'rw': 0.05, 'skin_factor': 0}
]

well_manager = WellManager(mesh, wells_config)
k = np.full((1, 10, 10), 9.869e-14)
well_manager.initialize_wells(k, physics.viscosity)

model = TwoPhaseIMPES(mesh, physics, {'cfl_factor': 0.8})

state = model.initialize_state({
    'initial_pressure': 30e6,
    'initial_saturation': 0.2
})

output_manager = OutputManager({'output_interval': 1})

dt = 86400.0
total_time = 864000.0
current_time = 0.0
time_step = 0

output_manager.save_timestep(0, current_time, state, well_manager)

while current_time < total_time:
    cfl_dt = model.compute_cfl_timestep(state['pressure'], state['saturation'], well_manager)
    actual_dt = min(dt, cfl_dt, total_time - current_time)

    state = model.solve_timestep(actual_dt, state, well_manager)

    if not model.validate_solution(state):
        print(f"Invalid solution at timestep {time_step}")
        break

    model.update_properties(state)
    current_time += actual_dt
    time_step += 1

    if time_step % 5 == 0:
        output_manager.save_timestep(time_step, current_time, state, well_manager)
        p = state['pressure']
        s = state['saturation']
        print(f"Step {time_step:3d}: t={current_time/86400:.1f}d  "
              f"P=[{np.min(p)/1e6:.1f}, {np.max(p)/1e6:.1f}] MPa  "
              f"Sw=[{np.min(s):.3f}, {np.max(s):.3f}]  "
              f"dt_cfl={cfl_dt/86400:.2f}d")

p = state['pressure']
s = state['saturation']
inj_cell = mesh.get_cell_index(0, 1, 1)
prod_cell = mesh.get_cell_index(0, 8, 8)

print(f"\nFinal results:")
print(f"  Pressure: {np.min(p)/1e6:.2f} - {np.max(p)/1e6:.2f} MPa")
print(f"  Saturation: {np.min(s):.4f} - {np.max(s):.4f}")
print(f"  Injector Sw: {s[inj_cell]:.4f}")
print(f"  Producer Sw: {s[prod_cell]:.4f}")
print(f"  CFL dt: {cfl_dt/86400:.2f} days")

assert np.all(p > 0), "Pressure must be positive"
assert np.all(s >= 0) and np.all(s <= 1), "Saturation must be in [0,1]"
assert s[inj_cell] > 0.2, "Injector should have increased water saturation"
print("\nTwo-phase waterflood test PASSED!")
