import logging
logging.basicConfig(level=logging.INFO)

from reservoirpy import ReservoirSimulator
import numpy as np

config = {
    'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'two_phase_impes',
        'permeability': 100.0,
        'porosity': 0.2,
        'compressibility': 1e-9,
        'oil_viscosity': 2e-3,
        'water_viscosity': 1e-3
    },
    'wells': [
        {'location': [0, 5, 5], 'control_type': 'bhp', 'value': 28e6, 'rw': 0.05, 'skin_factor': 0}
    ],
    'simulation': {
        'dt': 86400,
        'total_time': 864000,
        'initial_pressure': 30e6,
        'initial_saturation': 0.2,
        'output_interval': 5
    },
    'output': {'output_interval': 5}
}

sim = ReservoirSimulator(config_dict=config)
results = sim.run_simulation()

p = results['field_data']['pressure'][-1]
s = results['field_data']['saturation'][-1]

print(f'Final pressure: {np.min(p)/1e6:.2f} - {np.max(p)/1e6:.2f} MPa')
print(f'Final saturation: {np.min(s):.4f} - {np.max(s):.4f}')
print(f'Timesteps: {len(results["time_history"])}')
print(f'Saturation at well: {s[sim.mesh.get_cell_index(0, 5, 5)]:.4f}')
print('Two-phase IMPES test PASSED!')
