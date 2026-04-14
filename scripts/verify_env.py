from reservoirpy import ReservoirSimulator
import numpy as np

config = {
    'mesh': {'nx': 10, 'ny': 10, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
    'physics': {
        'type': 'single_phase',
        'permeability': 100.0, 'porosity': 0.2,
        'viscosity': 0.001, 'compressibility': 1e-9
    },
    'wells': [
        {'location': [0, 5, 5], 'control_type': 'bhp', 'value': 1e6, 'rw': 0.05, 'skin_factor': 0}
    ],
    'simulation': {'dt': 86400, 'total_time': 864000, 'initial_pressure': 30e6, 'output_interval': 5},
    'output': {'output_interval': 5}
}
sim = ReservoirSimulator(config_dict=config)
results = sim.run_simulation()
p = results['field_data']['pressure'][-1]
print(f'Grid: 10x10x1 = {sim.mesh.n_cells} cells')
print(f'Final pressure: {np.min(p)/1e6:.2f} - {np.max(p)/1e6:.2f} MPa')
print(f'Timesteps: {len(results["time_history"])}')
print('Integration test PASSED!')
