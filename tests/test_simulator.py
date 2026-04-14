import pytest
import numpy as np
from reservoirpy.core.simulator import ReservoirSimulator


class TestReservoirSimulator:
    @pytest.fixture
    def single_phase_config(self):
        return {
            'mesh': {'nx': 5, 'ny': 5, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
            'physics': {
                'type': 'single_phase',
                'permeability': 100.0,
                'porosity': 0.2,
                'viscosity': 0.001,
                'compressibility': 1e-9
            },
            'wells': [
                {'location': [0, 2, 2], 'control_type': 'bhp', 'value': 1e6,
                 'rw': 0.05, 'skin_factor': 0}
            ],
            'simulation': {
                'dt': 86400, 'total_time': 864000,
                'initial_pressure': 30e6, 'output_interval': 5
            },
            'output': {'output_interval': 5}
        }

    def test_simulator_creation(self, single_phase_config):
        sim = ReservoirSimulator(config_dict=single_phase_config)
        assert sim.mesh.n_cells == 25
        assert len(sim.wells) == 1

    def test_run_simulation(self, single_phase_config):
        sim = ReservoirSimulator(config_dict=single_phase_config)
        results = sim.run_simulation()
        assert 'time_history' in results
        assert 'field_data' in results
        assert 'pressure' in results['field_data']
        assert len(results['time_history']) > 1

    def test_pressure_positive(self, single_phase_config):
        sim = ReservoirSimulator(config_dict=single_phase_config)
        results = sim.run_simulation()
        final_pressure = results['field_data']['pressure'][-1]
        assert np.all(final_pressure > 0)

    def test_get_pressure_field(self, single_phase_config):
        sim = ReservoirSimulator(config_dict=single_phase_config)
        sim.run_simulation()
        p = sim.get_pressure_field()
        assert len(p) == 25

    def test_get_model_info(self, single_phase_config):
        sim = ReservoirSimulator(config_dict=single_phase_config)
        info = sim.get_model_info()
        assert 'model_type' in info
        assert info['model_type'] == 'SinglePhaseModel'

    def test_missing_config(self):
        with pytest.raises(ValueError):
            ReservoirSimulator()

    def test_invalid_physics_type(self):
        config = {
            'mesh': {'nx': 5, 'ny': 5, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
            'physics': {'type': 'invalid', 'permeability': 100},
            'simulation': {'dt': 86400, 'total_time': 864000}
        }
        with pytest.raises(ValueError):
            ReservoirSimulator(config_dict=config)

    def test_well_data_in_output(self, single_phase_config):
        sim = ReservoirSimulator(config_dict=single_phase_config)
        results = sim.run_simulation()
        assert 'well_data' in results
        assert len(results['well_data']) > 0
