import pytest
import numpy as np
from reservoirpy.models.two_phase_impes import TwoPhaseIMPES
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import TwoPhaseProperties
from reservoirpy.core.well_model import WellManager


class TestTwoPhaseIMPES:
    @pytest.fixture
    def model_and_wells(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = TwoPhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'compressibility': 1e-9,
            'oil_viscosity': 5e-3, 'water_viscosity': 1e-3
        })
        model = TwoPhaseIMPES(mesh, physics, {'cfl_factor': 0.8})

        wells_config = [
            {'location': [0, 0, 0], 'control_type': 'bhp',
             'value': 35e6, 'rw': 0.05, 'skin_factor': 0},
            {'location': [0, 4, 4], 'control_type': 'bhp',
             'value': 25e6, 'rw': 0.05, 'skin_factor': 0}
        ]
        well_manager = WellManager(mesh, wells_config)
        k = np.full((1, 5, 5), 9.869e-14)
        well_manager.initialize_wells(k, physics.viscosity)

        return model, well_manager

    def test_state_variables(self, model_and_wells):
        model, _ = model_and_wells
        vars = model.get_state_variables()
        assert 'pressure' in vars
        assert 'saturation' in vars

    def test_initialize_state(self, model_and_wells):
        model, _ = model_and_wells
        state = model.initialize_state({
            'initial_pressure': 30e6,
            'initial_saturation': 0.2
        })
        assert np.all(state['pressure'] == pytest.approx(30e6))
        assert np.all(state['saturation'] == pytest.approx(0.2))

    def test_solve_timestep(self, model_and_wells):
        model, well_manager = model_and_wells
        state = model.initialize_state({
            'initial_pressure': 30e6,
            'initial_saturation': 0.2
        })
        new_state = model.solve_timestep(86400.0, state, well_manager)
        assert np.all(new_state['pressure'] > 0)
        assert np.all(new_state['saturation'] >= 0)
        assert np.all(new_state['saturation'] <= 1)

    def test_waterflood_injection(self, model_and_wells):
        model, well_manager = model_and_wells
        state = model.initialize_state({
            'initial_pressure': 30e6,
            'initial_saturation': 0.2
        })

        for _ in range(5):
            state = model.solve_timestep(86400.0, state, well_manager)
            model.update_properties(state)

        inj_cell = model.mesh.get_cell_index(0, 0, 0)
        assert state['saturation'][inj_cell] > 0.2

    def test_cfl_timestep(self, model_and_wells):
        model, well_manager = model_and_wells
        state = model.initialize_state({
            'initial_pressure': 30e6,
            'initial_saturation': 0.2
        })
        state = model.solve_timestep(86400.0, state, well_manager)
        cfl_dt = model.compute_cfl_timestep(
            state['pressure'], state['saturation'], well_manager)
        assert cfl_dt > 0
        assert cfl_dt < np.inf

    def test_validate_solution(self, model_and_wells):
        model, _ = model_and_wells
        good = {'pressure': np.full(25, 30e6), 'saturation': np.full(25, 0.3)}
        assert model.validate_solution(good) is True

        bad_p = {'pressure': np.full(25, -1.0), 'saturation': np.full(25, 0.3)}
        assert model.validate_solution(bad_p) is False

        bad_s = {'pressure': np.full(25, 30e6), 'saturation': np.full(25, 2.0)}
        result = model.validate_solution(bad_s)
        assert result is True
        assert np.all(bad_s['saturation'] <= 1.0)

    def test_assemble_system(self, model_and_wells):
        model, well_manager = model_and_wells
        state = model.initialize_state({
            'initial_pressure': 30e6,
            'initial_saturation': 0.2
        })
        A, b = model.assemble_system(86400.0, state, well_manager)
        assert A.shape == (25, 25)
        assert b.shape == (25,)
        assert np.all(np.diag(A.toarray()) > 0)


class TestTwoPhaseViaSimulator:
    def test_impes_simulation(self):
        from reservoirpy import ReservoirSimulator

        config = {
            'mesh': {'nx': 5, 'ny': 5, 'nz': 1, 'dx': 10, 'dy': 10, 'dz': 10},
            'physics': {
                'type': 'two_phase_impes',
                'permeability': 100.0, 'porosity': 0.2,
                'compressibility': 1e-9,
                'oil_viscosity': 5e-3, 'water_viscosity': 1e-3
            },
            'wells': [
                {'location': [0, 0, 0], 'control_type': 'bhp',
                 'value': 35e6, 'rw': 0.05, 'skin_factor': 0},
                {'location': [0, 4, 4], 'control_type': 'bhp',
                 'value': 25e6, 'rw': 0.05, 'skin_factor': 0}
            ],
            'simulation': {
                'dt': 86400, 'total_time': 864000,
                'initial_pressure': 30e6,
                'initial_saturation': 0.2,
                'output_interval': 5
            },
            'output': {'output_interval': 5}
        }

        sim = ReservoirSimulator(config_dict=config)
        results = sim.run_simulation()

        assert 'pressure' in results['field_data']
        assert 'saturation' in results['field_data']
        assert len(results['time_history']) > 1
