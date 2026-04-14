import pytest
import numpy as np
from reservoirpy.models.model_factory import ModelFactory
from reservoirpy.models.base_model import BaseModel
from reservoirpy.models.single_phase.single_phase_model import SinglePhaseModel
from reservoirpy.models.two_phase_impes import TwoPhaseIMPES
from reservoirpy.models.two_phase_fim import TwoPhaseFIM
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.physics.physics import SinglePhaseProperties, TwoPhaseProperties


class TestModelFactory:
    def test_all_models_registered(self):
        assert ModelFactory.is_registered('single_phase')
        assert ModelFactory.is_registered('two_phase_impes')
        assert ModelFactory.is_registered('two_phase_fim')

    def test_create_single_phase_model(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = SinglePhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'viscosity': 0.001, 'compressibility': 1e-9
        })
        model = ModelFactory.create_model('single_phase', mesh, physics, {})
        assert isinstance(model, SinglePhaseModel)
        assert isinstance(model, BaseModel)

    def test_create_impes_model(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = TwoPhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'viscosity': 0.001, 'compressibility': 1e-9,
            'oil_viscosity': 2e-3, 'water_viscosity': 1e-3
        })
        model = ModelFactory.create_model('two_phase_impes', mesh, physics, {})
        assert isinstance(model, TwoPhaseIMPES)
        assert isinstance(model, BaseModel)

    def test_create_fim_model(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = TwoPhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'viscosity': 0.001, 'compressibility': 1e-9,
            'oil_viscosity': 2e-3, 'water_viscosity': 1e-3
        })
        model = ModelFactory.create_model('two_phase_fim', mesh, physics, {})
        assert isinstance(model, TwoPhaseFIM)
        assert isinstance(model, BaseModel)

    def test_unknown_model_type(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = SinglePhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'viscosity': 0.001, 'compressibility': 1e-9
        })
        with pytest.raises(ValueError):
            ModelFactory.create_model('unknown', mesh, physics, {})


class TestSinglePhaseModel:
    @pytest.fixture
    def model(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = SinglePhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'viscosity': 0.001, 'compressibility': 1e-9
        })
        return SinglePhaseModel(mesh, physics, {})

    def test_state_variables(self, model):
        assert model.get_state_variables() == ['pressure']

    def test_initialize_state(self, model):
        state = model.initialize_state({'initial_pressure': 30e6})
        assert 'pressure' in state
        assert np.all(state['pressure'] == pytest.approx(30e6))

    def test_validate_solution(self, model):
        good_state = {'pressure': np.full(25, 30e6)}
        assert model.validate_solution(good_state) is True

        bad_state = {'pressure': np.full(25, np.nan)}
        assert model.validate_solution(bad_state) is False

        neg_state = {'pressure': np.full(25, -1.0)}
        assert model.validate_solution(neg_state) is False


class TestTwoPhaseIMPES:
    @pytest.fixture
    def model(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = TwoPhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'viscosity': 0.001, 'compressibility': 1e-9,
            'oil_viscosity': 2e-3, 'water_viscosity': 1e-3
        })
        return TwoPhaseIMPES(mesh, physics, {})

    def test_state_variables(self, model):
        vars = model.get_state_variables()
        assert 'pressure' in vars
        assert 'saturation' in vars

    def test_initialize_state(self, model):
        state = model.initialize_state({
            'initial_pressure': 30e6,
            'initial_saturation': 0.2
        })
        assert 'pressure' in state
        assert 'saturation' in state
        assert np.all(state['saturation'] == pytest.approx(0.2))


class TestTwoPhaseFIM:
    @pytest.fixture
    def model(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        physics = TwoPhaseProperties(mesh, {
            'permeability': 100.0, 'porosity': 0.2,
            'viscosity': 0.001, 'compressibility': 1e-9,
            'oil_viscosity': 2e-3, 'water_viscosity': 1e-3
        })
        return TwoPhaseFIM(mesh, physics, {})

    def test_assemble_raises_not_implemented(self, model):
        state = {'pressure': np.full(25, 30e6), 'saturation': np.full(25, 0.2)}
        with pytest.raises(NotImplementedError):
            model.assemble_system(86400.0, state, None)
