import pytest
import numpy as np
from reservoirpy.physics.physics import (
    SinglePhaseProperties, TwoPhaseProperties, PropertyManager
)
from reservoirpy.mesh.mesh import StructuredMesh


class TestPropertyManager:
    @pytest.fixture
    def mesh(self):
        return StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)

    def test_homogeneous_permeability(self, mesh):
        pm = PropertyManager(mesh, {'permeability': 100.0, 'porosity': 0.2})
        k = pm.properties['permeability']
        assert isinstance(k, float)
        assert k == pytest.approx(100.0 * 9.869233e-16)

    def test_homogeneous_porosity(self, mesh):
        pm = PropertyManager(mesh, {'permeability': 100.0, 'porosity': 0.2})
        phi = pm.properties['porosity']
        assert isinstance(phi, float)
        assert phi == pytest.approx(0.2)

    def test_get_cell_property_scalar(self, mesh):
        pm = PropertyManager(mesh, {'permeability': 100.0, 'porosity': 0.2})
        phi = pm.get_cell_property(0, 'porosity')
        assert phi == pytest.approx(0.2)

    def test_get_cell_property_invalid(self, mesh):
        pm = PropertyManager(mesh, {'permeability': 100.0, 'porosity': 0.2})
        with pytest.raises(ValueError):
            pm.get_cell_property(0, 'nonexistent')


class TestSinglePhaseProperties:
    @pytest.fixture
    def physics(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        return SinglePhaseProperties(mesh, {
            'permeability': 100.0,
            'porosity': 0.2,
            'viscosity': 0.001,
            'compressibility': 1e-9
        })

    def test_viscosity(self, physics):
        assert physics.viscosity == pytest.approx(0.001)

    def test_compressibility(self, physics):
        assert physics.compressibility == pytest.approx(1e-9)

    def test_transmissibility(self, physics):
        trans = physics.get_transmissibility(0, 1, 'x')
        assert trans > 0

    def test_fluid_density(self, physics):
        rho = physics.get_fluid_density(30e6)
        assert rho > 0


class TestTwoPhaseProperties:
    @pytest.fixture
    def physics(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        return TwoPhaseProperties(mesh, {
            'permeability': 100.0,
            'porosity': 0.2,
            'viscosity': 0.001,
            'compressibility': 1e-9,
            'oil_viscosity': 2e-3,
            'water_viscosity': 1e-3
        })

    def test_viscosity_compatibility(self, physics):
        assert physics.viscosity == pytest.approx(1e-3)
        assert physics.mu_o == pytest.approx(2e-3)
        assert physics.mu_w == pytest.approx(1e-3)

    def test_relative_permeability(self, physics):
        kro = physics.get_relative_permeability(0.2, 'oil')
        krw = physics.get_relative_permeability(0.2, 'water')
        assert 0 <= kro <= 1
        assert 0 <= krw <= 1

    def test_capillary_pressure(self, physics):
        pc = physics.get_capillary_pressure(0.5)
        assert pc >= 0

    def test_invalid_phase(self, physics):
        with pytest.raises(ValueError):
            physics.get_relative_permeability(0.5, 'gas')
