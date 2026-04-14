import pytest
import numpy as np
from reservoirpy.core.well_model import Well, WellManager, validate_well_config
from reservoirpy.mesh.mesh import StructuredMesh


class TestWell:
    def test_well_creation(self):
        well = Well(location=[0, 2, 2], control_type='bhp', value=1e6)
        assert well.location == [0, 2, 2]
        assert well.control_type == 'bhp'
        assert well.value == 1e6
        assert well.rw == 0.05
        assert well.skin_factor == 0

    def test_compute_well_index(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        well = Well(location=[0, 2, 2], control_type='bhp', value=1e6)
        wi = well.compute_well_index(mesh, 1e-13, 0.001)
        assert wi > 0
        assert well.well_index is not None
        assert well.re is not None

    def test_compute_well_term_bhp(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        well = Well(location=[0, 2, 2], control_type='bhp', value=1e6)
        well.compute_well_index(mesh, 1e-13, 0.001)
        term = well.compute_well_term(30e6)
        assert term > 0

    def test_compute_well_term_rate(self):
        well = Well(location=[0, 2, 2], control_type='rate', value=0.001)
        term = well.compute_well_term(30e6)
        assert term == 0.001

    def test_well_term_without_index(self):
        well = Well(location=[0, 2, 2], control_type='bhp', value=1e6)
        with pytest.raises(ValueError):
            well.compute_well_term(30e6)


class TestWellManager:
    @pytest.fixture
    def mesh(self):
        return StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)

    def test_well_manager_creation(self, mesh):
        wells_config = [
            {'location': [0, 2, 2], 'control_type': 'bhp', 'value': 1e6},
            {'location': [0, 0, 0], 'control_type': 'rate', 'value': 0.001}
        ]
        wm = WellManager(mesh, wells_config)
        assert len(wm.wells) == 2

    def test_initialize_wells(self, mesh):
        wells_config = [
            {'location': [0, 2, 2], 'control_type': 'bhp', 'value': 1e6}
        ]
        wm = WellManager(mesh, wells_config)
        k = np.full((1, 5, 5), 1e-13)
        wm.initialize_wells(k, 0.001)
        assert wm.wells[0].well_index is not None
        assert wm.wells[0].well_index > 0

    def test_get_well_cells(self, mesh):
        wells_config = [
            {'location': [0, 2, 2], 'control_type': 'bhp', 'value': 1e6}
        ]
        wm = WellManager(mesh, wells_config)
        cells = wm.get_well_cells()
        assert len(cells) == 1
        assert cells[0] == mesh.get_cell_index(0, 2, 2)


class TestValidateWellConfig:
    def test_valid_config(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        config = {'location': [0, 2, 2], 'control_type': 'bhp', 'value': 1e6}
        assert validate_well_config(config, mesh) is True

    def test_invalid_location(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        config = {'location': [0, 10, 2], 'control_type': 'bhp', 'value': 1e6}
        assert validate_well_config(config, mesh) is False

    def test_invalid_control_type(self):
        mesh = StructuredMesh(nx=5, ny=5, nz=1, dx=10, dy=10, dz=10)
        config = {'location': [0, 2, 2], 'control_type': 'cutoff', 'value': 1e6}
        assert validate_well_config(config, mesh) is False
