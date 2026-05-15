import numpy as np
import pytest

jax = pytest.importorskip("jax")

from reservoirpy.core.well_model import WellManager
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.models.jax_single_phase import (
    JaxSinglePhaseCG,
    interpolate_to_fine,
    remap_well_positions,
    resample_perm_field,
)
from reservoirpy.models.single_phase.single_phase_model import SinglePhaseModel
from reservoirpy.physics.physics import SinglePhaseProperties


def test_jax_single_phase_matches_scipy_single_step():
    nx = ny = 8
    dx = dy = 50.0
    dz = 10.0
    dt = 86400.0
    initial_pressure = 30e6
    permeability_mD = np.linspace(80.0, 180.0, nx * ny).reshape(ny, nx)
    wells_config = [
        {
            "location": [0, ny // 2, nx // 2],
            "control_type": "bhp",
            "value": 25e6,
            "rw": 0.1,
            "skin_factor": 0.0,
        }
    ]

    jax_solver = JaxSinglePhaseCG(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dz=dz,
        permeability_mD=permeability_mD,
        porosity=0.2,
        viscosity=1e-3,
        compressibility=1e-9,
        wells_config=wells_config,
        cg_tolerance=1e-12,
        cg_maxiter=500,
    )
    jax_pressure, info = jax_solver.solve_timestep(
        np.full((ny, nx), initial_pressure), dt
    )

    mesh = StructuredMesh(nx=nx, ny=ny, nz=1, dx=dx, dy=dy, dz=dz)
    physics = SinglePhaseProperties(
        mesh,
        {
            "type": "single_phase",
            "permeability": permeability_mD.reshape(1, ny, nx),
            "porosity": 0.2,
            "viscosity": 1e-3,
            "compressibility": 1e-9,
        },
    )
    well_manager = WellManager(mesh, wells_config)
    well_manager.initialize_wells(
        physics.property_manager.properties["permeability"], physics.viscosity
    )
    scipy_model = SinglePhaseModel(
        mesh, physics, {"linear_solver": {"method": "direct"}}
    )
    scipy_state = scipy_model.solve_timestep(
        dt, {"pressure": np.full(nx * ny, initial_pressure)}, well_manager
    )
    scipy_pressure = scipy_state["pressure"].reshape(ny, nx)

    assert info.iterations < 500
    assert np.allclose(jax_pressure, scipy_pressure, rtol=1e-9, atol=1e-2)


def test_jax_single_phase_default_grid_is_64x64():
    solver = JaxSinglePhaseCG(wells_config=[])
    pressure = solver.initialize_pressure()
    assert pressure.shape == (64, 64)


class TestResamplePermField:
    def test_homogeneous_field_unchanged(self):
        perm = np.full((1, 64, 64), 100.0)
        coarse = resample_perm_field(perm, 16, 16)
        assert coarse.shape == (1, 16, 16)
        np.testing.assert_allclose(coarse, 100.0)

    def test_geometric_mean_upscaling(self):
        block = np.array([[1.0, 4.0], [9.0, 36.0]])
        perm = block.reshape(1, 2, 2)
        coarse = resample_perm_field(perm, 1, 1)
        expected = np.exp(np.mean(np.log(block)))
        np.testing.assert_allclose(coarse.ravel(), [expected], rtol=1e-12)

    def test_2d_input(self):
        perm = np.full((64, 64), 50.0)
        coarse = resample_perm_field(perm, 32, 32)
        assert coarse.shape == (32, 32)
        np.testing.assert_allclose(coarse, 50.0)

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError):
            resample_perm_field(np.zeros((10, 10)), 3, 3)


class TestRemapWellPositions:
    def test_identity_mapping(self):
        wells = [{"location": [0, 4, 8], "control_type": "bhp", "value": 1e6}]
        out = remap_well_positions(wells, 16, 16, 16, 16)
        assert out[0]["location"] == [0, 4, 8]

    def test_fine_to_coarse_center_well(self):
        wells = [{"location": [0, 32, 32], "control_type": "bhp", "value": 1e6}]
        out = remap_well_positions(wells, 64, 64, 16, 16)
        assert out[0]["location"] == [0, 8, 8]

    def test_fine_to_coarse_corner_well(self):
        wells = [{"location": [0, 0, 0], "control_type": "bhp", "value": 1e6}]
        out = remap_well_positions(wells, 64, 64, 16, 16)
        assert out[0]["location"] == [0, 0, 0]

    def test_preserves_other_keys(self):
        wells = [{"location": [0, 10, 10], "control_type": "bhp",
                  "value": 1e6, "rw": 0.1, "skin_factor": 2.0}]
        out = remap_well_positions(wells, 64, 64, 16, 16)
        assert out[0]["control_type"] == "bhp"
        assert out[0]["value"] == 1e6
        assert out[0]["rw"] == 0.1


class TestInterpolateToFine:
    def test_constant_field(self):
        coarse = np.full((4, 4), 30.0)
        fine = interpolate_to_fine(coarse, 8, 8)
        assert fine.shape == (8, 8)
        np.testing.assert_allclose(fine, 30.0)

    def test_3d_temporal_stack(self):
        coarse = np.full((5, 4, 4), 30.0)
        fine = interpolate_to_fine(coarse, 8, 8)
        assert fine.shape == (5, 8, 8)
