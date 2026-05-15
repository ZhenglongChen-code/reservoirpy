import numpy as np
import pytest

jax = pytest.importorskip("jax")

from reservoirpy.core.well_model import WellManager
from reservoirpy.mesh.mesh import StructuredMesh
from reservoirpy.models.jax_single_phase import JaxSinglePhaseCG
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
